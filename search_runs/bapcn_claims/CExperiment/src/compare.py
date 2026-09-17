"""Summarize C ablations and compare with an optional verified TVT baseline."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


VARIANTS = ("baseline", "no_agreement", "fused_alignment")
TAGS = ("best", "final")
METRICS = ("accuracy", "macro_f1")
MATCHED_FIELDS = (
    "profile", "split", "net_params", "initial_digest", "warmup_digest",
    "model_source_sha256", "torch_version", "cuda_version", "device", "selection_policy",
)


def _manifest(path, variant, external=False):
    if not path.is_file():
        raise ValueError(f"Missing manifest: {path}")
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    if result.get("status") != "complete":
        raise ValueError(f"Manifest status must be complete: {path}")
    ablation = result.get("ablation", "baseline" if external else None)
    if ablation != variant:
        raise ValueError(f"Wrong or missing ablation for {variant}: {path}")
    if result.get("prototype_space") != "band":
        raise ValueError(f"C variants and baseline must use prototype_space='band': {path}")
    for key in MATCHED_FIELDS:
        if key not in result:
            raise ValueError(f"Missing manifest field {key}: {path}")
    for key in ("profile", "split", "net_params"):
        if not isinstance(result[key], dict) or not result[key]:
            raise ValueError(f"Invalid manifest field {key}: {path}")
    if result["profile"].get("protocol") != "tvt":
        raise ValueError(f"C comparison requires TVT: {path}")
    if result["selection_policy"] != "validation_macro_f1_after_warmup":
        raise ValueError(f"Unsupported selection_policy: {path}")
    for key in ("initial_digest", "warmup_digest", "model_source_sha256"):
        value = result[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"Invalid SHA256 field {key}: {path}")
    _test_subjects(result)
    return result


def _test_subjects(manifest):
    subjects = manifest["split"].get("test")
    if (not isinstance(subjects, list) or not subjects
            or any(type(subject) is not int or subject < 1 for subject in subjects)
            or len(set(subjects)) != len(subjects)):
        raise ValueError("split.test must contain distinct positive integer test subjects")
    return set(subjects)


def _matched_value(manifest, field):
    if field == "profile":
        return {key: value for key, value in manifest[field].items() if key != "dataset_path"}
    return manifest[field]


def _metrics(path, expected_subjects):
    if not path.is_file():
        raise ValueError(f"Missing results: {path}")
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                subject = int(row["subject_id"])
                values = {metric: float(row[metric]) for metric in METRICS}
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid metric or subject at {path}:{line}") from error
            if subject in rows:
                raise ValueError(f"Duplicate test subject {subject}: {path}")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError(f"Metrics must be finite and within [0, 1] at {path}:{line}")
            rows[subject] = values
    if set(rows) != expected_subjects:
        raise ValueError(f"Test subjects differ from manifest: {path}")
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_run(run_root, baseline_dir=None, variants=None):
    """Check recorded pairing, then summarize test-subject metrics (ddof=0).

    Positive deltas mean baseline minus ablation. An optional baseline_dir
    points to B's baseline/tvt. Checkpoint tensors are not loaded here.
    """
    run_root = Path(run_root).resolve()
    if variants is None:
        variants = [name for name in VARIANTS if (run_root / name / "tvt" / "manifest.json").is_file()]
    else:
        variants = list(variants)
    if not variants or len(set(variants)) != len(variants) or any(name not in VARIANTS for name in variants):
        raise ValueError(f"Specify distinct available C variants from {VARIANTS}")
    roots = {variant: run_root / variant / "tvt" for variant in variants}
    if baseline_dir is not None:
        if "baseline" in roots:
            raise ValueError("Choose a local baseline or baseline_dir, not both")
        roots["baseline"] = Path(baseline_dir).resolve()
    elif "baseline" not in roots and (run_root / "baseline" / "tvt" / "manifest.json").is_file():
        roots["baseline"] = run_root / "baseline" / "tvt"
    roots = {variant: roots[variant] for variant in VARIANTS if variant in roots}
    manifests, inputs = {}, []
    for variant, root in roots.items():
        path = root / "manifest.json"
        manifests[variant] = _manifest(path, variant, external=variant == "baseline" and baseline_dir is not None)
        inputs.append(path)
    reference_name = next(iter(manifests))
    reference = manifests[reference_name]
    for variant, manifest in manifests.items():
        for field in MATCHED_FIELDS:
            if _matched_value(reference, field) != _matched_value(manifest, field):
                raise ValueError(f"Paired manifest mismatch: {field} ({reference_name} vs {variant})")
    subjects = _test_subjects(reference)
    has_baseline = "baseline" in roots
    summary_rows, paired_rows = [], []
    for tag in TAGS:
        by_variant = {}
        for variant, root in roots.items():
            path = root / "metrics" / f"test_{tag}.csv"
            by_variant[variant] = _metrics(path, subjects)
            inputs.append(path)
        for variant, rows in by_variant.items():
            for metric in METRICS:
                values = [rows[subject][metric] for subject in sorted(subjects)]
                row = {"variant": variant, "checkpoint": tag, "metric": metric,
                       "n_subjects": len(subjects), "mean": statistics.fmean(values),
                       "std": statistics.pstdev(values)}
                if has_baseline:
                    baseline = [by_variant["baseline"][subject][metric] for subject in sorted(subjects)]
                    delta = [left - right for left, right in zip(baseline, values)]
                    row.update(baseline_mean=statistics.fmean(baseline),
                               baseline_std=statistics.pstdev(baseline),
                               delta_mean=statistics.fmean(delta), delta_std=statistics.pstdev(delta))
                summary_rows.append(row)
            if has_baseline and variant != "baseline":
                for subject in sorted(subjects):
                    row = {"variant": variant, "checkpoint": tag, "subject_id": subject}
                    for metric in METRICS:
                        baseline = by_variant["baseline"][subject][metric]
                        value = rows[subject][metric]
                        row.update({f"baseline_{metric}": baseline,
                                    f"variant_{metric}": value, f"delta_{metric}": baseline - value})
                    paired_rows.append(row)
    checks = {
        "status": "passed" if has_baseline else "baseline_not_provided",
        "run_root": str(run_root), "variant_roots": {key: str(value) for key, value in roots.items()},
        "baseline_dir": str(roots["baseline"]) if has_baseline else None,
        "delta_direction": "baseline_minus_variant" if has_baseline else None,
        "metric_unit": "fraction", "aggregation": "equal_weight_per_test_subject", "std_ddof": 0,
        "std_interpretation": "test_subject_variability_not_seed_stability",
        "checkpoint_tags": list(TAGS), "test_subjects": sorted(subjects),
        "matched_fields": list(MATCHED_FIELDS) if len(roots) > 1 else [],
        "ignored_profile_fields": ["dataset_path"],
        "state_parity": ("training_recorded_initial_and_warmup_digests_equal" if len(roots) > 1
                         else "not_checked_single_variant"),
        "checkpoint_tensors_loaded": False,
        "initial_digest": reference["initial_digest"], "warmup_digest": reference["warmup_digest"],
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }
    destination = run_root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / "summary.csv"
    comparison = destination / "comparison.csv"
    check_path = destination / "comparison_checks.json"
    _write_csv(summary, summary_rows)
    outputs = {"summary_csv": str(summary), "checks_json": str(check_path)}
    if paired_rows:
        _write_csv(comparison, paired_rows)
        outputs["comparison_csv"] = str(comparison)
    elif comparison.exists():
        comparison.unlink()
    check_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    args = parser.parse_args()
    try:
        outputs = compare_run(args.run_root, args.baseline_dir, args.variants)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
