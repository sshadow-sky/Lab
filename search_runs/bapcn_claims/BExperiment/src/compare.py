"""Compare complete TVT baseline/fused pairs after strict protocol checks."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


VARIANTS = {"baseline": "band", "fused_proto": "fused"}
TAGS = ("best", "final")
METRICS = ("accuracy", "macro_f1")
MATCHED_FIELDS = (
    "profile", "split", "initial_digest", "warmup_digest", "net_params",
    "model_source_sha256",
)


def _manifest(path):
    if not path.is_file():
        raise ValueError(f"Missing manifest: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid manifest {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    for key in (*MATCHED_FIELDS, "prototype_space"):
        if key not in result:
            raise ValueError(f"Missing manifest field {key}: {path}")
    for key in ("profile", "split", "net_params"):
        if not isinstance(result[key], dict) or not result[key]:
            raise ValueError(f"Invalid manifest field {key}: {path}")
    for key in ("initial_digest", "warmup_digest", "model_source_sha256"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Invalid manifest field {key}: {path}")
    return result


def _test_subjects(manifest):
    subjects = manifest["split"].get("test")
    if (not isinstance(subjects, list) or not subjects
            or any(type(subject) is not int or subject < 1 for subject in subjects)
            or len(set(subjects)) != len(subjects)):
        raise ValueError("split.test must contain distinct positive integer test subjects")
    return set(subjects)


def _metrics(path, expected_subjects):
    if not path.is_file():
        raise ValueError(f"Missing paired results: {path}")
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                subject = int(row["subject_id"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid subject_id at {path}:{line}") from error
            if subject in rows:
                raise ValueError(f"Duplicate subject {subject}: {path}")
            values = {}
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid {metric} at {path}:{line}") from error
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{metric} must be finite and within [0, 1] at {path}:{line}")
                values[metric] = value
            rows[subject] = values
    actual_subjects = set(rows)
    if actual_subjects != expected_subjects:
        raise ValueError(
            f"Test subjects differ from manifest in {path}: "
            f"missing={sorted(expected_subjects - actual_subjects)}, "
            f"unexpected={sorted(actual_subjects - expected_subjects)}"
        )
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_run(run_root, output_dir=None):
    """Return output paths; positive deltas mean baseline exceeds fused.

    State parity uses digests recorded by training. Checkpoint tensors are not
    loaded here. Reported standard deviations use ddof=0 over test subjects;
    they describe subject variability, not variation over repeated seeds.
    """
    run_root = Path(run_root).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else run_root / "comparison"
    manifests = {}
    input_paths = []
    for variant, expected_space in VARIANTS.items():
        path = run_root / variant / "tvt" / "manifest.json"
        manifest = _manifest(path)
        if manifest["prototype_space"] != expected_space:
            raise ValueError(f"Wrong prototype_space for {variant}: expected {expected_space!r}")
        manifests[variant] = manifest
        input_paths.append(path)
    for key in MATCHED_FIELDS:
        if manifests["baseline"][key] != manifests["fused_proto"][key]:
            raise ValueError(f"Paired manifest mismatch: {key}")
    subjects = _test_subjects(manifests["baseline"])
    paired_rows, summary_rows = [], []
    for tag in TAGS:
        by_variant = {}
        for variant in VARIANTS:
            path = run_root / variant / "tvt" / "metrics" / f"test_{tag}.csv"
            by_variant[variant] = _metrics(path, subjects)
            input_paths.append(path)
        for subject in sorted(subjects):
            row = {"checkpoint": tag, "subject_id": subject}
            for metric in METRICS:
                baseline = by_variant["baseline"][subject][metric]
                fused = by_variant["fused_proto"][subject][metric]
                row[f"baseline_{metric}"] = baseline
                row[f"fused_{metric}"] = fused
                row[f"delta_{metric}"] = baseline - fused
            paired_rows.append(row)
        for metric in METRICS:
            baseline = [by_variant["baseline"][s][metric] for s in sorted(subjects)]
            fused = [by_variant["fused_proto"][s][metric] for s in sorted(subjects)]
            deltas = [b - f for b, f in zip(baseline, fused)]
            summary_rows.append({
                "checkpoint": tag, "metric": metric, "n_subjects": len(subjects),
                "baseline_mean": statistics.fmean(baseline),
                "baseline_std": statistics.pstdev(baseline),
                "fused_mean": statistics.fmean(fused),
                "fused_std": statistics.pstdev(fused),
                "delta_mean": statistics.fmean(deltas),
                "delta_std": statistics.pstdev(deltas),
            })
    checks = {
        "status": "passed", "run_root": str(run_root),
        "delta_direction": "baseline_minus_fused", "metric_unit": "fraction",
        "aggregation": "equal_weight_per_test_subject", "std_ddof": 0,
        "std_interpretation": "test_subject_variability_not_seed_stability",
        "checkpoint_tags": list(TAGS), "test_subjects": sorted(subjects),
        "matched_fields": list(MATCHED_FIELDS),
        "prototype_spaces": VARIANTS,
        "state_parity": "training_recorded_initial_and_warmup_digests_equal",
        "checkpoint_tensors_loaded": False,
        "initial_digest": manifests["baseline"]["initial_digest"],
        "warmup_digest": manifests["baseline"]["warmup_digest"],
        "input_sha256": {
            str(path.relative_to(run_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_paths
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    comparison = destination / "comparison.csv"
    summary = destination / "summary.csv"
    check_path = destination / "comparison_checks.json"
    _write_csv(comparison, paired_rows)
    _write_csv(summary, summary_rows)
    check_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return {"comparison_csv": str(comparison), "summary_csv": str(summary), "checks_json": str(check_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        outputs = compare_run(args.run_root, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
