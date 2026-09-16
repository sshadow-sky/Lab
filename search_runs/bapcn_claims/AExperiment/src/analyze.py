"""Analyze held-out SEED LOSO band evidence; never select or alter checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


BANDS = ("delta", "theta", "alpha", "beta", "gamma")
BAND_LABELS = (r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$")
BAND_COLORS = ("#4477AA", "#EE9944", "#228833", "#CC6677", "#AA66AA")
ARRAY_KEYS = ("band_posteriors", "band_weights", "consensus_posteriors", "labels", "subject_ids")
EXPORT_IDENTITY_FIELDS = ("protocol", "dataset", "epoch", "batch_size", "prototype_space",
                          "temperature", "epsilon", "run_id", "seed")


def _probabilities(name, values):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")


def summarize_correlation(heterogeneity, gain):
    x, y = np.asarray(heterogeneity, dtype=float), np.asarray(gain, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Correlation requires matching finite one-dimensional arrays")
    result = {"n_subjects": len(x), "pearson_r": None, "pearson_p_two_sided": None,
              "ols_slope": None, "ols_intercept": None, "unavailable_reason": None}
    x_constant = len(x) == 0 or np.ptp(x) == 0
    y_constant = len(y) == 0 or np.ptp(y) == 0
    if len(x) >= 2 and not x_constant:
        centered = x - x.mean()
        slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
        result.update(ols_slope=slope, ols_intercept=float(y.mean() - slope * x.mean()))
    if len(x) < 3:
        result["unavailable_reason"] = "fewer_than_three_subjects"
    elif x_constant:
        result["unavailable_reason"] = "constant_heterogeneity"
    elif y_constant:
        result["unavailable_reason"] = "constant_gain"
    else:
        correlation = stats.pearsonr(x, y)
        result.update(pearson_r=float(correlation.statistic),
                      pearson_p_two_sided=float(correlation.pvalue))
    return result


def analyze_arrays(band_posteriors, band_weights, consensus_posteriors, labels, subject_ids,
                   expected_subjects=15, allow_partial=False):
    """Return sample-pooled subject statistics, with accuracy in percent and gaps in pp."""
    band = np.asarray(band_posteriors, dtype=np.float64)
    weights = np.asarray(band_weights, dtype=np.float64)
    consensus = np.asarray(consensus_posteriors, dtype=np.float64)
    labels, subject_ids = np.asarray(labels), np.asarray(subject_ids)
    if band.ndim != 3 or band.shape[1] != 5 or band.shape[2] < 2 or band.shape[0] == 0:
        raise ValueError("band_posteriors must have nonempty shape [N, 5, K], K >= 2")
    n, _, classes = band.shape
    if weights.shape != (n, 5) or consensus.shape != (n, classes):
        raise ValueError("band_weights and consensus_posteriors have incompatible shapes")
    if labels.shape != (n,) or subject_ids.shape != (n,):
        raise ValueError("labels and subject_ids must have shape [N]")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(subject_ids.dtype, np.integer):
        raise ValueError("labels and subject_ids must be integer arrays")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("labels must be zero-based class indices")
    if expected_subjects < 1 or np.any(subject_ids < 1) or np.any(subject_ids > expected_subjects):
        raise ValueError("subject_ids must be one-based within expected_subjects")
    for name, values in (("band_posteriors", band), ("band_weights", weights),
                         ("consensus_posteriors", consensus)):
        _probabilities(name, values)
    reconstructed = np.sum(weights[..., None] * band, axis=1)
    if not np.allclose(consensus, reconstructed, rtol=1e-5, atol=1e-6):
        discrepancy = float(np.max(np.abs(consensus - reconstructed)))
        raise ValueError(f"Exported consensus differs from weighted band posteriors (max error {discrepancy})")

    subjects = np.unique(subject_ids)
    missing = sorted(set(range(1, expected_subjects + 1)) - set(subjects.tolist()))
    if missing and not allow_partial:
        raise ValueError(f"Incomplete LOSO export: missing subjects {missing}; use --allow-partial only for diagnostics")
    band_predictions, consensus_predictions = band.argmax(axis=-1), consensus.argmax(axis=-1)
    band_rows, rank_rows, subject_rows, accuracies = [], [], [], []
    hardest_counts = np.zeros(5, dtype=int)
    hardest_fractional = np.zeros(5, dtype=float)
    hardest_unambiguous = np.zeros(5, dtype=int)
    for subject in subjects:
        selected = subject_ids == subject
        n_subject = int(selected.sum())
        correct = (band_predictions[selected] == labels[selected, None]).sum(axis=0)
        accuracy = correct.astype(float) / n_subject * 100
        consensus_correct = int((consensus_predictions[selected] == labels[selected]).sum())
        consensus_accuracy = consensus_correct / n_subject * 100
        ranks = stats.rankdata(correct, method="average")
        order = np.argsort(correct, kind="stable")
        hardest = correct == correct.min()
        hardest_counts += hardest
        hardest_fractional += hardest / hardest.sum()
        if hardest.sum() == 1:
            hardest_unambiguous += hardest
        rows = []
        for index, band_name in enumerate(BANDS):
            row = {"subject_id": int(subject), "band_index": index, "band": band_name,
                   "n_samples": n_subject, "n_correct": int(correct[index]),
                   "accuracy_pct": float(accuracy[index]), "rank_average": float(ranks[index]),
                   "is_hardest": bool(hardest[index]), "hardest_tie_count": int(hardest.sum())}
            rows.append(row)
            band_rows.append(row)
        for position, index in enumerate(order, start=1):
            rank_rows.append({"subject_id": int(subject), "sorted_position": position,
                              **rows[index]})
        mean_accuracy = float(accuracy.mean())
        subject_rows.append({"subject_id": int(subject), "n_samples": n_subject,
                             "consensus_n_correct": consensus_correct,
                             "consensus_accuracy_pct": consensus_accuracy,
                             "mean_band_accuracy_pct": mean_accuracy,
                             "heterogeneity_pp": float(np.std(accuracy, ddof=0)),
                             "consensus_gain_pp": consensus_accuracy - mean_accuracy,
                             "hardest_bands": "|".join(np.asarray(BANDS)[hardest]),
                             "hardest_tie_count": int(hardest.sum()),
                             "hardest_to_easiest": "|".join(np.asarray(BANDS)[order])})
        accuracies.append(accuracy)
    hardest_rows = [{"band_index": i, "band": name,
                     "subjects_including_ties": int(hardest_counts[i]),
                     "fractional_subject_count": float(hardest_fractional[i]),
                     "unambiguous_subject_count": int(hardest_unambiguous[i]),
                     "n_subjects": len(subjects)} for i, name in enumerate(BANDS)]
    correlation = summarize_correlation([s["heterogeneity_pp"] for s in subject_rows],
                                        [s["consensus_gain_pp"] for s in subject_rows])
    return {"band_rows": band_rows, "rank_rows": rank_rows, "subject_rows": subject_rows,
            "hardest_rows": hardest_rows, "band_accuracy": np.asarray(accuracies),
            "subject_ids": subjects, "correlation": correlation, "n_samples": n,
            "expected_subjects": int(expected_subjects), "missing_subjects": missing,
            "partial": bool(missing), "consensus_max_abs_error": float(np.max(np.abs(consensus - reconstructed)))}


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(input_root, checkpoint_tag=None):
    """Load one run's per-fold exports; checkpoint tags are never pooled together."""
    root = Path(input_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    paths = [root] if root.is_file() else sorted(root.rglob("predictions.npz"))
    sources = []
    for path in paths:
        metadata_path = path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Export metadata must be a JSON object: {metadata_path}")
        metadata_tag = metadata.get("checkpoint_tag")
        if metadata_tag is not None and (not isinstance(metadata_tag, str) or not metadata_tag):
            raise ValueError(f"Invalid checkpoint_tag in {metadata_path}")
        with np.load(path, allow_pickle=False) as data:
            embedded_tag = None
            if "checkpoint_tag" in data.files:
                raw_tag = data["checkpoint_tag"]
                if raw_tag.ndim != 0 or raw_tag.dtype.kind not in "US":
                    raise ValueError(f"NPZ checkpoint_tag must be a scalar string: {path}")
                embedded_tag = raw_tag.item()
                if isinstance(embedded_tag, bytes):
                    embedded_tag = embedded_tag.decode("utf-8")
                if not embedded_tag:
                    raise ValueError(f"NPZ checkpoint_tag cannot be empty: {path}")
        if metadata_tag is not None and embedded_tag is not None and metadata_tag != embedded_tag:
            raise ValueError(f"checkpoint_tag mismatch between NPZ ({embedded_tag}) and metadata ({metadata_tag}): {path}")
        tag = embedded_tag or metadata_tag or path.parent.name
        if checkpoint_tag is None or tag == checkpoint_tag:
            sources.append((path, tag, metadata))
    if not sources:
        raise ValueError(f"No predictions.npz exports found for checkpoint tag {checkpoint_tag!r} under {root}")
    tags = sorted({tag for _, tag, _ in sources})
    if len(tags) != 1:
        raise ValueError(f"Multiple checkpoint tags found: {tags}; specify --checkpoint-tag")
    pieces = {key: [] for key in ARRAY_KEYS}
    provenance, missing_id_paths = [], []
    seen_sample_ids, export_identity = set(), {}
    for path, _, metadata in sources:
        for field in EXPORT_IDENTITY_FIELDS:
            value = metadata.get(field)
            if value is None:
                continue
            if field in export_identity and export_identity[field] != value:
                raise ValueError(f"Incompatible export {field}: {export_identity[field]!r} versus {value!r} in {path}")
            export_identity[field] = value
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(ARRAY_KEYS) - set(data.files))
            if missing:
                raise ValueError(f"Missing arrays {missing} in {path}")
            labels = data["labels"]
            if labels.ndim != 1 or len(labels) == 0:
                raise ValueError(f"labels must have nonempty shape [N]: {path}")
            count = len(labels)
            for key in ARRAY_KEYS:
                values = data[key]
                if values.ndim == 0 or len(values) != count:
                    raise ValueError(f"Array {key} has inconsistent sample count in {path}")
                pieces[key].append(values)
            if "sample_ids" in data.files:
                sample_ids = data["sample_ids"]
                if sample_ids.shape != (count,) or sample_ids.dtype.kind not in "USiu":
                    raise ValueError(f"sample_ids must be a string or integer array of shape [N]: {path}")
                ids = sample_ids.astype(str).tolist()
                unique_ids = set(ids)
                if "" in unique_ids:
                    raise ValueError(f"Empty sample_ids in {path}")
                if len(unique_ids) != count or seen_sample_ids.intersection(unique_ids):
                    raise ValueError(f"Duplicate sample_ids within or across exports: {path}; select one run and disjoint sample chunks")
                seen_sample_ids.update(unique_ids)
            else:
                missing_id_paths.append(str(path))
        provenance.append({"path": str(path), "sha256": _file_digest(path),
                           "n_samples": count, "metadata": metadata})
    arrays = {key: np.concatenate(values, axis=0) for key, values in pieces.items()}
    id_status = "complete" if not missing_id_paths else ("partial" if seen_sample_ids else "unavailable")
    return arrays, {"input_root": str(root), "checkpoint_tag": tags[0], "sources": provenance,
                    "export_identity": export_identity, "sample_id_validation": id_status,
                    "files_without_sample_ids": missing_id_paths}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure, stem):
    for extension in ("png", "pdf"):
        figure.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight", facecolor="white")


def plot_results(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    subjects = result["subject_ids"]
    accuracy = result["band_accuracy"]
    positions = np.arange(len(subjects))
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
             "legend.fontsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        figure, axes = plt.subplots(figsize=(10.2, 3.1), layout="constrained")
        order = np.argsort(accuracy, axis=1, kind="stable")
        for rank in range(5):
            indices = order[:, rank]
            axes.bar(positions + (rank - 2) * 0.155, accuracy[np.arange(len(subjects)), indices],
                     width=0.145, color=[BAND_COLORS[i] for i in indices], linewidth=0)
        axes.set(xticks=positions, xticklabels=[str(s) for s in subjects], xlabel="Subject",
                 ylabel="Accuracy (%)", ylim=(0, 104), xlim=(-0.6, len(subjects) - 0.4))
        axes.set_yticks(np.arange(0, 101, 20))
        axes.set_axisbelow(True)
        axes.grid(axis="y", color="#E3E3E3", linewidth=0.5)
        axes.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(BAND_COLORS, BAND_LABELS)],
                    ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18),
                    handlelength=1.2, columnspacing=2.1)
        if result["partial"]:
            figure.suptitle(f"Partial LOSO: {len(subjects)}/{result['expected_subjects']} subjects", fontsize=10)
        _save_figure(figure, output_dir / "fig1_subject_band_ranking")
        plt.close(figure)

        figure, (heatmap, scatter) = plt.subplots(1, 2, figsize=(10.2, 3.35), layout="constrained",
                                                 gridspec_kw={"width_ratios": [1.65, 1]})
        image = heatmap.imshow(accuracy.T, cmap="cividis", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        heatmap.set(xticks=positions, xticklabels=[str(s) for s in subjects], yticks=np.arange(5),
                    yticklabels=BAND_LABELS, xlabel="Subject", ylabel="Band")
        heatmap.set_title("(a)", loc="left")
        heatmap.tick_params(length=0)
        figure.colorbar(image, ax=heatmap, fraction=0.04, pad=0.025, label="Accuracy (%)")
        x = np.asarray([s["heterogeneity_pp"] for s in result["subject_rows"]])
        y = np.asarray([s["consensus_gain_pp"] for s in result["subject_rows"]])
        scatter.scatter(x, y, s=34, c="#4477AA", edgecolors="white", linewidths=0.5, zorder=3)
        for subject, xi, yi in zip(subjects, x, y):
            scatter.annotate(str(subject), (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        correlation = result["correlation"]
        if correlation["ols_slope"] is not None:
            line_x = np.linspace(float(x.min()), float(x.max()), 100)
            scatter.plot(line_x, correlation["ols_intercept"] + correlation["ols_slope"] * line_x,
                         color="#AA4455", linewidth=1.25, zorder=2)
        if correlation["pearson_r"] is None:
            statistic = "$r$ = NA, $p$ = NA"
        else:
            statistic = f"$r$ = {correlation['pearson_r']:.3f}, $p$ = {correlation['pearson_p_two_sided']:.3g}"
        scatter.text(0.04, 0.96, statistic, transform=scatter.transAxes, va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
        scatter.set(xlabel=r"Band heterogeneity $H_s$ (pp)", ylabel=r"Consensus gain $G_s$ (pp)")
        scatter.set_title("(b)", loc="left")
        scatter.margins(x=0.17, y=0.25)
        if np.ptp(x) == 0:
            scatter.set_xlim(x[0] - 1, x[0] + 1)
        if np.ptp(y) == 0:
            scatter.set_ylim(y[0] - 1, y[0] + 1)
        scatter.set_axisbelow(True)
        scatter.grid(color="#E3E3E3", linewidth=0.5)
        if result["partial"]:
            figure.suptitle(f"Partial LOSO: {len(subjects)}/{result['expected_subjects']} subjects", fontsize=10)
        _save_figure(figure, output_dir / "fig2_heterogeneity_consensus")
        plt.close(figure)


def write_outputs(result, output_dir, provenance=None, make_plots=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in (("band_accuracy.csv", "band_rows"), ("band_ranking.csv", "rank_rows"),
                          ("subject_metrics.csv", "subject_rows"), ("hardest_band_frequency.csv", "hardest_rows")):
        _write_csv(output_dir / filename, result[key])
    metadata = dict(provenance or {})
    metadata.update({"protocol": "SEED LOSO", "band_order": list(BANDS), "subject_ids_one_based": True,
                     "class_labels_zero_based": True, "expected_subjects": result["expected_subjects"],
                     "analyzed_subjects": result["subject_ids"].tolist(), "n_samples": result["n_samples"],
                     "partial": result["partial"], "missing_subjects": result["missing_subjects"],
                     "accuracy_unit": "percent", "difference_unit": "percentage_points", "heterogeneity_ddof": 0,
                     "aggregation": "Pool samples within each subject across sessions; no unweighted batch/session averaging.",
                     "prediction_ties": "Argmax chooses the first zero-based class index.",
                     "rank_ties": "Average tied rank; plotted order breaks ties by canonical band order.",
                     "hardest_ties": "Include every tied band; fractional count assigns 1/tie_count to each.",
                     "sample_selection": "All exported held-out target samples; no reliability threshold filtering.",
                     "consensus_definition": "sum_m band_weights[j,m] * band_posteriors[j,m,c]",
                     "consensus_max_abs_error": result["consensus_max_abs_error"],
                     "interpretation_limits": [
                         "These are diagnostic accuracies in learned class evidence, not a direct measurement of raw inter-subject distortion.",
                         "H_s and G_s share the same band accuracies; their mathematical coupling prevents treating correlation as causal proof.",
                         "Pearson p is descriptive: LOSO training sets overlap and subject points need not be independent.",
                         "Hardest-band differences can reflect task evidence as well as subject shift; no universal ordering is assumed."]})
    for filename, content in (("correlation.json", result["correlation"]), ("analysis_metadata.json", metadata)):
        (output_dir / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if make_plots:
        plot_results(result, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="One run directory, or one predictions.npz file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-tag", help="Analyze one export tag (e.g. final or warmup15), never pooled together")
    parser.add_argument("--expected-subjects", type=int, default=15)
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic output for incomplete LOSO; figures marked partial")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    arrays, provenance = load_predictions(args.input_root, checkpoint_tag=args.checkpoint_tag)
    result = analyze_arrays(**arrays, expected_subjects=args.expected_subjects, allow_partial=args.allow_partial)
    write_outputs(result, args.output_dir, provenance, make_plots=not args.no_plots)
    print(f"Analyzed {len(result['subject_ids'])}/{args.expected_subjects} subjects, {result['n_samples']} samples: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
