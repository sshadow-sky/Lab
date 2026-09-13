"""Export fixed warm-up snapshots for the baseline/fused-proto comparison."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


def _collect(model, loader, device):
    scales = None
    finals = []
    labels = []
    weights = []
    model.eval()
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            sid = sid.to(device).long()
            sessions = torch.zeros_like(sid)
            enc = model._encode_all(x, sid, sessions)
            batch_scales = [value.detach().cpu() for value in enc["shared_scales"]]
            if scales is None:
                scales = [[] for _ in batch_scales]
            for index, value in enumerate(batch_scales):
                scales[index].append(value)
            finals.append(enc["final_feat"].detach().cpu())
            labels.append(y.detach().cpu().argmax(dim=1))
            weights.append(enc["scale_weights"].detach().cpu())
    if scales is None:
        raise RuntimeError("Cannot export an empty loader")
    return (
        [torch.cat(values, dim=0).to(device) for values in scales],
        torch.cat(finals, dim=0).to(device),
        torch.cat(labels, dim=0).to(device),
        torch.cat(weights, dim=0).to(device),
    )


def _curve(scores, correct):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    thresholds = np.unique(np.r_[0.0, scores, 1.0])
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.searchsorted(sorted_scores, thresholds, side="left")
    counts = len(scores) - starts
    cumulative = np.r_[0, np.cumsum(correct[order])]
    hits = cumulative[-1] - cumulative[starts]
    precision = np.divide(
        hits,
        counts,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return [
        {
            "threshold": float(threshold),
            "selected_count": int(count),
            "coverage": float(count / len(scores)),
            "pseudo_label_accuracy": (
                float(value) if np.isfinite(value) else None
            ),
        }
        for threshold, count, value in zip(thresholds, counts, precision)
    ]


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_warmup_snapshot(
    output_dir,
    model,
    source_loader,
    target_loader,
    device,
    configured_threshold,
    dataset,
    variant,
    epoch=15,
):
    """Export both rules from the same pre-PConA checkpoint."""
    output_dir = Path(output_dir)
    source_scales, source_final, source_labels, _ = _collect(model, source_loader, device)
    target_scales, target_final, target_labels, target_weights = _collect(model, target_loader, device)

    consensus_state = model._ugfcda_reliability_and_pseudo(
        source_scales,
        target_scales,
        source_labels,
        target_weights,
    )
    fused_state = model._ugfcda_reliability_and_pseudo(
        [source_final],
        [target_final],
        source_labels,
        None,
    )

    truth = target_labels.detach().cpu().numpy()
    rules = {
        "Consensus": consensus_state,
        "Fused Proto.": fused_state,
    }
    curve_rows = []
    summary_rows = []
    archive = {
        "target_label": truth,
        "dataset": np.asarray(dataset),
        "variant": np.asarray(variant),
        "epoch": np.asarray(epoch),
    }
    for name, state in rules.items():
        pseudo = state["pseudo_labels"].detach().cpu().numpy()
        reliability = state["reliability"].detach().cpu().numpy()
        correct = pseudo == truth
        for row in _curve(reliability, correct):
            curve_rows.append({"rule": name, **row})
        keep = reliability >= float(configured_threshold)
        selected = int(keep.sum())
        summary_rows.append(
            {
                "rule": name,
                "threshold": float(configured_threshold),
                "selected_count": selected,
                "coverage": float(keep.mean()),
                "pseudo_label_accuracy": (
                    float(correct[keep].mean()) if selected else None
                ),
            }
        )
        archive[f"{name.replace(' ', '_').replace('.', '').lower()}_pseudo"] = pseudo
        archive[f"{name.replace(' ', '_').replace('.', '').lower()}_reliability"] = reliability

    _write_csv(output_dir / "warmup_selection_curves.csv", curve_rows)
    _write_csv(output_dir / "warmup_selection_summary.csv", summary_rows)
    np.savez_compressed(output_dir / "warmup_selection_evidence.npz", **archive)
    (output_dir / "warmup_selection_metadata.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "checkpoint_epoch": int(epoch),
                "warmup_epochs": 15,
                "configured_threshold": float(configured_threshold),
                "rules": list(rules),
                "interpretation": "Fixed pre-PConA checkpoint; labels are used offline only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
