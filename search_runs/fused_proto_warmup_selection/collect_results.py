"""Collect final and epoch-15 artifacts from the isolated fused-prototype runs.

The script is intentionally filesystem-only: it never reads labels or reruns a
model.  It writes one compact CSV that can be copied into the experiment log.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_RE = re.compile(r"(acc|macro-f1|macro_f1|f1)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _find_files(root: Path, name: str):
    return sorted(root.rglob(name))


def _read_last_metrics(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = {}
    for match in METRIC_RE.finditer(text):
        key = match.group(1).replace("_", "-")
        found[key] = float(match.group(2))
    return found


def _read_csv_row(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def collect(root: Path, output: Path):
    rows = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        variant = variant_dir.name
        if variant not in {"baseline", "fused_proto"}:
            continue
        output_dirs = [p for p in variant_dir.rglob("checkpoint-warmup15") if p.is_file()]
        for checkpoint in output_dirs:
            run_dir = checkpoint.parent
            final_checkpoints = sorted(run_dir.glob("checkpoint-best*"))
            final_checkpoint = next(
                (p for p in final_checkpoints if "macro-f1" in p.name),
                final_checkpoints[0] if final_checkpoints else None,
            )
            warmup = run_dir / "warmup_selection_summary.csv"
            warmup_metrics = run_dir / "warmup_test_metrics.csv"
            final_metrics = run_dir / "final_test_metrics.csv"
            result_files = sorted(run_dir.glob("result_*.txt"))
            final_row = _read_csv_row(final_metrics) if final_metrics.exists() else {}
            metrics = {
                "acc": float(final_row["acc"]) if final_row.get("acc") else None,
                "macro-f1": float(final_row["macro-f1"]) if final_row.get("macro-f1") else None,
            }
            if not any(value is not None for value in metrics.values()) and result_files:
                metrics = _read_last_metrics(result_files[-1])
            summary = _read_csv_row(warmup) if warmup.exists() else {}
            diagnostic = _read_csv_row(warmup_metrics) if warmup_metrics.exists() else {}
            rows.append(
                {
                    "dataset": diagnostic.get("dataset", ""),
                    "variant": variant,
                    "run_dir": str(run_dir),
                    "warmup_checkpoint": str(checkpoint),
                    "warmup_checkpoint_exists": checkpoint.exists(),
                    "final_checkpoint": str(final_checkpoint) if final_checkpoint else "",
                    "final_checkpoint_exists": bool(final_checkpoint and final_checkpoint.exists()),
                    "final_acc": metrics.get("acc", ""),
                    "final_macro_f1": metrics.get("macro-f1", metrics.get("f1", "")),
                    "warmup_threshold": summary.get("threshold", ""),
                    "consensus_coverage": "",
                    "consensus_pseudo_label_accuracy": "",
                    "fused_proto_coverage": "",
                    "fused_proto_pseudo_label_accuracy": "",
                }
            )
            # Keep one row per variant while retaining both rules as columns.
            if warmup.exists():
                with warmup.open(newline="", encoding="utf-8") as stream:
                    for item in csv.DictReader(stream):
                        prefix = "consensus" if item.get("rule") == "Consensus" else "fused_proto"
                        rows[-1][f"{prefix}_coverage"] = item.get("coverage", "")
                        rows[-1][f"{prefix}_pseudo_label_accuracy"] = item.get("pseudo_label_accuracy", "")

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "variant", "run_dir", "warmup_checkpoint", "warmup_checkpoint_exists",
        "final_checkpoint", "final_checkpoint_exists", "final_acc", "final_macro_f1",
        "warmup_threshold", "consensus_coverage", "consensus_pseudo_label_accuracy",
        "fused_proto_coverage", "fused_proto_pseudo_label_accuracy",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="one run root under outputs/")
    parser.add_argument("--output", type=Path, default=None, help="summary CSV path")
    args = parser.parse_args()
    output = args.output or args.root / "fused_proto_warmup_selection_summary.csv"
    count = collect(args.root, output)
    print(f"Wrote {count} variant rows to {output}")


if __name__ == "__main__":
    main()
