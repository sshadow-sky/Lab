"""Check the deap_a protocol against the recorded DEAP-A baseline run.

The recorded run is search_20260727_162630_765369 job 51/72. Its
`run_param_map` is the authoritative, de-duplicated view of what it ran, so the
ablation variants and that baseline are compared under the same budget.

    python ablation/verify_deap_baseline.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_config as cfg  # noqa: E402

# run_param_map of search_20260727_162630_765369 / runs/0051, verbatim
RECORDED = {
    "batch_size": 2048,
    "bounds": [5.0, 5.0],
    "dataset": "deap",
    "device": "cuda",
    "dms_sgpan_dropout": 0.3,
    "dms_sgpan_frequency_band_groups": [[0], [1], [2], [3], [4]],
    "dms_sgpan_loss_align": 0.1,
    "dms_sgpan_loss_gcl": 0.2,
    "dms_sgpan_loss_orth": 0.2,
    "dms_sgpan_loss_subject": 0.2,
    "dms_sgpan_ugfcda_eps": 1e-06,
    "dms_sgpan_ugfcda_keep_ratio_step_epochs": 20,
    "dms_sgpan_ugfcda_reliability_threshold": 0.9,
    "dms_sgpan_ugfcda_warmup_epochs": 10,
    "epochs": 40,
    "feature_type": "de_lds",
    "label_used": ["arousal"],
    "lr": 0.01,
    "metric_choose": "macro-f1",
    "metrics": ["acc", "macro-f1"],
    "model": "DMS_SGPAN",
    "num_workers": 16,
    "onehot": True,
    "sample_length": 1,
    "seed": 2024,
    "setting": "deap_sub_independent_train_val_test_setting",
    "stride": 1,
    "time_window": 1.0,
}

# not part of the protocol: paths are supplied per run
IGNORE = {"output_dir", "log_dir", "dataset_path"}
# passed by the recorded run, read by no model or training script
DEAD_FLAGS = {"dms_sgpan_ugfcda_keep_ratio_step_epochs"}
# our config states the YAML default explicitly instead of relying on it
EQUIVALENT = {"dms_sgpan_temperature": 0.2}

NUMERIC = {
    "time_window", "lr", "batch_size", "epochs", "sample_length", "stride",
    "seed", "num_workers", "dms_sgpan_dropout", "dms_sgpan_loss_align",
    "dms_sgpan_loss_gcl", "dms_sgpan_loss_orth", "dms_sgpan_loss_subject",
    "dms_sgpan_ugfcda_eps", "dms_sgpan_ugfcda_warmup_epochs",
    "dms_sgpan_ugfcda_reliability_threshold", "dms_sgpan_temperature",
}
ASTEXT = {"dataset", "device", "feature_type", "metric_choose", "model",
          "setting"}


def parse_args(argv):
    """Turn a flat -flag value... list into {name: values}, honouring nargs."""
    out, i = {}, 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-"):
            name = token.lstrip("-")
            values, j = [], i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                values.append(argv[j])
                j += 1
            out.setdefault(name, values if values else True)
            i = j
            continue
        i += 1
    return out


def normalize(name, value):
    """Make a recorded value and a command-line value comparable."""
    if name == "dms_sgpan_frequency_band_groups":
        raw = value[0] if isinstance(value, list) and len(value) == 1 else value
        return json.loads(raw) if isinstance(raw, str) else raw
    if name == "bounds":
        values = value if isinstance(value, list) else [value]
        return [float(v) for v in values]
    if name in ("metrics", "label_used"):
        return list(value) if isinstance(value, list) else [value]
    if name == "onehot":
        return True
    if name in NUMERIC:
        return float(value[0] if isinstance(value, list) else value)
    if name in ASTEXT:
        return value[0] if isinstance(value, list) else value
    raise AssertionError("unhandled parameter %r" % name)


def main():
    argv = cfg.build_args("deap_a", "baseline", "/o", "/l", dataset_path="/d")
    ours = parse_args(argv)

    names = sorted(set(ours) | set(RECORDED))
    problems = 0
    print("%-46s %-22s %s" % ("parameter", "recorded baseline", "deap_a protocol"))
    for name in names:
        if name in IGNORE or name in DEAD_FLAGS:
            continue
        a = RECORDED.get(name, "<absent>")
        b = ours.get(name, "<absent>")
        if a == "<absent>" and name in EQUIVALENT:
            verdict = "matches the YAML default"
            if normalize(name, b) != EQUIVALENT[name]:
                verdict = "DIFFERENT from the YAML default"
                problems += 1
            print("%-46s %-22s %-22s %s" % (name, "not overridden", b, verdict))
            continue
        if a == "<absent>" or b == "<absent>":
            print("%-46s %-22s %-22s MISSING" % (name, a, b))
            problems += 1
            continue
        na, nb = normalize(name, a), normalize(name, b)
        same = na == nb
        problems += 0 if same else 1
        print("%-46s %-22s %-22s %s" % (name, na, nb, "" if same else "<- DIFFERENT"))

    print()
    print("skipped: %s (paths, supplied per run)" % ", ".join(sorted(IGNORE)))
    print("skipped: %s (declared but read by nothing)"
          % ", ".join(sorted(DEAD_FLAGS)))
    print()
    print("RESULT: %s" % ("the deap_a protocol reproduces the recorded baseline"
                          if problems == 0
                          else "%d parameter(s) differ" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
