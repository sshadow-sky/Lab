#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import itertools
import json
import os
import queue
import shlex
import yaml
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


# Per-model branch defaults.
# - fixed_args: always passed for this model unless overridden
# - search_space: cartesian-product values for this model unless overridden
MODEL_BRANCHES = {
    "DGCNN": {
        "entry": "DGCNN_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64, 128],
            "lr": [0.001, 0.0005],
            "epochs": [80, 150],
        },
    },
    "EEGNet": {
        "entry": "EEGNet_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [64, 128, 256],
            "lr": [0.001, 0.0005, 0.0002],
            "epochs": [80, 120],
        },
    },
    "TSception": {
        "entry": "TSception_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64, 128],
            "lr": [0.001, 0.0005],
            "epochs": [80, 120],
        },
    },
    "GCBNet": {
        "entry": "GCBNet_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64, 128],
            "lr": [0.001, 0.0005],
            "epochs": [80, 120],
        },
    },
    "DSAGC": {
        "entry": "DSAGC_train.py",
        "fixed_args": {
            "feature_type": "de_lds",
            "time_window": 1,
            "sample_length": 1,
            "stride": 1,
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64],
            "lr": [0.001, 0.0005],
            "epochs": [80, 120],
        },
    },
    "MAET": {
        "entry": "MAET_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64],
            "lr": [0.001, 0.0005],
            "epochs": [80, 120],
        },
    },
    "FAT": {
        "entry": "FAT_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64, 128],
            "lr": [0.001, 0.0005],
            "epochs": [80, 120],
        },
    },
    "PCLTDGCN": {
        "entry": "PCLTDGCN_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [32, 64],
            "lr": [0.001, 0.0005],
            "epochs": [100, 150],
        },
    },
    "FDGCL": {
        "entry": "FDGCL_train.py",
        "fixed_args": {
            "feature_type": "de_lds",
            "time_window": 1,
            "sample_length": 1,
            "stride": 1,
            "onehot": True,
            "fdgcl_ugfcda_warmup_epochs": 20,
            "fdgcl_ugfcda_eps": 0.000001,
            "fdgcl_ugfcda_keep_ratio_step_epochs": 20,
        },
        "search_space": {
            "batch_size": [32, 64],
            "lr": [0.001, 0.0005],
            "epochs": [100],

            "fdgcl_loss_align": [0.3, 0.5],
            "fdgcl_loss_subject": [0.1, 0.3],
        },
    },
    "RGNN_official": {
        "entry": "RGNN_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [16, 32, 64],
            "lr": [0.001, 0.0005],
            "epochs": [100, 150],
        },
    },
    "DMMR": {
        "entry": "DMMR_train.py",
        "fixed_args": {
            # Keep DMMR onehot behavior consistent with its original implementation.
            "onehot": True,
        },
        "search_space": {
            "batch_size": [512],
            "lr": [0.001],
            "epochs": [200],
        },
    },
    "DMMR_GATTF": {
        "entry": "DMMR_GATTF_train.py",
        "fixed_args": {
            "onehot": True,
        },
        "search_space": {
            "batch_size": [512],
            "lr": [0.001],
            "epochs": [200],
        },
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search runner for LibEER with multi-GPU scheduling and clear run records"
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BRANCHES.keys()), help="Model branch to search")
    parser.add_argument("--dataset", required=True, help="LibEER dataset name, e.g. seed_de_lds / deap")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument(
        "--setting",
        required=True,
        help="Preset setting name used by LibEER, e.g. seed_sub_dependent_train_val_test_setting",
    )

    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Comma-separated GPU ids available for this search")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Requested max concurrent trainings. Keep this small to avoid occupying all resources",
    )
    parser.add_argument(
        "--worker-hard-cap",
        type=int,
        default=3,
        help="Hard cap for concurrent trainings, used to reserve resources for others",
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds to include in the grid")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["acc", "macro-f1"],
        help="Metrics passed to LibEER training scripts",
    )
    parser.add_argument("--metric-choose", default="macro-f1", help="Best-checkpoint selection metric")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers for each run")
    parser.add_argument("--device", default="cuda", help="Device argument passed to train script")

    # Common training/data args that many LibEER scripts accept.
    parser.add_argument("--feature-type", default=None, help="Passed as -feature_type")
    parser.add_argument("--time-window", type=float, default=None, help="Passed as -time_window")
    parser.add_argument("--sample-length", type=int, default=None, help="Passed as -sample_length")
    parser.add_argument("--stride", type=int, default=None, help="Passed as -stride")
    parser.add_argument("--bounds", type=float, nargs="+", default=None, help="Passed as -bounds values")
    parser.add_argument("--label-used", nargs="+", default=None, help="Passed as -label_used values")
    parser.add_argument("--cross-trail", default=None, help="Passed as -cross_trail")
    parser.add_argument("--split-type", default=None, help="Passed as -split_type")
    parser.add_argument("--test-size", type=float, default=None, help="Passed as -test_size")
    parser.add_argument("--val-size", type=float, default=None, help="Passed as -val_size")

    onehot_group = parser.add_mutually_exclusive_group()
    onehot_group.add_argument("--onehot", action="store_true", help="Explicitly pass -onehot")
    onehot_group.add_argument("--no-onehot", action="store_true", help="Disable -onehot in fixed args")

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override branch batch_size search values",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=None,
        help="Override branch lr search values",
    )
    parser.add_argument(
        "--epochs-list",
        type=int,
        nargs="+",
        default=None,
        help="Override branch epochs search values",
    )
    parser.add_argument(
        "--seed-grid",
        type=int,
        nargs="+",
        default=None,
        help="Override seed grid in search space",
    )

    # Generic extension points for model-specific key hyperparameters and other train args.
    # Accept either a json file path or inline json text.
    parser.add_argument(
        "--space-json",
        default=None,
        help='Extra/override search space json. Example: {"bounds": [[4.5,5.5]], "label_used": [["valence"],["arousal"]]}',
    )
    parser.add_argument(
        "--jobs-json",
        default=None,
        help=(
            "Explicit grouped hyperparameter jobs as an inline JSON list or JSON file. "
            "Each item is one complete hyperparameter dict; if seed is omitted, --seeds are applied to every item."
        ),
    )
    parser.add_argument(
        "--fixed-json",
        default=None,
        help='Extra/override fixed args json. Example: {"feature_type":"de_lds","time_window":1,"sample_length":1,"stride":1}',
    )

    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra raw CLI args appended to each command, e.g. '-onehot -label_used valence'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only generate plans/commands, do not execute")

    parser.add_argument(
        "--save-root",
        default="./search_runs",
        help="Root directory to save all logs/results/manifests for this search",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_str):
    ids = []
    for token in gpu_str.split(","):
        token = token.strip()
        if token == "":
            continue
        ids.append(int(token))
    if not ids:
        raise ValueError("No GPU ids found. Please set --gpu-ids")
    return ids


def _load_json_any_maybe_file(raw):
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    if text[0] in "[{":
        return json.loads(text)

    p = Path(text)
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(text)


def _load_json_maybe_file(raw):
    obj = _load_json_any_maybe_file(raw)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("JSON must be an object (dict)")
    return obj


def _pick_not_none(**kwargs):
    out = {}
    for k, v in kwargs.items():
        if v is not None:
            out[k] = v
    return out


def _infer_dmmr_variant(dataset_name):
    name = (dataset_name or "").lower()
    if name == "seediv" or name.startswith("seediv_"):
        return "seediv"
    if name == "seed" or name.startswith("seed_"):
        return "seed"
    raise ValueError(f"DMMR search only supports SEED / SEED-IV datasets, got '{dataset_name}'.")


def _load_dmmr_yaml_defaults(dataset_name):
    cfg_path = Path(__file__).resolve().parent / "config" / "model_param" / "DMMR.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"DMMR yaml not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    variant = _infer_dmmr_variant(dataset_name)
    train_table = cfg.get("train", {})
    param_cfg = cfg.get("params", {})
    train_cfg = train_table.get(variant, {})
    if len(train_cfg) == 0:
        raise ValueError(f"DMMR train defaults for variant '{variant}' are missing in DMMR.yaml")

    fixed = {
        "dmmr_num_classes": int(train_cfg.get("num_classes", 3)),
        "dmmr_time_steps": int(train_cfg.get("time_steps", 30)),
        "dmmr_iteration": int(train_cfg.get("iteration", 7)),
        "dmmr_epoch_pretraining": int(train_cfg.get("epoch_pretraining", 300)),
        "dmmr_epoch_finetuning": int(train_cfg.get("epoch_finetuning", 500)),
        "dmmr_input_dim": int(train_cfg.get("input_dim", param_cfg.get("input_dim", 310))),
        "dmmr_hid_dim": int(param_cfg.get("hid_dim", 64)),
        "dmmr_n_layers": int(param_cfg.get("n_layers", 1)),
        "dmmr_beta": float(param_cfg.get("beta", 0.05)),
        "dmmr_weight_decay": float(train_cfg.get("weight_decay", 0.0005)),
    }
    search = {
        "batch_size": [int(train_cfg.get("batch_size", 512))],
        "lr": [float(train_cfg.get("lr", 0.001))],
        "epochs": [int(train_cfg.get("epoch_finetuning", 200))],
    }
    return fixed, search


def _load_dmmr_gattf_yaml_defaults(dataset_name):
    cfg_path = Path(__file__).resolve().parent / "config" / "model_param" / "DMMR_GATTF.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"DMMR_GATTF yaml not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    variant = _infer_dmmr_variant(dataset_name)
    train_table = cfg.get("train", {})
    param_cfg = cfg.get("params", {})
    train_cfg = train_table.get(variant, {})
    if len(train_cfg) == 0:
        raise ValueError(f"DMMR_GATTF train defaults for variant '{variant}' are missing in DMMR_GATTF.yaml")

    fixed = {
        "gattf_num_classes": int(train_cfg.get("num_classes", 3)),
        "gattf_time_steps": int(train_cfg.get("time_steps", 30)),
        "gattf_iteration": int(train_cfg.get("iteration", 7)),
        "gattf_epoch_pretraining": int(train_cfg.get("epoch_pretraining", 60)),
        "gattf_epoch_finetuning": int(train_cfg.get("epoch_finetuning", 100)),
        "gattf_input_dim": int(param_cfg.get("input_dim", 310)),
        "gattf_num_channels": int(param_cfg.get("num_channels", 62)),
        "gattf_num_bands": int(param_cfg.get("num_bands", 5)),
        "gattf_gat_hidden_dim": int(param_cfg.get("gat_hidden_dim", 16)),
        "gattf_gat_heads": int(param_cfg.get("gat_heads", 4)),
        "gattf_model_dim": int(param_cfg.get("model_dim", 128)),
        "gattf_transformer_heads": int(param_cfg.get("transformer_heads", 4)),
        "gattf_transformer_layers": int(param_cfg.get("transformer_layers", 2)),
        "gattf_decoder_layers": int(param_cfg.get("decoder_layers", 2)),
        "gattf_feedforward_dim": int(param_cfg.get("feedforward_dim", 256)),
        "gattf_dropout": float(param_cfg.get("dropout", 0.1)),
        "gattf_beta": float(param_cfg.get("beta", 0.05)),
        "gattf_gamma": float(param_cfg.get("gamma", 0.1)),
        "gattf_temperature": float(param_cfg.get("temperature", 0.2)),
        "gattf_weight_decay": float(train_cfg.get("weight_decay", 0.0005)),
    }
    search = {
        "batch_size": [int(train_cfg.get("batch_size", 512))],
        "lr": [float(train_cfg.get("lr", 0.001))],
        "epochs": [int(train_cfg.get("epoch_finetuning", 100))],
    }
    return fixed, search


def build_spaces(model_name, args):
    branch = MODEL_BRANCHES[model_name]
    fixed_args = dict(branch.get("fixed_args", {}))
    search_space = dict(branch.get("search_space", {}))
    dropped_args = {}

    if model_name == "DMMR":
        dmmr_fixed, dmmr_search = _load_dmmr_yaml_defaults(args.dataset)
        for k, v in dmmr_fixed.items():
            fixed_args.setdefault(k, v)
        for k, v in dmmr_search.items():
            search_space.setdefault(k, v)

    if model_name == "DMMR_GATTF":
        gattf_fixed, gattf_search = _load_dmmr_gattf_yaml_defaults(args.dataset)
        for k, v in gattf_fixed.items():
            fixed_args.setdefault(k, v)
        for k, v in gattf_search.items():
            search_space.setdefault(k, v)

    if args.batch_sizes is not None:
        search_space["batch_size"] = args.batch_sizes
    if args.lrs is not None:
        search_space["lr"] = args.lrs
    if args.epochs_list is not None:
        search_space["epochs"] = args.epochs_list
    if args.seed_grid is not None:
        search_space["seed"] = args.seed_grid

    # Add seed into the cartesian product unless already overridden.
    if "seed" not in search_space:
        search_space["seed"] = args.seeds

    cli_fixed = _pick_not_none(
        feature_type=args.feature_type,
        time_window=args.time_window,
        sample_length=args.sample_length,
        stride=args.stride,
        bounds=args.bounds,
        label_used=args.label_used,
        cross_trail=args.cross_trail,
        split_type=args.split_type,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    fixed_args.update(cli_fixed)

    # onehot explicit control
    if args.onehot:
        fixed_args["onehot"] = True
    elif args.no_onehot and "onehot" in fixed_args:
        del fixed_args["onehot"]

    external_space = _load_json_maybe_file(args.space_json)
    external_fixed = _load_json_maybe_file(args.fixed_json)
    # external settings have highest priority.
    search_space.update(external_space)
    fixed_args.update(external_fixed)
    for k in external_fixed:
        if k not in external_space and k in search_space:
            del search_space[k]

    for k, vals in search_space.items():
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"Search space key '{k}' must map to a non-empty list")

    return branch["entry"], fixed_args, search_space


def cartesian_product(space):
    keys = list(space.keys())
    values = [space[k] for k in keys]
    jobs = []
    for combo in itertools.product(*values):
        jobs.append(dict(zip(keys, combo)))
    return jobs


def expand_explicit_jobs(job_specs, default_seeds):
    if not isinstance(job_specs, list) or len(job_specs) == 0:
        raise ValueError("--jobs-json must be a non-empty JSON list of hyperparameter dicts")

    jobs = []
    for idx, spec in enumerate(job_specs, 1):
        if not isinstance(spec, dict):
            raise ValueError(f"--jobs-json item {idx} must be a dict, got {type(spec).__name__}")

        base = dict(spec)
        seed_value = base.pop("seed", None)
        if seed_value is None:
            seeds = default_seeds
        elif isinstance(seed_value, list):
            seeds = seed_value
        else:
            seeds = [seed_value]

        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError(f"--jobs-json item {idx} has an empty seed list")

        for seed in seeds:
            hp = dict(base)
            hp["seed"] = int(seed)
            missing = [k for k in ("batch_size", "lr", "epochs", "seed") if k not in hp]
            if missing:
                raise ValueError(f"--jobs-json item {idx} missing required hyperparameters: {missing}")
            jobs.append(hp)

    return jobs


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sanitize_for_path(s):
    text = str(s)
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text


def _value_to_text(v):
    if isinstance(v, (list, tuple)):
        return "-".join(str(x) for x in v)
    return str(v)


def build_run_param_map(model_name, hp, fixed_args, args):
    merged = dict(fixed_args)
    merged.update(hp)
    merged.update(
        {
            "model": model_name,
            "dataset": args.dataset,
            "setting": args.setting,
            "metric_choose": args.metric_choose,
            "metrics": args.metrics,
            "num_workers": args.num_workers,
            "device": args.device,
        }
    )
    return {k: merged[k] for k in sorted(merged.keys())}


def build_run_path_suffix(param_map):
    # Encode all key params in path segments to make each run human-traceable.
    # Keep each segment shorter than common filename limits.
    segments = []
    for k, v in param_map.items():
        raw = f"{k}={_value_to_text(v)}"
        seg = sanitize_for_path(raw)
        if len(seg) > 120:
            digest = hashlib.md5(seg.encode("utf-8")).hexdigest()[:8]
            seg = f"{seg[:100]}__{digest}"
        segments.append(seg)
    return Path(*segments)


def build_run_name(model_name, hp):
    parts = []
    for k in sorted(hp.keys()):
        v = hp[k]
        v_text = _value_to_text(v)
        parts.append(f"{k}={v_text}")
    text = "__".join(sanitize_for_path(p) for p in parts)
    digest = hashlib.md5(json.dumps(hp, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    if len(text) > 120:
        text = text[:120]
    return f"{model_name}__{text}__{digest}"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_cli_arg(cmd, key, value):
    flag = f"-{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        cmd.append(flag)
        cmd.extend(str(x) for x in value)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_command(
    python_exe,
    entry_script,
    model_name,
    hp,
    fixed_args,
    args,
    run_output_dir,
    run_log_dir,
):
    cmd = [
        python_exe,
        entry_script,
        "-model",
        model_name,
        "-batch_size",
        str(hp["batch_size"]),
        "-lr",
        str(hp["lr"]),
        "-epochs",
        str(hp["epochs"]),
        "-seed",
        str(hp["seed"]),
        "-dataset",
        args.dataset,
        "-dataset_path",
        args.dataset_path,
        "-setting",
        args.setting,
        "-metrics",
        *args.metrics,
        "-metric_choose",
        args.metric_choose,
        "-num_workers",
        str(args.num_workers),
        "-device",
        args.device,
        "-output_dir",
        str(run_output_dir),
        "-log_dir",
        str(run_log_dir),
    ]

    merged = dict(fixed_args)
    merged.update(hp)
    reserved_keys = {
        "model",
        "dataset",
        "dataset_path",
        "setting",
        "metrics",
        "metric_choose",
        "num_workers",
        "device",
        "output_dir",
        "log_dir",
    }
    for k in sorted(merged.keys()):
        if k in reserved_keys:
            continue
        _append_cli_arg(cmd, k, merged[k])

    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    return cmd


def run_single_job(job_idx, total_jobs, task, gpu_queue, lock, search_dir, args):
    hp = task["hp"]
    fixed_args = task["fixed_args"]
    model_name = task["model"]
    entry_script = task["entry"]

    run_name = build_run_name(model_name, hp)
    run_param_map = build_run_param_map(model_name, hp, fixed_args, args)
    run_param_suffix = build_run_path_suffix(run_param_map)
    run_dir = search_dir / "runs" / f"{job_idx:04d}" / run_param_suffix
    if run_dir.exists():
        # Extra protection against accidental collisions (e.g., repeated dry-runs).
        run_dir = run_dir / f"rerun_{datetime.now().strftime('%H%M%S_%f')}"
    run_output_dir = run_dir / "output"
    run_log_dir = run_dir / "log"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = gpu_queue.get()
    start_time = time.time()
    status = "success"
    return_code = 0

    try:
        cmd = build_command(
            python_exe=sys.executable,
            entry_script=entry_script,
            model_name=model_name,
            hp=hp,
            fixed_args=fixed_args,
            args=args,
            run_output_dir=run_output_dir,
            run_log_dir=run_log_dir,
        )

        # Save run manifest first so every run is traceable.
        run_manifest = {
            "job_index": job_idx,
            "total_jobs": total_jobs,
            "model": model_name,
            "entry_script": entry_script,
            "hyper_params": hp,
            "fixed_args": fixed_args,
            "run_param_map": run_param_map,
            "dataset": args.dataset,
            "dataset_path": args.dataset_path,
            "setting": args.setting,
            "metrics": args.metrics,
            "metric_choose": args.metric_choose,
            "num_workers": args.num_workers,
            "gpu_id": gpu_id,
            "command": cmd,
            "cwd": str(Path(__file__).resolve().parent),
            "start_time": datetime.now().isoformat(),
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

        with open(run_dir / "train.log", "w", encoding="utf-8") as logf:
            if args.dry_run:
                logf.write("DRY RUN\n")
                logf.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
                return_code = 0
            else:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                return_code = proc.returncode
                if return_code != 0:
                    status = "failed"

    except Exception as exc:
        status = "failed"
        return_code = -1
        with open(run_dir / "train.log", "a", encoding="utf-8") as logf:
            logf.write("\nException:\n")
            logf.write(str(exc) + "\n")
    finally:
        gpu_queue.put(gpu_id)

    end_time = time.time()
    elapsed_sec = round(end_time - start_time, 2)

    result = {
        "job_index": job_idx,
        "run_name": run_name,
        "status": status,
        "return_code": return_code,
        "elapsed_sec": elapsed_sec,
        "gpu_id": gpu_id,
        "run_dir": str(run_dir),
        "hyper_params": hp,
    }

    with lock:
        return result


def main():
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    entry_script, fixed_args, search_space = build_spaces(args.model, args)
    explicit_job_specs = _load_json_any_maybe_file(args.jobs_json)
    if args.jobs_json is not None:
        default_seeds = args.seed_grid if args.seed_grid is not None else args.seeds
        jobs_hp = expand_explicit_jobs(explicit_job_specs, default_seeds)
        search_mode = "explicit_jobs"
    else:
        jobs_hp = cartesian_product(search_space)
        search_mode = "cartesian"

    if not jobs_hp:
        raise RuntimeError("No search tasks generated. Please check your search space.")

    # Apply user request and safety cap together.
    effective_workers = min(args.max_workers, args.worker_hard_cap, len(gpu_ids), len(jobs_hp))
    if effective_workers <= 0:
        raise RuntimeError("No available workers. Please check --max-workers and --worker-hard-cap.")

    ts = timestamp_now()
    search_dir = Path(args.save_root) / args.model / f"search_{ts}"
    search_dir.mkdir(parents=True, exist_ok=True)

    task_manifest = {
        "timestamp": ts,
        "model": args.model,
        "entry_script": entry_script,
        "fixed_args": fixed_args,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "setting": args.setting,
        "gpu_ids": gpu_ids,
        "requested_max_workers": args.max_workers,
        "worker_hard_cap": args.worker_hard_cap,
        "effective_workers": effective_workers,
        "search_space": search_space,
        "metrics": args.metrics,
        "metric_choose": args.metric_choose,
        "num_workers": args.num_workers,
        "device": args.device,
        "extra_args": args.extra_args,
        "dry_run": args.dry_run,
        "search_mode": search_mode,
        "explicit_job_specs": explicit_job_specs,
        "total_jobs": len(jobs_hp),
    }
    write_json(search_dir / "search_manifest.json", task_manifest)

    tasks = []
    for hp in jobs_hp:
        tasks.append({"model": args.model, "entry": entry_script, "hp": hp, "fixed_args": fixed_args})

    gpu_queue = queue.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    lock = threading.Lock()
    results = []

    print("=" * 80)
    print(f"Start LibEER hyperparameter search: {args.model}")
    print(f"Entry script: {entry_script}")
    print(f"Total jobs: {len(tasks)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Effective concurrent workers: {effective_workers}")
    print(f"Search directory: {search_dir}")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                run_single_job,
                idx + 1,
                len(tasks),
                task,
                gpu_queue,
                lock,
                search_dir,
                args,
            ): idx
            for idx, task in enumerate(tasks)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Searching {args.model}"):
            res = future.result()
            results.append(res)

    # Sort by job index for deterministic summary.
    results.sort(key=lambda x: x["job_index"])
    write_json(search_dir / "search_results.json", results)

    success_cnt = sum(1 for r in results if r["status"] == "success")
    fail_cnt = len(results) - success_cnt

    print("=" * 80)
    print("Search finished")
    print(f"Success: {success_cnt} | Failed: {fail_cnt} | Total: {len(results)}")
    print(f"Manifest: {search_dir / 'search_manifest.json'}")
    print(f"Results:  {search_dir / 'search_results.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
