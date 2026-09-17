"""Run agreement-factor and alignment-space ablations with LibEER TVT."""

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "no_agreement", "fused_alignment")


def load_local_trainer():
    source = ROOT / "src"
    for name in ("baseline_model.py", "experiment_model.py", "DMS_SGPAN_train.py", "evidence.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing isolated file: {source / name}")
    for name in ("baseline_model", "experiment_model", "DMS_SGPAN_train"):
        path = source / (name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"Local module: {module.__file__}", flush=True)
    return sys.modules["DMS_SGPAN_train"]


def check_outputs(output, variant, tags, skip_export):
    root = output / "tvt"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("ablation") != variant:
        raise RuntimeError(f"Incomplete or wrong variant: {root / 'manifest.json'}")
    required = [root / "checkpoints" / name for name in ("epoch_015.pt", "best.pt", "final.pt")]
    required.extend(root / "metrics" / name for name in ("history.csv", "test_best.csv", "test_final.csv"))
    if not skip_export:
        required.extend(root / "evidence" / tag / cohort / name for tag in tags
                        for cohort in ("target", "test") for name in ("predictions.npz", "sample_predictions.csv", "selection_curve.csv"))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Expected output is missing: {path}")


def native_arguments(profile, dataset_path, output_dir, device, subjects=None):
    values = ["-model", "DMS_SGPAN", "-dataset", profile["dataset"],
              "-setting", profile["setting"], "-dataset_path", str(dataset_path),
              "-output_dir", str(output_dir), "-log_dir", str(output_dir / "log"),
              "-device", device, "-metrics", "acc", "macro-f1", "-metric_choose", "macro-f1",
              "-feature_type", "de_lds", "-time_window", "1", "-sample_length", "1",
              "-stride", "1", "-onehot", "-sessions", *map(str, profile["sessions"])]
    for key in ("seed", "epochs", "batch_size", "lr"):
        values.extend(("-" + key, str(profile[key])))
    aliases = {"GLalpha": "gl_alpha", "K": "cheb_k"}
    for key, value in profile["model"].items():
        name = "loss_" + key[2:] if key.startswith("w_") else aliases.get(key, key)
        values.extend(("-dms_sgpan_" + name,
                       json.dumps(value) if isinstance(value, list) else str(value)))
    if subjects:
        values.extend(("-sr", *map(str, subjects)))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="seed", choices=["seed", "seediv", "seedv"])
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--setting", "-setting")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--libeer-dir", type=Path,
                        default=Path(os.environ.get("LIBEER_DIR", ROOT.parents[2] / "LibEER/LibEER")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["no_agreement", "fused_alignment"])
    parser.add_argument("--baseline-dir", type=Path, help="Existing B baseline/tvt for paired comparison")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--checkpoint-tags", nargs="+", choices=["warmup15", "best", "final"],
                        default=["warmup15", "final"])
    parser.add_argument("--export-batch-size", type=int, default=512)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    profiles = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.dataset not in profiles:
        parser.error(f"This experiment supports {', '.join(profiles)}")
    profile = profiles[args.dataset]
    if args.setting and args.setting != profile["setting"]:
        parser.error(f"This experiment requires -setting {profile['setting']}")
    variants = args.variants
    if len(set(variants)) != len(variants):
        parser.error("Duplicate variants")
    if profile["protocol"] != "tvt":
        parser.error("CExperiment requires the original TVT settings")
    if args.baseline_dir and "baseline" in variants:
        parser.error("Choose an existing --baseline-dir or run --variants baseline, not both")
    if args.export_batch_size < 1 or (args.export_only and args.skip_export):
        parser.error("Invalid export options")
    if args.export_only and args.run_dir is None:
        parser.error("--export-only requires the original --run-dir")
    run_root = (args.run_dir or ROOT / "outputs" / args.dataset / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    libeer = args.libeer_dir.resolve()
    dataset_path = (args.dataset_path or Path(profile["dataset_path"])).resolve()
    profile["dataset_path"] = str(dataset_path)
    for name in ("baseline_model.py", "experiment_model.py", "DMS_SGPAN_train.py", "evidence.py", "compare.py"):
        if not (ROOT / "src" / name).is_file():
            parser.error(f"Missing isolated file: {ROOT / 'src' / name}")
    if args.baseline_dir and not (args.baseline_dir / "manifest.json").is_file():
        parser.error("--baseline-dir must point to an existing baseline/tvt directory")
    if args.dry_run:
        for variant in variants:
            print(f"variant={variant}; prototype_space=band; alignment_space={'fused' if variant == 'fused_alignment' else 'band'}")
            print("LibEER:", libeer)
            print("Native arguments:", shlex.join(native_arguments(profile, dataset_path, run_root / variant,
                                                                     args.device)))
            print("Training log:", run_root / variant / "train.log")
        return
    if not (libeer / "config/setting.py").is_file():
        parser.error(f"LibEER not found: {libeer}; set LIBEER_DIR")
    if not dataset_path.is_dir():
        parser.error(f"Dataset directory not found: {dataset_path}")
    if args.worker:
        if len(variants) != 1:
            parser.error("A worker runs one variant")
        sys.path[:0] = [str(ROOT / "src"), str(libeer)]
        os.chdir(libeer)
        from utils.args import get_args_parser
        trainer = load_local_trainer()
        native = get_args_parser().parse_args(native_arguments(profile, dataset_path, run_root / variants[0],
                                                                args.device))
        native.profile = profile
        native.prototype_space = "band"
        native.ablation = variants[0]
        native.baseline_dir = args.baseline_dir.resolve() if args.baseline_dir else None
        for key in ("export_only", "skip_export", "checkpoint_tags", "export_batch_size"):
            setattr(native, key, getattr(args, key))
        trainer.main(native)
        return
    for variant in variants:
        output = run_root / variant
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", "-u", str(ROOT / "run.py"), "--worker", "--dataset", args.dataset,
                   "--dataset-path", str(dataset_path), "--libeer-dir", str(libeer), "--run-dir", str(run_root),
                   "--device", args.device, "--variants", variant, "--export-batch-size", str(args.export_batch_size),
                   "--checkpoint-tags", *args.checkpoint_tags]
        if args.baseline_dir:
            command.extend(("--baseline-dir", str(args.baseline_dir.resolve())))
        for key in ("export_only", "skip_export"):
            if getattr(args, key):
                command.append("--" + key.replace("_", "-"))
        log_path = output / ("export.log" if args.export_only else "train.log")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {shlex.join(command)}\n")
            log.flush()
            print(f"START {variant}: {log_path}", flush=True)
            completed = subprocess.run(command, cwd=libeer, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise SystemExit(f"FAILED {variant} (exit={completed.returncode}); see {log_path}")
        check_outputs(output, variant, args.checkpoint_tags, args.skip_export)
        print(f"DONE {variant}", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    from compare import compare_run
    compare_run(run_root, baseline_dir=args.baseline_dir)
    print(f"Completed: {run_root}", flush=True)


if __name__ == "__main__":
    main()
