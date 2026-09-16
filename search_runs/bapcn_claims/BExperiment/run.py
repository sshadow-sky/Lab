"""Launch isolated BAPCN training with the existing LibEER pipeline."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


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
    parser.add_argument("--variants", nargs="+", choices=["baseline", "fused_proto"])
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--checkpoint-tags", nargs="+", choices=["warmup15", "final"],
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
    allowed = ["baseline"] if profile["protocol"] == "loso" else ["baseline", "fused_proto"]
    variants = args.variants or allowed
    if len(set(variants)) != len(variants) or any(v not in allowed for v in variants):
        parser.error(f"Allowed variants: {allowed}")
    if args.subjects and (profile["protocol"] != "loso" or len(set(args.subjects)) != len(args.subjects)
                          or any(s < 1 or s > profile["subjects"] for s in args.subjects)):
        parser.error("--subjects requires distinct SEED LOSO subject IDs from 1 to 15")
    if args.export_batch_size < 1 or (args.export_only and args.skip_export):
        parser.error("Invalid export options")
    if args.export_only and args.run_dir is None:
        parser.error("--export-only requires the original --run-dir")
    run_root = (args.run_dir or ROOT / "outputs" / args.dataset / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    libeer = args.libeer_dir.resolve()
    dataset_path = (args.dataset_path or Path(profile["dataset_path"])).resolve()
    profile["dataset_path"] = str(dataset_path)
    if args.dry_run:
        for variant in variants:
            print(f"variant={variant}; prototype_space={'band' if variant == 'baseline' else 'fused'}")
            print("LibEER:", libeer)
            print("Native arguments:", shlex.join(native_arguments(profile, dataset_path, run_root / variant,
                                                                     args.device, args.subjects)))
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
        import DMS_SGPAN_train as trainer
        native = get_args_parser().parse_args(native_arguments(profile, dataset_path, run_root / variants[0],
                                                                args.device, args.subjects))
        native.profile = profile
        native.prototype_space = "band" if variants[0] == "baseline" else "fused"
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
        if args.subjects:
            command.extend(("--subjects", *map(str, args.subjects)))
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
        print(f"DONE {variant}", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    if profile["protocol"] == "tvt":
        if all((run_root / v / "tvt/manifest.json").is_file() for v in allowed):
            from compare import compare_run
            compare_run(run_root)
    elif not args.skip_export and all((run_root / "baseline" / f"subject_{s:02d}" /
                                      "evidence/final/predictions.npz").is_file() for s in range(1, 16)):
        from analyze import main as analyze_main
        analyze_main(["--input-root", str(run_root / "baseline"), "--output-dir",
                      str(run_root / "analysis/final"), "--checkpoint-tag", "final"])
    print(f"Completed: {run_root}", flush=True)


if __name__ == "__main__":
    main()
