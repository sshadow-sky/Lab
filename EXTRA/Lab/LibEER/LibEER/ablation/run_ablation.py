"""Dataset-agnostic BAPCN ablation runner.

    python run_ablation.py --protocol deap_a  --gpu 0 --num-workers 16
    python run_ablation.py --protocol seedv   --variants baseline,no_gcl
    python run_ablation.py --protocol deap_v  --dry-run          # write command.txt only

Every run lands in ``<run-root>/<variant>/{output,log,train.log,command.txt}``,
the same layout the existing SEED launchers produce, so the result tables can be
read by the same scripts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation import ablation_config as cfg  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BAPCN ablation runner")
    p.add_argument("--protocol", default=None, choices=sorted(cfg.PROTOCOLS),
                   help="which dataset / protocol to run")
    p.add_argument("--libeer-dir", default=os.getcwd(),
                   help="directory holding the training entry points")
    p.add_argument("--entry-prefix", default="DMS_SGPAN",
                   choices=["DMS_SGPAN", "BAPCN"],
                   help="'BAPCN' uses the renamed entry points")
    p.add_argument("--variants", default="default",
                   help="'default', 'all' (incl. unimplemented), or a comma list")
    p.add_argument("--run-root", default=None,
                   help="default: <cwd>/<search_name>/<timestamp>")
    p.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"))
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="write command.txt but do not train")
    p.add_argument("--list", action="store_true",
                   help="print the Table II coverage and exit")
    return p.parse_args(argv)


def resolve_variants(spec: str) -> list:
    if spec == "default":
        return cfg.runnable_variants()
    if spec == "all":
        return cfg.runnable_variants(include_unimplemented=True,
                                     include_extra=True)
    wanted = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [v for v in wanted if v != "baseline" and v not in cfg.BY_KEY]
    if unknown:
        raise SystemExit("unknown variant(s): %s" % ", ".join(unknown))
    blocked = [v for v in wanted
               if cfg.BY_KEY.get(v) is not None
               and cfg.BY_KEY[v].status == cfg.STATUS_NEEDS_MODEL]
    if blocked:
        raise SystemExit(
            "these variants have no implementation yet: %s\n"
            "see ablation_config.VARIANTS notes (alignment_space / "
            "use_scale_agreement)" % ", ".join(blocked))
    return wanted


def main(argv=None) -> int:
    a = parse_args(argv)

    if a.list:
        print(cfg.describe_table())
        return 0

    if not a.protocol:
        p.error("--protocol is required unless --list is given")

    proto = cfg.PROTOCOLS[a.protocol]
    variants = resolve_variants(a.variants)

    run_root = a.run_root or os.path.join(
        os.getcwd(), proto.search_name,
        _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    print("protocol      : %s  (dataset=%s, setting=%s)"
          % (proto.key, proto.dataset, proto.setting))
    if proto.extra_args:
        print("extra args    : %s" % " ".join(proto.extra_args))
    print("lambda_align  : %s    delta: %s    dropout: %s"
          % (proto.loss_align, proto.reliability_threshold, proto.dropout))
    if not proto.confirmed:
        print("!! values marked UNCONFIRMED: %s" % proto.note)
        print("!! confirm them on the %s validation split before publishing "
              "the numbers." % proto.key)
    missing = cfg.missing_variants()
    if missing:
        print("not run (no implementation yet): %s"
              % ", ".join("%s [%s]" % (v.label, v.key) for v in missing))
    print("run root      : %s" % run_root)
    print("variants (%d) : %s" % (len(variants), ", ".join(variants)))
    print()

    failed = []
    for variant in variants:
        vdir = os.path.join(run_root, variant)
        out_dir = os.path.join(vdir, "output")
        log_dir = os.path.join(vdir, "log")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        entry = cfg.entry_script(a.libeer_dir, variant, a.entry_prefix)
        argv_run = cfg.build_args(proto.key, variant, out_dir, log_dir,
                                 num_workers=a.num_workers,
                                 dataset_path=a.dataset_path)

        cmd_file = os.path.join(vdir, "command.txt")
        with open(cmd_file, "w", encoding="utf-8") as fh:
            fh.write("protocol=%s\n" % proto.key)
            fh.write("dataset=%s\nsetting=%s\n" % (proto.dataset, proto.setting))
            fh.write("variant=%s\n" % variant)
            fh.write("gpu=%s\n" % a.gpu)
            fh.write("entry=%s\n" % entry)
            fh.write("command: CUDA_VISIBLE_DEVICES=%s python3 %s %s\n"
                     % (a.gpu, entry, " ".join(argv_run)))

        stamp = _dt.datetime.now().strftime("%F %T")
        if a.dry_run:
            print("[%s] DRY-RUN %-18s -> %s" % (stamp, variant, cmd_file))
            continue

        if not os.path.isfile(entry):
            print("[%s] FAIL    %-18s entry point missing: %s"
                  % (stamp, variant, entry), file=sys.stderr)
            failed.append(variant)
            continue

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
        print("[%s] START   %-18s (GPU %s)" % (stamp, variant, a.gpu))
        with open(os.path.join(vdir, "train.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, entry] + argv_run,
                                 cwd=a.libeer_dir, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0:
            print("[%s] DONE    %-18s" % (_dt.datetime.now().strftime("%F %T"),
                                          variant))
        else:
            print("[%s] FAIL    %-18s exit=%d" % (
                _dt.datetime.now().strftime("%F %T"), variant, rc),
                file=sys.stderr)
            failed.append(variant)

    if failed:
        print("\nfailed: %s" % ", ".join(failed), file=sys.stderr)
        return 1
    print("\nall requested variants finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
