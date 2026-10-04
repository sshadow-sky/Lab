"""Check that ablation_config.PROTOCOLS still reproduces the hand-written launchers.

The three SEED launchers in search_runs/ are the setups that produced the
published numbers. This script extracts their command line and diffs it,
flag by flag, against what build_args() generates for the same protocol. Any
difference means the unified config has drifted from the runs it replaces.

    python ablation/verify_launchers.py
    python ablation/verify_launchers.py --search-runs /path/to/search_runs
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_config as cfg  # noqa: E402

DEFAULT_SEARCH_RUNS = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "search_runs"))

PAIRS = [("seed", "run_dms_sgpan_ablation_seed.sh"),
         ("seediv", "run_dms_sgpan_ablation_seediv.sh"),
         ("seedv", "run_dms_sgpan_ablation_seedv.sh")]

# supplied by the caller at run time, so not part of a protocol definition
IGNORE = {"-dataset_path", "-num_workers", "-output_dir", "-log_dir",
          "-dms_sgpan_ablation"}


def strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def as_pairs(tokens):
    """Turn a flat token list into {flag: value} for comparable flags."""
    out = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and not nxt.startswith("-"):
                out[t] = strip_quotes(nxt)
                i += 2
                continue
            out[t] = ""
        i += 1
    return out


def launcher_pairs(path):
    text = open(path, encoding="utf-8").read()
    m = re.search(r"COMMON_ARGS=\((.*?)\n\s*\)", text, re.S)
    if not m:
        return None
    return as_pairs(re.sub(r"\\\n", " ", m.group(1)).split())


def config_pairs(protocol):
    args = cfg.build_args(protocol, "baseline", output_dir="/o", log_dir="/l",
                          dataset_path="/d")
    return as_pairs([a.replace("DMS_SGPAN", "DMS_SGPAN") for a in args])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search-runs", default=DEFAULT_SEARCH_RUNS)
    a = p.parse_args(argv)

    problems = 0
    for protocol, script in PAIRS:
        path = os.path.join(a.search_runs, script)
        if not os.path.exists(path):
            print("!! missing launcher: %s" % path)
            problems += 1
            continue
        script_args = launcher_pairs(path)
        config_args = config_pairs(protocol)
        diffs = []
        for flag in sorted(set(script_args) | set(config_args)):
            if flag in IGNORE:
                continue
            if flag not in script_args:
                diffs.append("  only in config : %s %s"
                             % (flag, config_args[flag]))
            elif flag not in config_args:
                diffs.append("  only in script : %s %s"
                             % (flag, script_args[flag]))
            elif script_args[flag] != config_args[flag]:
                diffs.append("  value differs  : %s  script=%s config=%s"
                             % (flag, script_args[flag], config_args[flag]))
        shared = len(set(script_args) & set(config_args))
        print("[%s] %s -> %d shared flags, %d diff(s)"
              % (protocol, script, shared, len(diffs)))
        for d in diffs:
            print(d)
        problems += len(diffs)

    print()
    print("RESULT: %s" % ("launchers and config agree" if problems == 0
                          else "%d mismatch(es)" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
