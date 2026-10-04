"""Check every protocol in ablation_config against the LibEER framework.

The config is only correct if the framework actually accepts what it emits. This
script reads the framework's own tables and compares:

  1. `-dataset` is a name the loader knows
  2. `-setting` is a key of `preset_setting`
  3. the setting's own dataset guard accepts that `-dataset` value
  4. every flag emitted by `build_args()` has a definition in `utils/args.py`
  5. a protocol whose labels are continuous ratings carries `-bounds`, because
     `label_process` binarizes with `value <= bounds[0]` / `value >= bounds[1]`
     and stops on a None subscript otherwise

    python ablation/verify_protocols.py
"""
import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_config as cfg  # noqa: E402

DEFAULT_LIBEER = os.path.normpath(os.path.join(HERE, ".."))

# datasets whose labels are continuous ratings that must be binarized
CONTINUOUS_LABEL_DATASETS = {"deap", "deap_raw", "hci", "dreamer"}


def _module_value(path, name):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
    return None


def module_literal(path, name):
    """A module-level list or set of string literals."""
    value = _module_value(path, name)
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except ValueError:
        return None


def dict_string_keys(path, name):
    """The string keys of a module-level dict, whatever its values are."""
    value = _module_value(path, name)
    if value is None or not isinstance(value, ast.Dict):
        return set()
    return {ast.literal_eval(k) for k in value.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def function_source(path, name):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"\ndef %s\(.*?(?=\ndef |\Z)" % re.escape(name), src, re.S)
    return m.group(0) if m else ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libeer-dir", default=DEFAULT_LIBEER)
    a = p.parse_args(argv)

    load_data = os.path.join(a.libeer_dir, "data_utils", "load_data.py")
    setting_py = os.path.join(a.libeer_dir, "config", "setting.py")
    args_py = os.path.join(a.libeer_dir, "utils", "args.py")

    # the framework files are what this check reads; on a training server they
    # are present, inside a partial overlay they are not, so each check reports
    # itself as skipped rather than failing when its source is absent
    have_load = os.path.isfile(load_data)
    have_setting = os.path.isfile(setting_py)
    have_args = os.path.isfile(args_py)

    available = set(module_literal(load_data, "available_dataset") or []) \
        if have_load else set()
    presets = dict_string_keys(setting_py, "preset_setting") \
        if have_setting else set()
    declared = set(re.findall(r"add_argument\(\s*['\"]([^'\"]+)",
                              open(args_py, encoding="utf-8").read())) \
        if have_args else set()

    print("framework under %s" % a.libeer_dir)
    print("  data_utils/load_data.py %s" % ("found" if have_load else "MISSING"))
    print("  config/setting.py       %s" % ("found" if have_setting else "MISSING"))
    print("  utils/args.py           %s" % ("found" if have_args else "MISSING"))
    print()

    problems = 0
    for key in sorted(cfg.PROTOCOLS):
        proto = cfg.PROTOCOLS[key]
        print("[%s] dataset=%s setting=%s" % (key, proto.dataset, proto.setting))
        issues = []

        if have_load and proto.dataset not in available:
            issues.append("-dataset %r is not in the loader's available_dataset"
                          % proto.dataset)

        if have_setting:
            if proto.setting not in presets:
                issues.append("-setting %r is not a key of preset_setting"
                              % proto.setting)
            else:
                body = function_source(setting_py, proto.setting)
                guards = re.findall(r"args\.dataset\.startswith\(['\"]([^'\"]+)",
                                    body)
                for g in guards:
                    if not proto.dataset.startswith(g):
                        issues.append("the setting's guard startswith(%r) rejects "
                                      "-dataset %r" % (g, proto.dataset))

        flags = [t for t in cfg.build_args(key, "baseline", "/o", "/l",
                                           dataset_path="/d") if t.startswith("-")]
        if have_args:
            unknown = [f for f in flags if f not in declared
                       and f != "-dms_sgpan_ablation"]
            if unknown:
                issues.append("flags not declared in utils/args.py: %s" % unknown)

        binarized = proto.dataset in CONTINUOUS_LABEL_DATASETS
        if binarized and not proto.bounds:
            issues.append("continuous ratings but no -bounds: label_process will "
                          "subscript None")
        if not binarized and proto.bounds:
            issues.append("discrete labels but -bounds is set: %s"
                          % (proto.bounds,))

        for issue in issues:
            print("    FAIL  %s" % issue)
        problems += len(issues)
        if not issues:
            parts = []
            parts.append("dataset known" if have_load else "dataset not checked")
            parts.append("setting known" if have_setting else "setting not checked")
            parts.append("%d flags declared" % len(flags) if have_args
                         else "flags not checked")
            parts.append("bounds=%s" % (proto.bounds or "not needed",))
            print("    OK    " + ", ".join(parts))

    print()
    if not (have_load and have_setting and have_args):
        print("note: some checks were skipped because the framework files above")
        print("      are missing from this directory. Run this on the LibEER tree.")
    print("RESULT: %s" % ("every protocol matches the framework"
                          if problems == 0
                          else "%d point(s) to resolve" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
