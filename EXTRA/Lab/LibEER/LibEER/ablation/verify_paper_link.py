"""Check that the unified ablation config matches the paper and the code.

Three independent comparisons, each parsed from source rather than retyped:

  A. TABLE II in the manuscript  vs  VARIANTS   -- row labels, group headings,
                                                   group order, row count
  B. DMS_SGPAN_ABLATION_SPECS    vs  VARIANTS   -- the specification dimension
                                                   each row flips, and its value
  C. DMS_SGPAN_ABLATION_MODELS   vs  VARIANTS   -- that the code variant each
                                                   row names really exists

    python ablation/verify_paper_link.py
    python ablation/verify_paper_link.py --paper ../Paper_PBPC/sections/04_experiments.tex
"""
import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_config as cfg  # noqa: E402

DEFAULT_PAPER = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "Paper_PBPC", "sections", "04_experiments.tex"))
DEFAULT_MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))


def normalize(text):
    return text.replace("--", "-").strip()


# --------------------------------------------------------------------------
# A. the manuscript's TABLE II
# --------------------------------------------------------------------------

def strip_caption(block):
    """Drop the \\caption{...} span, which spans several lines of plain prose."""
    start = block.find("\\caption{")
    if start < 0:
        return block
    depth = 0
    for i in range(start + len("\\caption"), len(block)):
        if block[i] == "{":
            depth += 1
        elif block[i] == "}":
            depth -= 1
            if depth == 0:
                return block[:start] + block[i + 1:]
    return block


def parse_table_two(path):
    text = open(path, encoding="utf-8").read()
    # the ablation table is the table* block carrying label tab:ablation
    block = None
    for m in re.finditer(r"\\begin\{table\*\}.*?\\end\{table\*\}", text, re.S):
        if "\\label{tab:ablation}" in m.group(0):
            block = m.group(0)
            break
    if block is None:
        return None, None
    block = strip_caption(block)
    groups, labels = [], []
    for raw in block.splitlines():
        line = raw.strip()
        g = re.match(r"\\multicolumn\{7\}\{c\}\{\\textbf\{\\textit\{(.*?)\}\}\}", line)
        if g:
            groups.append(normalize(g.group(1)))
            continue
        if not line or line.startswith("\\") or line.startswith("&"):
            continue
        labels.append(normalize(line))
    return groups, labels


# --------------------------------------------------------------------------
# B. the code's specification table
# --------------------------------------------------------------------------

def parse_specs(path):
    """Return (field defaults, {variant: [(field, value), ...]})."""
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)

    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DMS_SGPANAblationSpec":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    defaults[stmt.target.id] = ast.literal_eval(stmt.value)

    specs = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == "DMS_SGPAN_ABLATION_SPECS"):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            name = ast.literal_eval(key)
            if isinstance(value, ast.Name):
                specs[name] = []
                continue
            changes = []
            for kw in value.keywords:
                changes.append((kw.arg, ast.literal_eval(kw.value)))
            specs[name] = changes
    return defaults, specs


# --------------------------------------------------------------------------
# C. the registered model classes
# --------------------------------------------------------------------------

def _module_dict(tree, name):
    """Return the value node of a module-level dict assignment, annotated or not."""
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
    return None


def parse_models(path):
    source = open(path, encoding="utf-8").read()
    value = _module_dict(ast.parse(source), "DMS_SGPAN_ABLATION_MODELS")
    out = {}
    if value is None:
        return out
    for key, item in zip(value.keys, value.values):
        out[ast.literal_eval(key)] = getattr(item, "id", None) or getattr(
            getattr(item, "func", None), "id", "?")
    return out


def parse_trainers(path):
    """Pull the registered variant keys out of the ablation training entry point."""
    source = open(path, encoding="utf-8").read()
    m = re.search(r"DMS_SGPAN_ABLATION_TRAINERS\s*=\s*\{(.*?)\n\}", source, re.S)
    if not m:
        return set()
    return set(re.findall(r'"([A-Za-z0-9_]+)"\s*:', m.group(1)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", default=DEFAULT_PAPER)
    p.add_argument("--models", default=DEFAULT_MODELS)
    a = p.parse_args(argv)

    rows = [v for v in cfg.VARIANTS if v.status != cfg.STATUS_EXTRA]
    problems = 0

    # -- A -----------------------------------------------------------------
    print("A. manuscript TABLE II vs config")
    if not os.path.isfile(a.paper):
        # this is the normal case on a training server, which usually has no
        # copy of the manuscript; checks B to E do not need it
        print("   SKIPPED: %s not found." % a.paper)
        print("   Point --paper at the section file to also compare the row")
        print("   labels and group headings. Checks B to E below still run.")
    else:
        groups, labels = parse_table_two(a.paper)
        if groups is None:
            print("   could not find the ablation table in %s" % a.paper)
            problems += 1
        else:
            paper_groups = list(dict.fromkeys(groups))
            config_groups = [g for g in cfg.GROUPS]
            config_labels = [normalize(v.label) for v in rows]
            print("   groups : paper=%d config=%d" % (len(paper_groups), len(config_groups)))
            print("   rows   : paper=%d config=%d" % (len(labels), len(config_labels)))
            for name, same in (("group order", paper_groups == config_groups),
                               ("row order", labels == config_labels)):
                print("   %-10s: %s" % (name, "identical" if same else "DIFFERENT"))
                if not same:
                    problems += 1
                    for i, (x, y) in enumerate(zip(labels, config_labels)):
                        if x != y:
                            print("      row %2d: paper=%r config=%r" % (i + 1, x, y))
                    for i, (x, y) in enumerate(zip(paper_groups, config_groups)):
                        if x != y:
                            print("      group %d: paper=%r config=%r" % (i + 1, x, y))
            missing = [l for l in labels if l not in config_labels]
            if missing:
                print("   in the paper but not in the config: %s" % missing)
                problems += 1

    # -- B and C -----------------------------------------------------------
    defaults, specs = parse_specs(os.path.join(a.models, "DMS_SGPAN_aba.py"))
    models = parse_models(os.path.join(a.models, "DMS_SGPAN_ablation.py"))

    print()
    print("B. code specification vs config")
    print("   %-46s %-46s %s" % ("TABLE II row", "config says", "code"))
    for v in rows:
        if not v.code_id:
            print("   %-46s %-46s %s" % (v.label, "(no implementation)", "-"))
            problems += 1
            continue
        changes = specs.get(v.code_id)
        if changes is None:
            print("   %-46s %-46s %s" % (v.label, v.flips, "NOT IN SPECS"))
            problems += 1
            continue
        if not changes:
            code_text = "baseline (no change)"
        else:
            code_text = "; ".join(
                "%s: %r -> %r" % (k, defaults.get(k), val) for k, val in changes)
        ok = len(changes) == 1
        print("   %-46s %-46s %s%s" % (v.label, v.flips, code_text,
                                       "" if ok else "   <- multi-dimension"))
        if not ok:
            problems += 1

    print()
    print("C. registered model classes vs config")
    for v in rows:
        if not v.code_id:
            continue
        cls = models.get(v.code_id)
        print("   %-46s %-22s %s" % (v.label, v.code_id,
                                     cls if cls else "NOT REGISTERED"))
        if not cls:
            problems += 1

    # -- D. the three registries must list the same variants ----------------
    trainers = parse_trainers(os.path.join(
        os.path.normpath(os.path.join(a.models, "..")), "DMS_SGPAN_ablation_train.py"))
    # baseline is trained by DMS_SGPAN_train.py, so it is absent from both the
    # model and trainer registries by design
    spec_keys, model_keys = set(specs) - {"baseline"}, set(models)
    print()
    print("D. spec table / model registry / trainer registry")
    print("   specs=%d variants (+baseline)  models=%d  trainers=%d"
          % (len(spec_keys), len(model_keys), len(trainers)))
    for name, left, right in (("spec vs model", spec_keys, model_keys),
                              ("spec vs trainer", spec_keys, trainers)):
        if left == right:
            print("   %-16s identical" % name)
        else:
            print("   %-16s DIFFERENT: %s" % (name, sorted(left ^ right)))
            problems += 1

    # -- E. the switches the config claims to flip are read by the model ----
    body = open(os.path.join(a.models, "DMS_SGPAN_aba.py"), encoding="utf-8").read()
    print()
    print("E. is each switched field actually read by the model?")
    for field in sorted(defaults):
        flag = "self.ablation_spec.%s" % field
        hits = body.count(flag)
        print("   %-24s read %d time(s)%s"
              % (field, hits, "" if hits else "   <- never read"))
        if not hits:
            problems += 1

    print()
    print("RESULT: %s" % ("paper, code and config agree"
                          if problems == 0
                          else "%d point(s) to resolve" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
