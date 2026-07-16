"""Patch an existing DS-AGC models_DS_AGC.py for dynamic channel counts.

Run this script from the DS-AGC repository root:
    python patch_models_DS_AGC_for_deap.py

It creates models_DS_AGC.py.bak before modifying the file.
"""
from pathlib import Path
import re
import shutil

MODEL_PATH = Path(__file__).resolve().parent / "models_DS_AGC.py"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Cannot find {MODEL_PATH}. Put this patch script beside models_DS_AGC.py."
    )

text = MODEL_PATH.read_text(encoding="utf-8")
pattern = re.compile(
    r"self\.fea_extrator_f\s*=\s*feature_extractor\(\s*310\s*,\s*64\s*,\s*64\s*\)"
)
replacement = (
    "self.fea_extrator_f = feature_extractor("
    "channel * net_params['num_of_features'], 64, 64)"
)
new_text, count = pattern.subn(replacement, text)

if count == 0:
    if "channel * net_params['num_of_features']" in text:
        print("models_DS_AGC.py is already patched.")
        raise SystemExit(0)
    raise RuntimeError(
        "Could not find the hard-coded feature_extractor(310, 64, 64). "
        "Check whether your models_DS_AGC.py differs from the public repository."
    )
if count != 1:
    raise RuntimeError(f"Expected one replacement, but found {count}.")

backup = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".bak")
if not backup.exists():
    shutil.copy2(MODEL_PATH, backup)
MODEL_PATH.write_text(new_text, encoding="utf-8")
print(f"Patched: {MODEL_PATH}")
print(f"Backup:  {backup}")
