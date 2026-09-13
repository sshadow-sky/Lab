"""Force this isolated experiment to use its own ``src/models`` package.

The support modules (config/data_utils/utils) may come from LibEER, but the
model implementation must never be selected by an ambient ``PYTHONPATH``.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_MODEL_DIR = Path(__file__).resolve().parent / "models"
_INIT = _MODEL_DIR / "__init__.py"
if not _INIT.exists():
    raise ImportError(f"Isolated model package is missing: {_MODEL_DIR}")

_spec = spec_from_file_location(
    "models",
    _INIT,
    submodule_search_locations=[str(_MODEL_DIR)],
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load isolated model package: {_MODEL_DIR}")

_package = module_from_spec(_spec)
_package.__path__ = [str(_MODEL_DIR)]
_package.__package__ = "models"
sys.modules["models"] = _package
_spec.loader.exec_module(_package)

