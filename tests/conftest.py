"""GRADWAVE_TEST_DEVICE=cuda runs the suite with systems on that device.

Patches setup_system at conftest-import time (before test modules are
collected, so their `from gradwave.scf.loop import setup_system` binds the
wrapper). Tests stay device-agnostic; CPU runs are unaffected.
"""

import os
import sys
from pathlib import Path

# Put the repo root on sys.path so `from tests.helpers import ...` resolves no
# matter which directory pytest was launched from (tests/ is a package).
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_dev = os.environ.get("GRADWAVE_TEST_DEVICE")
if _dev and _dev != "cpu":
    import gradwave.scf.loop as _loop
    import gradwave.scf.uspp as _uspp

    _orig_setup = _loop.setup_system
    _orig_setup_uspp = _uspp.setup_uspp

    def _setup_on_device(*args, **kwargs):
        return _orig_setup(*args, **kwargs).to(_dev)

    def _setup_uspp_on_device(*args, **kwargs):
        return _orig_setup_uspp(*args, **kwargs).to(_dev)

    _loop.setup_system = _setup_on_device
    _uspp.setup_uspp = _setup_uspp_on_device
