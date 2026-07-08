"""GRADWAVE_TEST_DEVICE=cuda runs the suite with systems on that device.

Patches setup_system at conftest-import time (before test modules are
collected, so their `from gradwave.scf.loop import setup_system` binds the
wrapper). Tests stay device-agnostic; CPU runs are unaffected.
"""

import os

_dev = os.environ.get("GRADWAVE_TEST_DEVICE")
if _dev and _dev != "cpu":
    import gradwave.scf.loop as _loop

    _orig_setup = _loop.setup_system

    def _setup_on_device(*args, **kwargs):
        return _orig_setup(*args, **kwargs).to(_dev)

    _loop.setup_system = _setup_on_device
