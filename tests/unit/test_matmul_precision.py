"""gradwave's float32 matmul-precision default (gradwave._matmul).

Verifies the precedence: GRADWAVE_TF32 wins; else a user-set precision is left
alone; else "high" (TF32) is applied. The setting is inert for the fp64 physics
path and on CPU — it only speeds the opt-in reduced-precision GPU draft GEMMs.
"""

import pytest
import torch

from gradwave._matmul import apply_default_matmul_precision


@pytest.fixture(autouse=True)
def _restore_precision():
    """Save/restore the process-global matmul precision around each test."""
    saved = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved)


def test_auto_default_enables_tf32(monkeypatch):
    monkeypatch.delenv("GRADWAVE_TF32", raising=False)
    torch.set_float32_matmul_precision("highest")  # torch default, untouched
    assert apply_default_matmul_precision() == "high"
    assert torch.get_float32_matmul_precision() == "high"


def test_env_off_forces_highest(monkeypatch):
    for val in ("off", "0", "false", "highest"):
        monkeypatch.setenv("GRADWAVE_TF32", val)
        torch.set_float32_matmul_precision("high")
        assert apply_default_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "highest"


def test_env_on_enables_tf32(monkeypatch):
    monkeypatch.setenv("GRADWAVE_TF32", "on")
    torch.set_float32_matmul_precision("highest")
    assert apply_default_matmul_precision() == "high"


def test_does_not_clobber_user_choice(monkeypatch):
    """No gradwave knob + a precision already moved off the torch default =>
    the user configured it themselves; leave torch untouched."""
    monkeypatch.delenv("GRADWAVE_TF32", raising=False)
    torch.set_float32_matmul_precision("medium")
    assert apply_default_matmul_precision() is None
    assert torch.get_float32_matmul_precision() == "medium"
