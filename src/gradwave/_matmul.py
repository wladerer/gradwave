"""Float32 matmul precision default for gradwave.

gradwave's physics runs in float64 (complex128), which ``set_float32_matmul_precision``
does not touch — so this default is inert for a normal fp64 SCF. It matters only for
the *opt-in reduced-precision draft paths* (``GRADWAVE_FP32_EXPANSION``,
``GRADWAVE_SUBSPACE_STORAGE=complex64``): a complex64 matmul decomposes into real
float32 matmuls, and on a tensor-core GPU (Ampere+) routing those through TF32 was
measured ~2x faster on the Rayleigh-Ritz build (RTX 3050) at no cost to the final
result, because the draft is re-certified in fp64 before any convergence claim. TF32
has no effect on CPU (no tensor cores), so this is a free GPU-draft win and a no-op
everywhere else.

torch's default is ``"highest"`` (full fp32, no TF32). gradwave sets ``"high"`` (TF32)
on import, honouring an explicit user choice:

- ``GRADWAVE_TF32=off`` (or ``0`` / ``false`` / ``highest``) — force ``"highest"``,
  disabling TF32.
- ``GRADWAVE_TF32`` unset and the current precision already moved off the torch
  default ``"highest"`` — the user configured it themselves; leave torch untouched.
- otherwise apply ``"high"``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("gradwave.matmul")

_OFF_VALUES = ("off", "0", "false", "no", "highest")


def apply_default_matmul_precision() -> str | None:
    """Apply gradwave's float32 matmul-precision policy once, at import.

    Returns the precision applied (``"high"`` / ``"highest"``), or ``None`` when
    gradwave deliberately left torch as-is because the user had already configured
    it. See the module docstring for precedence."""
    import torch

    raw = os.environ.get("GRADWAVE_TF32")
    if raw is not None:
        want = "highest" if raw.strip().lower() in _OFF_VALUES else "high"
        torch.set_float32_matmul_precision(want)
        logger.debug("matmul: GRADWAVE_TF32=%r -> %s", raw, want)
        return want

    # no gradwave knob: don't clobber a user who already moved it off the default
    if torch.get_float32_matmul_precision() != "highest":
        logger.debug("matmul: user-set precision %r, leaving as-is",
                     torch.get_float32_matmul_precision())
        return None

    torch.set_float32_matmul_precision("high")
    logger.debug("matmul: auto-default -> high (TF32 for fp32 draft GEMMs)")
    return "high"
