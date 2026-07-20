"""Range-separated Coulomb kernels for exact and screened exchange (Layer C).

The exchange interaction between two co-densities of crystal momentum q is a
multiplicative kernel in reciprocal space,

    K(q+G) = 4π e² · screen(|q+G|²) / |q+G|²,

so an exchange build is `Σ_G |ρ̃(q+G)|² K(q+G)`. Three range separations, the
substrate for screened hybrids (HSE) and for making the mixing/range parameters
learnable:

- ``"full"``      bare 1/|q+G|²  — exact (Hartree–Fock) exchange, PBE0-style.
- ``"short_range"`` erfc(ω r)/r  — the screened exchange of HSE. The screening
  removes the q+G→0 divergence: the (q+G)=0 cell is *finite*, π e²/ω².
- ``"long_range"``  erf(ω r)/r   — the complement; diverges at q+G→0 like the
  bare kernel, so that cell is excluded (as for ``"full"``).

ω [Å⁻¹] is the range-separation parameter (differentiable — pass it as a tensor
to carry gradients into a learnable hybrid). The divergent (q+G)=0 cell of the
``full`` / ``long_range`` kernels is set to 0 here, matching the G=0 exclusion
in ``core/energies/hartree.py``; a production EXX would replace that with a
Gygi–Baldereschi auxiliary-function correction (future work).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import G2_ZERO_TOL

_MODES = ("full", "short_range", "long_range")


def coulomb_kernel(
    qg2: torch.Tensor, mode: str = "full", omega: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """K(q+G) [eV·Å²] given |q+G|² [Å⁻²] on the dense box.

    qg2: any shape. mode ∈ {"full", "short_range", "long_range"}. omega required
    (and > 0) for the screened modes. Returns the kernel with the (q+G)=0 cell
    handled per mode: finite π e²/ω² for short_range, 0 for full/long_range.
    Differentiable in ``omega``.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown Coulomb-kernel mode {mode!r}, expected one of {_MODES}")
    zero = qg2 <= G2_ZERO_TOL
    safe = torch.where(zero, torch.ones_like(qg2), qg2)
    base = 4.0 * math.pi * E2 / safe  # 4π e² / |q+G|², with the 0-cell placeheld

    if mode == "full":
        k = base
        zero_val: torch.Tensor | float = 0.0
    else:
        if omega is None:
            raise ValueError(f"mode {mode!r} needs omega (the range-separation length)")
        omega = torch.as_tensor(omega, dtype=qg2.dtype, device=qg2.device)
        gauss = torch.exp(-qg2 / (4.0 * omega ** 2))
        if mode == "short_range":
            k = base * (1.0 - gauss)
            # lim_{q+G→0} 4π e² (1−e^{−x/4ω²})/x = 4π e² /(4ω²) = π e²/ω²
            zero_val = math.pi * E2 / omega ** 2
        else:  # long_range
            k = base * gauss
            zero_val = 0.0

    zero_val = torch.as_tensor(zero_val, dtype=k.dtype, device=k.device)
    return torch.where(zero, zero_val.expand_as(k), k)
