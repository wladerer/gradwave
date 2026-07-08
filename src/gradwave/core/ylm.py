"""Real spherical harmonics l ≤ 4, as differentiable torch functions (Layer A).

l = 4 exists for the USPP/PAW augmentation channels (L up to 2·l_max_beta);
KB projectors themselves stop at l = 3.

Ordering within each l mirrors QE: m = 0, +1, −1, +2, −2, +3, −3.
Coefficients are verified against sympy/scipy by scripts/gen_ylm.py and by
tests/unit/test_ylm.py (orthonormality on a spherical quadrature).

Sign conventions per m are internally consistent; physical observables only
ever contain products Y_lm(ĝ)·Y_lm(ĝ'), so a global sign per (l,m) channel
cancels — but it must be the SAME function everywhere, hence one module.

Polynomials are written in terms of r² = x²+y²+z² (which is 1 for real
directions) so that the |g| = 0 rows — where we zero the unit components —
give exactly 0 for every l ≥ 1 and Y₀₀ = 1/(2√π): the correct q → 0 limit
for KB projectors (j_l(0) = 0 kills l ≥ 1 there anyway).
"""

from __future__ import annotations

import math

import torch

C00 = 0.5 * math.sqrt(1.0 / math.pi)
C1 = math.sqrt(3.0 / (4.0 * math.pi))
C20 = 0.25 * math.sqrt(5.0 / math.pi)
C21 = 0.5 * math.sqrt(15.0 / math.pi)
C22 = 0.25 * math.sqrt(15.0 / math.pi)
C30 = 0.25 * math.sqrt(7.0 / math.pi)
C31 = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
C32 = 0.25 * math.sqrt(105.0 / math.pi)
C32M = 0.5 * math.sqrt(105.0 / math.pi)
C33 = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
C40 = 3.0 / 16.0 / math.sqrt(math.pi)
C41 = 3.0 / 8.0 * math.sqrt(10.0 / math.pi)
C42 = 3.0 / 8.0 * math.sqrt(5.0 / math.pi)
C42M = 3.0 / 4.0 * math.sqrt(5.0 / math.pi)
C43 = 3.0 / 8.0 * math.sqrt(70.0 / math.pi)
C44 = 3.0 / 16.0 * math.sqrt(35.0 / math.pi)
C44M = 3.0 / 4.0 * math.sqrt(35.0 / math.pi)


def ylm_all(lmax: int, g: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """All Y_lm for l = 0..lmax at directions ĝ.

    g: (..., 3) — need not be normalized (zero vectors allowed).
    Returns (..., (lmax+1)²), ordered (0,0),(1,0),(1,1),(1,-1),(2,0),...
    """
    if lmax > 4:
        raise ValueError("ylm_all supports lmax <= 4")
    norm = torch.linalg.norm(g, dim=-1, keepdim=True)
    unit = g / torch.clamp(norm, min=eps)
    zero = (norm < eps).squeeze(-1)
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    x = torch.where(zero, torch.zeros_like(x), x)
    y = torch.where(zero, torch.zeros_like(y), y)
    z = torch.where(zero, torch.zeros_like(z), z)
    r2 = x * x + y * y + z * z  # 1 for real directions, 0 for zero-vector rows

    out = [torch.full_like(x, C00)]
    if lmax >= 1:
        out += [C1 * z, C1 * x, C1 * y]
    if lmax >= 2:
        out += [
            C20 * (3.0 * z * z - r2),
            C21 * x * z,
            C21 * y * z,
            C22 * (x * x - y * y),
            C21 * x * y,
        ]
    if lmax >= 3:
        out += [
            C30 * z * (5.0 * z * z - 3.0 * r2),
            C31 * x * (5.0 * z * z - r2),
            C31 * y * (5.0 * z * z - r2),
            C32 * z * (x * x - y * y),
            C32M * x * y * z,
            C33 * x * (x * x - 3.0 * y * y),
            C33 * y * (3.0 * x * x - y * y),
        ]
    if lmax >= 4:
        z2, x2, y2 = z * z, x * x, y * y
        out += [
            C40 * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r2 * r2),
            C41 * x * z * (7.0 * z2 - 3.0 * r2),
            C41 * y * z * (7.0 * z2 - 3.0 * r2),
            C42 * (x2 - y2) * (7.0 * z2 - r2),
            C42M * x * y * (7.0 * z2 - r2),
            C43 * x * z * (x2 - 3.0 * y2),
            C43 * y * z * (3.0 * x2 - y2),
            C44 * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2),
            C44M * x * y * (x2 - y2),
        ]
    return torch.stack(out, dim=-1)
