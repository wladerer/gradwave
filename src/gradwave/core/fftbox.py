"""r ↔ G transforms on the dense FFT box (Layer A — differentiable).

## Conventions (normative for the whole package)

Wavefunctions:  ψ_nk(r) = (1/√Ω) Σ_G c_nk(G) e^{i(k+G)·r},  Σ_G |c(G)|² = 1.
Fields:         f(r) = Σ_G f̃(G) e^{iG·r}  (Fourier series; f̃ carries f's units).

With N grid points r_j:
    sphere → r-grid:  f(r_j) = Σ_G c(G) e^{iG·r_j}        = N·ifftn(box)
    r-grid → coeffs:  f̃(G)  = (1/N) Σ_j f(r_j) e^{−iG·r_j} = fftn(f)/N

Complex autograd: torch uses the Wirtinger convention — autograd.grad of a
real scalar w.r.t. complex input returns the CONJUGATE Wirtinger derivative
(∂L/∂z̄ doubled appropriately), i.e. the steepest-ascent direction. All
hand-written backward passes (scf/implicit.py) must match inner products
against this convention. Decided once, here.

All ops are out-of-place; this module must stay traceable by autograd and
usable under torch.func transforms.
"""

from __future__ import annotations

import torch


def sphere_to_box(coeffs: torch.Tensor, flat_idx: torch.Tensor, shape) -> torch.Tensor:
    """Scatter sphere coefficients (..., npw) into dense boxes (..., n1, n2, n3)."""
    batch = coeffs.shape[:-1]
    n = shape[0] * shape[1] * shape[2]
    flat = torch.zeros(*batch, n, dtype=coeffs.dtype, device=coeffs.device)
    flat = flat.index_add(-1, flat_idx, coeffs)
    return flat.reshape(*batch, *shape)


def box_to_sphere(box: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
    """Gather sphere coefficients (..., npw) from dense boxes (..., n1, n2, n3)."""
    flat = box.reshape(*box.shape[:-3], -1)
    return flat.index_select(-1, flat_idx)


def g_to_r_box(f_g: torch.Tensor, *, real: bool = False) -> torch.Tensor:
    """Box Fourier coefficients f̃(G) → periodic function values on the r-grid.

    The inverse of :func:`r_to_g` for a field already on the dense box:
    f(r_j) = Σ_G f̃(G) e^{iG·r_j} = N·ifftn(f̃), over the last three (box) axes
    with N their product. ``f_g`` must be a dense box (..., n1, n2, n3) — scatter
    a masked/flat G-vector with :func:`sphere_to_box` (or ``box[mask] = vec;
    box.reshape(shape)``) first. Pass ``real=True`` to return the real part, for
    a field known to describe a real function (density, local/Hartree potential,
    augmentation charge). Out-of-place; stays autograd- and torch.func-traceable.
    """
    n = f_g.shape[-3] * f_g.shape[-2] * f_g.shape[-1]
    out = torch.fft.ifftn(f_g, dim=(-3, -2, -1)) * n
    return out.real if real else out


def g_to_r(coeffs: torch.Tensor, flat_idx: torch.Tensor, shape) -> torch.Tensor:
    """Sphere coefficients → periodic function values on the r-grid.

    Returns f(r_j) = Σ_G c(G) e^{iG·r_j}, shape (..., n1, n2, n3), complex.
    """
    return g_to_r_box(sphere_to_box(coeffs, flat_idx, shape))


def r_to_g(f: torch.Tensor) -> torch.Tensor:
    """Function values on the r-grid → Fourier coefficients f̃(G) on the box."""
    n = f.shape[-3] * f.shape[-2] * f.shape[-1]
    return torch.fft.fftn(f, dim=(-3, -2, -1)) / n
