"""Polar vector-field symmetrization (gradwave.symmetry), the machinery the
E-field DFPT dielectric/Born response uses to run with IBZ reduction.

VectorFieldSymmetrizer folds the density response Δρ_α — a POLAR vector field
— through the point group (component mixing by S = Aᵀ W A⁻ᵀ, no det factor);
symmetrize_tensor / symmetrize_atom_tensor take the point-group star sum of the
ε∞ and Born tensors. On a cubic group these must project any tensor onto the
isotropic subspace, which is the structural guarantee behind the with-vs-without
symmetry dielectric oracle.
"""

import numpy as np
import torch

from gradwave.grids import build_fft_grid
from gradwave.symmetry import (
    VectorFieldSymmetrizer,
    find_spacegroup,
    symmetrize_atom_tensor,
    symmetrize_tensor,
)
from tests.helpers import RY, si_fcc

SI_CELL, SI_POS = si_fcc()


def _si_group():
    frac = SI_POS @ np.linalg.inv(SI_CELL)
    return find_spacegroup(SI_CELL, frac, [0, 0])


def test_vector_field_symmetrizer_idempotent():
    """Group-averaging a polar vector field is a projector: applying it twice
    equals applying it once (exact on the density sphere)."""
    sg = _si_group()
    grid = build_fft_grid(SI_CELL, 15 * RY, equal_dims=True)
    vsym = VectorFieldSymmetrizer(grid.shape, sg, SI_CELL, dens_mask=grid.dens_mask)
    gen = torch.Generator().manual_seed(7)
    raw = torch.randn(3, *grid.shape, generator=gen, dtype=torch.float64)
    vg = torch.fft.fftn(raw, dim=(-3, -2, -1)).to(torch.complex128) / raw[0].numel()
    once = vsym.apply(vg)
    twice = vsym.apply(once)
    assert torch.allclose(once, twice, atol=1e-13)
    # the fold keeps each component a real field (Δρ_α(r) is real)
    rho_r = torch.fft.ifftn(once * once[0].numel(), dim=(-3, -2, -1))
    assert float(rho_r.imag.abs().max()) < 1e-11 * float(rho_r.real.abs().max())


def test_vector_fold_differs_from_scalar_fold():
    """The polar fold is NOT three independent scalar folds — the components
    mix. On a cubic group a field pointing along x picks up y/z weight from the
    3-fold [111] rotations, so a per-component scalar fold gives a different
    (wrong) answer. Guards against a naive 'symmetrize each Δρ_α alone'."""
    from gradwave.symmetry import RhoSymmetrizer

    sg = _si_group()
    grid = build_fft_grid(SI_CELL, 15 * RY, equal_dims=True)
    vsym = VectorFieldSymmetrizer(grid.shape, sg, SI_CELL, dens_mask=grid.dens_mask)
    rsym = RhoSymmetrizer(grid.shape, sg, dens_mask=grid.dens_mask)
    gen = torch.Generator().manual_seed(11)
    raw = torch.randn(3, *grid.shape, generator=gen, dtype=torch.float64)
    vg = torch.fft.fftn(raw, dim=(-3, -2, -1)).to(torch.complex128) / raw[0].numel()
    vec = vsym.apply(vg)
    scal = torch.stack([rsym.apply(vg[a]) for a in range(3)])
    assert float((vec - scal).abs().max()) > 1e-3  # genuinely different operators


def test_symmetrize_tensor_cubic_isotropic():
    """On the cubic Si point group, symmetrize_tensor projects any rank-2
    tensor onto a scalar multiple of the identity (isotropic ε∞)."""
    sg = _si_group()
    gen = torch.Generator().manual_seed(3)
    t = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    ts = symmetrize_tensor(t, sg, SI_CELL)
    iso = float(torch.diagonal(ts).mean())
    assert torch.allclose(ts, iso * torch.eye(3, dtype=torch.float64), atol=1e-12)
    # trace preserved (the invariant of the projection)
    assert abs(float(torch.trace(ts)) - float(torch.trace(t))) < 1e-10


def test_symmetrize_atom_tensor_born_constraints():
    """Per-atom Born tensor on cubic Si: symmetrize_atom_tensor makes each
    site's tensor isotropic-diagonal and both sublattices equal (the sites are
    related by the space group)."""
    sg = _si_group()
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(2, 3, 3, generator=gen, dtype=torch.float64)
    zs = symmetrize_atom_tensor(z, sg, SI_CELL)
    for s in range(2):
        d = float(torch.diagonal(zs[s]).mean())
        assert torch.allclose(zs[s], d * torch.eye(3, dtype=torch.float64), atol=1e-10)
    # the two Si atoms are symmetry-equivalent → identical Born tensors
    assert torch.allclose(zs[0], zs[1], atol=1e-10)
