"""Metamorphic / physical-invariant tests for the SCF and its operators.

Oracle-free checks: each asserts that a transformation (relabeling the grid under
the space group, integrating the density, doubling the cell) leaves a computed
result invariant to the theory's *exact* identity — no reference value needed. A
convention/phase/sign/normalization bug breaks them by orders of magnitude, while
the tiny cell's physical inaccuracy is irrelevant to the identity being tested.

Complements ``tests/property/test_scf_invariants.py`` (translation floor, ΣF = 0,
inversion, rotation/stress co-rotation, spin flip) — this file adds the invariants
that file does NOT cover: **RhoSymmetrizer idempotence** (P² = P), **charge
conservation** (∫ρ = N_e), **stress-tensor symmetry** (σ = σᵀ), and **supercell
energy additivity** (E[2×1×1] = 2·E[1×1×1] at matched k-density).

Systems are the smallest that still exercise the full chain: 2-atom diamond Si
(insulator, smearing='none') at 12 Ry, coarse k. The identities hold at any
cutoff/grid/k-mesh, so this stays cheap while keeping teeth.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.grids import build_fft_grid
from gradwave.postscf.stress import stress as compute_stress
from gradwave.scf.loop import scf, setup_system
from gradwave.symmetry import RhoSymmetrizer, find_spacegroup
from tests.helpers import RY, si_fcc, si_upf

_ECUT = 12 * RY
# smearing='none' (Si insulator) keeps every SCF sub-second; tight tols so the
# invariants hold well below their assertion thresholds.
_SCF_KW = dict(smearing="none", width=0.0, etol=1e-9, rhotol=1e-8,
               diago_tol=1e-11, verbose=False)


@pytest.fixture(autouse=True)
def _limit_threads():
    torch.set_num_threads(4)


@pytest.fixture(scope="module")
def _si():
    return si_upf()


@pytest.fixture(scope="module")
def _si_prim_scf(_si):
    """One converged primitive 2-atom Si SCF at kmesh (2,1,1), shared by the
    charge-conservation, stress-symmetry, and supercell-additivity checks so the
    module runs a single primitive SCF (plus one supercell SCF)."""
    cell, pos = si_fcc()
    sys = setup_system(cell, pos, [0, 0], [_si], ecut=_ECUT, kmesh=(2, 1, 1))
    res = scf(sys, LDA_PW92(), **_SCF_KW)
    assert res.converged
    return res


# --- density-operator idempotence (unit-scale, no SCF) -----------------------

def test_rho_symmetrizer_idempotent(_si):
    """The G-space density symmetrizer is a projector onto the space-group-
    invariant subspace, so applying it twice equals applying it once (P² = P).
    Exact on the density sphere (the mask drops the Nyquist shell where the
    Miller fold is ill-defined). A wrong rotation map or phase breaks P² = P at
    O(1). No SCF: acts on a seeded random ρ(G), so this is a fast unit check."""
    cell, pos = si_fcc()
    frac = pos @ np.linalg.inv(cell)
    sg = find_spacegroup(cell, frac, [0, 0])
    grid = build_fft_grid(cell, 15 * RY, equal_dims=True)  # box must be closed under the group
    rsym = RhoSymmetrizer(grid.shape, sg, dens_mask=grid.dens_mask)

    gen = torch.Generator().manual_seed(7)
    raw = torch.randn(*grid.shape, generator=gen, dtype=torch.float64)
    rho_g = torch.fft.fftn(raw).to(torch.complex128) / raw.numel()
    once = rsym.apply(rho_g)
    twice = rsym.apply(once)
    # exact projector on the masked subspace → machine-precision idempotence
    assert torch.allclose(once, twice, atol=1e-13)


# --- converged-SCF invariants ------------------------------------------------

@pytest.mark.slow
def test_charge_conservation(_si_prim_scf):
    """∫ρ(r) dr = N_electrons: the converged total density integrates to the
    electron count. Guards the density normalization / grid-quadrature weight
    (a wrong Ω/N_G factor or a lost k-weight breaks it). Tolerance well above the
    ~1e-10 the per-step normalization achieves but far below any real defect."""
    res = _si_prim_scf
    grid = res.system.grid
    dvol = float(grid.volume) / float(grid.n_points)
    n_int = float(res.rho.sum()) * dvol
    assert abs(n_int - res.system.n_electrons) < 1e-6


@pytest.mark.slow
def test_stress_tensor_symmetric(_si_prim_scf):
    """The stress tensor is symmetric, σ_ij = σ_ji. NOTE: ``stress()`` assembles
    σ = ½(∂E/∂ε + (∂E/∂ε)ᵀ)/Ω and the space-group projection preserves symmetry,
    so this is largely a *structural* guard (it fails only if that symmetrization
    is dropped or the projection is wrong) rather than a deep physical test.
    Tight tol because the construction makes it exact to rounding."""
    sigma = compute_stress(_si_prim_scf, LDA_PW92()).detach().cpu().numpy()
    assert np.abs(sigma - sigma.T).max() < 1e-8  # eV/Å³; construction-exact


@pytest.mark.slow
def test_supercell_energy_additivity(_si_prim_scf, _si):
    """Metamorphic size-extensivity: a 2×1×1 supercell at half the k-density
    along the doubled axis has exactly twice the primitive energy. The primitive
    kmesh (2,1,1) = {0, ½}·b₁ folds *exactly* onto the supercell Γ point
    (Γ-centered MP), so E[super] = 2·E[prim] up to FFT-grid discretization of the
    doubled cell. A per-cell (extensive vs intensive) bug in any energy term —
    Ewald, Hartree, band sum — breaks additivity at O(1).

    TOLERANCE UNCERTAIN: the deviation floor is set by whether ``build_fft_grid``
    rounds the doubled axis to exactly 2× the primitive sampling; if not, the XC
    quadrature differs slightly between the two grids. Set conservatively at
    2e-3 eV/atom and flagged for careful validation on the big machine."""
    cell, pos = si_fcc()
    # double along the first lattice vector a₁ = cell[0]
    cell_s = cell.copy()
    cell_s[0] = 2.0 * cell[0]
    pos_s = np.concatenate([pos, pos + cell[0]], axis=0)
    sys_s = setup_system(cell_s, pos_s, [0, 0, 0, 0], [_si],
                         ecut=_ECUT, kmesh=(1, 1, 1))
    res_s = scf(sys_s, LDA_PW92(), **_SCF_KW)
    assert res_s.converged

    e_prim_per_atom = float(_si_prim_scf.energies.free_energy) / 2.0
    e_super_per_atom = float(res_s.energies.free_energy) / 4.0
    assert abs(e_super_per_atom - e_prim_per_atom) < 2e-3
