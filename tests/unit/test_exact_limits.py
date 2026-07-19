"""Exactly solvable limits (Tier-2 gates, docs/verification.md).

Analytic oracles for the eigenproblem stack (sphere construction, FFT
apply, Davidson) with no pseudopotential and no reference code:

- Empty lattice: with V=0 the exact spectrum at any k is ħ²|k+G|²/2m.
  Run on a triclinic cell at a generic k so no lattice symmetry can hide
  an indexing/convention error.
- Cosine potential: V(x) = V0·cos(2πx/a) maps to Mathieu's equation.
  Band edges at Γ and X are Mathieu characteristic values (scipy
  mathieu_a/mathieu_b): Γ carries the π-periodic solutions
  (a_0, b_2, a_2, ...), X the 2π-anti-periodic ones (b_1, a_1, b_3, ...).
  A nontrivial analytic band structure through the full 3D machinery; the
  plane-wave error for a single-harmonic potential is superexponentially
  small at this cutoff, so agreement is at solver tolerance.

The transverse directions are kept short (L = 1.2 Å) so transverse-excited
states sit ~100 eV up and the low spectrum is purely one-dimensional.
"""

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.batch import BatchedHamiltonian, build_batched
from gradwave.core.hamiltonian import ProjectorData
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.solvers.davidson import davidson_batched

RY = 13.605693122994

torch.set_num_threads(4)


def _solve(cell, ecut, k_frac, v_of_grid, nb, seed=0, tol=1e-11):
    """Lowest nb eigenvalues of T + V at one k via the production stack."""
    grid = build_fft_grid(np.asarray(cell, dtype=np.float64), ecut)
    sph = build_gsphere(grid, ecut, k_frac)
    pd = ProjectorData(
        atom_index=torch.zeros(0, dtype=torch.int64),
        f_ylm_phase_free=torch.zeros((0, sph.npw), dtype=CDTYPE),
        kpg=sph.kpg,
        dij_full=torch.zeros((0, 0), dtype=RDTYPE),
    )
    bk = build_batched([sph], [pd])
    v_r = v_of_grid(grid)
    h = BatchedHamiltonian(bk, grid.shape, v_r, bk.proj_phase_free)
    gen = torch.Generator().manual_seed(seed)
    x0 = (torch.randn(1, nb, sph.npw, generator=gen, dtype=torch.float64)
          + 1j * torch.randn(1, nb, sph.npw, generator=gen, dtype=torch.float64))
    x0 = (x0 / (1.0 + HBAR2_2M * sph.kpg2)).to(CDTYPE)
    x0 = x0 / torch.linalg.norm(x0, dim=-1, keepdim=True)
    res = davidson_batched(h.apply, x0, bk.t, bk.mask, tol=tol, max_iter=200)
    return res.eigenvalues[0], sph


def test_empty_lattice_free_electron_bands():
    cell = [[3.1, 0.0, 0.0], [0.4, 2.7, 0.0], [0.2, -0.3, 3.3]]  # triclinic
    k = (0.31, 0.17, -0.23)  # generic k, no symmetry
    nb = 12
    eigs, sph = _solve(cell, 30 * RY, k, lambda g: torch.zeros(g.shape,
                                                               dtype=RDTYPE), nb)
    exact = torch.sort(HBAR2_2M * sph.kpg2).values[:nb]
    assert float((eigs - exact).abs().max()) < 1e-9


def test_cosine_potential_mathieu_bands():
    from scipy.special import mathieu_a, mathieu_b

    a, L = 3.0, 1.2
    unit = HBAR2_2M * (np.pi / a) ** 2  # Mathieu energy scale [eV]
    q = 1.5
    v0 = 2.0 * q * unit  # V = 2q·unit·cos(2πx/a) ↔ Mathieu parameter q

    def v_cos(grid):
        n1 = grid.shape[0]
        x = torch.arange(n1, dtype=RDTYPE) / n1
        v = v0 * torch.cos(2.0 * np.pi * x)
        return v[:, None, None].expand(grid.shape).contiguous()

    # keep only levels safely below the first transverse excitation (~104 eV)
    e_cap = 80.0
    cases = {
        (0.0, 0.0, 0.0): [mathieu_a(0, q), mathieu_b(2, q), mathieu_a(2, q),
                          mathieu_b(4, q), mathieu_a(4, q)],
        (0.5, 0.0, 0.0): [mathieu_b(1, q), mathieu_a(1, q), mathieu_b(3, q),
                          mathieu_a(3, q)],
    }
    cell = np.diag([a, L, L])
    for k, chars in cases.items():
        exact = np.array([unit * c for c in chars])
        exact = exact[exact < e_cap]
        assert len(exact) >= 4
        eigs, _ = _solve(cell, 60 * RY, k, v_cos, len(exact) + 2)
        err = np.abs(eigs.numpy()[: len(exact)] - exact).max()
        assert err < 1e-7, f"k={k}: max |E - Mathieu| = {err:.2e} eV"
