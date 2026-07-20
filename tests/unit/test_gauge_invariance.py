"""U(N) gauge invariance (Tier-1 metamorphic, docs/verification.md).

The KS density and total energy are functionals of the density matrix
Σ_n f_n |ψ_n⟩⟨ψ_n|, so a unitary rotation among bands with EQUAL
occupations is a pure gauge change: ρ, every energy term, and all becp
contractions must be invariant. A per-band quantity that leaks into a
supposedly gauge-invariant assembly (a missing conjugate, an occ applied
on the wrong side of a contraction) breaks this at O(1).

Unit-level and off-stationarity on purpose: random non-eigenstate
coefficients, a different random U(3) at every k, rattled P1 geometry —
nothing cancels by symmetry or by stationarity.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import becp_b, density_b, projectors_b
from gradwave.core.energies.total import total_energy
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import _stack_dij, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(autouse=True)
def _limit_threads():
    torch.set_num_threads(4)


def _random_unitary(n, gen):
    m = (torch.randn(n, n, generator=gen, dtype=torch.float64)
         + 1j * torch.randn(n, n, generator=gen, dtype=torch.float64))
    q, r = torch.linalg.qr(m)
    # fix the QR phase ambiguity so q is a haar-ish unitary
    return (q * (r.diagonal() / r.diagonal().abs())[None, :]).to(CDTYPE)


def _energy_and_rho(system, xc, c, occ):
    grid, bk, spheres = system.grid, system.batch, system.spheres
    nk = bk.nk
    kw = system.kweights
    projs = projectors_b(bk, system.positions)
    rho = density_b(c, occ, kw, bk, grid.shape, grid.volume)
    eb = total_energy(
        coeffs_per_k=[c[ik, :, : int(bk.npw[ik])] for ik in range(nk)],
        occ=occ, kweights=kw, spheres=spheres, grid=grid, rho=rho,
        positions=system.positions, charges=system.charges,
        species_index=system.species_index, vloc_tables=system.vloc_tables,
        becp_per_k=[becp_b(projs, c)[ik] for ik in range(nk)],
        dij_full=_stack_dij(system), xc=xc, rho_core=system.rho_core,
    )
    return float(eb.total), rho


def test_gauge_rotation_invariance():
    from gradwave.core.xc.pbe import PBE
    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(lattice, pos, [0, 0], [si], ecut=12 * RY,
                          kmesh=(2, 1, 1))
    bk = system.batch
    nb, ndeg = 5, 3  # bands 0..2 share f=2.0 — the gauge-degenerate block
    occ = torch.tensor([2.0, 2.0, 2.0, 0.8, 0.3],
                       dtype=RDTYPE)[None, :].repeat(bk.nk, 1)

    gen = torch.Generator().manual_seed(11)
    c = (torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE)
         + 1j * torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE))
    c = c.to(CDTYPE) / (1.0 + bk.t)[:, None, :] * bk.mask[:, None, :]
    c = c / torch.linalg.norm(c, dim=-1, keepdim=True)

    c_rot = c.clone()
    for ik in range(bk.nk):  # independent gauge per k — it is a per-k freedom
        u = _random_unitary(ndeg, gen)
        c_rot[ik, :ndeg] = u @ c[ik, :ndeg]

    e_a, rho_a = _energy_and_rho(system, PBE(), c, occ)
    e_b, rho_b = _energy_and_rho(system, PBE(), c_rot, occ)
    assert float((rho_b - rho_a).abs().max() / rho_a.abs().max()) < 1e-12
    assert abs(e_b - e_a) < 1e-10, f"gauge rotation changed E by {e_b - e_a:.2e}"

    # control: rotating across UNEQUAL occupations is not a gauge freedom —
    # the same machinery must see it (the test has teeth)
    c_bad = c.clone()
    u = _random_unitary(2, gen)
    c_bad[0, 3:5] = u @ c[0, 3:5]  # f = 0.8 vs 0.3
    e_c, _ = _energy_and_rho(system, PBE(), c_bad, occ)
    assert abs(e_c - e_a) > 1e-3
