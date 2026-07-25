"""Phase 1 of the differentiable-composition channel.

A per-atom weight lambda blends two endpoint pseudopotentials for the local
potential and the ionic charge, and dE/dlambda is the alchemical derivative of
those terms. The tests check endpoint exactness (lambda=0 is species A, lambda=1
is species B) and the derivative against finite difference on both the Ewald
charge channel and the Hellmann-Feynman local term. This is not the virtual
crystal approximation. The endpoints are real species and the derivative is
taken there. Si and Ge are the pair, so both the local potential (different form
factors) and the charge (4 vs 14 valence) vary with lambda.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.hamiltonian import becp, projectors
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import (
    alchemical_charges,
    blend_local_table,
    blend_projector_data,
    endpoint_local_tables,
    per_atom_local_tables,
)
from gradwave.scf.loop import setup_system
from gradwave.scf.setup_common import _unique_shells

RY = 13.605693122994
DG = Path(__file__).resolve().parents[2] / "benchmarks" / "delta_gauge" / "pseudos"
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "qe" / "pseudos"


def _si_cell():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return cell, pos


def _setup():
    cell, pos = _si_cell()
    si = parse_upf(DG / "Si.upf")
    ge = parse_upf(DG / "Ge.upf")
    system = setup_system(cell, pos, [0, 0], [si], ecut=20 * RY, kmesh=(1, 1, 1))
    return system, si, ge, cell, pos


def _shells(grid):
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    return _unique_shells(g_flat)


def test_endpoint_tables_match_plain_setup():
    # the A endpoint table must equal the per-species table a plain setup builds
    system, si, ge, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, _ = endpoint_local_tables(si, ge, uniq, inv, system.grid.shape)
    assert torch.allclose(si_tab, system.vloc_tables[0], atol=1e-10)


def test_blend_endpoints_are_exact():
    system, si, ge, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, ge_tab = endpoint_local_tables(si, ge, uniq, inv, system.grid.shape)
    b0 = blend_local_table(si_tab, ge_tab, torch.tensor(0.0, dtype=RDTYPE))
    b1 = blend_local_table(si_tab, ge_tab, torch.tensor(1.0, dtype=RDTYPE))
    assert torch.allclose(b0, si_tab, atol=1e-12)
    assert torch.allclose(b1, ge_tab, atol=1e-12)


def test_identity_blend_has_zero_gradient():
    # blending a species with itself is lambda-independent
    system, si, _, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, _ = endpoint_local_tables(si, si, uniq, inv, system.grid.shape)
    lam = torch.tensor(0.4, dtype=RDTYPE, requires_grad=True)
    blend_local_table(si_tab, si_tab, lam).sum().backward()
    assert lam.grad.abs().item() < 1e-12


def test_charge_channel_ewald_gradient_vs_fd():
    # atom 0 transmutes Si -> Ge (Z 4 -> 14); atom 1 stays Si
    system, si, ge, cell, pos = _setup()
    pos_t = torch.tensor(pos, dtype=RDTYPE)
    z_si, z_ge = float(si.z_valence), float(ge.z_valence)

    def e_ewald(x):
        q = torch.stack([alchemical_charges(z_si, z_ge, x),
                         torch.tensor(z_si, dtype=RDTYPE)])
        return ewald_energy(pos_t, q, cell)

    lam0 = 0.3
    lam = torch.tensor(lam0, dtype=RDTYPE, requires_grad=True)
    e_ewald(lam).backward()
    g_ad = lam.grad.item()
    h = 1e-5
    g_fd = (e_ewald(torch.tensor(lam0 + h, dtype=RDTYPE)).item()
            - e_ewald(torch.tensor(lam0 - h, dtype=RDTYPE)).item()) / (2 * h)
    assert abs(g_ad - g_fd) < 1e-4 * max(1.0, abs(g_fd)), (g_ad, g_fd)


def test_local_potential_gradient_vs_fd():
    # d(E_loc)/dlambda at a fixed density, autograd vs finite difference
    system, si, ge, _, pos = _setup()
    grid = system.grid
    uniq, inv = _shells(grid)
    si_tab, ge_tab = endpoint_local_tables(si, ge, uniq, inv, grid.shape)
    torch.manual_seed(0)
    rho_g = torch.randn(grid.shape, dtype=torch.complex128)

    def e_loc(x):
        vatom = per_atom_local_tables(system.vloc_tables, system.species_index,
                                      {0: (si_tab, ge_tab, x)})
        vg = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume,
                               vloc_atom=vatom)
        return local_energy(rho_g, vg, grid.volume)

    lam0 = 0.35
    lam = torch.tensor(lam0, dtype=RDTYPE, requires_grad=True)
    e_loc(lam).backward()
    g_ad = lam.grad.item()
    h = 1e-5
    g_fd = (e_loc(torch.tensor(lam0 + h, dtype=RDTYPE)).item()
            - e_loc(torch.tensor(lam0 - h, dtype=RDTYPE)).item()) / (2 * h)
    assert abs(g_ad - g_fd) < 1e-5 * max(1.0, abs(g_fd)), (g_ad, g_fd)


# --------------------------------------------------------------------------- #
#  phase 2: the nonlocal (KB) projector channel                               #
# --------------------------------------------------------------------------- #
def _one_atom_pd(element: str):
    """Gamma-point ProjectorData and Bloch coeffs for a single `element` atom in
    a fixed cubic cell, so Si and Ge share the same k-sphere."""
    cell = 5.0 * np.eye(3)
    pos = np.zeros((1, 3))
    upf = parse_upf(DG / f"{element}.upf")
    system = setup_system(cell, pos, [0], [upf], ecut=16 * RY, kmesh=(1, 1, 1))
    return system.proj_data[0], system.positions


def _e_nl(pd, pos, c):
    p = projectors(pd, pos)
    b = becp(p, c)
    occ = torch.ones(1, c.shape[0], dtype=RDTYPE)
    return nonlocal_energy([b], pd.dij_full, occ, torch.ones(1, dtype=RDTYPE))


def test_nonlocal_blend_endpoints_are_exact():
    pd_si, pos = _one_atom_pd("Si")
    pd_ge, _ = _one_atom_pd("Ge")
    torch.manual_seed(1)
    c = torch.randn(4, pd_si.kpg.shape[0], dtype=torch.complex128)
    e_si, e_ge = _e_nl(pd_si, pos, c), _e_nl(pd_ge, pos, c)
    e0 = _e_nl(blend_projector_data(pd_si, pd_ge, torch.tensor(0.0, dtype=RDTYPE)), pos, c)
    e1 = _e_nl(blend_projector_data(pd_si, pd_ge, torch.tensor(1.0, dtype=RDTYPE)), pos, c)
    assert abs(e0.item() - e_si.item()) < 1e-9 * max(1.0, abs(e_si.item()))
    assert abs(e1.item() - e_ge.item()) < 1e-9 * max(1.0, abs(e_ge.item()))


def test_nonlocal_blend_gradient_vs_fd():
    pd_si, pos = _one_atom_pd("Si")
    pd_ge, _ = _one_atom_pd("Ge")
    torch.manual_seed(2)
    c = torch.randn(4, pd_si.kpg.shape[0], dtype=torch.complex128)

    def e(x):
        return _e_nl(blend_projector_data(pd_si, pd_ge, x), pos, c)

    lam0 = 0.4
    lam = torch.tensor(lam0, dtype=RDTYPE, requires_grad=True)
    e(lam).backward()
    g_ad = lam.grad.item()
    h = 1e-5
    g_fd = (e(torch.tensor(lam0 + h, dtype=RDTYPE)).item()
            - e(torch.tensor(lam0 - h, dtype=RDTYPE)).item()) / (2 * h)
    assert abs(g_ad - g_fd) < 1e-6 * max(1.0, abs(g_fd)), (g_ad, g_fd)
    # the alchemical nonlocal derivative is E_nl(B) - E_nl(A) by linearity in D
    e_si, e_ge = _e_nl(pd_si, pos, c), _e_nl(pd_ge, pos, c)
    assert abs(g_ad - (e_ge.item() - e_si.item())) < 1e-7 * max(1.0, abs(g_ad))


# --------------------------------------------------------------------------- #
#  phase 3: a full alchemical SCF, endpoint exactness through convergence      #
# --------------------------------------------------------------------------- #
def test_alchemical_scf_endpoints_match_pure():
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.alchemical import setup_alchemical_system
    from gradwave.scf.loop import scf

    # C <-> Si (both 4 valence, so the electron count is fixed across lambda)
    c_upf = parse_upf(FIX / "C_ONCV_PBE-1.2.upf")
    si_upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    ecut = 20 * RY
    kw = dict(smearing="gaussian", width=0.2, etol=1e-8, rhotol=1e-7,
              max_iter=200, verbose=False)

    def pure(upf, nb):
        return scf(setup_system(cell, pos, [0, 0], [upf], ecut=ecut,
                                kmesh=(1, 1, 1), nbands=nb), PBE(), **kw)

    # lambda = 1 reproduces pure Si
    alc1 = setup_alchemical_system(cell, pos, c_upf, si_upf, 1.0, ecut=ecut)
    r1 = scf(alc1, PBE(), **kw)
    r_si = pure(si_upf, alc1.nbands)
    assert r1.converged and r_si.converged
    assert abs(float(r1.energies.total) - float(r_si.energies.total)) < 1e-5, \
        (float(r1.energies.total), float(r_si.energies.total))

    # lambda = 0 reproduces pure C (same cell)
    alc0 = setup_alchemical_system(cell, pos, c_upf, si_upf, 0.0, ecut=ecut)
    r0 = scf(alc0, PBE(), **kw)
    r_c = pure(c_upf, alc0.nbands)
    assert abs(float(r0.energies.total) - float(r_c.energies.total)) < 1e-5, \
        (float(r0.energies.total), float(r_c.energies.total))


@pytest.mark.standard
def test_alchemical_scf_gradient_hellmann_feynman():
    # dE/dlambda through the converged SCF vs a full-SCF finite difference
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.alchemical import alchemical_energy_gradient, setup_alchemical_system
    from gradwave.scf.loop import scf

    c_upf = parse_upf(FIX / "C_ONCV_PBE-1.2.upf")
    si_upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    ecut = 20 * RY
    kw = dict(smearing="gaussian", width=0.2, etol=1e-9, rhotol=1e-8,
              max_iter=300, verbose=False)

    def e(lam):
        s = setup_alchemical_system(cell, pos, c_upf, si_upf, lam, ecut=ecut)
        return float(scf(s, PBE(), **kw).energies.free_energy)

    lam0 = 0.5
    res = scf(setup_alchemical_system(cell, pos, c_upf, si_upf, lam0, ecut=ecut),
              PBE(), **kw)
    g_hf = float(alchemical_energy_gradient(res, lam0))
    h = 2e-3
    g_fd = (e(lam0 + h) - e(lam0 - h)) / (2 * h)
    assert abs(g_hf - g_fd) < 0.02, (g_hf, g_fd)


@pytest.mark.standard
def test_per_site_alchemical_heterovalent_endpoint():
    # per-site lambda = [1, 0] on a Si/Ge cell must reproduce the real ordered
    # Ge/Si cell (heterovalent, Z 14 and 4, neutral integer electron count)
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.alchemical import setup_alchemical_system
    from gradwave.scf.loop import scf

    si = parse_upf(DG / "Si.upf")
    ge = parse_upf(DG / "Ge.upf")
    a = 5.5
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    ecut = 30 * RY
    kw = dict(smearing="gaussian", width=0.2, etol=1e-8, rhotol=1e-7,
              max_iter=300, verbose=False)

    alc = setup_alchemical_system(cell, pos, si, ge, [1.0, 0.0], ecut=ecut)
    r_alc = scf(alc, PBE(), **kw)
    # real ordered cell: atom 0 = Ge (species 1), atom 1 = Si (species 0)
    real = setup_system(cell, pos, [1, 0], [si, ge], ecut=ecut, kmesh=(1, 1, 1),
                        nbands=alc.nbands)
    r_real = scf(real, PBE(), **kw)
    assert r_alc.converged and r_real.converged
    assert abs(float(r_alc.energies.total) - float(r_real.energies.total)) < 1e-5, \
        (float(r_alc.energies.total), float(r_real.energies.total))


@pytest.mark.standard
def test_per_site_alchemical_gradient():
    # per-site dE/dlambda_i vs finite difference on each site (C <-> Si)
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.alchemical import alchemical_energy_gradient, setup_alchemical_system
    from gradwave.scf.loop import scf

    c_upf = parse_upf(FIX / "C_ONCV_PBE-1.2.upf")
    si_upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    ecut = 20 * RY
    kw = dict(smearing="gaussian", width=0.2, etol=1e-9, rhotol=1e-8,
              max_iter=300, verbose=False)
    lam0 = np.array([0.5, 0.3])

    def e(vec):
        s = setup_alchemical_system(cell, pos, c_upf, si_upf, vec, ecut=ecut)
        return float(scf(s, PBE(), **kw).energies.free_energy)

    res = scf(setup_alchemical_system(cell, pos, c_upf, si_upf, lam0, ecut=ecut),
              PBE(), **kw)
    g_hf = alchemical_energy_gradient(res, lam0).numpy()
    h = 2e-3
    for i in range(2):
        up, dn = lam0.copy(), lam0.copy()
        up[i] += h
        dn[i] -= h
        g_fd = (e(up) - e(dn)) / (2 * h)
        assert abs(g_hf[i] - g_fd) < 0.02, (i, g_hf[i], g_fd)


@pytest.mark.standard
def test_heterovalent_gradient_core_and_janak():
    # Si -> Ge (Z 4 -> 14, NLCC semicore): the gradient needs both the core-
    # correction XC term and the Janak chemical-potential term mu*dN/dlambda
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.alchemical import alchemical_energy_gradient, setup_alchemical_system
    from gradwave.scf.loop import scf

    si = parse_upf(DG / "Si.upf")
    ge = parse_upf(DG / "Ge.upf")
    a = 5.5
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    ecut = 30 * RY
    kw = dict(smearing="gaussian", width=0.2, etol=1e-9, rhotol=1e-8,
              max_iter=300, verbose=False)

    def e(lam):
        s = setup_alchemical_system(cell, pos, si, ge, lam, ecut=ecut)
        return float(scf(s, PBE(), **kw).energies.free_energy)

    lam0 = 0.5
    res = scf(setup_alchemical_system(cell, pos, si, ge, lam0, ecut=ecut),
              PBE(), **kw)
    g_hf = float(alchemical_energy_gradient(res, lam0, xc=PBE()))
    h = 2e-3
    g_fd = (e(lam0 + h) - e(lam0 - h)) / (2 * h)
    assert abs(g_hf - g_fd) < 0.05, (g_hf, g_fd)
