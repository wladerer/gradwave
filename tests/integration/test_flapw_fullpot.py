"""Full-potential non-spherical potential in the LAPW Hamiltonian (EFG step 3).

With ``v_nsph`` absent the Hamiltonian is exactly the muffin-tin one; a synthetic axial V_20 in the
sphere adds the l-channel coupling ⟨u_l Y_lm|V_20 Y_20|u_l' Y_l'm'⟩ and splits the degenerate 2p
triplet into an m=0 singlet and an m=±1 doublet — the crystal-field splitting, which matches
first-order perturbation theory ΔE = ∫u_1 V_20 u_1 dr · [G(1,0,2,0,1,0) − G(1,1,2,0,1,1)].
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG
from gradwave.flapw.atom import atomic_scf
from gradwave.flapw.efg import gaunt_matrix
from gradwave.flapw.radial import log_mesh
from gradwave.flapw.scf import _lapw_multi_k, _radial_u

_ECUT, _R = 220.0, 1.4


@pytest.fixture(scope="module")
def neon():
    """A single Ne atom at the cell centre (atomic potential); the setup for the LAPW solves."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    lc = 6.0 * BOHR_ANG
    at, v_at = atomic_scf("Ne", r, dx)
    v0 = float(v_at.numpy()[np.argmin(np.abs(r.numpy() - _R))])
    v_mt = torch.where(r <= _R, v_at - v0, torch.zeros_like(r))
    el = {0: at["2s"] - v0, 1: at["2p"] - v0, 2: -5.0 - v0}
    species = {"Ne": {"R": _R, "v": v_mt, "El": el}}
    atoms = [(np.array([lc / 2, lc / 2, lc / 2]), "Ne")]
    return dict(r=r, dx=dx, cell=np.eye(3) * lc, species=species, atoms=atoms, v_mt=v_mt, el=el)


def _solve(neon, v_nsph=None):
    return _lapw_multi_k((0, 0, 0), neon["cell"], neon["atoms"], neon["species"],
                         2, _ECUT, neon["r"], neon["dx"], 6, v_nsph=v_nsph)[0]


def test_nonspherical_reduces_to_muffin_tin(neon):
    """An empty non-spherical potential leaves the Hamiltonian exactly the muffin-tin one."""
    ev0 = _solve(neon)
    ev_empty = _solve(neon, v_nsph={"Ne": {}})
    assert float(np.abs(ev_empty - ev0).max()) < 1e-10


def test_axial_v20_crystal_field_splitting(neon):
    """A synthetic axial V_20 splits the 2p triplet into a doublet + singlet matching first-order
    perturbation theory (magnitude, multiplicity, and sign)."""
    ev0 = np.sort(_solve(neon)[1:4])
    assert float(ev0[2] - ev0[0]) < 1e-6                    # 2p is 3-fold degenerate at the start

    rr = neon["r"].numpy()[neon["r"].numpy() <= _R]
    v20 = np.full_like(rr, 2.0)                             # eV, uniform axial component
    p = np.sort(_solve(neon, v_nsph={"Ne": {(2, 0): v20}})[1:4])
    assert abs(p[0] - p[1]) < 1e-4                          # m=±1 doublet
    assert p[2] - p[1] > 0.3                                # m=0 singlet split off, higher

    u1, _ = _radial_u(1, neon["el"][1], neon["r"], neon["dx"], neon["v_mt"], _R)
    i_aa = float(np.sum(u1 * v20 * u1 * (rr * neon["dx"])))
    g = gaunt_matrix(1, 2, 0, 1)
    pt_split = i_aa * float((g[1, 1] - g[0, 0]).real)       # E(m=0) − E(m=±1)
    assert abs((p[2] - p[0]) - pt_split) < 0.1 * pt_split   # within 10% of perturbation theory


@pytest.mark.slow
def test_fullpot_cubic_reduces_to_muffin_tin():
    """The self-consistent full-potential loop converges and reproduces the muffin-tin splitting
    for cubic Ne; the non-spherical potential self-consistently vanishes at a cubic closed shell."""
    from gradwave.flapw import crystal_scf_multi
    mt, _ = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4}, ecut=140.0, iters=25)
    fp, _ = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                              ecut=140.0, iters=25, fullpot=True)
    s_mt = np.array(mt["ev"])[1:4].mean() - np.array(mt["ev"])[0]
    s_fp = np.array(fp["ev"])[1:4].mean() - np.array(fp["ev"])[0]
    assert abs(float(s_fp - s_mt)) < 0.02


@pytest.mark.slow
def test_oxygen_efg_quadrupolar_coupling():
    """Open-shell O (2p⁴) in a tetragonal cell has a nonzero EFG (a p-hole), unlike a closed-shell
    atom. Compressed along z the p_x,p_y doublet fills, leaving an axial p_z hole (η≈0), giving a
    physical ¹⁷O quadrupolar coupling — the end-to-end solid-state-NMR observable."""
    from gradwave.flapw import crystal_scf_multi
    from gradwave.flapw.nmr import quadrupolar_coupling
    _, info = crystal_scf_multi([7.0, 7.0, 5.0], [((0.5, 0.5, 0.5), "O")], {"O": 1.3},
                                ecut=200.0, iters=30, efg=True, smearing=0.15)
    site = info["efg"]["a0"]
    assert abs(site["V_zz"]) > 5.0                 # nonzero EFG (open shell)
    assert site["eta"] < 0.05                      # axial (tetragonal, doublet-filled)
    cq = quadrupolar_coupling(site["V_zz"], site["eta"], "17O")
    assert 0.5 < cq["abs_C_Q_MHz"] < 20.0          # physical ¹⁷O range
