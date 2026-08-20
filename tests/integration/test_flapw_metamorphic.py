"""Metamorphic relations for the FLAPW stack — invariances that must hold with NO reference data.

Every silent failure this subsystem has produced violated a property the code must satisfy
regardless of the physics being right: equivalent atoms disagreeing (the smeared-TiO2 instability),
a result changing under a resolution knob far past its formal convergence (the fullpot lmax=2 EFG
artifact), an implementation path changing the answer (fork-pool corruption; IBZ-vs-full-mesh
drift). Reference comparisons (Elk, NIST) catch none of these cheaply; metamorphic relations catch
all of them. Each test names its relation (MR) explicitly.

Protocol note: SCF fixed points can be multistable (documented for ionic cells), so path-comparison
MRs (IBZ vs full mesh, pool vs serial, warm restart) are anchored at ONE converged potential via
``v_start`` — they test the mechanics, not basin selection. Absolute eigenvalues wander under
threaded BLAS (see crystal_scf docstring); MRs therefore compare splittings, spans, and EFG
tensors, never absolute levels.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import torch

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.radial import log_mesh
from gradwave.flapw.scf import SolveKArgs, _lapw_multi_k, _solve_k_args

# a small tetragonal O crystal: open shell -> real, axial EFG; two body-centred atoms are
# translation-equivalent, so their EFG tensors must match exactly by symmetry.
BCT_A = [5.4, 5.4, 6.6]                     # Bohr; c/a breaks cubic symmetry -> nonzero EFG
BCT_ATOMS = [((0.0, 0.0, 0.0), "O"), ((0.5, 0.5, 0.5), "O")]
BCT_R = {"O": 0.75}
BCT_KW = dict(ecut=150.0, lmax=2, iters=18, kmesh=(2, 2, 2), smearing=0.0, efg=True)


@pytest.fixture(scope="module")
def bct_converged():
    """One converged full-mesh reference state shared by the path-comparison MRs."""
    bands, info = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, use_symmetry=False, **BCT_KW)
    return bands, info


@pytest.mark.standard
def test_equivalent_atoms_agree(bct_converged):
    """MR: symmetry-equivalent atoms report identical EFG tensors — with NO symmetrization in the
    loop (use_symmetry=False), so this checks the raw pipeline, not the averaging that could mask
    a per-atom bug. The smeared-TiO2 instability first showed up as exactly this violation."""
    _, info = bct_converged
    t0 = info["efg"]["a0"]["tensor"]
    t1 = info["efg"]["a1"]["tensor"]
    assert np.abs(t0 - t1).max() < 5e-2 * max(1.0, np.abs(t0).max())


@pytest.mark.standard
def test_ibz_matches_full_mesh_at_fixed_point(bct_converged):
    """MR: from one converged potential, a few IBZ iterations and a few full-mesh iterations must
    stay on the same fixed point (span and EFG agree). Anchoring at the fixed point isolates the
    symmetry machinery from SCF basin multistability. This is the regression guard for the
    IBZ-vs-full drift found on TiO2 (the original BeO check was vacuous: its group had 2 ops, so
    IBZ == full mesh and a real reduction was never exercised)."""
    bands0, info0 = bct_converged
    v = info0["v_by_key"]
    b_full, i_full = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, use_symmetry=False, v_start=v,
                                       **{**BCT_KW, "iters": 6})
    b_ibz, i_ibz = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, use_symmetry=True, v_start=v,
                                     **{**BCT_KW, "iters": 6})
    assert abs(b_full["span"] - b_ibz["span"]) < 5e-2
    tf = i_full["efg"]["a0"]["tensor"]
    ti = i_ibz["efg"]["a0"]["tensor"]
    assert np.abs(tf - ti).max() < 5e-2 * max(1.0, np.abs(tf).max())


@pytest.mark.standard
def test_warm_restart_stays_on_fixed_point(bct_converged):
    """MR: restarting from a converged potential must not move the answer (a drifting restart
    means the potential round-trip or the initialisation path is lossy)."""
    bands0, info0 = bct_converged
    b1, _ = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, use_symmetry=False,
                              v_start=info0["v_by_key"], **{**BCT_KW, "iters": 4})
    assert abs(b1["span"] - bands0["span"]) < 5e-2


@pytest.mark.standard
def test_cell_units_equivalence():
    """MR: the same crystal specified via the legacy Bohr positional (a_bohr) and via the Å
    keyword (cell=) must be numerically identical — the Bohr-cell/Å-radii convention split caused
    a real bug (overlapping muffin tins from misread units)."""
    from gradwave.constants import BOHR_ANG
    kw = dict(ecut=140.0, iters=8, kmesh=(1, 1, 1), smearing=0.0, efg=True, use_symmetry=False)
    b1, i1 = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4}, **kw)
    b2, i2 = crystal_scf_multi(atoms=[((0.5, 0.5, 0.5), "Ne")], radii={"Ne": 1.4},
                               cell=6.0 * BOHR_ANG, **kw)
    assert abs(b1["span"] - b2["span"]) < 1e-9
    assert abs(i1["efg"]["a0"]["V_zz"] - i2["efg"]["a0"]["V_zz"]) < 1e-9


def test_pool_single_solve_bit_equal():
    """MR: one secular solve through a spawn worker is BIT-identical to in-process (the fork pool
    silently returned wrong eigenvalues — this pins the pool mechanics at zero tolerance)."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    v = torch.where(r <= 1.4, -8.0 / r + 3.0, torch.zeros_like(r))
    species = {"a0": {"R": 1.4, "v": v, "El": {0: -3.0, 1: -1.5, 2: -0.5}}}
    acart = [(np.array([3.0, 3.0, 3.0]), "a0")]
    args = SolveKArgs(kf=(0.25, 0.0, 0.0), cell=6.0 * 0.529177, acart=acart,
                      species=species, lmax=2, ecut=120.0, r=r, dx=dx, nb_solve=4,
                      v_nsph=None, chan=None, lodat=None, nsph_int=None, c_prev=None,
                      subspace_tol=1e-5, warp=None)
    ev_local = _lapw_multi_k(*args[:9], v_nsph=None, chan=None, lodat=None, nsph_int=None)[0]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        ev_pool = list(ex.map(_solve_k_args, [args]))[0][0]
    assert np.abs(ev_local - ev_pool).max() == 0.0


@pytest.mark.slow
def test_efg_rotation_equivariance():
    """MR: relabelling the crystal axes (c along z vs c along x) must permute the EFG tensor's
    principal frame and change nothing else — V_zz and eta are frame-invariants. Catches any
    hidden frame convention slip in the aspherical chain (density, Poisson, boundary, tensor)."""
    kw = dict(ecut=150.0, lmax=2, iters=18, kmesh=(2, 2, 2), smearing=0.0, efg=True,
              use_symmetry=False)
    _, iz = crystal_scf_multi([5.4, 5.4, 6.6], [((0.0, 0.0, 0.0), "O"),
                                                ((0.5, 0.5, 0.5), "O")], BCT_R, **kw)
    _, ix = crystal_scf_multi([6.6, 5.4, 5.4], [((0.0, 0.0, 0.0), "O"),
                                                ((0.5, 0.5, 0.5), "O")], BCT_R, **kw)
    z, x = iz["efg"]["a0"], ix["efg"]["a0"]
    assert abs(abs(z["V_zz"]) - abs(x["V_zz"])) < 5e-2 * max(1.0, abs(z["V_zz"]))
    assert abs(z["eta"] - x["eta"]) < 5e-2
    wz = np.sort(np.linalg.eigvalsh(z["tensor"]))
    wx = np.sort(np.linalg.eigvalsh(x["tensor"]))
    assert np.abs(wz - wx).max() < 5e-2 * max(1.0, np.abs(wz).max())


@pytest.mark.slow
def test_resolution_knob_stability():
    """MR: a result quoted as converged must be stable under raising a resolution knob one notch
    past its quoted setting — here the aspherical-potential cutoff fullpot_lmax (the fullpot
    lmax=2 EFG artifact was a 3x error invisible at a single setting)."""
    kw = dict(ecut=150.0, lmax=2, iters=14, kmesh=(1, 1, 2), smearing=0.0, efg=True,
              fullpot=True, use_symmetry=False)
    _, i2 = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, fullpot_lmax=2, **kw)
    _, i4 = crystal_scf_multi(BCT_A, BCT_ATOMS, BCT_R, fullpot_lmax=4, **kw)
    v2 = abs(i2["efg"]["a0"]["V_zz"])
    v4 = abs(i4["efg"]["a0"]["V_zz"])
    assert abs(v2 - v4) < 0.35 * max(1.0, v2)   # flag order-unity artifacts, allow refinement
