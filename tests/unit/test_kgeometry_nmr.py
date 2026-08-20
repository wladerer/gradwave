"""Finite-q velocity perturbations (postscf.kgeometry_nmr, milestone 7).

Route equivalence is the load-bearing validation: the batched k+q
Sternheimer solves at the SCF mesh (kpq_map umklapp embedding + the
½(v^(k)+v^(k+q)) operator + cg_sternheimer) must reproduce the dense twin
built from explicit BlochHK matrices at k and the UNFOLDED k+q — measured
rel ~1e-9 at q = 0 and at commensurate q where 6 of 8 mesh points wrap
through the zone boundary. TR symmetry of Si gives P(q) = P(−q)ᵀ =
conj(P(−q)); at q = 0 the tensor is Hermitian and the machinery reduces to
the M5 velocity solves.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf._response import insulator_window
from gradwave.postscf.kgeometry_nmr import (
    paramagnetic_tensor,
    paramagnetic_tensor_dense,
    velocity_perturbation_q,
)
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


@pytest.fixture(scope="module")
def si_mesh():
    """A 2×1×1 unreduced mesh: q = (½,0,0) makes one of the two k wrap —
    the smallest system that exercises the umklapp embedding."""
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(2, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


def test_umklapped_q_matches_dense(si_mesh):
    res = si_mesh
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    q = (0.5, 0.0, 0.0)
    sol = velocity_perturbation_q(res, q, cg_tol=1e-9)
    assert int((np.abs(sol.g0).sum(axis=1) > 0).sum()) == 1  # 1/2 k wraps
    p_st = paramagnetic_tensor(sol)
    for ik in range(2):  # both: the wrapping and the non-wrapping point
        kf = res.system.spheres[ik].k_frac
        p_d = paramagnetic_tensor_dense(res, kf, q, nocc)
        rel = (p_st[ik] - p_d).abs().max().item() / p_d.abs().max().item()
        assert rel < 1e-7  # measured ~1e-9


def test_q_zero_reduces_to_velocity_solves(si_mesh):
    res = si_mesh
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    sol = velocity_perturbation_q(res, (0.0, 0.0, 0.0), cg_tol=1e-9)
    assert np.abs(sol.g0).max() == 0 and np.array_equal(
        sol.jidx, np.arange(len(res.system.spheres))
    )
    p_st = paramagnetic_tensor(sol)
    # Hermitian at q = 0 (P_μν = Σ ⟨v_μu|G|v_νu⟩ with real G weights) …
    assert (p_st - p_st.mH).abs().max().item() < 1e-8 * p_st.abs().max().item()
    # … and equal to the dense reference
    p_d = paramagnetic_tensor_dense(res, res.system.spheres[1].k_frac,
                                    (0.0, 0.0, 0.0), nocc)
    rel = (p_st[1] - p_d).abs().max().item() / p_d.abs().max().item()
    assert rel < 1e-7  # measured ~4e-10


def test_rejects_incommensurate_and_reduced():
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-8, rhotol=1e-7, verbose=False, max_iter=80)
    with pytest.raises(ValueError):
        velocity_perturbation_q(res, (0.3, 0.0, 0.0))  # not on the mesh
