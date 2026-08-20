"""Sternheimer-route orbital magnetization: the matrix-free velocity apply
and the dense-eigh-free CTVR evaluation (milestone 5).

- kernel identities: jl_scaled_x2_t(l, x²)·x^l == jl_t(l, x) across the
  series/closed-form branch boundary, and ylm_all(solid=True) == |g|^l·Y_lm
  (finite, exact-polynomial at g = 0);
- the analytic forward-mode velocity apply v_μ|ψ⟩ matches an INDEPENDENT
  implementation — postscf.dielectric's central-FD-in-k ∂H/∂k
  (``_dhdk_psi``) — to FD truncation error, INCLUDING at Γ where the
  pre-smooth-build cusp masking used to drop the G = 0 row's l ≥ 1
  derivative (a ~10% error caught by exactly this cross-check);
- route equivalence: the per-k CTVR tensors from cg_sternheimer solves at
  the SCF's own mesh match the dense-eigh route's tensors to ~1e-9
  (limited by the SCF's splined beta tables vs the direct transform), and
  the assembled Si orbital magnetization is the TR null.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.ylm import ylm_all
from gradwave.postscf._response import insulator_window, pad_coeffs
from gradwave.postscf.dielectric import _dhdk_psi
from gradwave.postscf.kgeometry import (
    BlochHK,
    VelocityApply,
    _ctvr_tensor_dense,
    _sternheimer_ctvr,
    orbital_magnetization_sternheimer,
)
from gradwave.pseudo.radial_torch import jl_scaled_x2_t, jl_t
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


def _si_scf(kmesh, use_symmetry=False):
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=kmesh, nbands=8, use_symmetry=use_symmetry,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


@pytest.fixture(scope="module")
def si_gamma():
    """Γ-only Si — Γ's G = 0 row is the velocity cusp regression point."""
    return _si_scf((1, 1, 1))


@pytest.fixture(scope="module")
def si_mesh():
    """Si on an unreduced 2×2×2 mesh (Γ included)."""
    return _si_scf((2, 2, 2))


# --------------------------------------------------------------------------- #
# smooth-kernel identities                                                    #
# --------------------------------------------------------------------------- #


def test_jl_scaled_matches_jl():
    x = torch.linspace(0.0, 12.0, 481, dtype=torch.float64)  # crosses SERIES_X = 4
    for l in range(4):
        got = jl_scaled_x2_t(l, x * x) * x.pow(l)
        ref = jl_t(l, x)
        assert (got - ref).abs().max().item() < 1e-14
        # exact, finite value at the origin: j_l(x)/x^l → 1/(2l+1)!!
        val0 = jl_scaled_x2_t(l, torch.zeros(1, dtype=torch.float64)).item()
        assert abs(val0 - 1.0 / float(np.prod(np.arange(1, 2 * l + 2, 2)))) < 1e-15


def test_solid_ylm_matches_scaled_ylm():
    gen = torch.Generator().manual_seed(3)
    g = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    norm = g.norm(dim=-1, keepdim=True)
    y = ylm_all(3, g)
    s = ylm_all(3, g, solid=True)
    for l in range(4):
        sl = slice(l * l, (l + 1) * (l + 1))
        assert (s[:, sl] - norm.pow(l) * y[:, sl]).abs().max().item() < 1e-12
    # at g = 0: l = 0 keeps Y00, every l ≥ 1 entry is exactly 0 (polynomial)
    s0 = ylm_all(3, torch.zeros(1, 3, dtype=torch.float64), solid=True)
    assert s0[0, 1:].abs().max().item() == 0.0
    assert torch.isfinite(s0).all()


# --------------------------------------------------------------------------- #
# velocity apply                                                              #
# --------------------------------------------------------------------------- #


def test_velocity_apply_matches_independent_fd(si_gamma):
    # cross-check against postscf.dielectric's central-FD-in-k ∂H/∂k — a
    # fully independent implementation (splined tables, FD instead of fwAD).
    # Γ-only: the G = 0 row is the regression that caught the cusp masking.
    res = si_gamma
    system = res.system
    bk = system.batch
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    c_occ = pad_coeffs(list(res.coeffs), bk.npw_max)[:, :nocc]
    v = VelocityApply(system)
    for mu in range(3):
        va = v.apply(c_occ, mu)
        vfd = _dhdk_psi(system, c_occ, mu, 1e-3)
        assert (va - vfd).abs().max().item() < 1e-6  # measured 7.1e-8


# --------------------------------------------------------------------------- #
# route equivalence and the physical null                                     #
# --------------------------------------------------------------------------- #


def test_sternheimer_route_matches_dense_and_null(si_mesh):
    res = si_mesh
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    mu_c = float(res.fermi)
    t_st, w = _sternheimer_ctvr(res, mu_c)
    assert abs(w.sum().item() - 1.0) < 1e-12

    # per-k CTVR tensor vs the dense-eigh route at Γ and a generic mesh k
    for ik in (0, len(res.system.spheres) - 1):
        sph = res.system.spheres[ik]
        hk = BlochHK.from_scf(res, sph.k_frac)
        t_d = _ctvr_tensor_dense(hk.h, hk.k_cart(sph.k_frac),
                                 list(range(nocc)), mu_c)
        rel = (t_st[ik] - t_d).abs().max().item() / t_d.abs().max().item()
        assert rel < 1e-7  # measured ~1e-9 (splined vs direct beta tables)

    # assembled moment: the TR null, same tensors — no extra solves
    t_avg = torch.einsum("k,kab->ab", w.to(t_st.dtype), t_st)
    i_vec = torch.stack([(t_avg[1, 2] - t_avg[2, 1]).imag,
                         (t_avg[2, 0] - t_avg[0, 2]).imag,
                         (t_avg[0, 1] - t_avg[1, 0]).imag])
    assert i_vec.abs().max().item() < 1e-10  # eV·Å²; measured ~1e-14


def test_sternheimer_rejects_symmetry_reduced_mesh():
    res = _si_scf((2, 2, 2), use_symmetry=True)
    with pytest.raises(NotImplementedError, match="full k-mesh"):
        orbital_magnetization_sternheimer(res, float(res.fermi))
