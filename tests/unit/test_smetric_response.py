"""S-metric magnetic-response bridge, Phase 1 (generalized velocity +
Sternheimer for H|ψ⟩ = ε S|ψ⟩).

The decisive gate is the NC-LIMIT REDUCTION: with a norm-conserving
pseudopotential S = I, ∂S/∂k = 0, so the new S-metric velocity + S-metric
Sternheimer must reproduce the shipped norm-conserving path.

- ``test_cg_sternheimer_smetric_bit_identical_at_S_eq_I`` proves it at the
  solver level to ROUND-OFF (bit-for-bit): the S-metric branch driven with an
  explicit identity S executes the same float operations as the plain lockstep
  solve, on a synthetic generalized eigenproblem. The same test shows the
  S ≠ I branch actually solves the generalized Sternheimer (residual and
  S-orthogonality near round-off).
- ``test_nc_limit_reduction_velocity_perturbation_q`` drives it through the
  real NMR entry point ``velocity_perturbation_q`` on a Si NC ground state:
  the identity-S δu matches the shipped δu to CG-certification precision (the
  reference runs the per-column + dense-draft fast path, so agreement is at
  ``cg_tol``, not bit-for-bit — the solver-level test above carries the exact
  claim).

Supporting checks:
- ``OverlapVelocity`` (the ∂S/∂k operator) matches a central finite difference
  of the nonlocal overlap S(k) on a real PAW system.
- The batched S-metric Sternheimer RUNS on a converged PAW ground state and
  returns a sane δu that is S-orthogonal to the occupied block (the Phase-1
  smoke bar; full physical validation of the PAW δu is Phase 2).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradwave.core.hamiltonian import projectors
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import cg_sternheimer, insulator_window, sternheimer_shift
from gradwave.postscf.kgeometry import OverlapVelocity, _overlap_kbprojectors
from gradwave.postscf.kgeometry_nmr import velocity_perturbation_q
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

PSEUDOS = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


# --------------------------------------------------------------------------- #
# Solver-level NC-limit reduction (bit-for-bit) + S != I correctness           #
# --------------------------------------------------------------------------- #


def _generalized_eigenpairs(Hs, Ss, nocc):
    nk, n = len(Hs), Hs[0].shape[0]
    c_occ = torch.zeros(nk, nocc, n, dtype=CDTYPE)
    eps = torch.zeros(nk, nocc, dtype=RDTYPE)
    for k in range(nk):
        Li = torch.linalg.inv(torch.linalg.cholesky(Ss[k]))
        w, vv = torch.linalg.eigh(Li @ Hs[k] @ Li.conj().T)
        c_occ[k] = (Li.conj().T @ vv).T[:nocc]  # S-orthonormal rows
        eps[k] = w[:nocc]
    return c_occ, eps


def test_cg_sternheimer_smetric_bit_identical_at_S_eq_I():
    """The S-metric cg_sternheimer branch, run with an explicit identity S,
    is bit-for-bit the plain norm-conserving solve; and with a genuine overlap
    it solves the generalized Sternheimer to round-off."""
    torch.manual_seed(0)
    nk, n, nocc = 3, 14, 4

    def rand_herm(n):
        a = torch.randn(n, n, dtype=CDTYPE)
        return (a + a.conj().T) / 2

    def make_spd(n, j):
        a = torch.randn(n, n, dtype=CDTYPE)
        return a @ a.conj().T + j * torch.eye(n, dtype=CDTYPE)

    # ---- S != I: the generalized solve is correct ----
    Hs = [rand_herm(n) for _ in range(nk)]
    Ss = [make_spd(n, 2.0) for _ in range(nk)]
    Ht, St = torch.stack(Hs), torch.stack(Ss)
    c_occ, eps = _generalized_eigenpairs(Hs, Ss, nocc)
    h = SimpleNamespace(apply=lambda x: torch.einsum("kij,kbj->kbi", Ht, x))

    def s_apply(x):
        return torch.einsum("kij,kbj->kbi", St, x)

    # bk shim: cg_sternheimer only reads bk.t (Teter table); no apply_cols, so
    # the norm-conserving reference below takes the lockstep path too.
    bk = SimpleNamespace(t=torch.stack([Hs[k].diagonal().real.clamp_min(1e-3)
                                        for k in range(nk)]))
    s_occ = s_apply(c_occ)
    o = torch.randn(nk, nocc, n, dtype=CDTYPE)
    ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), o)
    rhs = -(o - torch.einsum("kbn,kng->kbg", ov, s_occ))  # in the P_S^† complement
    shift = sternheimer_shift(eps)
    dpsi = cg_sternheimer(h, bk, c_occ, eps, rhs, torch.zeros_like(rhs), shift,
                          tol=1e-13, max_iter=4000, s_apply=s_apply, s_occ=s_occ)
    phys = h.apply(dpsi) - eps[..., None] * s_apply(dpsi)
    ovp = torch.einsum("kng,kbg->kbn", c_occ.conj(), phys)
    res_cond = (phys - torch.einsum("kbn,kng->kbg", ovp, s_occ)) - rhs
    assert res_cond.norm().item() < 1e-11 * rhs.norm().item()
    sortho = torch.einsum("kng,kbg->kbn", s_occ.conj(), dpsi).abs().max().item()
    assert sortho < 1e-12 * dpsi.abs().max().item()

    # ---- S = I: bit-for-bit reduction ----
    Hs = [rand_herm(n) for _ in range(nk)]
    Ht = torch.stack(Hs)
    eye = [torch.eye(n, dtype=CDTYPE) for _ in range(nk)]
    c_occ, eps = _generalized_eigenpairs(Hs, eye, nocc)
    h = SimpleNamespace(apply=lambda x: torch.einsum("kij,kbj->kbi", Ht, x))
    bk = SimpleNamespace(t=torch.stack([Hs[k].diagonal().real.clamp_min(1e-3)
                                        for k in range(nk)]))
    o = torch.randn(nk, nocc, n, dtype=CDTYPE)
    ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), o)
    rhs = -(o - torch.einsum("kbn,kng->kbg", ov, c_occ))
    shift = sternheimer_shift(eps)
    d_nc = cg_sternheimer(h, bk, c_occ, eps, rhs, torch.zeros_like(rhs), shift,
                          tol=1e-10, max_iter=4000)
    d_id = cg_sternheimer(h, bk, c_occ, eps, rhs, torch.zeros_like(rhs), shift,
                          tol=1e-10, max_iter=4000,
                          s_apply=lambda x: x, s_occ=c_occ)
    assert (d_nc - d_id).abs().max().item() == 0.0  # identical float ops


# --------------------------------------------------------------------------- #
# NC-limit reduction through the real NMR entry point                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def si_nc_mesh():
    """Si NC SCF on a 2×1×1 unreduced mesh (mirrors test_kgeometry_nmr)."""
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(2, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
def test_nc_limit_reduction_velocity_perturbation_q(si_nc_mesh, q):
    """velocity_perturbation_q driven with an explicit identity overlap S (and
    ∂S/∂k = 0) reproduces the shipped norm-conserving δu to CG-certification
    precision — the S-metric plumbing is wired through the NMR entry point."""
    res = si_nc_mesh
    ref = velocity_perturbation_q(res, q, cg_tol=1e-10)
    sm = velocity_perturbation_q(res, q, cg_tol=1e-10, s_apply=lambda x: x)
    rel = (sm.dpsi - ref.dpsi).abs().max().item() / ref.dpsi.abs().max().item()
    assert rel < 1e-7, f"NC-limit reduction rel {rel:.3e} at q={q}"


# --------------------------------------------------------------------------- #
# PAW system + converged state (shared by the ∂S/∂k and smoke tests)           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def si_paw():
    """A converged Si PAW ground state + its USPPSystem (loose ecut)."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(2)
    paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell, pos = si_fcc()
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=12 * RY,
                        kmesh=(2, 1, 1), ecutrho=48 * RY, nbands=8)
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=60)
    assert res["converged"]
    return res


def test_overlap_velocity_matches_finite_difference(si_paw):
    """OverlapVelocity's ∂S/∂k_μ apply matches a central finite difference of
    the nonlocal overlap S_nl(k) c = Σ_ij q_ij⟨β_i(k)|c⟩ β_j(k) at every mesh
    k (the batched apply carries per-k projectors, so ``c`` is the full
    (nk, nb, npw_max) block)."""
    res = si_paw
    system = res["system"]
    ov = OverlapVelocity(system)
    nk = len(system.spheres)
    nb = 3
    torch.manual_seed(0)
    c = torch.zeros(nk, nb, ov.npw_max, dtype=CDTYPE)
    for ik in range(nk):
        c[ik, :, : ov.npw[ik]] = torch.randn(nb, ov.npw[ik], dtype=CDTYPE)

    h = 1e-5
    for mu in range(3):
        an = ov.apply(c, mu)  # (nk, nb, npw_max)
        e = torch.zeros(3, dtype=RDTYPE)
        e[mu] = 1.0
        for ik in range(nk):
            sph = system.spheres[ik]
            npw = int(ov.npw[ik])
            kb = _overlap_kbprojectors(system, sph)
            q = kb.dij_full.to(CDTYPE)
            k0 = sph.k_cart.to(RDTYPE)

            def s_nl(k_cart, _kb=kb, _q=q, _ik=ik, _npw=npw):
                p = _kb.p(k_cart)  # (nproj, npw)
                b = torch.einsum("pg,bg->bp", p.conj(), c[_ik, :, :_npw])
                return torch.einsum("bp,pq,qg->bg", b, _q, p)

            fd = (s_nl(k0 + h * e) - s_nl(k0 - h * e)) / (2 * h)
            rel = (an[ik, :, :npw] - fd).abs().max().item() / max(
                fd.abs().max().item(), 1e-30)
            assert rel < 1e-6, f"∂S/∂k_{mu} k={ik} vs FD rel {rel:.3e}"


def test_smetric_sternheimer_runs_on_paw(si_paw):
    """Phase-1 smoke: the batched S-metric Sternheimer converges on a real PAW
    ground state and returns a δu that is S-orthogonal to the occupied block."""
    from gradwave.constants import HBAR2_2M
    from gradwave.postscf.uspp_frozen import frozen_veff, screened_dscr
    from gradwave.scf.uspp_loop import _HkS

    res = si_paw
    system = res["system"]
    shape = system.grid.shape
    coeffs = res["coeffs"]
    nocc = insulator_window(res["occupations"], 2.0, "insulating")
    veff = frozen_veff(res, PBE())
    dscr = screened_dscr(res, PBE(), veff)

    spheres = system.spheres
    nk = len(spheres)
    npw = [int(s.kpg2.shape[0]) for s in spheres]
    npw_max = max(npw)

    hks, c_occ = [], torch.zeros(nk, nocc, npw_max, dtype=CDTYPE)
    tt = torch.zeros(nk, npw_max, dtype=RDTYPE)
    kpg = torch.zeros(nk, npw_max, 3, dtype=RDTYPE)
    for ik, sph in enumerate(spheres):
        p = projectors(system.proj_data[ik], system.positions)
        hks.append(_HkS(sph, shape, veff[0], system.proj_data[ik], p,
                        dscr[0], system.q_full))
        c_occ[ik, :, :npw[ik]] = coeffs[ik][:nocc]
        tt[ik, :npw[ik]] = HBAR2_2M * sph.kpg2
        kpg[ik, :npw[ik]] = sph.kpg.to(RDTYPE)

    def _perk(fn, x):
        out = torch.zeros_like(x)
        for ik in range(nk):
            out[ik, :, :npw[ik]] = fn(ik, x[ik, :, :npw[ik]])
        return out

    h = SimpleNamespace(apply=lambda x: _perk(lambda ik, c: hks[ik].h(c), x))

    def s_apply(x):
        return _perk(lambda ik, c: hks[ik].s(c), x)

    bk = SimpleNamespace(t=tt)
    eps = res["eigenvalues"][:, :nocc].to(RDTYPE)
    s_occ = s_apply(c_occ)
    shift = sternheimer_shift(eps)

    # a sane S-orthogonal perturbation: the kinetic velocity along x, projected
    # off the occupied block with the S-metric conduction projector.
    vkin = (2.0 * HBAR2_2M) * kpg[:, None, :, 0] * c_occ
    o = torch.einsum("kng,kbg->kbn", c_occ.conj(), vkin)
    rhs = -(vkin - torch.einsum("kbn,kng->kbg", o, s_occ))

    dpsi = cg_sternheimer(h, bk, c_occ, eps, rhs, torch.zeros_like(rhs), shift,
                          tol=1e-9, max_iter=400, s_apply=s_apply, s_occ=s_occ)

    assert torch.isfinite(dpsi).all()
    assert dpsi.abs().max().item() > 1e-8  # a nontrivial response
    phys = h.apply(dpsi) - eps[..., None] * s_apply(dpsi)
    ovp = torch.einsum("kng,kbg->kbn", c_occ.conj(), phys)
    res_cond = (phys - torch.einsum("kbn,kng->kbg", ovp, s_occ)) - rhs
    assert res_cond.norm().item() < 1e-6 * rhs.norm().item()
    sortho = torch.einsum("kng,kbg->kbn", s_occ.conj(), dpsi).abs().max().item()
    assert sortho < 1e-7 * dpsi.abs().max().item(), f"S-ortho {sortho:.3e}"


# --------------------------------------------------------------------------- #
# Phase 2: velocity_perturbation_q end-to-end on the PAW path (USPP backend)    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
def test_uspp_velocity_perturbation_q_runs(si_paw, q):
    """Phase-2 gate 1: the full ``velocity_perturbation_q`` entry point runs on
    a real PAW ground state through the USPP S-metric backend (dsvel/s_apply on)
    and returns an S-orthogonal δu. The eigen residual of the batched operator
    on the SCF state certifies that the KBProjectors β + screened D + q_full
    reproduce the SCF Hamiltonian/overlap (consistent column orderings)."""
    from gradwave.postscf.kgeometry_nmr import (
        build_uspp_response_ctx,
        velocity_perturbation_q,
    )

    res = si_paw
    system = res["system"]
    ctx = build_uspp_response_ctx(res, PBE())
    nocc = insulator_window(res["occupations"], 2.0, "insulating")

    # eigen residual ||H c − ε S c|| / ||ε S c|| on the k-spheres (q=0 fold=id)
    c_all = torch.zeros(len(system.spheres), res["coeffs"][0].shape[0],
                        ctx.bk.npw_max, dtype=CDTYPE)
    for ik, c in enumerate(res["coeffs"]):
        c_all[ik, :, : c.shape[1]] = c
    c_occ = c_all[:, :nocc]
    eps = res["eigenvalues"][:, :nocc].to(RDTYPE)
    h0, _ = ctx.build_hkq(torch.arange(len(system.spheres)))
    resid = h0.apply(c_occ) - eps[..., None].to(CDTYPE) * h0.s_apply(c_occ)
    denom = (eps[..., None].to(CDTYPE) * h0.s_apply(c_occ)).norm().item()
    assert resid.norm().item() / denom < 1e-6, "operator does not reproduce SCF state"

    sol = velocity_perturbation_q(res, q, cg_tol=1e-9, max_iter=600, uspp=ctx)
    assert torch.isfinite(sol.dpsi).all()
    assert sol.dpsi.abs().max().item() > 1e-6  # nontrivial response
    hkq, _ = ctx.build_hkq(torch.as_tensor(sol.jidx, dtype=torch.long))
    s_occ_kq = hkq.s_apply(sol.c_occ_kq)
    for mu in range(3):
        sortho = torch.einsum(
            "kng,kbg->kbn", s_occ_kq.conj(), sol.dpsi[mu]).abs().max().item()
        assert sortho < 1e-6 * sol.dpsi[mu].abs().max().item(), \
            f"S-ortho μ={mu} {sortho:.3e}"


def test_uspp_smooth_continuity_closure(si_paw):
    """Phase-2 gate 2: the PAW smooth+nonlocal induced current satisfies the
    continuity equation q·mean[j_kin + j_nl] = mean s at G=0. The kinetic
    current ALONE is broken by the [V_NL, e^{-iqr}] commutator; adding the KB
    nonlocal current (with the SCREENED D) closes it to the ecut-vanishing
    truncation + augmentation-charge remainder — the physical validation of the
    S-metric PAW δu."""
    from gradwave.postscf.kgeometry_nmr import (
        build_uspp_response_ctx,
        uspp_smooth_continuity,
        velocity_perturbation_q,
    )

    res = si_paw
    ctx = build_uspp_response_ctx(res, PBE())
    q_frac = np.array([0.5, 0.0, 0.0])
    e_pol = np.array([0.0, 1.0, 0.0])
    sol = velocity_perturbation_q(res, q_frac, cg_tol=1e-11, max_iter=800, uspp=ctx)
    c = uspp_smooth_continuity(res, ctx, sol, e_pol)
    scale = abs(c["source"])
    rel_kin = abs(c["q_j_kin"] - c["source"]) / scale
    rel_full = abs(c["q_j_kin"] + c["q_j_nl"] - c["source"]) / scale
    assert rel_kin > 1e-2, f"KB term should matter (kin-only rel {rel_kin:.2e})"
    assert rel_full < 3e-3, f"smooth continuity not closed (rel {rel_full:.2e})"


# --------------------------------------------------------------------------- #
# Phase 2: the USPP-routed sigma_shielding NC-limit reduction (the PR-A gate)   #
# --------------------------------------------------------------------------- #


class _ZeroOverlapVel:
    """∂S/∂k stand-in for a norm-conserving system (S = I ⇒ ∂S/∂k = 0). Carries
    the batched KB β-projectors ``p`` (the source ``USPPResponseCtx.build_hkq``
    reads for the folded k+q Hamiltonian) and returns 0 for its apply."""

    def __init__(self, p: torch.Tensor) -> None:
        self.p = p  # (nk, nproj, npw_max)

    def apply(self, c, mu, k_index=None):
        return torch.zeros_like(c)


def _nc_uspp_ctx(res):
    """A :class:`USPPResponseCtx` built from a norm-conserving SCF result whose
    every operator IS the NC operator: S = I (``q_full`` = 0 ⇒ ``s_apply`` = id),
    the bare D as the "screened" D, the NC batched β-projectors (in the shared
    atom→channel→m column order, so ``Σ p†·dscr·p`` reproduces the NC KB term),
    the NC velocity ∂H/∂k (``VelocityApply``), ∂S/∂k = 0, and the smooth ρ = the
    NC density. Routing ``sigma_shielding`` through this ctx must reproduce the
    plain NC ``sigma_shielding`` — the machine-precision NC-limit reduction gate
    (the #349 pattern, one layer up at the shielding assembly)."""
    from gradwave.core.batch import projectors_b
    from gradwave.postscf.kgeometry import VelocityApply
    from gradwave.postscf.kgeometry_nmr import USPPResponseCtx

    system = res.system
    bk = system.batch
    assert bk is not None
    p = projectors_b(bk, system.positions)  # (nk, nproj, npw_max)
    veff = res.v_eff if res.v_eff.dim() == 3 else res.v_eff[0]
    return USPPResponseCtx(
        system=system, bk=bk, shape=tuple(system.grid.shape),
        v_eff_r=veff, dscr=bk.dij_full, q_full=torch.zeros_like(bk.dij_full),
        ov=_ZeroOverlapVel(p), v_h=VelocityApply(system), rho_smooth=res.rho)


@pytest.fixture(scope="module")
def si_nc_mesh_2axis():
    """Si NC SCF on a 2×2×1 unreduced mesh — the smallest mesh with ≥2 axes of
    n>1, so ``sigma_shielding`` is well-determined. Tight rho tol so the stored
    density equals the coeff-built one (the uspp diamagnetic ρ) to ~round-off."""
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=10 * RY,
                          kmesh=(2, 2, 1), nbands=8, use_symmetry=False,
                          fft_shape=(18, 18, 18))
    res = scf(system, PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=120)
    assert res.converged
    return res


@pytest.mark.standard
def test_uspp_sigma_shielding_nc_limit_reduction(si_nc_mesh_2axis):
    """PR-A correctness gate: the USPP/PAW-routed ``sigma_shielding`` (S-metric
    velocity + Sternheimer, screened-D KB nonlocal current, smooth diamagnetic ρ)
    reduces to the plain norm-conserving ``sigma_shielding`` when the ctx encodes
    S = I / bare D. Reproduces the NC shielding tensor to the CG-certification
    tolerance — no external reference needed (the #349 NC-limit pattern, applied
    end-to-end at the shielding assembly)."""
    from gradwave.postscf.kgeometry_nmr import sigma_shielding

    res = si_nc_mesh_2axis
    ref = sigma_shielding(res, cg_tol=1e-10, cg_max_iter=800)
    ctx = _nc_uspp_ctx(res)
    got = sigma_shielding(res, cg_tol=1e-10, cg_max_iter=800, uspp=ctx)

    scale = ref.abs().max().item()
    rel = (got - ref).abs().max().item() / scale
    assert rel < 1e-6, f"USPP-routed σ vs NC σ rel {rel:.3e} (scale {scale:.3e} ppm)"
