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


# --------------------------------------------------------------------------- #
# induced current density + screening (milestone 8)                           #
# --------------------------------------------------------------------------- #


def test_current_fsum_per_k(si_mesh):
    # exact per-k f-sum: 2Re[Ω·mean_r j_para,μ(k)/f] + 2·HBAR2_2M·δ_μν·nocc
    # equals d/dk_ν Σ_occ⟨v_kin,μ⟩ (dense FD reference) — pins the whole
    # (u, δu) → j(r) assembly including the covariant-derivative structure
    from gradwave.constants import HBAR2_2M
    from gradwave.postscf.kgeometry import BlochHK
    from gradwave.postscf.kgeometry_nmr import induced_current_q

    res = si_mesh
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    vol = res.system.grid.volume
    nu = 1
    cur = induced_current_q(res, (0.0, 0.0, 0.0), nu, cg_tol=1e-10)

    def g_vec(kf):
        hk = BlochHK.from_scf(res, kf)
        _, u = torch.linalg.eigh(hk.h(hk.k_cart(kf)))
        kpg = hk.g_cart + hk.k_cart(kf)
        occ = u[:, :nocc]
        dens = (occ.conj() * occ).real.sum(dim=1)
        return (2.0 * HBAR2_2M) * (dens[:, None] * kpg).sum(dim=0)

    cell, _ = si_fcc()
    b = 2.0 * np.pi * np.linalg.inv(cell).T
    e_nu_frac = np.linalg.solve(b.T, np.eye(3)[nu])  # Cartesian step in frac
    h = 1e-4
    for ik, sph in enumerate(res.system.spheres):
        kf = np.asarray(sph.k_frac, float)
        dg = (g_vec(kf + h * e_nu_frac) - g_vec(kf - h * e_nu_frac)) / (2.0 * h)
        for mu in range(3):
            lhs = 2.0 * (vol * cur.j_para_k[ik, mu].mean() / 2.0).real.item()
            lhs += 2.0 * HBAR2_2M * nocc * (1.0 if mu == nu else 0.0)
            assert abs(lhs - dg[mu].item()) < 1e-3  # measured ≤ 2e-5 on |G'| ~ 100


def test_q0_tr_null_and_inert_screening(si_mesh):
    # TR: the physical density response to the q=0 velocity perturbation
    # vanishes (Re of the branch sum); screening is then a no-op
    from gradwave.core.xc.pbe import PBE as XC
    from gradwave.postscf.kgeometry_nmr import induced_current_q

    res = si_mesh
    cur_b = induced_current_q(res, (0.0, 0.0, 0.0), 1, cg_tol=1e-10)
    assert cur_b.drho_bare.real.abs().max().item() < 1e-10  # measured 2e-13
    assert cur_b.drho_bare.imag.abs().max().item() > 0.1  # the branch artifact
    cur_s = induced_current_q(res, (0.0, 0.0, 0.0), 1, xc=XC(), screen=True,
                              cg_tol=1e-10)
    assert cur_s.n_dyson <= 2
    assert (cur_s.j_para - cur_b.j_para).abs().max().item() < 1e-8


def test_finite_q_screening_converges(si_mesh):
    from gradwave.core.xc.pbe import PBE as XC
    from gradwave.postscf.kgeometry_nmr import induced_current_q

    res = si_mesh
    cur = induced_current_q(res, (0.5, 0.0, 0.0), 1, xc=XC(), screen=True,
                            cg_tol=1e-9, dyson_tol=1e-7)
    assert cur.n_dyson < 30
    # genuine self-consistent screening at finite q (measured ~16% of bare)
    rel = (cur.drho - cur.drho_bare).abs().max().item() / cur.drho_bare.abs().max().item()
    assert 0.01 < rel < 1.0
    assert torch.isfinite(cur.j_para).all() and torch.isfinite(cur.j_dia).all()


def test_rejects_incommensurate_and_reduced():
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-8, rhotol=1e-7, verbose=False, max_iter=80)
    with pytest.raises(ValueError):
        velocity_perturbation_q(res, (0.3, 0.0, 0.0))  # not on the mesh


# --------------------------------------------------------------------------- #
# KB nonlocal current + bare shielding assembly (milestone 9)                 #
# --------------------------------------------------------------------------- #


def test_continuity_closure_kb(si_mesh):
    # THE validation gate of the KB nonlocal current: for the single-branch
    # response pair at finite q the exact identity is
    #   q·mean[j_para + j_nl] = mean s + T_trunc
    # (s = continuity_source_q, T_trunc the explicit basis-truncation term).
    # Without the KB term the closure fails by the [V_NL, e^{-iqr}]
    # commutator — measured rel 6.9e-2 broken vs 5.6e-11 closed (the
    # KB-corrected-but-unaccounted-truncation midpoint is 2.4e-3, pure
    # sphere-boundary leakage that vanishes with ecut).
    from gradwave.grids import reciprocal_cell
    from gradwave.postscf.kgeometry_nmr import (
        continuity_source_q,
        continuity_truncation_term,
        induced_current_q,
        velocity_perturbation_q,
    )

    res = si_mesh
    b = reciprocal_cell(res.system.grid.cell)
    q_frac = np.array([0.5, 0.0, 0.0])
    q_cart = torch.as_tensor(q_frac @ b)
    e_pol = np.array([0.0, 1.0, 0.0])
    sol = velocity_perturbation_q(res, q_frac, cg_tol=1e-11)
    cur = induced_current_q(res, q_frac, e_pol, sol=sol, cg_tol=1e-11)

    rhs = complex(continuity_source_q(res, sol, e_pol).mean())
    jp = torch.stack([cur.j_para[m].mean() for m in range(3)])
    jn = torch.stack([cur.j_nl[m].mean() for m in range(3)])
    lhs_kin = complex((q_cart.to(jp.dtype) * jp).sum())
    lhs_full = lhs_kin + complex((q_cart.to(jn.dtype) * jn).sum())
    t_tr = complex(continuity_truncation_term(res, sol, cur.dpsi))

    scale = abs(rhs)
    rel_broken = abs(lhs_kin - rhs) / scale
    rel_kb = abs(lhs_full - rhs) / scale
    rel_closed = abs(lhs_full - rhs - t_tr) / scale
    assert rel_broken > 1e-2  # measured 6.9e-2: the KB term is load-bearing
    assert rel_kb < 1e-2  # measured 2.4e-3: truncation-only remainder
    assert rel_closed < 1e-8  # measured 5.6e-11: exact closure
    assert rel_closed < 1e-4 * rel_broken


def test_gauge_longitudinal_null(si_mesh):
    # A longitudinal polarization (e_pol ∥ q̂) is a pure-gauge vector
    # potential: the antisymmetrized physical assembly must produce no
    # induced field. Measured ratio ~2.5e-14 vs the transverse response.
    from gradwave.dtypes import RDTYPE
    from gradwave.grids import reciprocal_cell
    from gradwave.postscf.kgeometry_nmr import (
        _biot_savart_sigma_cols,
        induced_current_q,
        velocity_perturbation_q,
    )

    res = si_mesh
    system = res.system
    b = reciprocal_cell(system.grid.cell)
    q_frac = np.array([0.5, 0.0, 0.0])
    q_cart = torch.as_tensor(q_frac @ b, dtype=RDTYPE)
    q_hat = (q_frac @ b) / np.linalg.norm(q_frac @ b)
    sites = system.positions.detach().cpu().to(RDTYPE)
    sol_p = velocity_perturbation_q(res, q_frac, cg_tol=1e-10)
    sol_m = velocity_perturbation_q(res, -q_frac, cg_tol=1e-10)

    def col(pol):
        cp = induced_current_q(res, q_frac, pol, sol=sol_p, cg_tol=1e-10)
        cm = induced_current_q(res, -q_frac, pol, sol=sol_m, cg_tol=1e-10)
        return _biot_savart_sigma_cols(cp.j_total, cm.j_total, q_cart,
                                       system.grid.g_cart, sites)

    t1 = np.array([0.0, 1.0, 0.0])
    t1 = t1 - (t1 @ q_hat) * q_hat
    t1 /= np.linalg.norm(t1)
    col_t = col(t1)
    col_l = col(q_hat)
    assert float(col_t.abs().max()) > 0
    assert float(col_l.abs().max() / col_t.abs().max()) < 1e-10  # ~2.5e-14


def test_lamb_prefactor_synthetic():
    # Biot–Savart prefactor + sign, no SCF: dia-only branch fields of a
    # Gaussian density at the A-wave node must reproduce the analytic
    # Landau-gauge Lamb term σ_dia = (2α²·HBAR2_2M/(3E2))·⟨1/r⟩ > 0
    # (diamagnetic shielding is positive). Measured +1.9% at q = L/8.
    import math

    from gradwave.constants import ALPHA_FS, E2, HBAR2_2M
    from gradwave.dtypes import CDTYPE, RDTYPE
    from gradwave.postscf.kgeometry_nmr import _biot_savart_sigma_cols

    length, n, sig, nelec = 10.0, 40, 0.6, 2.0
    ax = torch.arange(n, dtype=torch.float64) / n * length
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
    r0 = torch.tensor([0.0, length / 2, length / 2], dtype=torch.float64)
    r2 = (xx - r0[0]) ** 2 + (yy - r0[1]) ** 2 + (zz - r0[2]) ** 2
    rho = torch.exp(-r2 / (2 * sig**2))
    rho = rho / (rho.sum() * (length / n) ** 3) * nelec

    m = torch.fft.fftfreq(n, d=1.0 / n).to(torch.float64)
    mx, my, mz = torch.meshgrid(m, m, m, indexing="ij")
    g_cart = torch.stack([mx, my, mz], dim=-1) * (2 * math.pi / length)

    e_pol = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    s_dia = torch.einsum("m,ijk->mijk", e_pol.to(CDTYPE),
                         (2.0 * HBAR2_2M) * rho.to(CDTYPE))
    q_cart = torch.tensor([0.125 * 2 * math.pi / length, 0.0, 0.0], dtype=RDTYPE)
    cols = _biot_savart_sigma_cols(s_dia, s_dia, q_cart, g_cart,
                                   r0[None, :].to(RDTYPE))
    lamb = 2.0 * ALPHA_FS**2 * HBAR2_2M / (3.0 * E2) \
        * nelec * math.sqrt(2.0 / math.pi) / sig
    assert abs(float(cols[0, 2]) - lamb) / lamb < 0.05  # measured 0.019
    assert abs(float(cols[0, 0])) < 1e-8 * lamb
    assert abs(float(cols[0, 1])) < 1e-8 * lamb


@pytest.mark.slow
def test_sigma_cubic_isotropy():
    # Full bare shielding tensor on a (2,2,2) Γ-centered mesh: the cubic
    # (T_d site) symmetry demands an isotropic tensor — measured off-diag
    # 4.9e-11 ppm and diagonal spread 1.4e-11 ppm on σ_iso ~ 34/59 ppm
    # (machine-exact isotropy). The two sites differ at finite q (25.4 ppm
    # at q = b/2, shrinking O(q²) as the mesh refines — the documented
    # finite-q systematic of the commensurate Pickard–Mauri assembly), so
    # site equivalence is NOT asserted here.
    from gradwave.postscf.kgeometry_nmr import sigma_shielding

    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(2, 2, 2), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    sig = sigma_shielding(res, cg_tol=1e-9)
    assert sig.shape == (2, 3, 3)
    assert torch.isfinite(sig).all()
    diag = torch.diagonal(sig, dim1=1, dim2=2)
    off = sig - torch.diag_embed(diag)
    iso = float(diag.mean())
    assert iso > 0  # diamagnetically-dominated bare σ for Si
    assert float(off.abs().max()) < 1e-6 * abs(iso)  # measured ~1e-12 rel
    for s in range(2):
        spread = float(diag[s].max() - diag[s].min())
        assert spread < 1e-6 * abs(iso)  # measured ~4e-13 rel


@pytest.mark.slow
def test_sigma_q_linearity_and_underdetermined():
    # q-linearity of the antisymmetric extraction: the single-axis shielding
    # column on a (4,1,1) TR=False mesh drifts by 4.8% between q = b/4 and
    # b/2 (the even-order O(q²) tail); the longitudinal gauge null holds at
    # 4.5e-13 also when +q and −q are genuinely different mesh points. A
    # single-axis mesh cannot determine the full tensor: sigma_shielding
    # raises.
    from gradwave.dtypes import RDTYPE
    from gradwave.grids import reciprocal_cell
    from gradwave.postscf.kgeometry_nmr import (
        _biot_savart_sigma_cols,
        _transverse_frame,
        induced_current_q,
        sigma_shielding,
        velocity_perturbation_q,
    )

    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(4, 1, 1), nbands=8, use_symmetry=False,
                          time_reversal=False, fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    b = reciprocal_cell(cell)
    sites = system.positions.detach().cpu().to(RDTYPE)

    def col_at(qi, pol=None):
        q_frac = np.array([qi, 0.0, 0.0])
        q_cart = torch.as_tensor(q_frac @ b, dtype=RDTYPE)
        q_hat = (q_frac @ b) / np.linalg.norm(q_frac @ b)
        sol_p = velocity_perturbation_q(res, q_frac, cg_tol=1e-9)
        sol_m = velocity_perturbation_q(res, -q_frac, cg_tol=1e-9)
        if pol is None:
            pol = _transverse_frame(q_hat)[0]
        cp = induced_current_q(res, q_frac, pol, sol=sol_p, cg_tol=1e-9)
        cm = induced_current_q(res, -q_frac, pol, sol=sol_m, cg_tol=1e-9)
        return _biot_savart_sigma_cols(cp.j_total, cm.j_total, q_cart,
                                       system.grid.g_cart, sites), q_hat

    c14, q_hat = col_at(0.25)
    c12, _ = col_at(0.5)
    drift = float((c14 - c12).abs().max() / c14.abs().max())
    assert drift < 0.15  # measured 0.048
    cl, _ = col_at(0.25, pol=q_hat)
    assert float(cl.abs().max() / c14.abs().max()) < 1e-9  # measured 4.5e-13

    with pytest.raises(ValueError, match="underdetermined"):
        sigma_shielding(res)


# --------------------------------------------------------------------------- #
# analytic ∂/∂q shielding (milestone 10)                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dq_engine(si_mesh):
    from gradwave.postscf.kgeometry_nmr import ShieldingDq

    return ShieldingDq(si_mesh)


def test_dq_branch_field_matches_mesh_at_q0(si_mesh, dq_engine):
    # S⁰ of the analytic engine equals the mesh route's conserved j_total at
    # q = 0 (both gauge-invariant; dense eigh vs Davidson+CG states) —
    # measured rel 3.5e-10 — and the fixed-sphere twin at s = 0 reproduces
    # the S⁰ assembly minus the diamagnetic term to machine precision
    # (measured 2.9e-16).
    from gradwave.constants import HBAR2_2M
    from gradwave.dtypes import CDTYPE, RDTYPE
    from gradwave.postscf.kgeometry_nmr import (
        _branch_field_fixed_sphere,
        induced_current_q,
    )

    res = si_mesh
    eng = dq_engine
    q_hat = np.array([0.0, 1.0, 0.0])
    pol = np.array([0.0, 0.0, 1.0])
    ((s0, _ds),) = eng.branch_fields_axis(q_hat, [pol])
    cur = induced_current_q(res, (0.0, 0.0, 0.0), pol, cg_tol=1e-11)
    rel = float((s0 - cur.j_total).abs().max() / cur.j_total.abs().max())
    assert rel < 1e-7
    rho = res.rho if res.rho.dim() == 3 else res.rho[0]
    dia = torch.einsum(
        "m,ijk->mijk",
        torch.as_tensor(pol, dtype=RDTYPE).to(CDTYPE),
        (2.0 * HBAR2_2M) * rho.to(CDTYPE),
    )
    tw0 = sum(
        _branch_field_fixed_sphere(eng, ik, q_hat, pol, 0.0)
        for ik in range(len(eng.ks))
    )
    assert float(((s0 - dia) - tw0).abs().max() / tw0.abs().max()) < 1e-12


def test_dq_derivative_matches_fd_twin(si_mesh, dq_engine):
    # THE validation gate of the analytic derivative chain (M second
    # derivatives, δu', covariant-derivative projections, Ā'): central
    # differences of the finite-s fixed-sphere twin converge to the analytic
    # ∂S/∂s as pure h² truncation — measured rel 1.2e-4 / 3.0e-5 / 7.4e-6 at
    # h = 2e-3 / 1e-3 / 5e-4 (exact 4× per halving), so one small h bounds
    # every term and the halved h certifies it is truncation, not a floor.
    from gradwave.postscf.kgeometry_nmr import _branch_field_fixed_sphere

    eng = dq_engine
    q_hat = np.array([0.0, 1.0, 0.0])
    pol = np.array([0.0, 0.0, 1.0])
    ((_s0, ds),) = eng.branch_fields_axis(q_hat, [pol])
    rels = []
    for h in (1e-3, 5e-4):
        sp = sum(
            _branch_field_fixed_sphere(eng, ik, q_hat, pol, +h)
            for ik in range(len(eng.ks))
        )
        sm = sum(
            _branch_field_fixed_sphere(eng, ik, q_hat, pol, -h)
            for ik in range(len(eng.ks))
        )
        fd = (sp - sm) / (2.0 * h)
        rels.append(float((ds - fd).abs().max() / fd.abs().max()))
    assert rels[0] < 5e-4
    assert rels[1] < rels[0]  # shrinking with h: truncation, not a floor


def test_dq_gauge_and_tr_null(si_mesh, dq_engine):
    # Longitudinal (pure gauge) polarization along the SAMPLED mesh axis b̂1:
    # the analytic column vanishes up to the mesh-superlattice Wannier
    # residual (the BZ sum of a k-derivative retains i(q̂·R)f_R over
    # superlattice vectors R; along b̂1 only R ∥ 2a₁ survives). The s = 0
    # Biot–Savart term — the coefficient of a spurious 1/q divergence, the
    # TR null — is machine-zero.
    from gradwave.dtypes import RDTYPE
    from gradwave.grids import reciprocal_cell
    from gradwave.postscf.kgeometry_nmr import (
        _biot_savart_sigma_cols_dq,
        _transverse_frame,
    )

    res = si_mesh
    eng = dq_engine
    b = reciprocal_cell(res.system.grid.cell)
    q_hat = np.asarray(b[0], dtype=float) / np.linalg.norm(b[0])
    pol_t = _transverse_frame(q_hat)[0]
    fields = eng.branch_fields_axis(q_hat, [pol_t, q_hat])
    sites = res.system.positions.detach().cpu().to(RDTYPE)
    qh_t = torch.as_tensor(q_hat, dtype=RDTYPE)
    col_t, null_t = _biot_savart_sigma_cols_dq(
        fields[0][0], fields[0][1], qh_t, res.system.grid.g_cart, sites
    )
    col_l, _ = _biot_savart_sigma_cols_dq(
        fields[1][0], fields[1][1], qh_t, res.system.grid.g_cart, sites
    )
    scale = float(col_t.abs().max())
    assert scale > 0
    assert float(col_l.abs().max()) / scale < 1e-2  # n=2 axis: f_{2a} residual
    tr_null = float(null_t.abs().max()) * float(np.linalg.norm(b[0])) / scale
    assert tr_null < 1e-10  # measured ~2e-17


def test_dq_biot_savart_route_equivalence_synthetic():
    # The analytic product-rule Biot–Savart is the EXACT q → 0 limit of the
    # validated finite ±q assembly — pinned on a synthetic dia-only field
    # (clean periodic Gaussian, minimum-image, so the box has no unpaired
    # Nyquist content; a boundary-truncated density puts weight on the
    # m = −n/2 planes, which have no +n/2 partner and pollute BOTH
    # assemblies with a spurious 1/q term — the trap the min-image build
    # avoids). Sites both at the Gaussian center (q̂·r_s = 0) and off-center
    # (q̂·r_s ≠ 0 — exercises the phase-derivative i(q̂·r_s) kernel term).
    # Measured: |analytic − finite| rel 6.2e-3 at q = L/8 → 3.9e-4 at L/32
    # (O(q²) approach), null 5e-18.
    import math

    from gradwave.constants import HBAR2_2M
    from gradwave.dtypes import CDTYPE, RDTYPE
    from gradwave.postscf.kgeometry_nmr import (
        _biot_savart_sigma_cols,
        _biot_savart_sigma_cols_dq,
    )

    length, n, sig, nelec = 10.0, 40, 0.6, 2.0
    ax = torch.arange(n, dtype=torch.float64) / n * length
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
    r0 = torch.tensor([0.0, length / 2, length / 2], dtype=torch.float64)

    def mi(d):
        return (d + length / 2) % length - length / 2

    r2 = mi(xx - r0[0]) ** 2 + mi(yy - r0[1]) ** 2 + mi(zz - r0[2]) ** 2
    rho = torch.exp(-r2 / (2 * sig**2))
    rho = rho / (rho.sum() * (length / n) ** 3) * nelec

    m = torch.fft.fftfreq(n, d=1.0 / n).to(torch.float64)
    mx, my, mz = torch.meshgrid(m, m, m, indexing="ij")
    g_cart = torch.stack([mx, my, mz], dim=-1) * (2 * math.pi / length)

    e_pol = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    s_dia = torch.einsum("m,ijk->mijk", e_pol.to(CDTYPE),
                         (2.0 * HBAR2_2M) * rho.to(CDTYPE))
    sites = torch.stack(
        [r0, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)]
    ).to(RDTYPE)
    q_hat = torch.tensor([1.0, 0.0, 0.0], dtype=RDTYPE)
    col_a, null = _biot_savart_sigma_cols_dq(
        s_dia, torch.zeros_like(s_dia), q_hat, g_cart, sites)
    scale = float(col_a.abs().max())
    assert scale > 0
    assert float(null.abs().max()) < 1e-12 * scale  # measured 5e-18 abs

    def fin(f):
        q_cart = torch.tensor([f * 2 * math.pi / length, 0.0, 0.0], dtype=RDTYPE)
        return _biot_savart_sigma_cols(s_dia, s_dia, q_cart, g_cart, sites)

    d8 = float((fin(0.125) - col_a).abs().max()) / scale
    d32 = float((fin(0.03125) - col_a).abs().max()) / scale
    assert d32 < 2e-3  # measured 3.9e-4
    assert d32 < 0.5 * d8  # O(q²) approach to the analytic limit


def test_sigma_dq_underdetermined(si_mesh):
    # a single-axis mesh cannot determine the tensor (and the analytic
    # BZ-derivative only converges along sampled axes): ValueError
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

    with pytest.raises(ValueError, match="underdetermined"):
        sigma_shielding_dq(si_mesh)


@pytest.mark.standard
def test_sigma_dq_cubic_isotropy_and_site_equivalence():
    # The analytic route's headline: on the (2,2,2) mesh where the finite-q
    # route's equivalent-site split is pure O(q²) error (25.4 ppm at 12 Ry),
    # the analytic tensor is machine-isotropic AND site-symmetric — measured
    # at 6 Ry: offdiag 3.3e-10 ppm, diag spread 6.2e-12 ppm, site split
    # 2.0e-3 ppm on σ_iso ≈ 20.8 ppm (the residual split is the n=2
    # mesh-superlattice Wannier term, not a q artifact).
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=6 * RY,
                          kmesh=(2, 2, 2), nbands=8, use_symmetry=False,
                          fft_shape=(15, 15, 15))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    sig = sigma_shielding_dq(res)
    assert sig.shape == (2, 3, 3)
    assert torch.isfinite(sig).all()
    diag = torch.diagonal(sig, dim1=1, dim2=2)
    off = sig - torch.diag_embed(diag)
    iso = float(diag.mean())
    assert iso > 0
    assert float(off.abs().max()) < 1e-6 * abs(iso)
    for s in range(2):
        spread = float(diag[s].max() - diag[s].min())
        assert spread < 1e-6 * abs(iso)
    split = float(abs(diag[0].mean() - diag[1].mean()))
    assert split < 1e-3 * abs(iso)  # measured ~1e-4 relative


# --------------------------------------------------------------------------- #
# little-group-of-q IBZ wedge reduction (opt-in, exact, symmetry-gated)        #
# --------------------------------------------------------------------------- #


def _sigma_dq_res(cell, pos, kmesh, *, ecut=6, fft=(15, 15, 15)):
    torch.set_num_threads(2)
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=ecut * RY,
                          kmesh=kmesh, nbands=8, use_symmetry=False,
                          fft_shape=fft)
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


@pytest.mark.slow
def test_sigma_dq_wedge_route_equivalence_si():
    # THE non-negotiable gate for the little-group-of-q wedge reduction on the
    # GO case (Si, Oh): the opt-in symmetry-reduced σ must equal the full-mesh
    # σ to solver tolerance. A 4³ mesh reduces per axis under the |G_q|=6 little
    # co-group on top of the TR fold (nk 36 → 13 wedge / axis, union ≈ 29).
    from gradwave.postscf.kgeometry_nmr import (
        _plan_wedge_reduction,
        sigma_shielding_dq,
    )

    cell, pos = si_fcc()
    res = _sigma_dq_res(cell, pos, (4, 4, 4))

    # the reduction actually engaged (not a silent full-mesh fallback)
    k_frac = np.stack([s.k_frac for s in res.system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    plan = _plan_wedge_reduction(res, axes, mesh_n, "auto")
    assert plan is not None
    union_kfrac, per_axis = plan
    assert len(union_kfrac) < len(res.system.spheres)  # setup reduced
    assert all(len(idx) < len(res.system.spheres) for _i, _q, idx, _w, _f in per_axis)

    sig_full = sigma_shielding_dq(res, use_symmetry=False)
    sig_red = sigma_shielding_dq(res, use_symmetry="auto")
    scale = float(sig_full.abs().max())
    resid = float((sig_red - sig_full).abs().max())
    assert resid < 1e-6 * scale  # bit-equivalent to the full-mesh route
    # symmetry-protected invariants survive the fold
    diag = torch.diagonal(sig_red, dim1=1, dim2=2)
    off = sig_red - torch.diag_embed(diag)
    iso = float(diag.mean())
    assert float(off.abs().max()) < 1e-6 * abs(iso)  # cubic isotropy
    assert float(abs(diag[0].mean() - diag[1].mean())) < 1e-3 * abs(iso)  # sites


@pytest.mark.slow
def test_sigma_dq_wedge_route_equivalence_tetragonal():
    # Stronger symmetry test on a genuinely tetragonal (D4h) cell where the
    # reciprocal axes are INEQUIVALENT (c-axis little group ≠ a-axis): the
    # rutile-class gate. Real rutile TiO2 needs a Ti pseudo absent from the
    # fixtures, so a [001]-strained diamond Si (space group I4₁/amd, D4h)
    # stands in — same point-group class, same per-axis-inequivalent little
    # groups, built from the available Si pseudo. Route-equivalence must still
    # be exact even though the per-axis factor is marginal.
    from gradwave.postscf.kgeometry_nmr import (
        _plan_wedge_reduction,
        sigma_shielding_dq,
    )

    cell, pos = si_fcc()
    strain = np.diag([1.0, 1.0, 1.06])  # uniaxial [001] → tetragonal D4h
    cell_t, pos_t = cell @ strain, pos @ strain
    res = _sigma_dq_res(cell_t, pos_t, (4, 4, 4))

    from gradwave.symmetry import find_spacegroup
    frac = res.system.positions.cpu().numpy() @ np.linalg.inv(np.asarray(cell_t))
    sg = find_spacegroup(np.asarray(cell_t), frac, res.system.species_of_atom)
    assert sg.n_ops < 48  # genuinely lower than the cubic Oh (48)

    k_frac = np.stack([s.k_frac for s in res.system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    assert _plan_wedge_reduction(res, axes, mesh_n, "auto") is not None

    sig_full = sigma_shielding_dq(res, use_symmetry=False)
    sig_red = sigma_shielding_dq(res, use_symmetry="auto")
    scale = float(sig_full.abs().max())
    assert float((sig_red - sig_full).abs().max()) < 1e-6 * scale


@pytest.mark.standard
def test_sigma_dq_wedge_no_slowdown_low_symmetry():
    # The auto-fallback guarantee: on a cell whose symmetry is broken (an atom
    # displaced off the diamond site → trivial point group), the planner
    # returns None and the reduced call runs the EXACT full-mesh path — so σ is
    # identical and no reduction machinery is engaged (never slower than
    # use_symmetry=False).
    from gradwave.postscf.kgeometry_nmr import (
        _plan_wedge_reduction,
        sigma_shielding_dq,
    )

    cell, pos = si_fcc()
    pos = pos.copy()
    pos[1] += np.array([0.31, 0.17, 0.09])  # break every point-group op
    res = _sigma_dq_res(cell, pos, (2, 2, 2))

    k_frac = np.stack([s.k_frac for s in res.system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    assert _plan_wedge_reduction(res, axes, mesh_n, "auto") is None  # falls back

    sig_full = sigma_shielding_dq(res, use_symmetry=False)
    sig_auto = sigma_shielding_dq(res, use_symmetry="auto")
    assert torch.equal(sig_full, sig_auto)  # identical: same full-mesh code path


@pytest.mark.standard
def test_sigma_driver_small(si_mesh):
    # End-to-end sigma_shielding driver at deliberately small scale
    # ((2,2,1) mesh, 6 Ry, loose CG): finite tensor, and the residual
    # x↔y mirror symmetry of the anisotropic mesh holds (σ_xx = σ_yy,
    # zero xz/yz blocks, symmetric xy) — measured to display precision;
    # the full cubic isotropy at (2,2,2) is the slow-tier test. The
    # single-axis si_mesh cannot determine the tensor: ValueError.
    from gradwave.postscf.kgeometry_nmr import sigma_shielding

    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=6 * RY,
                          kmesh=(2, 2, 1), nbands=8, use_symmetry=False,
                          fft_shape=(15, 15, 15))
    res = scf(system, PBE(), etol=1e-8, rhotol=1e-7, verbose=False, max_iter=80)
    assert res.converged
    sig = sigma_shielding(res, cg_tol=1e-6, nl_quad=3)
    assert sig.shape == (2, 3, 3)
    assert torch.isfinite(sig).all()
    scale = float(sig.abs().max())
    for s in range(2):
        assert abs(float(sig[s, 0, 0] - sig[s, 1, 1])) < 1e-3 * scale
        assert float(sig[s, 0, 2].abs() + sig[s, 1, 2].abs()
                     + sig[s, 2, 0].abs() + sig[s, 2, 1].abs()) < 1e-3 * scale
        assert abs(float(sig[s, 0, 1] - sig[s, 1, 0])) < 1e-3 * scale

    with pytest.raises(ValueError, match="underdetermined"):
        sigma_shielding(si_mesh)


# --------------------------------------------------------------------------- #
# perf machinery: dense eigh-prepass drafts, per-column CG, spawn pool         #
# --------------------------------------------------------------------------- #


def test_dense_draft_matches_cg_route(si_mesh, monkeypatch):
    # The eigh-prepass draft + CG certification must land on the same solution
    # as the plain CG route to the shared tolerance, and the draft must
    # actually be exercised (solve called once per Cartesian mu). The dense
    # path is a measured-negative opt-in (default off), so force it on.
    from gradwave.postscf._response import (
        DenseSternheimerSolver,
        dense_sternheimer_for,
    )

    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "on")
    res = si_mesh
    q = (0.5, 0.0, 0.0)
    sol_cg = velocity_perturbation_q(res, q, cg_tol=1e-9, dense=None)
    solver = dense_sternheimer_for(res)
    assert solver is not None
    calls = {"n": 0}
    orig = DenseSternheimerSolver.solve

    def counting(self, jsel, eps, rhs, shift, nocc):
        calls["n"] += 1
        return orig(self, jsel, eps, rhs, shift, nocc)

    try:
        DenseSternheimerSolver.solve = counting
        sol_d = velocity_perturbation_q(res, q, cg_tol=1e-9, dense=solver)
    finally:
        DenseSternheimerSolver.solve = orig
    assert calls["n"] == 3
    scale = float(sol_cg.dpsi.abs().max())
    assert float((sol_d.dpsi - sol_cg.dpsi).abs().max()) < 1e-6 * max(scale, 1.0)
    p_cg = paramagnetic_tensor(sol_cg)
    p_d = paramagnetic_tensor(sol_d)
    assert float((p_cg - p_d).abs().max() / p_cg.abs().max()) < 1e-7


def test_dense_draft_certified_against_lying_solver(si_mesh, monkeypatch):
    # THE draft-then-certify contract (the fp32-expansion-Davidson style
    # deliberately-lying-operator check): a solver returning pure garbage
    # drafts must still produce the CG-route answer, because every column is
    # re-measured against the true batched operator and iterated to cg_tol.
    from gradwave.postscf._response import (
        DenseSternheimerSolver,
        dense_sternheimer_for,
    )

    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "on")
    res = si_mesh
    q = (0.5, 0.0, 0.0)
    sol_cg = velocity_perturbation_q(res, q, cg_tol=1e-9, dense=None)
    solver = dense_sternheimer_for(res)
    assert solver is not None
    orig = DenseSternheimerSolver.solve

    def lying(self, jsel, eps, rhs, shift, nocc):
        torch.manual_seed(0)
        return torch.randn_like(rhs)  # garbage draft, O(1) amplitude

    try:
        DenseSternheimerSolver.solve = lying
        sol_lie = velocity_perturbation_q(res, q, cg_tol=1e-9, dense=solver)
    finally:
        DenseSternheimerSolver.solve = orig
    scale = float(sol_cg.dpsi.abs().max())
    assert float((sol_lie.dpsi - sol_cg.dpsi).abs().max()) < 1e-6 * max(scale, 1.0)


def test_dense_gate_env(si_mesh, monkeypatch):
    from gradwave.postscf._response import dense_sternheimer_for

    res = si_mesh
    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "off")
    assert dense_sternheimer_for(res) is None
    # measured-negative default: "auto" declines too — the dense path is
    # opt-in only, never built on the production default.
    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "auto")
    assert dense_sternheimer_for(res) is None
    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "bogus")
    with pytest.raises(ValueError, match="GRADWAVE_DENSE_STERNHEIMER"):
        dense_sternheimer_for(res)
    # explicit opt-in builds it …
    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER", "on")
    assert dense_sternheimer_for(res) is not None
    # … but still declines when the factorization would blow the budget
    monkeypatch.setenv("GRADWAVE_DENSE_STERNHEIMER_BUDGET", "1")
    assert dense_sternheimer_for(res) is None


def test_apply_cols_matches_apply(si_mesh):
    from gradwave.core.batch import BatchedHamiltonian, projectors_b

    res = si_mesh
    system = res.system
    bk = system.batch
    h = BatchedHamiltonian(bk, system.grid.shape, res.v_eff,
                           projectors_b(bk, system.positions))
    torch.manual_seed(3)
    c = torch.randn(bk.nk, 3, bk.npw_max, dtype=torch.complex128)
    c = c * bk.mask[:, None, :]
    full = h.apply(c)
    kcol = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    bcol = torch.tensor([0, 2, 0, 1, 2], dtype=torch.long)
    cols = h.apply_cols(c[kcol, bcol], kcol)
    assert float((cols - full[kcol, bcol]).abs().max()) < 1e-10


def test_percolumn_cg_matches_lockstep_and_certifies(si_mesh):
    # The per-column/compaction path (BatchedHamiltonian has apply_cols) must
    # agree with the historical lockstep loop (forced via an apply-only shim)
    # AND return solutions whose TRUE per-column residuals all sit below tol
    # (the final top-up pass makes this structural).
    from types import SimpleNamespace

    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.postscf._response import (
        cg_sternheimer,
        insulator_window,
        pad_coeffs,
        sternheimer_shift,
    )
    from gradwave.postscf.kgeometry import VelocityApply

    res = si_mesh
    system = res.system
    bk = system.batch
    nocc = insulator_window(res.occupations, 2.0, "insulating")
    c_occ = pad_coeffs(res.coeffs, bk.npw_max)[:, :nocc]
    eps = res.eigenvalues[:, :nocc].double()
    shift = sternheimer_shift(eps)
    h = BatchedHamiltonian(bk, system.grid.shape, res.v_eff,
                           projectors_b(bk, system.positions))

    def p_occ(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return torch.einsum("kbn,kng->kbg", ov, c_occ)

    v = VelocityApply(system)
    rhs = v.apply(c_occ, 0)
    rhs = -(rhs - p_occ(rhs))
    tol = 1e-9
    x_new = cg_sternheimer(h, bk, c_occ, eps, rhs, torch.zeros_like(rhs),
                           shift, tol=tol)
    shim = SimpleNamespace(apply=h.apply)  # no apply_cols -> legacy lockstep
    x_old = cg_sternheimer(shim, bk, c_occ, eps, rhs, torch.zeros_like(rhs),
                           shift, tol=tol)
    scale = float(x_old.abs().max())
    assert float((x_new - x_old).abs().max()) < 1e-6 * max(scale, 1.0)

    def a_apply(x):
        hx = h.apply(x) - eps[..., None] * x
        return hx - p_occ(hx) + shift * p_occ(x)

    rn = torch.linalg.norm(rhs - a_apply(x_new), dim=-1)
    assert float(rn.max()) < 5.0 * tol  # every column certified
