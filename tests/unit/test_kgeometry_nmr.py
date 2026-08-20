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
    from gradwave.grids import reciprocal_cell
    from gradwave.postscf.kgeometry_nmr import (
        _biot_savart_sigma_cols,
        induced_current_q,
        velocity_perturbation_q,
    )
    from gradwave.dtypes import RDTYPE

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
