"""Finite-s twin check of the S-metric analytic du1 on MgO.

delta_u(s) = R_{S(k+s)}(eps_n(k)) O(s) u_n on the FIXED k-sphere, computed by
direct generalized eigh at k+s*qhat (no implicit differentiation), with
O(s) = (1/2) sum_mu pol_mu [v_gen(0) + v_gen(s)] and v_gen(s) evaluated at
frozen eps_n(k). Central differences (delta_u(+h) - delta_u(-h))/2h vs the
analytic du1 of ShieldingDq's assembly:

  MATCH    -> the analytic formula correctly differentiates its own finite-s
              definition; the divergence is in the DEFINITION (missing USPP
              augmentation physics in O(s) / the characterization),
  MISMATCH -> derivation/implementation bug in the du1 assembly.

Also prints |du_fd| at both ecut rungs: if the finite-s definition itself
produces a growing derivative, the definition is the sick part.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def geigh(hks, kc):
    with torch.no_grad():
        hmat = hks.h(kc)
        smat = hks.s(kc)
        ell = torch.linalg.cholesky(0.5 * (smat + smat.mH))
        ell_inv = torch.linalg.solve_triangular(
            ell, torch.eye(hks.npw, dtype=CDTYPE), upper=False)
        a = ell_inv @ hmat @ ell_inv.mH
        a = 0.5 * (a + a.mH)
        w, y = torch.linalg.eigh(a)
        u = ell_inv.mH @ y
    return w, u, smat


def dh_ds_at(hks, kc):
    dh, ds = [], []
    for mu in range(3):
        t = torch.zeros(3, dtype=RDTYPE)
        t[mu] = 1.0
        dh.append(torch.func.jvp(hks.h, (kc,), (t,))[1])
        ds.append(torch.func.jvp(hks.s, (kc,), (t,))[1])
    return dh, ds


def run(ecut_ry: float, ecutrho_ry: float) -> None:
    from gradwave.postscf import kgeometry_nmr as kg
    from gradwave.postscf._response import insulator_window

    torch.set_num_threads(5)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(CELL, POS, [0, 1], [paw_mg, paw_o], ecut=ecut_ry * RY,
                        kmesh=(2, 2, 2), ecutrho=ecutrho_ry * RY, nbands=12)
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"]
    ctx = kg.build_uspp_response_ctx(res, PBE())
    nocc = insulator_window(res["occupations"], 2.0, "insulating required")

    b = np.linalg.inv(CELL).T * 2 * np.pi
    q_hat = np.asarray(b[0], dtype=float) / np.linalg.norm(b[0])
    qh = torch.as_tensor(q_hat, dtype=RDTYPE)
    pol = list(kg._transverse_frame(q_hat))[0]
    ep = torch.as_tensor(np.asarray(pol, dtype=float), dtype=CDTYPE)

    ik = 3
    kf = system.spheres[ik].k_frac
    hks = kg.BlochHKS.from_uspp_ctx(ctx, kf)
    kc = hks.k_ref_cart

    # ---- analytic du0/du1 (mirrors _accumulate_k S-metric branch) ----
    w, u, dh, ds = kg._geigh_and_dhds(hks, kc)
    s0m = hks.s(kc)
    uo, wo = u[:, :nocc], w[:nocc]
    uc, wc = u[:, nocc:], w[nocc:]
    wo_c = wo.to(CDTYPE)
    qh_c = qh.to(CDTYPE)
    m_h = kg._d2_mats(hks.h, kc, qh)
    m_s = kg._d2_mats(hks.s, kc, qh)
    dtu = kg._resolvent_apply_s(
        uc, wc, wo, kg._gen_velocity(dh, ds, uo, wo_c, qh_c), s0m)
    o0u = kg._gen_velocity(dh, ds, uo, wo_c, ep)
    du0 = kg._resolvent_apply_s(uc, wc, wo, o0u, s0m)
    me_u = 0.5 * (
        ep[0] * (m_h[0] @ uo - (m_s[0] @ uo) * wo_c)
        + ep[1] * (m_h[1] @ uo - (m_s[1] @ uo) * wo_c)
        + ep[2] * (m_h[2] @ uo - (m_s[2] @ uo) * wo_c))
    inner = kg._gen_velocity(dh, ds, du0, wo_c, qh_c) - dtu @ (
        uo.mH @ (s0m @ o0u)) + me_u
    du1 = kg._resolvent_apply_s(uc, wc, wo, inner, s0m) - uo @ (
        dtu.mH @ (s0m @ du0))

    # ---- finite-s twin ----
    def delta_u(s: float) -> torch.Tensor:
        ks = kc + s * qh
        ws, us, ss = geigh(hks, ks)
        dh_s, ds_s = dh_ds_at(hks, ks)
        # O(s) u = 1/2 sum_mu ep_mu [(dh0 - eps ds0) + (dh_s - eps ds_s)] u
        acc = torch.zeros_like(uo)
        for mu in range(3):
            v0 = dh[mu] @ uo - (ds[mu] @ uo) * wo_c
            vs = dh_s[mu] @ uo - (ds_s[mu] @ uo) * wo_c
            acc = acc + ep[mu] * 0.5 * (v0 + vs)
        ucs, wcs = us[:, nocc:], ws[nocc:]
        return kg._resolvent_apply_s(ucs, wcs, wo, acc, ss)

    h_fd = 1e-3
    d0 = delta_u(0.0)
    dp = delta_u(+h_fd)
    dm = delta_u(-h_fd)
    du_fd = (dp - dm) / (2.0 * h_fd)

    def fro(x):
        return float(torch.linalg.norm(x.reshape(-1)))

    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry  ik={ik} npw={hks.npw} ===",
          flush=True)
    print(f"  |delta_u(0) - du0| / |du0| = {fro(d0 - du0) / fro(du0):.3e}"
          f"   (|du0| = {fro(du0):.3e})", flush=True)
    print(f"  |du_fd|      = {fro(du_fd):.4e}", flush=True)
    print(f"  |du1|        = {fro(du1):.4e}", flush=True)
    print(f"  |du_fd - du1| / |du1| = {fro(du_fd - du1) / fro(du1):.3e}",
          flush=True)
    pb_fd = torch.linalg.norm(du_fd, dim=0)
    pb_an = torch.linalg.norm(du1, dim=0)
    print(f"  per-band |du_fd|: {[f'{float(x):.2e}' for x in pb_fd]}", flush=True)
    print(f"  per-band |du1|  : {[f'{float(x):.2e}' for x in pb_an]}", flush=True)
    # richardson: halve h to expose FD error scale
    dp2 = delta_u(+0.5 * h_fd)
    dm2 = delta_u(-0.5 * h_fd)
    du_fd2 = (dp2 - dm2) / h_fd
    print(f"  |du_fd(h/2) - du1| / |du1| = {fro(du_fd2 - du1) / fro(du1):.3e}",
          flush=True)


if __name__ == "__main__":
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        run(e, er)
    print("FD_PROBE_DONE", flush=True)
