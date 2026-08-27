"""Pinpoint which object in the S-metric analytic response derivative diverges
with ecut on MgO (round-2 probe; round-1 showed the ∂S/∂s field grows ~10x at
fixed G going 40 -> 60 Ry).

Per ecut rung and per mesh-k:
  1. dense generalized eigenvalues vs the SCF's iterative eigenvalues
     (validates the BlochHKS.from_uspp_ctx dscr/qint layout end-to-end),
  2. eig(S(k)) min/max (S-conditioning collapse check),
  3. norms of every intermediate of the ds/ds assembly for one axis/pol:
     dtu, du0, du1, me_u, M^H, M^S, and the du1 sub-terms.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import HBAR2_2M, RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def fro(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)))


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
    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry  nocc={nocc} ===", flush=True)

    b = np.linalg.inv(CELL).T * 2 * np.pi
    q_hat = np.asarray(b[0], dtype=float) / np.linalg.norm(b[0])
    qh = torch.as_tensor(q_hat, dtype=RDTYPE)
    pol = list(kg._transverse_frame(q_hat))[0]

    for ik in [0, 3]:
        kf = system.spheres[ik].k_frac
        hks = kg.BlochHKS.from_uspp_ctx(ctx, kf)
        kc = hks.k_ref_cart
        w, u, dh, ds = kg._geigh_and_dhds(hks, kc)
        eps_scf = res["eigenvalues"][ik]
        n_cmp = min(len(eps_scf), 12)
        dmax = float((w[:n_cmp].cpu() - eps_scf[:n_cmp].cpu()).abs().max())
        s0m = hks.s(kc)
        sev = torch.linalg.eigvalsh(0.5 * (s0m + s0m.mH))
        print(f" k[{ik}] npw={hks.npw}  |eps_dense - eps_scf|max(12) = {dmax:.3e}"
              f"   eig(S): min={float(sev.min()):.4e} max={float(sev.max()):.4e}",
              flush=True)

        uo, wo = u[:, :nocc], w[:nocc]
        uc, wc = u[:, nocc:], w[nocc:]
        wo_c = wo.to(CDTYPE)
        qh_c = qh.to(CDTYPE)
        m_h = kg._d2_mats(hks.h, kc, qh)
        m_s = kg._d2_mats(hks.s, kc, qh)
        dtu = kg._resolvent_apply_s(
            uc, wc, wo, kg._gen_velocity(dh, ds, uo, wo_c, qh_c), s0m)
        ep = torch.as_tensor(np.asarray(pol, dtype=float), dtype=CDTYPE)
        o0u = kg._gen_velocity(dh, ds, uo, wo_c, ep)
        du0 = kg._resolvent_apply_s(uc, wc, wo, o0u, s0m)
        me_u = 0.5 * (
            ep[0] * (m_h[0] @ uo - (m_s[0] @ uo) * wo_c)
            + ep[1] * (m_h[1] @ uo - (m_s[1] @ uo) * wo_c)
            + ep[2] * (m_h[2] @ uo - (m_s[2] @ uo) * wo_c))
        t_gv = kg._gen_velocity(dh, ds, du0, wo_c, qh_c)
        t_proj = dtu @ (uo.mH @ (s0m @ o0u))
        inner = t_gv - t_proj + me_u
        du1 = kg._resolvent_apply_s(uc, wc, wo, inner, s0m) - uo @ (
            dtu.mH @ (s0m @ du0))
        print(f"   |dH|={fro(torch.stack(dh)):.3e} |dS|={fro(torch.stack(ds)):.3e}"
              f" |M_H|={fro(torch.stack(m_h)):.3e} |M_S|={fro(torch.stack(m_s)):.3e}",
              flush=True)
        print(f"   |o0u|={fro(o0u):.3e} |du0|={fro(du0):.3e} |dtu|={fro(dtu):.3e}"
              f" |me_u|={fro(me_u):.3e}", flush=True)
        print(f"   |t_gv|={fro(t_gv):.3e} |t_proj|={fro(t_proj):.3e}"
              f" |inner|={fro(inner):.3e} |du1|={fro(du1):.3e}", flush=True)
        # the current-field ingredients that feed cross1
        vdu1 = torch.stack([dh[mu] @ du1 for mu in range(3)])
        dop_du0 = torch.stack([0.5 * (m_h[mu] @ du0)
                               + HBAR2_2M * float(qh[mu]) * du0 for mu in range(3)])
        print(f"   |v@du1|={fro(vdu1):.3e} |dop@du0|={fro(dop_du0):.3e}",
              flush=True)
        # spectral placement: worst-case per-band norms
        pb_du1 = torch.linalg.norm(du1, dim=0)
        print(f"   per-band |du1|: {[f'{float(x):.2e}' for x in pb_du1]}",
              flush=True)
        print(f"   eps_occ(dense) [eV]: {[f'{float(x):.2f}' for x in wo]}",
              flush=True)


if __name__ == "__main__":
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        run(e, er)
    print("PROBE_DONE", flush=True)
