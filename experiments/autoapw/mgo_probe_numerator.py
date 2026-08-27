"""Split the growing MgO perturbation numerator into kinetic / KB / eps*dS
parts, and compare against the shipped CG finite-q response.

Part 1: for band 4 (O 2s) and the dominant conduction state (eps ~ +21 eV),
print |<m|dKin|4>|, |<m|dH_NL|4>|, |<m|eps dS|4>| (pol-contracted) at both
rungs -> which operator part doubles.

Part 2: velocity_perturbation_q (USPP ctx, mesh q) per-band |dpsi| at both
rungs -> does the matrix-free +-q S-metric route grow the same way?
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
    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry ===", flush=True)

    b = np.linalg.inv(CELL).T * 2 * np.pi
    q_hat = np.asarray(b[0], dtype=float) / np.linalg.norm(b[0])
    pol = list(kg._transverse_frame(q_hat))[0]
    ep = torch.as_tensor(np.asarray(pol, dtype=float), dtype=CDTYPE)

    ik = 3
    kf = system.spheres[ik].k_frac
    hks = kg.BlochHKS.from_uspp_ctx(ctx, kf)
    kc = hks.k_ref_cart
    w, u, dh, ds = kg._geigh_and_dhds(hks, kc)
    uo, wo = u[:, :nocc], w[:nocc]
    uc, wc = u[:, nocc:], w[nocc:]
    n_band = 4
    u4 = uo[:, n_band]
    e4 = wo[n_band].to(CDTYPE)
    # dominant conduction state: nearest to +21.4 eV
    m_idx = int(torch.argmin((wc - 21.4).abs()))
    um = uc[:, m_idx]
    print(f" band4 eps={float(wo[n_band]):+.2f}  m eps={float(wc[m_idx]):+.2f}",
          flush=True)

    # operator split: dh = dkin + dNL (dkin diagonal 2*kappa*(k+G)_mu)
    kpg = hks.g_cart + kc
    for mu in range(3):
        if abs(float(ep[mu].real)) < 1e-12:
            continue
        dkin_u4 = (2.0 * HBAR2_2M) * kpg[:, mu].to(CDTYPE) * u4
        dnl_u4 = dh[mu] @ u4 - dkin_u4
        ds_u4 = (ds[mu] @ u4) * e4
        me_kin = complex(um.conj() @ dkin_u4)
        me_nl = complex(um.conj() @ dnl_u4)
        me_ds = complex(um.conj() @ ds_u4)
        print(f"  mu={mu} ep={float(ep[mu].real):+.3f}: "
              f"|<m|dKin|4>|={abs(me_kin):.4e}  |<m|dNL|4>|={abs(me_nl):.4e}  "
              f"|<m|eps dS|4>|={abs(me_ds):.4e}  "
              f"|<m|gen_v|4>|={abs(me_kin + me_nl - me_ds):.4e}", flush=True)
    # also the S-weighted numerator the resolvent actually uses
    o0u4 = kg._gen_velocity(dh, ds, uo[:, n_band:n_band + 1], e4[None], ep)
    s0m = hks.s(kc)
    num = complex((um.conj() @ (s0m @ o0u4[:, 0])))
    print(f"  resolvent numerator |<m|S O0|4>| = {abs(num):.4e}", flush=True)
    # how much of u4 lives on the added high-G shells: |u4(G)| profile
    gn = torch.sqrt((kpg ** 2).sum(-1))
    for cut in (8.0, 10.0, 12.0):
        m = gn > cut
        print(f"  |u4(G>{cut})| = {float(torch.linalg.norm(u4[m])):.4e}"
              f"  |um(G>{cut})| = {float(torch.linalg.norm(um[m])):.4e}",
              flush=True)

    # part 2: CG finite-q response per-band norms
    q_frac = np.zeros(3)
    q_frac[0] = 0.5  # mesh q on the 2x2x2 mesh
    sol = kg.velocity_perturbation_q(res, q_frac, cg_tol=1e-9, max_iter=800,
                                     uspp=ctx)
    dpsi = torch.einsum("m,mkng->kng", ep, sol.dpsi)  # pol-contracted
    pb = torch.linalg.norm(dpsi[ik], dim=-1)
    print(f" CG finite-q per-band |dpsi| (ik={ik}): "
          f"{[f'{float(x):.2e}' for x in pb]}", flush=True)
    print(f" CG finite-q total |dpsi| = {float(torch.linalg.norm(dpsi)):.4e}",
          flush=True)


if __name__ == "__main__":
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        run(e, er)
    print("NUMERATOR_PROBE_DONE", flush=True)
