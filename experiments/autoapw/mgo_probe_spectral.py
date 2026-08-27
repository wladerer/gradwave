"""Spectral decomposition of the growing MgO S-metric response + S(ecut) scan.

Part 1 (per ecut rung, one k): decompose du0 and du1 over the conduction
eigenstates of the generalized problem: dominant contributions' eps_m,
denominator eps_n - eps_m, S-coefficient |c_m|, and the state's EUCLIDEAN norm
(S-orthonormal states with large euclidean norm = near-null-S / ghost
character). Also per-band |du0| and the |du0(G)| high-G profile.

Part 2 (no SCF): min-eig of S(k) at Gamma vs ecut 40..120 Ry — does the
overlap head to singular/negative (the PAW ghost precursor)?

Part 3: all-NC MgO control (plain NC analytic route) at 40/60/80 Ry.
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


def fro(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)))


def part1(ecut_ry: float, ecutrho_ry: float) -> None:
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
    w, u, dh, ds = kg._geigh_and_dhds(hks, kc)
    s0m = hks.s(kc)
    uo, wo = u[:, :nocc], w[:nocc]
    uc, wc = u[:, nocc:], w[nocc:]
    wo_c = wo.to(CDTYPE)
    o0u = kg._gen_velocity(dh, ds, uo, wo_c, ep)
    du0 = kg._resolvent_apply_s(uc, wc, wo, o0u, s0m)

    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry ik={ik} npw={hks.npw} ===",
          flush=True)
    pb0 = torch.linalg.norm(du0, dim=0)
    print(f" per-band |du0|: {[f'{float(x):.2e}' for x in pb0]}", flush=True)

    # euclidean norms of the S-orthonormal conduction states (ghost character)
    eun = torch.linalg.norm(uc, dim=0)
    top = torch.argsort(eun, descending=True)[:6]
    print(" top-euclid conduction states (idx, eps[eV], |u|_2):", flush=True)
    for t in top:
        print(f"   m={int(t)}  eps={float(wc[t]):+9.2f}  |u|={float(eun[t]):.3f}",
              flush=True)

    # spectral decomposition of du0 for the O 2s band (band 4)
    n_band = 4
    a = uc.mH @ (s0m @ o0u[:, n_band])  # S-coefficients
    denom = (wo[n_band] - wc).to(CDTYPE)
    c = a / denom
    contrib = torch.abs(c) * eun  # euclidean size of each state's contribution
    order = torch.argsort(contrib, descending=True)[:10]
    print(f" band {n_band} (eps={float(wo[n_band]):+.2f} eV) du0 spectral "
          "decomposition (top 10):", flush=True)
    for m in order:
        print(f"   eps_m={float(wc[m]):+10.2f}  denom={float(wo[n_band] - wc[m]):+10.2f}"
              f"  |c|={float(torch.abs(c[m])):.3e}  |u|_2={float(eun[m]):.3f}"
              f"  contrib={float(contrib[m]):.3e}", flush=True)

    # |du0(G)| profile vs |G| for band 4
    g2 = ((hks.g_cart + kc) ** 2).sum(-1)
    gn = torch.sqrt(g2)
    edges = torch.linspace(0, float(gn.max()) * 1.001, 13)
    prof = []
    for i in range(12):
        m = (gn >= edges[i]) & (gn < edges[i + 1])
        prof.append(float(torch.linalg.norm(du0[m, n_band])) if m.any() else 0.0)
    print(f" |du0(G)| band-4 shell profile: {[f'{x:.2e}' for x in prof]}",
          flush=True)


def part2() -> None:
    from gradwave.postscf.kgeometry_nmr import BlochHKS, build_uspp_response_ctx  # noqa: F401
    from gradwave.postscf.kgeometry_nmr import _overlap_kbprojectors

    print("\n=== part 2: min-eig S(Gamma) vs ecut (no SCF) ===", flush=True)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    for e in (40.0, 60.0, 80.0, 100.0, 120.0):
        system = setup_uspp(CELL, POS, [0, 1], [paw_mg, paw_o], ecut=e * RY,
                            kmesh=(1, 1, 1), ecutrho=6 * e * RY, nbands=12)
        kb = _overlap_kbprojectors(system, system.spheres[0])
        npw = kb.g_cart.shape[0]
        p = kb.p(torch.zeros(3, dtype=RDTYPE))
        smat = torch.eye(npw, dtype=CDTYPE) + p.mT @ (
            kb.dij_full.to(CDTYPE) @ p.conj())
        ev = torch.linalg.eigvalsh(0.5 * (smat + smat.mH))
        print(f"  ecut {e:5.1f} Ry npw={npw}: min-eig S = {float(ev.min()):+.5e}"
              f"  max = {float(ev.max()):.3f}", flush=True)


def part3() -> None:
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    print("\n=== part 3: all-NC MgO control ===", flush=True)
    upfs = [parse_upf("tests/fixtures/qe/pseudos/Mg_ONCV_PBE-1.2.upf"),
            parse_upf("tests/fixtures/qe/pseudos/O_ONCV_PBE-1.2.upf")]
    for e in (40.0, 60.0, 80.0):
        system = setup_system(CELL, POS, [0, 1], upfs, ecut=e * RY,
                              kmesh=(2, 2, 2), nbands=12, use_symmetry=False)
        res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False,
                  max_iter=120)
        assert res.converged
        sig = sigma_shielding_dq(res, use_symmetry=False)
        iso = [float(torch.diagonal(sig[a]).mean()) for a in range(2)]
        print(f"  NC ecut {e} Ry: bare sigma_iso  Mg = {iso[0]:+.2f}  "
              f"O = {iso[1]:+.2f} ppm", flush=True)


if __name__ == "__main__":
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        part1(e, er)
    part2()
    part3()
    print("SPECTRAL_PROBE_DONE", flush=True)
