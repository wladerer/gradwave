"""Capture the analytic-USPP branch fields for MgO at one ecut rung and save
them (plus rho_full / rho_smooth) to a .pt so every sub-term decomposition is a
cheap offline post-processing step.

Decompositions done inline at the end:
  A. full assembly (reference; must reproduce the recorded bare sigma_iso),
  B. dia term REMOVED entirely from S0,
  C. dia term with the SMOOTH pseudo density only (rho_full - rho_aug),
i.e. B-A = total dia contribution, C-A = the augmentation-density part of dia.

Usage: python mgo_field_capture.py <Mg.UPF> <ecut_ry> <ecutrho_ry> <out.pt>
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

MG = sys.argv[1]
ECUT = float(sys.argv[2])
ECUTRHO = float(sys.argv[3])
OUT = sys.argv[4]
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def main() -> None:
    from gradwave.core.batch import g_to_r_b
    from gradwave.postscf import kgeometry_nmr as kg

    torch.set_num_threads(5)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(CELL, POS, [0, 1], [paw_mg, paw_o], ecut=ECUT * RY,
                        kmesh=(2, 2, 2), ecutrho=ECUTRHO * RY, nbands=12)
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"]
    ctx = kg.build_uspp_response_ctx(res, PBE())
    shape = tuple(system.grid.shape)
    vol = float(system.grid.volume)

    # smooth pseudo density from the occupied coefficients (no augmentation)
    bk = ctx.bk
    from gradwave.postscf._response import insulator_window, pad_coeffs
    nocc = insulator_window(res["occupations"], 2.0, "insulating required")
    c_all = pad_coeffs(list(res["coeffs"]), bk.npw_max)
    c_occ = c_all[:, :nocc]
    psi_r = g_to_r_b(c_occ, bk, shape)  # (nk, nocc, *shape)
    w = (2.0 * system.kweights.to(RDTYPE))
    rho_smooth = torch.einsum(
        "k,kbxyz->xyz", w, (psi_r.conj() * psi_r).real.to(RDTYPE)) / vol
    rho_full = res["rho"] if res["rho"].dim() == 3 else res["rho"][0]
    print(f"rho_full integral  = {float(rho_full.mean()) * vol * 1.0:.6f}*npts/vol "
          f"-> N_e = {float(rho_full.sum()) * vol / rho_full.numel():.4f}")
    print(f"rho_smooth N_e     = {float(rho_smooth.sum()) * vol / rho_smooth.numel():.4f}")

    captured: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]] = []
    orig = kg._biot_savart_sigma_cols_dq

    def spy(s0, ds, q_hat_cart, g_cart, sites):
        captured.append((s0.detach().clone(), ds.detach().clone(),
                         np.asarray(q_hat_cart.detach().cpu().numpy())))
        return orig(s0, ds, q_hat_cart, g_cart, sites)

    kg._biot_savart_sigma_cols_dq = spy
    try:
        sig_bare = kg.sigma_shielding_dq(res, uspp=ctx, use_symmetry=False)
    finally:
        kg._biot_savart_sigma_cols_dq = orig
    iso = [float(torch.diagonal(sig_bare[a]).mean()) for a in range(2)]
    print(f"A. full bare sigma_iso:  Mg = {iso[0]:+.2f}   O = {iso[1]:+.2f} ppm")

    torch.save({
        "fields": captured, "rho_full": rho_full.cpu(),
        "rho_smooth": rho_smooth.cpu(), "shape": shape, "vol": vol,
        "g_cart": torch.as_tensor(np.asarray(system.grid.g_cart)),
        "sites": torch.as_tensor(POS, dtype=torch.float64),
        "cell": CELL, "sig_bare": sig_bare.cpu(),
        "ecut": ECUT, "ecutrho": ECUTRHO,
    }, OUT)
    print(f"saved fields -> {OUT}")

    # ---- decompositions B and C (reassemble sigma from modified fields) ----
    g_cart = torch.as_tensor(np.asarray(system.grid.g_cart))
    sites = torch.as_tensor(POS, dtype=RDTYPE)
    # the transverse-pol bookkeeping mirrors sigma_shielding_dq's lstsq
    from gradwave.core.fftbox import r_to_g  # noqa: F401  (used by helper)

    def assemble(delta_rho: torch.Tensor | None) -> list[float]:
        """Recompute sigma_iso per site with `delta_rho` SUBTRACTED from the
        dia term of every branch's S0 (None = reference)."""
        b = np.linalg.inv(CELL).T * 2 * np.pi
        k_frac = np.stack([sph.k_frac for sph in system.spheres])
        mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
        axes = [i for i in range(3) if mesh_n[i] > 1]
        pols_per_axis = []
        for i in axes:
            qh = np.asarray(b[i], dtype=float) / np.linalg.norm(b[i])
            pols_per_axis.append((qh, list(kg._transverse_frame(qh))))
        b_rows, m_rows = [], []
        ci = 0
        for qh, pols in pols_per_axis:
            for pol in pols:
                s0f, dsf, qh_c = captured[ci]
                ci += 1
                s0m = s0f
                if delta_rho is not None:
                    ep_t = torch.as_tensor(np.asarray(pol, dtype=float),
                                           dtype=RDTYPE)
                    s0m = s0f - torch.einsum(
                        "m,ijk->mijk", ep_t.to(CDTYPE),
                        (2.0 * HBAR2_2M) * delta_rho.to(CDTYPE))
                col, _ = kg._biot_savart_sigma_cols_dq(
                    s0m, dsf, torch.as_tensor(qh, dtype=RDTYPE), g_cart, sites)
                m_rows.append(col)
                b_rows.append(np.cross(qh, pol))
        bmat = torch.as_tensor(np.stack(b_rows), dtype=RDTYPE)
        mmat = torch.stack(m_rows)
        outs = []
        for s in range(2):
            t = torch.linalg.lstsq(bmat, mmat[:, s, :]).solution.mT * 1e6
            outs.append(float(torch.diagonal(t).mean()))
        return outs

    ref = assemble(None)
    no_dia = assemble(rho_full)
    smooth_dia = assemble(rho_full - rho_smooth)
    print(f"A. reference (recheck):  Mg = {ref[0]:+.2f}   O = {ref[1]:+.2f}")
    print(f"B. dia removed:          Mg = {no_dia[0]:+.2f}   O = {no_dia[1]:+.2f}")
    print(f"C. dia smooth-rho only:  Mg = {smooth_dia[0]:+.2f}   O = {smooth_dia[1]:+.2f}")
    print(f"   dia total (A-B):      Mg = {ref[0] - no_dia[0]:+.2f}   "
          f"O = {ref[1] - no_dia[1]:+.2f}")
    print(f"   dia from n_aug (A-C): Mg = {ref[0] - smooth_dia[0]:+.2f}   "
          f"O = {ref[1] - smooth_dia[1]:+.2f}")


if __name__ == "__main__":
    main()
