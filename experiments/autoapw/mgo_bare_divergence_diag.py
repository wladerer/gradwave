"""Diagnose the ecut divergence of the analytic-USPP bare shielding on MgO.

Builds MgO rocksalt PAW ground states at 2-3 ecut rungs and instruments the
bare term (`sigma_shielding_dq` through the S-metric ctx):

  1. the per-site bare sigma_iso at each rung (confirm divergence),
  2. the Biot-Savart shell decay: for the O site, the cumulative shielding
     column as a function of a |G| cutoff, and the per-shell magnitude, to
     distinguish a decaying tail (physical missing term, converges with ecut)
     from a non-decaying / growing tail (diverging evaluation).

Run on asus, OMP_NUM_THREADS=5, one at a time.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.xc import PBE

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"

# MgO rocksalt, a = 4.21 Angstrom, 2-atom primitive (fcc)
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL  # Mg at 0, O at (1/2)


def build(ecut_ry: float, ecutrho_ry: float):
    torch.set_num_threads(5)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(
        CELL, POS, [0, 1], [paw_mg, paw_o],
        ecut=ecut_ry * RY, kmesh=(2, 2, 2), ecutrho=ecutrho_ry * RY, nbands=12,
    )
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"], f"SCF not converged at {ecut_ry}/{ecutrho_ry}"
    return [paw_mg, paw_o], res


def analyze(ecut_ry: float, ecutrho_ry: float):
    from gradwave.postscf import kgeometry_nmr as kg

    paws, res = build(ecut_ry, ecutrho_ry)
    ctx = kg.build_uspp_response_ctx(res, PBE())
    system = ctx.system

    # capture the (s0f, dsf, q_hat) fields going into every Biot-Savart branch
    captured = []
    orig = kg._biot_savart_sigma_cols_dq

    def spy(s0, ds, q_hat_cart, g_cart, sites):
        captured.append((s0.detach().clone(), ds.detach().clone(),
                         q_hat_cart.detach().clone().numpy()))
        return orig(s0, ds, q_hat_cart, g_cart, sites)

    kg._biot_savart_sigma_cols_dq = spy
    try:
        sig_bare = kg.sigma_shielding_dq(res, uspp=ctx, use_symmetry=False)
    finally:
        kg._biot_savart_sigma_cols_dq = orig

    iso = [float(torch.diagonal(sig_bare[a]).mean()) for a in range(2)]
    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry   shape={tuple(system.grid.shape)} "
          f"npw~{system.spheres[0].miller.shape[0]} ===")
    print(f"  bare sigma_iso:  Mg = {iso[0]:+.2f}   O = {iso[1]:+.2f} ppm")

    # --- Biot-Savart shell decay for the O site (index 1) ---
    import math

    from gradwave.constants import ALPHA_FS, E2
    from gradwave.postscf.kgeometry_nmr import _antisym_field_g

    g = torch.as_tensor(np.asarray(system.grid.g_cart), dtype=torch.float64)
    gnorm = g.norm(dim=-1)
    o_site = torch.as_tensor(POS[1], dtype=torch.float64)
    pref = 4.0 * math.pi * ALPHA_FS**2 / E2

    # accumulate |per-G contribution to the O column| over all branches, and
    # its split by sub-term (cross1 from response ds, crossq from s0, kernel).
    gmax = float(gnorm.max())
    nsh = 24
    edges = np.linspace(0.0, gmax * 1.001, nsh + 1)
    shell = np.digitize(gnorm.reshape(-1).numpy(), edges) - 1
    per_shell = {k: np.zeros(nsh) for k in ("total", "cross1", "crossq", "kernel")}

    for s0f, dsf, q_hat in captured:
        f0 = _antisym_field_g(s0f, s0f).permute(1, 2, 3, 0)
        f1 = _antisym_field_g(dsf, -dsf).permute(1, 2, 3, 0)
        g2 = (g * g).sum(-1).clone()
        g2[0, 0, 0] = 1.0
        qh = torch.as_tensor(q_hat, dtype=torch.float64)
        gc128 = g.to(torch.complex128)
        c0 = 1j * torch.linalg.cross(gc128, f0.to(torch.complex128), dim=-1)
        c1 = 1j * torch.linalg.cross(gc128, f1.to(torch.complex128), dim=-1)
        cq = 1j * torch.linalg.cross(qh.to(torch.complex128).expand_as(f0),
                                     f0.to(torch.complex128), dim=-1)
        g2c = g2[..., None].to(torch.complex128)
        kern = -(2.0 * (g @ qh) / g2)[..., None].to(torch.complex128) * c0
        gdotr = torch.einsum("ijka,a->ijk", g, o_site)
        phase = torch.exp(1j * gdotr.to(torch.complex128))
        parts = {"cross1": c1 / g2c, "crossq": cq / g2c, "kernel": kern / g2c}
        for name, fld in parts.items():
            contrib = pref * 2.0 * (phase[..., None] * fld).real  # (n1,n2,n3,3)
            mag = contrib.norm(dim=-1).reshape(-1).numpy()
            mag[0] = 0.0  # G=0 omitted
            for k in range(nsh):
                per_shell[name][k] += mag[shell == k].sum()
        tot = pref * 2.0 * (phase[..., None] * (parts["cross1"] + parts["crossq"] + parts["kernel"])).real
        magt = tot.norm(dim=-1).reshape(-1).numpy()
        magt[0] = 0.0
        for k in range(nsh):
            per_shell["total"][k] += magt[shell == k].sum()

    print("  O-site Biot-Savart |contribution| per |G| shell (summed over branches):")
    print("   |G|(1/A)   total      cross1(dS)  crossq(S0)  kernel")
    for k in range(nsh):
        gc = 0.5 * (edges[k] + edges[k + 1])
        print(f"   {gc:7.3f}  {per_shell['total'][k]:10.3e}  "
              f"{per_shell['cross1'][k]:10.3e}  {per_shell['crossq'][k]:10.3e}  "
              f"{per_shell['kernel'][k]:10.3e}")
    print(f"  cumulative total |contribution| = {per_shell['total'].sum():.4e}")
    return iso


if __name__ == "__main__":
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        analyze(e, er)
