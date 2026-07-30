"""Step 4: one real step against the WIEN2k all-electron Si reference.

- gradwave's current Delta(Si) at the probe settings (theta=0), and at the
  converged delta-gauge settings (48 Ry / 8^3) as a basis-error reference.
- a few Adam steps on theta to match the WIEN2k E(V) SHAPE (mean-subtracted,
  the offset-invariant analogue of the calcDelta shift-to-minimum), at the probe
  settings.
- Delta(Si) before/after at the probe settings.
- off-training force check: forces on a displaced Si cell, uncorrected vs the
  trained correction baked into vloc_tables (the question is whether the EOS fix
  degrades forces).

Honesty: the probe ecut conflates basis error with pseudization error. The
|Delta_probe - Delta_converged| gap is the basis-error estimate to compare
against any Delta improvement.
"""
import os

import torch

torch.set_num_threads(int(os.environ.get("GW_THREADS", "2")))
import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from correction import default_centers, dv_table  # noqa: E402
from probe import (  # noqa: E402
    apply_correction,
    build_si,
    build_si_displaced,
    energy_of_theta,
    fixed_grid,
    run_scf,
)

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.dtypes import RDTYPE  # noqa: E402
from gradwave.postscf.eos import (  # noqa: E402
    EV_A3_TO_GPA,
    birch_murnaghan,
    delta_value,
    fit_bm3,
)
from gradwave.postscf.forces import forces  # noqa: E402

ECUT = float(os.environ.get("GW_ECUT_RY", "28"))
K = int(os.environ.get("GW_K", "6"))
ECUT_CONV = float(os.environ.get("GW_ECUT_CONV", "48"))
K_CONV = int(os.environ.get("GW_K_CONV", "8"))
NSTEP = int(os.environ.get("GW_NSTEP", "10"))
LR = float(os.environ.get("GW_LR", "0.05"))
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
V0_W, B0_W, B1_W = 20.4530, 88.545, 4.31  # WIEN2k Si (cases.WIEN2K)


def volumes(scales):
    return [V0_W * s for s in scales]  # Ang^3/atom (diamond: 2 atoms/cell)


def eos_curve(systems, theta, centers, prev=None):
    es, results = [], []
    warm = None
    for i, s in enumerate(systems):
        corr = apply_correction(s, theta, centers)
        r = run_scf(corr, start_from=(prev[i] if prev is not None else warm))
        es.append(float(r.energies.total) / 2.0)  # per atom (2 atoms/cell)
        results.append(r)
        warm = r
    return es, results


def delta_vs_wien(vols, es):
    fit = fit_bm3(np.array(vols), np.array(es))
    ref = (0.0, V0_W, B0_W / EV_A3_TO_GPA, B1_W)
    return delta_value(fit, ref), fit


def main():
    t0 = time.time()
    centers = default_centers(4)
    vols = volumes(SCALES)
    scf_count = 0

    # --- basis-error reference: Delta at converged settings, theta=0 ---
    grid_c = fixed_grid(SCALES, ECUT_CONV, K_CONV)
    sys_c = [build_si(s, ECUT_CONV, K_CONV, fft_shape=grid_c) for s in SCALES]
    e_conv, _ = eos_curve(sys_c, torch.zeros(4), centers)
    scf_count += len(SCALES)
    d_conv, fit_conv = delta_vs_wien(vols, e_conv)

    # --- probe settings, theta=0 ---
    grid_p = fixed_grid(SCALES, ECUT, K)
    sys_p = [build_si(s, ECUT, K, fft_shape=grid_p) for s in SCALES]
    e0, res0 = eos_curve(sys_p, torch.zeros(4), centers)
    scf_count += len(SCALES)
    d_before, fit_before = delta_vs_wien(vols, e0)

    # --- WIEN2k reference E(V) shape to train against ---
    e_ref = birch_murnaghan(np.array(vols), 0.0, V0_W,
                            B0_W / EV_A3_TO_GPA, B1_W)
    e_ref_t = torch.as_tensor(e_ref, dtype=RDTYPE)

    # --- train theta at probe settings against the WIEN2k shape ---
    theta = torch.zeros(4, dtype=RDTYPE, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    prev = None
    hist = []
    for step in range(NSTEP):
        theta_cur = theta.detach().clone()
        e_gw, results = eos_curve(sys_p, theta_cur, centers, prev=prev)
        scf_count += len(SCALES)
        prev = results
        e_terms = [energy_of_theta(results[i], apply_correction(
            sys_p[i], theta, centers), theta, centers,
            theta_ref=theta_cur) / 2.0 for i in range(len(SCALES))]
        e_stack = torch.stack(e_terms)
        resid = (e_stack - e_stack.mean()) - (e_ref_t - e_ref_t.mean())
        loss = (resid**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        d_now, _ = delta_vs_wien(vols, e_gw)
        hist.append({"step": step, "delta": d_now,
                     "theta": theta.detach().tolist()})
        print(f"step {step:2d} delta={d_now:.4f} meV/at theta={theta.detach().tolist()}")

    e_after, _ = eos_curve(sys_p, theta.detach(), centers, prev=prev)
    scf_count += len(SCALES)
    d_after, fit_after = delta_vs_wien(vols, e_after)

    # --- off-training force check on a displaced cell (probe settings) ---
    # built with use_symmetry=False so the reduced k-mesh matches the broken
    # symmetry of the displaced cell (the correction vs uncorrected comparison
    # is otherwise apples-to-apples, but absolute forces need the true cell)
    base_disp = build_si_displaced(1.0, ECUT, K, [0.10, 0.0, 0.0])
    r_unc = run_scf(base_disp)
    f_unc = forces(r_unc, xc=PBE())  # Si.upf carries an NLCC core charge
    # bake trained correction into vloc_tables so forces() sees it
    dv = dv_table(base_disp.grid, theta.detach(), centers)
    vt = base_disp.vloc_tables.clone()
    vt[0] = vt[0] + dv
    sys_corr = dataclasses.replace(base_disp, vloc_tables=vt)
    r_cor = run_scf(sys_corr)
    f_cor = forces(r_cor, xc=PBE())
    scf_count += 2
    df = (f_cor - f_unc)
    fmax_unc = float(f_unc.abs().max())
    fmax_diff = float(df.abs().max())

    out = {
        "settings": {"ecut_probe": ECUT, "k_probe": K,
                     "ecut_conv": ECUT_CONV, "k_conv": K_CONV,
                     "grid_probe": list(grid_p), "grid_conv": list(grid_c)},
        "delta_converged_meV": d_conv,
        "delta_probe_before_meV": d_before,
        "delta_probe_after_meV": d_after,
        "basis_error_estimate_meV": abs(d_before - d_conv),
        "v0_before": fit_before.v0, "b0_before_GPa": fit_before.b0_GPa,
        "v0_after": fit_after.v0, "b0_after_GPa": fit_after.b0_GPa,
        "v0_conv": fit_conv.v0, "b0_conv_GPa": fit_conv.b0_GPa,
        "v0_wien2k": V0_W, "b0_wien2k_GPa": B0_W,
        "theta_final": theta.detach().tolist(),
        "force_check": {"displacement_A": 0.10,
                        "fmax_uncorrected_eV_A": fmax_unc,
                        "fmax_force_change_eV_A": fmax_diff,
                        "f_uncorrected": f_unc.tolist(),
                        "f_corrected": f_cor.tolist()},
        "scf_count": scf_count, "wall_s": round(time.time() - t0, 1),
        "history": hist,
    }
    (Path(__file__).parent / "results_step4.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nDelta(Si) converged({ECUT_CONV}Ry/{K_CONV}^3) = {d_conv:.4f} meV/at")
    print(f"Delta(Si) probe({ECUT}Ry/{K}^3) before = {d_before:.4f}, "
          f"after = {d_after:.4f} meV/at")
    print(f"basis-error estimate |probe-conv| = {abs(d_before-d_conv):.4f} meV/at")
    print(f"force: max|F|_unc = {fmax_unc:.4f}, max|dF| = {fmax_diff:.4f} eV/A")
    print(f"SCF count {scf_count}, wall {out['wall_s']}s")


if __name__ == "__main__":
    main()
