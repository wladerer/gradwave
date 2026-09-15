"""Validate the differentiable force-theorem MAE-vs-strain gradient.

Proves the autograd gradient dMAE/dε (postscf.mae.mae_strain_gradient) is
correct by two independent checks on a CHEAP FePt config (the autograd-vs-FD
agreement is basis-independent, so a coarse mesh / low ecut suffices — this is
NOT a converged physical MAE):

  1. VALUE cross-check: mae_strained at ε=0 vs the established band-energy force
     theorem (force_theorem_mae) — same frozen-density anisotropy, so they agree
     up to the total-vs-band force-theorem accounting.
  2. GRADIENT check (the headline): the analytic dMAE/dη for a volume-conserving
     tetragonal strain η (ε = η·diag(-1,-1,2), c/a → c/a·(1+3η)) vs a central
     finite difference of the SAME frozen-orbital functional. Must agree to ~1e-5.

    GW_THREADS=8 uv run python benchmarks/mae_inverse/gradient_check.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.dtypes import RDTYPE
from gradwave.postscf.mae import (
    _fband_strained,
    _frozen_oneshot,
    force_theorem_mae,
    mae_strained,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

torch.set_num_threads(int(os.environ.get("GW_THREADS", "8")))
sys.stdout.reconfigure(line_buffering=True)

RY = 13.605693122994
PSE = Path(os.environ.get("GW_PSE",
           str(Path(__file__).parents[2] / "tests/fixtures/qe/pseudos")))
FE = parse_upf(str(PSE / "Fe_ONCV_PBE_FR-1.0.upf"))
PT = parse_upf(str(PSE / "Pt_ONCV_PBE_FR-1.0.upf"))
FRAC = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])

A0, C0 = 2.723, 3.712                 # L1_0 FePt reference [Å]
# CHEAP probe config — coarse mesh + low ecut. Not physically converged; the
# autograd==FD identity we test does not depend on convergence.
ECUT = 30 * RY
KMESH = (2, 2, 2)
INIT = [[0, 0, 3.0], [0, 0, 0.4]]
DELTA = 1e-4                          # tetragonal-strain FD step


def build():
    cell = np.diag([A0, A0, C0])
    return setup_system(cell, FRAC @ cell, [0, 1], [FE, PT], ecut=ECUT,
                        kmesh=KMESH, nbands=30, use_symmetry=False,
                        time_reversal=False)


def main():
    t0 = time.time()
    xc = NoncollinearXC(LSDA_PW92())
    print(f"FePt MAE-gradient check (ecut {ECUT/RY:.0f} Ry, mesh {KMESH})",
          flush=True)
    res = scf_noncollinear(build(), xc, mag_vec_init=INIT, smearing="gaussian",
                           width=0.1, etol=1e-9, rhotol=1e-7, max_iter=300,
                           mixing_alpha=0.3, mixing_history=12, verbose=False)
    assert res.converged, "SCF not converged"
    print(f"SCF converged in {res.n_iter} it ({time.time()-t0:.0f}s)", flush=True)

    # reference band energies from the established force theorem (no fold)
    ft = force_theorem_mae(res, xc, [[0, 0, 1.0], [1.0, 0, 0]], verbose=False)
    ref = res.mag_vec / np.linalg.norm(res.mag_vec)
    ref_t = torch.as_tensor(ref, dtype=RDTYPE)
    prep_h = _frozen_oneshot(res, xc, (1, 0, 0), ref_t, smearing="gaussian",
                             width=0.1, diago_tol=1e-10)
    prep_e = _frozen_oneshot(res, xc, (0, 0, 1), ref_t, smearing="gaussian",
                             width=0.1, diago_tol=1e-10)

    # (1) per-direction F_band(ε=0) vs force_theorem_mae — validates the
    # band-energy potential-expectation assembly directly (tightest gate).
    eps0 = torch.zeros(3, 3, dtype=RDTYPE)
    fb_h = float(_fband_strained(res, xc, prep_h, eps0))
    fb_e = float(_fband_strained(res, xc, prep_e, eps0))
    ft_h, ft_e = float(ft.band_free_energies[1]), float(ft.band_free_energies[0])
    d_h, d_e = abs(fb_h - ft_h), abs(fb_e - ft_e)
    print(f"\nF_band[hard]  mine={fb_h:+.6f}  ft={ft_h:+.6f}  Δ={d_h:.2e} eV",
          flush=True)
    print(f"F_band[easy]  mine={fb_e:+.6f}  ft={ft_e:+.6f}  Δ={d_e:.2e} eV",
          flush=True)
    mae_mine = (fb_h - fb_e) * 1000.0
    mae_ft = float(ft.mae[1]) * 1000.0
    print(f"MAE  mine={mae_mine:+.4f} meV   ft={mae_ft:+.4f} meV", flush=True)
    value_ok = d_h < 1e-4 and d_e < 1e-4

    # (2) gradient: analytic dMAE/dη vs central FD (SAME frozen orbitals),
    # volume-conserving tetragonal strain ε = η·diag(-1,-1,2), c/a → c/a·(1+3η)
    tetra = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 2.0]], dtype=RDTYPE)
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=True)
    mae0 = mae_strained(res, xc, prep_h, prep_e, eps)
    (g,) = torch.autograd.grad(mae0, eps)
    g = 0.5 * (g + g.T)
    dmae_deta_ag = float(-g[0, 0] - g[1, 1] + 2 * g[2, 2]) * 1000.0   # meV/η

    def E(eta):
        return float(mae_strained(res, xc, prep_h, prep_e,
                                  eta * tetra)) * 1000.0
    dmae_deta_fd = (E(DELTA) - E(-DELTA)) / (2 * DELTA)               # meV/η
    rel = abs(dmae_deta_ag - dmae_deta_fd) / (abs(dmae_deta_fd) + 1e-12)
    print(f"\ndMAE/dη  autograd = {dmae_deta_ag:+.6f} meV", flush=True)
    print(f"dMAE/dη  central-FD = {dmae_deta_fd:+.6f} meV", flush=True)
    print(f"relative mismatch = {rel:.2e}", flush=True)
    print(f"dMAE/d(c/a) = {dmae_deta_ag/(3*C0/A0):+.4f} meV  "
          f"(c/a = {C0/A0:.3f})", flush=True)
    grad_ok = rel < 1e-4

    ok = value_ok and grad_ok
    print(f"\nVALUE_CHECK {'PASS' if value_ok else 'FAIL'}  "
          f"GRADIENT_CHECK {'PASS' if grad_ok else 'FAIL'}  "
          f"total {time.time()-t0:.0f}s", flush=True)
    print("MAE_GRAD_DONE", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
