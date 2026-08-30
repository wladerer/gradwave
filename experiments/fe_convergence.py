"""bcc-Fe moment convergence: smearing-scheme x width x k-mesh.

Resolves whether the Fe moment's 6^3 -> 8^3 drift (2.22 -> 2.40 uB at gaussian
0.1 eV) is a finite-smearing artifact or real mesh-convergence physics. Gaussian
smearing carries an O(width^2) finite-temperature error; Methfessel-Paxton (mp1)
and Marzari-Vanderbilt cold smearing cancel it to higher order, so their moment
should be far more stable across width and mesh. If gaussian drifts while
mp1/cold converge to a consistent value, the drift is a gaussian artifact and
that consistent value is the trustworthy moment.

For each scheme the moment and free energy are extrapolated to zero width. The
error_estimate (numerical-error UQ) block is emitted for the anchor via the
supported api path.

Writes experiments/fe_convergence_data.json.

Run (asus, 8 threads):
    OMP_NUM_THREADS=8 uv run python experiments/fe_convergence.py --threads 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

HERE = Path(__file__).resolve().parent
PSE = HERE.parent / "tests" / "fixtures" / "qe" / "pseudos"
RY = 13.605693122994

A = 2.87
CELL = A / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
FFT = (24, 24, 24)  # match the CI fixture grid
NBANDS = 12

SCHEMES = ["gaussian", "mp1", "cold"]
WIDTHS = [0.05, 0.10, 0.20]        # eV
MESHES = [(6, 6, 6), (8, 8, 8), (10, 10, 10)]
FINE = (10, 10, 10)                # extrapolation mesh


def _run(scheme: str, width: float, mesh) -> dict[str, Any]:
    upf = parse_upf(str(PSE / "Fe_ONCV_PBE-1.2.upf"))
    system = setup_system(CELL, np.zeros((1, 3)), [0], [upf], ecut=60 * RY,
                          kmesh=mesh, nbands=NBANDS, fft_shape=FFT)
    t0 = time.perf_counter()
    res = scf(system, SpinPBE(), smearing=scheme, width=width, nspin=2,
              start_mag=[0.4], etol=1e-9, rhotol=1e-8, max_iter=200,
              verbose=False)
    wall = time.perf_counter() - t0
    return {
        "scheme": scheme, "width_eV": width, "mesh": list(mesh),
        "m_tot": float(res.mag_total), "m_abs": float(res.mag_abs),
        "fermi_eV": float(res.fermi),
        "E_free_eV": float(res.energies.free_energy),
        "E_internal_eV": float(res.energies.total),
        "n_iter": int(res.n_iter), "converged": bool(res.converged),
        "wall_s": round(wall, 1),
    }


def _extrap_zero_width(rows: list[dict], scheme: str, mesh) -> dict[str, Any]:
    """Fit m(width) and F(width) at fixed mesh and extrapolate to width -> 0.

    Quadratic in width (3 points); the O(width^2) gaussian error is captured by
    the width^2 term, so the intercept is the zero-broadening value."""
    sub = sorted((r for r in rows
                  if r["scheme"] == scheme and tuple(r["mesh"]) == tuple(mesh)),
                 key=lambda r: r["width_eV"])
    w = np.array([r["width_eV"] for r in sub])
    deg = 2 if len(w) >= 3 else 1
    m0 = float(np.polyval(np.polyfit(w, [r["m_tot"] for r in sub], deg), 0.0))
    f0 = float(np.polyval(np.polyfit(w, [r["E_free_eV"] for r in sub], deg), 0.0))
    return {"scheme": scheme, "mesh": list(mesh),
            "m_tot_width0": m0, "E_free_width0_eV": f0,
            "widths": w.tolist(), "m_at_width": [r["m_tot"] for r in sub]}


def _error_estimate() -> dict[str, Any] | None:
    """Numerical-error UQ block for the anchor, via the supported api path."""
    try:
        from gradwave.api.scf import run_scf
        from gradwave.api.summary import _error_estimate_block
        from gradwave.inputs import load_input

        inp = load_input(str(HERE / "fe_anchor.yaml"))
        res = run_scf(inp, verbose=False)
        block = _error_estimate_block(res, inp)
        return {"anchor_m_tot": float(res.mag_total), "block": block}
    except Exception as exc:  # record, don't abort the study
        return {"error": repr(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    print("=== Fe smearing x width x mesh convergence ===", flush=True)
    grid: list[dict[str, Any]] = []
    for mesh in MESHES:
        for scheme in SCHEMES:
            for width in WIDTHS:
                r = _run(scheme, width, mesh)
                grid.append(r)
                print(f"{scheme:8s} w={width:.2f} k={mesh}: "
                      f"m_tot={r['m_tot']:.3f} Ef={r['fermi_eV']:.3f} "
                      f"F={r['E_free_eV']:.4f} it={r['n_iter']} "
                      f"conv={r['converged']} ({r['wall_s']:.0f}s)", flush=True)

    extrap = [_extrap_zero_width(grid, s, FINE) for s in SCHEMES]
    for e in extrap:
        print(f"extrap {e['scheme']:8s} k={tuple(e['mesh'])}: "
              f"m(w->0)={e['m_tot_width0']:.3f} uB  "
              f"F(w->0)={e['E_free_width0_eV']:.4f} eV", flush=True)
    ms = [e["m_tot_width0"] for e in extrap]
    spread = float(max(ms) - min(ms))
    consistent = float(np.mean(ms))
    print(f"cross-scheme w->0 moment at k={FINE}: "
          f"{consistent:.3f} +/- {spread/2:.3f} uB (spread {spread:.3f})",
          flush=True)

    err = _error_estimate()

    out = {
        "grid": grid,
        "extrapolation": extrap,
        "fine_mesh": list(FINE),
        "cross_scheme_width0": {"mean_m_tot": consistent,
                                "half_spread": spread / 2.0,
                                "full_spread": spread,
                                "schemes": SCHEMES},
        "error_estimate": err,
        "settings": {"a_ang": A, "ecut_Ry": 60, "nbands": NBANDS,
                     "fft_shape": list(FFT), "start_mag": 0.4,
                     "etol_eV": 1e-9, "rhotol": 1e-8, "symmetry": False,
                     "pseudo": "Fe_ONCV_PBE-1.2.upf"},
    }
    path = Path(args.outdir) / "fe_convergence_data.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
