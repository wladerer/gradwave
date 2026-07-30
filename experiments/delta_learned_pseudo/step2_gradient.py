"""Step 2: gradient credential. dE/dtheta by Hellmann-Feynman (autograd of the
differentiable E(theta) at the converged density) vs central finite difference
over theta with full SCF re-convergence at each displaced theta.

The repo's derivative-oracle convention (benchmarks/derivatives/README.md): a
first-derivative FD floors near ~1e-5 relative; match to that or explain.
"""
import os

import torch

torch.set_num_threads(2)
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from correction import default_centers  # noqa: E402
from probe import (  # noqa: E402
    apply_correction,
    build_si,
    energy_of_theta,
    run_scf,
)

from gradwave.dtypes import RDTYPE  # noqa: E402

ECUT = float(os.environ.get("GW_ECUT_RY", "28"))
K = int(os.environ.get("GW_K", "6"))
# a nonzero base theta so the check is not at the trivial theta=0 point
THETA0 = torch.tensor([0.15, -0.10, 0.05, -0.02], dtype=RDTYPE)
EPS = float(os.environ.get("GW_FD_EPS", "1e-3"))


def total_energy_at(base, theta, centers, start_from=None):
    corr = apply_correction(base, theta, centers)
    r = run_scf(corr, start_from=start_from)
    return float(r.energies.total), r


def main():
    centers = default_centers(4)
    base = build_si(1.0, ECUT, K)

    # analytic HF gradient at THETA0
    _, r0 = total_energy_at(base, THETA0, centers)
    corr = apply_correction(base, THETA0, centers)
    theta = THETA0.clone().detach().requires_grad_(True)
    e = energy_of_theta(r0, corr, theta, centers, theta_ref=THETA0)
    (g_hf,) = torch.autograd.grad(e, theta)
    g_hf = g_hf.detach()

    # central FD, re-converging the SCF at each displaced theta (warm-started)
    g_fd = torch.zeros_like(THETA0)
    for k in range(len(THETA0)):
        tp = THETA0.clone()
        tp[k] += EPS
        tm = THETA0.clone()
        tm[k] -= EPS
        ep, _ = total_energy_at(base, tp, centers, start_from=r0)
        em, _ = total_energy_at(base, tm, centers, start_from=r0)
        g_fd[k] = (ep - em) / (2 * EPS)

    rel = (g_hf - g_fd).abs() / g_fd.abs().clamp_min(1e-30)
    out = {
        "ecut_ry": ECUT, "k": K, "grid": list(base.grid.shape),
        "theta0": THETA0.tolist(), "fd_eps": EPS,
        "grad_hf": g_hf.tolist(), "grad_fd": g_fd.tolist(),
        "rel_err": rel.tolist(), "max_rel_err": float(rel.max()),
    }
    print(json.dumps(out, indent=2))
    (Path(__file__).parent / "results_step2.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nGRADIENT max rel err {rel.max():.2e} "
          f"({'PASS' if rel.max() < 1e-4 else 'CHECK'})")


if __name__ == "__main__":
    main()
