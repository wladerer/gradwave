"""Step 1: endpoint exactness. theta=0 must reproduce the uncorrected SCF energy
bit-for-bit. Run the plain SCF and the theta=0 corrected SCF and diff.
"""
import os

import torch

torch.set_num_threads(2)
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from correction import default_centers  # noqa: E402
from probe import apply_correction, build_si, run_scf, zeros_theta  # noqa: E402

ECUT = float(os.environ.get("GW_ECUT_RY", "28"))
K = int(os.environ.get("GW_K", "6"))


def main():
    centers = default_centers(4)
    theta0 = zeros_theta(4)

    base = build_si(1.0, ECUT, K)
    r_base = run_scf(base)

    corr = apply_correction(base, theta0, centers)
    r_corr = run_scf(corr)

    e_base = float(r_base.energies.total)
    e_corr = float(r_corr.energies.total)
    diff = abs(e_base - e_corr)
    out = {
        "ecut_ry": ECUT, "k": K, "grid": list(base.grid.shape),
        "e_base_eV": e_base, "e_theta0_eV": e_corr,
        "abs_diff_eV": diff, "rel_diff": diff / abs(e_base),
        "n_iter_base": int(r_base.n_iter), "n_iter_corr": int(r_corr.n_iter),
    }
    print(json.dumps(out, indent=2))
    (Path(__file__).parent / "results_step1.json").write_text(
        json.dumps(out, indent=2))
    tol = 1e-9 * abs(e_base)
    print(f"\nENDPOINT {'PASS' if diff <= max(tol, 1e-9) else 'FAIL'} "
          f"(|dE|={diff:.3e} eV, rel={diff/abs(e_base):.2e})")


if __name__ == "__main__":
    main()
