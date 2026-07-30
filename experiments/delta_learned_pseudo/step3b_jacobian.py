"""Step 3b: diagnose the step-3 recovery failure with the EOS-loss Jacobian.

At the converged density each volume's energy is exactly linear in theta
(E_local is linear in dv), so the mean-subtracted EOS shape observable is a
near-linear map r = A theta + O(theta^2). Recovery of theta* from the EOS alone
is well-posed only if A is well-conditioned. This script computes
A[i, k] = d(e_i - mean(e))/dtheta_k by autograd of the Hellmann-Feynman energy
at theta=0 (5 SCFs, warm-started) and reports its singular values, the
condition number, and the component of (theta_fit - theta*) inside the
null space, which is the quantitative explanation of the step-3 outcome.
"""
import torch

torch.set_num_threads(2)
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from correction import default_centers  # noqa: E402
from probe import (  # noqa: E402
    apply_correction,
    build_si,
    energy_of_theta,
    fixed_grid,
    run_scf,
)

from gradwave.dtypes import RDTYPE  # noqa: E402

ECUT = float(os.environ.get("GW_ECUT_RY", "28"))
K = int(os.environ.get("GW_K", "6"))
SCALES = [0.94, 0.97, 1.00, 1.03, 1.06]
THETA_STAR = torch.tensor([0.30, -0.20, 0.10, -0.04], dtype=RDTYPE)
# step-3 outcome (results_step3.json theta_final); fallback to the recorded run
_res3 = Path(__file__).parent / "results_step3.json"
if _res3.exists():
    THETA_FIT = torch.tensor(json.loads(_res3.read_text())["theta_final"],
                             dtype=RDTYPE)
else:
    THETA_FIT = torch.tensor([0.07157, 0.05923, 0.07143, 0.07207], dtype=RDTYPE)


def main():
    centers = default_centers(4)
    grid = fixed_grid(SCALES, ECUT, K)
    systems = [build_si(s, ECUT, K, fft_shape=grid) for s in SCALES]

    theta0 = torch.zeros(4, dtype=RDTYPE)
    rows = []
    warm = None
    for s in systems:
        corr = apply_correction(s, theta0, centers)
        r = run_scf(corr, start_from=warm)
        warm = r
        th = theta0.clone().requires_grad_(True)
        e = energy_of_theta(r, corr, th, centers, theta_ref=theta0)
        (g,) = torch.autograd.grad(e, th)
        rows.append(g.detach())
    J = torch.stack(rows)  # (nvol, 4) dE_i/dtheta_k [eV per unit theta]
    A = J - J.mean(dim=0, keepdim=True)  # mean-subtracted shape Jacobian

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    s = S.tolist()
    cond = s[0] / max(s[-1], 1e-300)

    # decompose the recovery error into row space vs null space of A
    derr = THETA_FIT - THETA_STAR
    # effective rank at a 1e-6 relative singular-value cut
    rank = int(sum(1 for x in s if x > 1e-6 * s[0]))
    V = Vh.T  # columns = right singular vectors
    row_p = V[:, :rank] @ (V[:, :rank].T @ derr)
    null_p = derr - row_p
    out = {
        "ecut_ry": ECUT, "k": K, "scales": SCALES,
        "jacobian_eV": J.tolist(),
        "singular_values": s,
        "condition_number": cond,
        "effective_rank_1e-6": rank,
        "theta_err": derr.tolist(),
        "theta_err_norm": float(derr.norm()),
        "rowspace_component_norm": float(row_p.norm()),
        "nullspace_component_norm": float(null_p.norm()),
        "right_singular_vectors": Vh.tolist(),
    }
    (Path(__file__).parent / "results_step3b.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("jacobian_eV", "right_singular_vectors")},
                     indent=2))
    print(f"\nJACOBIAN singular values {['%.3e' % x for x in s]}")
    print(f"condition number {cond:.3e}; "
          f"|err| {out['theta_err_norm']:.3e} = rowspace "
          f"{out['rowspace_component_norm']:.3e} + nullspace "
          f"{out['nullspace_component_norm']:.3e}")


if __name__ == "__main__":
    main()
