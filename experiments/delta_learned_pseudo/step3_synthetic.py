"""Step 3: synthetic recovery oracle -- the go/no-go signal.

Perturb Si's v_loc by a KNOWN theta* inside the parameterization span, generate a
synthetic reference E(V) over 5 volumes with the perturbed potential, then train
theta from zero to minimize the E(V) shape mismatch (each curve referenced to its
own mean energy, the offset-invariant analogue of the calcDelta shift-to-minimum
convention). Success = recovering theta* (parameter-space distance) and driving
the EOS residual to zero.

Gradient path: at each Adam step every volume is reconverged at the current theta
(warm-started), then E_i(theta) is the Hellmann-Feynman-differentiable energy at
that frozen density; the loss over volumes backprops to theta exactly.
"""
import os

import torch

torch.set_num_threads(2)
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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
NSTEP = int(os.environ.get("GW_NSTEP", "40"))
LR = float(os.environ.get("GW_LR", "0.05"))
SCALES = [0.94, 0.97, 1.00, 1.03, 1.06]
# known perturbation inside the span [eV.Ang^3 on each bump]
THETA_STAR = torch.tensor([0.30, -0.20, 0.10, -0.04], dtype=RDTYPE)


def eos_curve(systems, theta, centers, prev=None):
    """Converge all volumes at `theta`; return (energies list, results list)."""
    es, results = [], []
    warm = None
    for i, s in enumerate(systems):
        corr = apply_correction(s, theta, centers)
        start = (prev[i] if prev is not None else warm)
        r = run_scf(corr, start_from=start)
        es.append(float(r.energies.total))
        results.append(r)
        warm = r
    return es, results


def shape_loss_value(e_gw, e_ref):
    a = torch.as_tensor(e_gw, dtype=RDTYPE)
    b = torch.as_tensor(e_ref, dtype=RDTYPE)
    r = (a - a.mean()) - (b - b.mean())
    return float((r**2).mean().sqrt())


def main():
    t_start = time.time()
    centers = default_centers(4)
    grid = fixed_grid(SCALES, ECUT, K)
    systems = [build_si(s, ECUT, K, fft_shape=grid) for s in SCALES]
    scf_count = 0

    # 1) synthetic reference curve at THETA_STAR
    e_ref, ref_results = eos_curve(systems, THETA_STAR, centers)
    scf_count += len(SCALES)
    e_ref_t = torch.as_tensor(e_ref, dtype=RDTYPE)

    # 2) train theta from zero
    theta = torch.zeros(4, dtype=RDTYPE, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    history = []
    prev = None
    for step in range(NSTEP):
        theta_cur = theta.detach().clone()
        e_gw, results = eos_curve(systems, theta_cur, centers, prev=prev)
        scf_count += len(SCALES)
        prev = results

        # differentiable loss at frozen densities (HF energies)
        e_terms = [energy_of_theta(results[i], apply_correction(
            systems[i], theta, centers), theta, centers, theta_ref=theta_cur)
            for i in range(len(SCALES))]
        e_stack = torch.stack(e_terms)
        resid = (e_stack - e_stack.mean()) - (e_ref_t - e_ref_t.mean())
        loss = (resid**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        pdist = float((theta.detach() - THETA_STAR).norm())
        lval = shape_loss_value(e_gw, e_ref)
        history.append({"step": step, "loss_meV": lval * 1000,
                        "param_dist": pdist, "theta": theta.detach().tolist()})
        if step % 5 == 0 or step == NSTEP - 1:
            print(f"step {step:3d} shape_rms={lval*1e3:8.4f} meV  "
                  f"|theta-theta*|={pdist:.4e}  theta={theta.detach().tolist()}")

    out = {
        "ecut_ry": ECUT, "k": K, "grid": list(grid), "scales": SCALES,
        "theta_star": THETA_STAR.tolist(), "theta_final": theta.detach().tolist(),
        "param_dist_final": float((theta.detach() - THETA_STAR).norm()),
        "shape_rms_meV_final": history[-1]["loss_meV"],
        "shape_rms_meV_init": history[0]["loss_meV"],
        "n_step": NSTEP, "lr": LR, "scf_count": scf_count,
        "wall_s": round(time.time() - t_start, 1),
        "history": history,
    }
    (Path(__file__).parent / "results_step3.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nRECOVERY  |theta-theta*|: {out['param_dist_final']:.4e}  "
          f"shape_rms: {out['shape_rms_meV_init']:.3f} -> "
          f"{out['shape_rms_meV_final']:.4f} meV/atom")
    print(f"theta*    = {THETA_STAR.tolist()}")
    print(f"theta_fit = {theta.detach().tolist()}")
    print(f"SCF count {scf_count}, wall {out['wall_s']}s")


if __name__ == "__main__":
    main()
