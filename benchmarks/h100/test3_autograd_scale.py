"""H100 Test 3 — DIFFERENTIABLE GRADIENT THROUGH THE SCF, at scale (GPU vs CPU).

gradwave's defining capability is autograd *through* the SCF fixed point. That path
(`scf.implicit.density_loss_param_grads` → adjoint solve + f_xc double-backward) has
never been characterized at large cell size on a full-fp64 GPU. This measures, per
cell size and device:

  * forward   — the converged SCF (`scf`, a torch.no_grad region), and
  * backward  — one scalar density-loss gradient dL/dθ w.r.t. the LearnableX XC
                parameters (κ, μ), which REQUIRES the SCF response (does not cancel
                by stationarity, unlike Hellmann-Feynman forces),
  * peak GPU memory across forward+backward (the real question for autograd-at-scale).

Insulator (diamond Si, smearing=none) so the adjoint is clean. `use_symmetry=False`
is required by the adjoint (setup_system defaults to it). Based on the validated
gradcheck `tests/gradcheck/test_implicit_scf.py`.

Run on the H100 box:
    uv run python benchmarks/h100/test3_autograd_scale.py --sizes 1,2,3
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO = _ROOT / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf"


def build_si(nrep: int):
    """Diamond-Si primitive (2 atoms) tiled nrep**3 → 2*nrep**3 atoms."""
    from ase import Atoms

    a = 5.43
    cell = (a / 2) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    at = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=cell, pbc=True).repeat((nrep, nrep, nrep))
    return np.asarray(at.cell.array), at.get_positions(), len(at)


def _loss(rho):
    # a scalar density functional whose gradient needs the SCF response
    return (rho * rho).sum()


def forward_scf(system, xc, device: str):
    from gradwave.scf.loop import scf

    sysd = system.to(device) if device != "cpu" else system
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    res = scf(sysd, xc, smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    return res, time.perf_counter() - t


def backward_grad(res, xc, device: str):
    from gradwave.scf.implicit import density_loss_param_grads

    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    loss, grads = density_loss_param_grads(res, xc, _loss)
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    gk = float(grads["raw_kappa"])
    return dt, gk


def run_one(nrep, ecut_ry, kmesh, device):
    from gradwave.core.xc.learnable import LearnableX
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    cell, pos, nat = build_si(nrep)
    si = parse_upf(_PSEUDO)
    system = setup_system(cell, pos, [0] * nat, [si], ecut=ecut_ry * RY, kmesh=kmesh)
    xc = LearnableX(kappa=0.70, mu=0.20)

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    res, t_fwd = forward_scf(system, xc, device)
    if not res.converged:
        return nat, system, float("nan"), float("nan"), float("nan"), False
    t_bwd, gk = backward_grad(res, xc, device)
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if device != "cpu" else float("nan")
    npw = max(int(s.npw) for s in system.spheres)
    return nat, npw, t_fwd, t_bwd, peak, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="1,2,3", help="nrep list; atoms = 2*nrep**3")
    ap.add_argument("--ecut", type=float, default=20.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--gamma-above", type=int, default=16, help="Gamma above this atom count")
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "no-CUDA"
    print(f"# H100 Test 3 — autograd-through-SCF | CUDA={has_cuda} ({dev_name}) | "
          f"ecut={args.ecut} Ry | reps={args.reps}")
    print(f"{'atoms':>6} {'npw':>7} {'dev':>5} {'fwd[s]':>9} {'bwd[s]':>9} "
          f"{'tot[s]':>9} {'peakGiB':>8} {'gpu/cpu':>8}")

    for nrep in [int(x) for x in args.sizes.split(",")]:
        nat = 2 * nrep**3
        km = 1 if nat > args.gamma_above else args.kmesh
        kmesh = (km, km, km)
        row_gpu_tot = float("nan")
        for device in (["cuda", "cpu"] if has_cuda else ["cpu"]):
            fwds, bwds, peak, npw, ok = [], [], float("nan"), 0, True
            for _ in range(args.reps):
                nat, npw, tf, tb, peak, ok = run_one(nrep, args.ecut, kmesh, device)
                if not ok:
                    break
                fwds.append(tf)
                bwds.append(tb)
            if not ok:
                print(f"{nat:>6} {npw:>7} {device:>5}  DID-NOT-CONVERGE")
                continue
            tf, tb = statistics.median(fwds), statistics.median(bwds)
            tot = tf + tb
            if device == "cuda":
                row_gpu_tot = tot
                ratio = "-"
            else:
                ratio = f"{row_gpu_tot / tot:.3f}" if row_gpu_tot == row_gpu_tot else "-"
            pk = f"{peak:.3f}" if peak == peak else "-"
            print(f"{nat:>6} {npw:>7} {device:>5} {tf:>9.3f} {tb:>9.3f} "
                  f"{tot:>9.3f} {pk:>8} {ratio:>8}")

    print("\n# READ: bwd = adjoint solve + f_xc double-backward (the response-carrying gradient).")
    print("# peakGiB = peak HBM across fwd+bwd. gpu/cpu<1 => H100 wins the differentiable path.")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
