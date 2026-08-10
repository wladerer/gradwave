"""H100 Test 6 — noncollinear / spin-orbit (SOC) SCF, GPU vs CPU.

The fully-relativistic spinor path DOUBLES the plane-wave basis (coeffs are
(nk, nb, 2*npw)) and runs j-resolved SOC projectors on the doubled axis — twice the
dense linear algebra per k-point, which is exactly the regime a full-fp64 GPU should
win most. `scf_noncollinear` has never been benched GPU-vs-CPU. This times a magnetic
SOC SCF (fcc Ni, FR pseudo) on each device across a k-mesh sweep.

Based on the converging setup in tests/integration/test_ni_soc_convergence.py. FR
pseudos live under tests/fixtures/qe/pseudos/. Requires time_reversal=False +
use_symmetry=False for a magnetic spinor run; a collinear scf() rejects FR pseudos.

Run on the H100 box (main branch is fine — noncollinear is already merged):
    uv run python benchmarks/h100/test6_soc_scale.py --kmeshes 4,6
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_NI_FR = _ROOT / "tests/fixtures/qe/pseudos/Ni_ONCV_PBE_FR-1.0.upf"


def build_ni(km: int, ecut_ry: float, nbands: int):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    a = 3.52
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    ni = parse_upf(_NI_FR)  # fully-relativistic -> SOC active
    return setup_system(cell, pos, [0], [ni], ecut=ecut_ry * RY, kmesh=(km, km, km),
                        nbands=nbands, use_symmetry=False, time_reversal=False)


def run_soc(system, device: str):
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear

    sysd = system.to(device) if device != "cpu" else system
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    res = scf_noncollinear(
        sysd, NoncollinearXC(SpinPBE()),
        mag_vec_init=[[0, 0, 1.0]],           # FM seed along z
        smearing="gaussian", width=0.1,
        max_iter=60, etol=1e-4, rhotol=5e-3,  # metal-appropriate gate
        spin_precond=True, mag_mixer="pulay",
        mag_diago_schedule="quadratic", mag_mixing_alpha=0.3,
        verbose=False,
    )
    if device != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter() - t, res


def _magz(res):
    return float(res.mag_vec[2]) if hasattr(res, "mag_vec") else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmeshes", default="4,6", help="comma list of k per axis")
    ap.add_argument("--ecut", type=float, default=40.0, help="Ry")
    ap.add_argument("--nbands", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "no-CUDA"
    print(f"# H100 Test 6 — noncollinear/SOC (fcc Ni FR) | CUDA={has_cuda} ({dev_name}) | "
          f"ecut={args.ecut} Ry | reps={args.reps}")
    print(f"{'kmesh':>6} {'nk':>4} {'npw':>7} {'gpu[s]':>9} {'cpu[s]':>9} "
          f"{'gpu/cpu':>8} {'iters g/c':>10} {'mag_z':>7}")

    for km in [int(x) for x in args.kmeshes.split(",")]:
        system = build_ni(km, args.ecut, args.nbands)
        nk = len(system.spheres)
        npw = max(int(s.npw) for s in system.spheres)

        if has_cuda:
            gres = [run_soc(system, "cuda") for _ in range(args.reps)]
            gpu = statistics.median([t for t, _ in gres])
            g_it = gres[-1][1].n_iter
            magz = _magz(gres[-1][1])
        else:
            gpu, g_it, magz = float("nan"), -1, float("nan")

        torch.set_num_threads(args.cpu_threads)
        cres = [run_soc(system, "cpu") for _ in range(args.reps)]
        cpu = statistics.median([t for t, _ in cres])
        c_it = cres[-1][1].n_iter
        if magz != magz:
            magz = _magz(cres[-1][1])

        ratio = f"{gpu / cpu:.3f}" if has_cuda else "-"
        gpu_s = f"{gpu:.3f}" if has_cuda else "-"
        print(f"{km:>6} {nk:>4} {npw:>7} {gpu_s:>9} {cpu:>9.3f} {ratio:>8} "
              f"{str(g_it) + '/' + str(c_it):>10} {magz:>7.3f}")

    print("\n# READ: npw = SPINOR basis (2*plane-waves). gpu/cpu<1 => H100 wins SOC.")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
