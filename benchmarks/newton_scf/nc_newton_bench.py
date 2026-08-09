"""NC inexact-Newton SCF finisher — measurement harness (prototype).

Compares the shipped linear density mixer against the exact-Jacobian Newton
finisher (scf.newton_nc, wired into scf/loop.py via `newton_finish=`) on a
norm-conserving metal. Two things are measured, both from identical starting
runs:

  Stage 1 (the lever, same-state single step): the default and newton runs are
  bit-identical until the residual first crosses `switch` at iteration k (the
  newton run only diverges in the UPDATE at step k, with the linear mixer at
  full history in the default run). So r[k] is shared, and r[k+1] compares one
  shipped mixer step vs one Newton step FROM THE SAME STATE.

  Stage 2 (iteration count): total SCF iterations and same-machine wall time to
  the same convergence gate, plus ΔE (must be < 1e-8 eV — same fixed point).

Usage:
  uv run python benchmarks/newton_scf/nc_newton_bench.py \
      --elem Al --struct fcc --ecut 30 --kmesh 8 --width 0.2 --switch 1e-3
  uv run python benchmarks/newton_scf/nc_newton_bench.py \
      --elem Fe --struct bcc --ecut 45 --kmesh 8 --width 0.2 --switch 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

RY = 13.605693122994
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks" / "delta_gauge"))
from lattices import geometry  # noqa: E402

# WIEN2k reference per-atom volumes (Å³/atom) for the cubic elemental metals.
_V0 = {"Al": 16.4796, "Fe": 11.3436, "Cu": 11.9511, "Ni": 10.8876,
       "Pd": 15.3148, "Ag": 17.8534, "Au": 18.0022, "Li": 20.2191,
       "Na": 37.4686, "V": 13.4520, "Nb": 18.0686, "Mo": 15.8382}


def _build(elem, struct, ecut, kmesh):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    cell, pos, _ = geometry(struct, elem, _V0[elem])
    upf = parse_upf(str(_ROOT / "benchmarks" / "delta_gauge" / "pseudos" / f"{elem}.upf"))
    system = setup_system(cell, pos, [0], [upf], ecut=ecut * RY,
                          kmesh=(kmesh, kmesh, kmesh), use_symmetry=False)
    return system


def _run(system, xc, *, newton, switch, width, newton_kwargs, max_iter):
    from gradwave.scf.loop import scf

    t0 = time.time()
    res = scf(system, xc, smearing="cold", width=width, max_iter=max_iter,
              etol=1e-11, rhotol=1e-9, verbose=False,
              newton_finish=newton, newton_switch=switch,
              newton_kwargs=newton_kwargs)
    wall = time.time() - t0
    traj = [float(h["res"]) for h in res.history]
    return res, wall, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elem", default="Al")
    ap.add_argument("--struct", default="fcc", choices=["fcc", "bcc"])
    ap.add_argument("--ecut", type=float, default=30.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=8)
    ap.add_argument("--width", type=float, default=0.2, help="cold-smearing width [eV]")
    ap.add_argument("--switch", type=float, default=1e-3, help="mixer→Newton residual threshold")
    ap.add_argument("--inner-tol", type=float, default=1e-6)
    ap.add_argument("--max-inner", type=int, default=60)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from gradwave.core.xc.pbe import PBE

    system = _build(a.elem, a.struct, a.ecut, a.kmesh)
    nk, nb = len(system.spheres), system.nbands
    print(f"# {a.elem} {a.struct}  ecut={a.ecut:.0f} Ry  kmesh={a.kmesh}^3 "
          f"(nk_TR={nk})  nbands={nb}  grid={tuple(system.grid.shape)}  "
          f"width={a.width} eV  switch={a.switch:g}  threads={a.threads}",
          flush=True)
    nkw = dict(inner_tol=a.inner_tol, max_inner=a.max_inner, beta=a.beta)

    resA, wA, tA = _run(system, PBE(), newton=False, switch=a.switch,
                        width=a.width, newton_kwargs=nkw, max_iter=a.max_iter)
    resB, wB, tB = _run(system, PBE(), newton=True, switch=a.switch,
                        width=a.width, newton_kwargs=nkw, max_iter=a.max_iter)

    FA = float(resA.energies.free_energy)
    FB = float(resB.energies.free_energy)

    print("\n== Stage 2: iteration count (PRIMARY) ==")
    print(f"  default mixer : n_iter={resA.n_iter:2d}  converged={resA.converged}  "
          f"F={FA:.9f} eV  wall={wA:.1f}s")
    print(f"  newton finish : n_iter={resB.n_iter:2d}  converged={resB.converged}  "
          f"F={FB:.9f} eV  wall={wB:.1f}s")
    print(f"  ΔE (newton − default) = {FB - FA:+.2e} eV  "
          f"(gate: |ΔE| < 1e-8 → {'PASS' if abs(FB - FA) < 1e-8 else 'FAIL'})")
    print(f"  iteration reduction: {resA.n_iter} → {resB.n_iter}  "
          f"(mixer wall is back-to-back on the same machine; iteration count is "
          f"the machine-independent metric)")

    # Stage 1: same-state single-step comparison. Runs are identical until the
    # first iteration k with res < switch; that step's update differs.
    print("\n== Stage 1: same-state single-step residual drop (the lever) ==")
    print("  default res traj:", ["%.2e" % x for x in tA])
    print("  newton  res traj:", ["%.2e" % x for x in tB])
    k = next((i for i, x in enumerate(tB) if x < a.switch), None)
    if k is None or k + 1 >= len(tA) or k + 1 >= len(tB):
        print("  (no clean single-step bracket — residual crossed switch on the "
              "last iteration; loosen --switch or tighten rhotol)")
        return
    # sanity: r[k] should match between runs (identical state up to the switch)
    shared = abs(tA[k] - tB[k]) / max(tA[k], 1e-300)
    print(f"  switch at iter k={k+1}: r[k] = {tB[k]:.3e}  "
          f"(runs agree to {shared:.1e} up to here)")
    print(f"    one mixer step : r[k+1] = {tA[k+1]:.3e}   "
          f"(drop ×{tB[k] / max(tA[k+1], 1e-300):.2f})")
    print(f"    one Newton step: r[k+1] = {tB[k+1]:.3e}   "
          f"(drop ×{tB[k] / max(tB[k+1], 1e-300):.2f})")
    print(f"    Newton beats mixer by ×{tA[k+1] / max(tB[k+1], 1e-300):.1f} "
          f"on this single step")


if __name__ == "__main__":
    main()
