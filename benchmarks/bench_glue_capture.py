"""CUDA-graph "glue capture" benchmark: does capturing the real-valued glue
kernels of one SCF step reclaim the per-kernel launch overhead on a consumer GPU?

Background. The GPU profile of a small-cell SCF (docs/manual/performance.md,
docs/ideas.md) is dense-linear-algebra-bound with a 32-46% host launch/sync gap.
An apply-only CUDA graph and a whole-Davidson-round capture both replayed at 1.0x
(the FFT kernels are back-to-back, no launch gap; `torch.linalg.eigh` is not
capturable). The open question the docs flag as UNMEASURED on the RTX 3050 is the
"PyGraph-style glue capture": graph only the many-tiny-kernel real-valued work
BETWEEN the FFTs and OUTSIDE the solver -- density build, Hartree, XC assembly --
and leave the uncapturable Fermi solve and charge mixer eager. This measures that.

Faithful, not synthetic: the glue tensors (rho, coeffs, occ, bk, grid) are the
real ones the NC SCF loop produces on iteration 2, snapshotted by hooking the loop,
not hand-built. We then time eager vs `CUDAGraph.replay()` for each glue kernel the
docs name, and for the composite (the realistic captured chunk).

Scope. XC is benchmarked as `energy_density` (the elementwise-transcendental
forward -- the kernel `compile_xc` got ~19x on, and the launch-overhead-bound part
of the glue). The full vxc *potential* adds an autograd backward of the same
elementwise kernels; capturing that needs make_graphed_callables and is out of
scope for this first measurement (noted in the printout). Mixing and the Fermi
solve are inherently uncapturable (host branches / data-dependent bisection /
linalg.solve) and stay eager -- exactly the PyGraph cut.

Usage: uv run python benchmarks/bench_glue_capture.py [cuda|cpu] [reps]
       (cpu runs the eager timings only, as a launch-overhead-free reference)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

import gradwave.core.batch as batch
import gradwave.scf.loop as loop
from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WARMUP = 10

RY = 13.605693122994
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0, 0], [A / 4] * 3])
ROOT = Path(__file__).parents[1]


class _Stop(Exception):
    """Sentinel to abort the SCF once iteration-2 glue tensors are snapshotted."""


def snapshot_glue(system, xc):
    """Run the real NC SCF; grab the live glue tensors on iteration 2, then stop.

    Hooks `effective_potentials` (loop-module global) for the per-spin density and
    `core.batch.density_b` (bound by scf's local import at call time) for the real
    coeffs/occ/bk. Iteration 2 avoids the atomic-guess transient of iteration 1.
    """
    snap: dict = {"n_eff": 0}
    orig_eff = loop.effective_potentials
    orig_db = batch.density_b

    def eff_hook(sys_, xc_, rho_s, vloc_r, tau=None):
        out = orig_eff(sys_, xc_, rho_s, vloc_r, tau=tau)
        snap["n_eff"] += 1
        if snap["n_eff"] == 2:
            snap["rho_tot"] = (rho_s[0] if len(rho_s) == 1 else rho_s[0] + rho_s[1]).clone()
        return out

    def db_hook(coeffs, occ, kweights, bk, shape, volume):
        if "coeffs" not in snap and snap["n_eff"] >= 2:
            snap.update(
                coeffs=coeffs.clone(), occ=occ.clone(), kweights=kweights.clone(),
                bk=bk, shape=shape, volume=volume,
            )
            raise _Stop
        return orig_db(coeffs, occ, kweights, bk, shape, volume)

    loop.effective_potentials = eff_hook
    batch.density_b = db_hook
    try:
        scf(system, xc, smearing="none", etol=1e-9, rhotol=1e-8, verbose=False, max_iter=6)
    except _Stop:
        pass
    finally:
        loop.effective_potentials = orig_eff
        batch.density_b = orig_db
    return snap


def build_kernels(snap, system, xc):
    """Return {name: thunk} for the glue kernels, closing over the real tensors."""
    grid = system.grid
    rho = snap["rho_tot"]
    g2, g_cart, vol = grid.g2, grid.g_cart, grid.volume
    coeffs, occ, kw, bk, shape = (
        snap["coeffs"], snap["occ"], snap["kweights"], snap["bk"], snap["shape"],
    )
    needs_grad = xc.needs_gradient

    def xc_kernel():
        sigma = sigma_from_rho(rho, g_cart) if needs_grad else None
        return xc.energy_density(rho, sigma)

    def hartree_kernel():
        return hartree_potential_r(rho, g2)

    def density_kernel():
        return batch.density_b(coeffs, occ, kw, bk, shape, vol)

    def composite():
        # The realistic captured chunk: XC assembly + Hartree + density build,
        # the real-valued glue that brackets the FFTs in one SCF step.
        e = xc_kernel()
        v = hartree_kernel()
        r = density_kernel()
        return e.sum() + v.sum() + r.sum()

    return {
        "xc.energy_density": xc_kernel,
        "hartree_potential_r": hartree_kernel,
        "density_b": density_kernel,
        "COMPOSITE (glue)": composite,
    }


def time_eager(fn, reps, cuda):
    for _ in range(WARMUP):
        fn()
    if cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def time_graph(fn, reps):
    # Standard capture protocol: warm up on a side stream, then capture once.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main():
    cuda = device != "cpu"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    torch.set_num_threads(8)

    si = parse_upf(ROOT / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    print(f"device={device} reps={REPS}  (RTX 3050 target)")
    if cuda:
        print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    for xc_cls, label in ((LDA_PW92, "LDA (no sigma; pure elementwise)"),
                          (PBE, "PBE (sigma via FFT gradient)")):
        system = setup_system(CELL, POS, [0, 0], [si], ecut=30 * RY,
                              kmesh=(4, 4, 4), use_symmetry=False)
        if cuda:
            system = system.to(device)
        xc = xc_cls()
        snap = snapshot_glue(system, xc)
        box = "x".join(str(int(s)) for s in system.grid.shape)
        nk, nb, npw = snap["coeffs"].shape
        print(f"\n=== {label} ===")
        print(f"    box {box}  nk={nk} nb={nb} npw_max={npw}")
        kernels = build_kernels(snap, system, xc)

        hdr = f"    {'kernel':22s} {'eager µs':>10s} {'graph µs':>10s} {'speedup':>9s}"
        print(hdr)
        for name, fn in kernels.items():
            eager_us = time_eager(fn, REPS, cuda) * 1e3
            if cuda:
                try:
                    graph_us = time_graph(fn, REPS) * 1e3
                    sp = f"{eager_us / graph_us:6.2f}x"
                    gcol = f"{graph_us:10.1f}"
                except Exception as exc:  # capture failed -> report why, keep going
                    gcol, sp = "  CAPFAIL", f"({type(exc).__name__})"
            else:
                gcol, sp = f"{'-':>10s}", "  (cpu)"
            print(f"    {name:22s} {eager_us:10.1f} {gcol} {sp:>9s}")

    print("\nnote: XC is energy_density (fwd); vxc adds a backward of the same "
          "elementwise kernels.\n      Fermi solve + charge mixer are uncapturable "
          "(host branches) and excluded -- they stay eager in any real step.")


if __name__ == "__main__":
    main()
