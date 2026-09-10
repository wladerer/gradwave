"""Native-vs-eager Davidson probe: does the ~48% eager-glue wall convert?

Loads a captured mid-SCF round-boundary state (capture_state.py), then runs the
SAME warm Davidson solve three ways on identical inputs:

  A. torch.profiler accounting of the eager solve — op self-time vs wall
     (grounds the "glue fraction" number on this exact system);
  B. the production ``davidson_batched`` (eager PyTorch, the real baseline);
  C. the native C kernel (FFTW + CBLAS + LAPACKE, zero framework dispatch).

Agreement gate: identical n_iter and H-apply tallies, eigenvalues to 1e-9 eV,
residual-norm max to 1e-8 — same inputs, same algorithm, different substrate.

Usage:  uv run python experiments/native_davidson_probe/run_probe.py [small|medium] [reps] [threads]
Needs:  ./build.sh first (compiles libdavnative.so via nix-shell).
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADWAVE_TOEPLITZ", "off")

import numpy as np
import torch

HERE = Path(__file__).parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "small"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def load_state():
    z = np.load(HERE / f"state_{TAG}.npz")
    return z


def build_hamiltonian(z):
    """Reconstruct a BatchedHamiltonian equivalent to the captured one."""
    from gradwave.core.batch import BatchedHamiltonian, BatchedK

    nk, nb, m = z["x0"].shape
    mask = torch.from_numpy(z["mask"])
    bk = BatchedK(
        npw=mask.sum(dim=1),
        mask=mask,
        flat_idx=torch.from_numpy(z["gather_idx"]),
        kpg=torch.zeros(nk, m, 3, dtype=torch.float64),
        t=torch.from_numpy(z["bk_t"]),
        proj_phase_free=torch.from_numpy(z["p"]),  # unused by apply (p passed to ctor)
        proj_atom_index=torch.zeros(z["p"].shape[1], dtype=torch.int64),
        dij_full=torch.from_numpy(z["dij"]),
    )
    shape = tuple(int(s) for s in z["shape"])
    h = BatchedHamiltonian(bk, shape, torch.from_numpy(z["v_eff"]),
                           torch.from_numpy(z["p"]))
    # sanity: reconstructed scatter/gather must equal the captured ones
    assert torch.equal(h.idx_scatter, torch.from_numpy(z["idx_scatter"]))
    assert torch.equal(h.gather_idx, torch.from_numpy(z["gather_idx"]))
    return h


def eager_solve(h, z, tally=False):
    from gradwave.core import batch as batchmod
    from gradwave.solvers.davidson import davidson_batched

    if tally:
        batchmod.reset_happly_tally()
    res = davidson_batched(
        h.apply, torch.from_numpy(z["x0"]).clone(),
        torch.from_numpy(z["t_solve"]), h.bk.mask, tol=float(z["tol"]))
    napply = batchmod.happly_tally() if tally else -1
    return res, napply


def native_lib():
    lib = ctypes.CDLL(str(HERE / "libdavnative.so"))
    lib.davidson_native.restype = ctypes.c_int
    lib.davidson_native.argtypes = (
        [ctypes.c_int64] * 9 + [ctypes.c_double, ctypes.c_int]
        + [ctypes.c_void_p] * 8
        + [ctypes.c_void_p] * 4)
    return lib


def native_solve(lib, z, max_dim):
    nk, nb, m = z["x0"].shape
    n1, n2, n3 = (int(s) for s in z["shape"])
    nproj = z["p"].shape[1]
    x0 = np.ascontiguousarray(z["x0"])
    t = np.ascontiguousarray(z["t_solve"])
    mask = np.ascontiguousarray(z["mask"].astype(np.uint8))
    isc = np.ascontiguousarray(z["idx_scatter"])
    iga = np.ascontiguousarray(z["gather_idx"])
    veff = np.ascontiguousarray(z["v_eff"].reshape(-1))
    p = np.ascontiguousarray(z["p"])
    dij = np.ascontiguousarray(z["dij"].astype(np.complex128))
    eig = np.zeros((nk, nb))
    x = np.zeros((nk, nb, m), dtype=np.complex128)
    rn = np.zeros((nk, nb))
    napply = np.zeros(1, dtype=np.int64)

    def ptr(a):
        return a.ctypes.data_as(ctypes.c_void_p)

    t0 = time.perf_counter()
    ret = lib.davidson_native(
        nk, nb, m, n1, n2, n3, nproj, max_dim, 40, float(z["tol"]), THREADS,
        ptr(x0), ptr(t), ptr(mask), ptr(isc), ptr(iga), ptr(veff), ptr(p),
        ptr(dij), ptr(eig), ptr(x), ptr(rn), ptr(napply))
    dt = time.perf_counter() - t0
    if ret < 0:
        raise RuntimeError(f"davidson_native error {ret} "
                           "(-1 degenerate row, -2 LAPACK, -3 restart needed)")
    return ret, eig, rn, int(napply[0]), dt


def main():
    torch.set_num_threads(THREADS)
    z = load_state()
    nk, nb, m = z["x0"].shape
    max_dim = min(4 * nb, int(z["mask"].sum(axis=1).min()))
    print(f"[{TAG}] nk={nk} nb={nb} npw_max={m} box={tuple(z['shape'])} "
          f"nproj={z['p'].shape[1]} tol={float(z['tol']):.2e} "
          f"threads={THREADS} reps={REPS}")

    h = build_hamiltonian(z)

    # ---- A: profiler accounting of one eager solve ----
    eager_solve(h, z)  # warm (allocator, verdicts, thread pools)
    from torch.profiler import ProfilerActivity, profile
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        res_a, _ = eager_solve(h, z)
    wall_prof = time.perf_counter() - t0
    op_time = sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e6
    print(f"A. profiler: wall={wall_prof * 1e3:.1f} ms, "
          f"sum(op self time)={op_time * 1e3:.1f} ms "
          f"(profiling overhead inflates both; ratio is the signal)")

    # ---- B: eager timing ----
    res_b, napply_b = eager_solve(h, z, tally=True)
    times_b = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        eager_solve(h, z)
        times_b.append(time.perf_counter() - t0)
    tb = min(times_b)
    print(f"B. eager davidson_batched: n_iter={res_b.n_iter} "
          f"napply={napply_b} best={tb * 1e3:.2f} ms "
          f"(median {np.median(times_b) * 1e3:.2f})")

    # ---- C: native timing ----
    lib = native_lib()
    n_iter_c, eig_c, rn_c, napply_c, _ = native_solve(lib, z, max_dim)  # warm
    times_c = []
    for _ in range(REPS):
        _, _, _, _, dt = native_solve(lib, z, max_dim)
        times_c.append(dt)
    tc = min(times_c)
    print(f"C. native C kernel:        n_iter={n_iter_c} "
          f"napply={napply_c} best={tc * 1e3:.2f} ms "
          f"(median {np.median(times_c) * 1e3:.2f})")

    # ---- agreement ----
    eig_b = res_b.eigenvalues.numpy()
    rn_b = res_b.residual_norms.numpy()
    d_eig = float(np.abs(eig_b - eig_c).max())
    d_rn = float(np.abs(rn_b - rn_c).max())
    ok = (res_b.n_iter == n_iter_c and napply_b == napply_c
          and d_eig < 1e-9 and d_rn < 1e-8)
    print(f"agreement: max|d eig|={d_eig:.2e}  max|d rn|={d_rn:.2e}  "
          f"n_iter {res_b.n_iter}=={n_iter_c}  napply {napply_b}=={napply_c}"
          f"  -> {'OK' if ok else 'MISMATCH'}")
    print(f"SPEEDUP native/eager: {tb / tc:.2f}x")

    out = {
        "tag": TAG, "nk": int(nk), "nb": int(nb), "m": int(m),
        "threads": THREADS, "reps": REPS,
        "eager_ms_best": tb * 1e3, "native_ms_best": tc * 1e3,
        "eager_ms_median": float(np.median(times_b)) * 1e3,
        "native_ms_median": float(np.median(times_c)) * 1e3,
        "speedup_best": tb / tc, "n_iter": int(res_b.n_iter),
        "napply_eager": int(napply_b), "napply_native": int(napply_c),
        "d_eig": d_eig, "d_rn": d_rn, "agreement_ok": bool(ok),
        "profiler_wall_ms": wall_prof * 1e3, "profiler_op_ms": op_time * 1e3,
    }
    (HERE / f"result_{TAG}_t{THREADS}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote result_{TAG}_t{THREADS}.json")


if __name__ == "__main__":
    main()
