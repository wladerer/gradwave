"""Probe B — s-step cegterg traffic existence test (GO-3 stage 1).

The native incremental Rayleigh-Ritz kernel
(src/gradwave/solvers/native/davidson_native.c, lines ~658-663) builds the new
columns of the reduced matrices hc, sc each round with two ZGEMMs:

    hc[:, dim:newdim] = V(m x newdim)^H  @  HV_new(m x notcnv)
    sc[:, dim:newdim] = V(m x newdim)^H  @   V_new(m x notcnv)

The big operand V (m=npw rows x newdim cols) is streamed from DRAM every round;
the Si-64 wall is ~43% bandwidth-bound dense subspace algebra. An s-step
(Chronopoulos-Gear) reassociation would generate s Krylov directions before one
RR, so the panel width goes notcnv -> s*notcnv while the streamed V is read once
instead of s times: memory-traffic-per-useful-flop should drop ~s x.

This is a MICROBENCH EXISTENCE TEST ONLY (does NOT patch the kernel). It issues
the same ZGEMM shape with a thin panel (p = b columns) vs a fat panel
(p = s*b columns) and reports flops and (under `perf stat`) DRAM/LLC bytes, so
the caller can read bytes-per-flop off the counters.

  dims                     : build Si-64@30Ry setup, print npw_max and nb
  gemm <npw> <newdim> <p> <reps>
                           : pure-BLAS zgemm loop (no gradwave import), the
                             region to wrap in `perf stat`

Design: `gemm` imports only numpy/scipy so the perf counters are pure BLAS
traffic; setup cost lives only in `dims`. Compare two perf-stat runs (thin vs
fat) at the same npw/newdim; the fixed per-process offset cancels in the trend.
"""

from __future__ import annotations

import sys


def run_dims():
    from pathlib import Path

    import numpy as np

    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    RY = 13.605693122994
    root = Path(__file__).resolve().parents[2]
    # Si-64: 2x2x2 supercell of the 8-atom conventional cubic diamond cell
    a = 5.43
    conv = np.array([
        [0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0],
        [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1],
    ], dtype=float) * (a / 4)
    reps = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                reps.append(conv + np.array([i, j, k]) * a)
    pos = np.concatenate(reps, axis=0)  # 64 atoms
    cell = np.eye(3) * (2 * a)
    si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0] * 64, [si], ecut=30 * RY,
                          kmesh=(1, 1, 1), use_symmetry=False)
    npw_max = int(system.batch.npw_max)
    # nb: occupied + a standard ~20% band buffer (256 electrons -> 128 occ)
    n_elec = 64 * 4
    nocc = n_elec // 2
    nb = int(nocc * 1.2 + 4)
    print(f"SI64 npw_max={npw_max} nocc={nocc} nb~={nb} max_dim(4nb)={4 * nb}")


def run_gemm(npw: int, newdim: int, p: int, reps: int):
    import numpy as np
    from scipy.linalg.blas import zgemm

    rng = np.random.default_rng(0)
    # column-major (Fortran) to match the C kernel's CblasColMajor path
    V = np.asfortranarray(rng.standard_normal((npw, newdim))
                          + 1j * rng.standard_normal((npw, newdim)))
    HVnew = np.asfortranarray(rng.standard_normal((npw, p))
                              + 1j * rng.standard_normal((npw, p)))
    Vnew = np.asfortranarray(rng.standard_normal((npw, p))
                             + 1j * rng.standard_normal((npw, p)))
    acc = 0.0
    for _ in range(reps):
        # hc block: V^H @ HVnew  (ConjTrans on V) -> (newdim x p)
        hc = zgemm(1.0, V, HVnew, trans_a=2)  # trans_a=2 => conjugate transpose
        # sc block: V^H @ Vnew
        sc = zgemm(1.0, V, Vnew, trans_a=2)
        acc += hc[0, 0].real + sc[0, 0].real
    # complex GEMM flops: 8*M*N*K each; two of them per rep
    flops = 2 * reps * 8.0 * newdim * p * npw
    bytes_stream = reps * (
        # per rep, two gemms; V read twice (once each), HVnew/Vnew once, outputs
        2 * (npw * newdim) + (npw * p) + (npw * p) + 2 * (newdim * p)
    ) * 16.0
    print(f"GEMMBENCH npw={npw} newdim={newdim} p={p} reps={reps} "
          f"flops={flops:.4e} model_bytes={bytes_stream:.4e} "
          f"model_bytes_per_flop={bytes_stream / flops:.4f} acc={acc:.3e}")


def run_sweep(npw: int, newdim: int, reps: int):
    """GFLOP/s vs panel width (wall-time, no perf needed): shows whether the two
    incremental-RR ZGEMMs are bandwidth- or compute-bound at Si-64 dims. A flat
    GFLOP/s across small->large panel means already compute-bound (s-step's
    traffic reduction cannot convert to speed -> KILL); a low-at-small,
    rising-at-large curve means a bandwidth-bound small-panel regime s-step could
    help. RUN ON AN IDLE BOX. Also prints analytic AI = flops/bytes."""
    import time

    import numpy as np
    from scipy.linalg.blas import zgemm

    rng = np.random.default_rng(0)
    V = np.asfortranarray(rng.standard_normal((npw, newdim))
                          + 1j * rng.standard_normal((npw, newdim)))
    print(f"SWEEP npw={npw} newdim={newdim} reps={reps} (V={npw * newdim * 16 / 1e6:.0f}MB)")
    for p in (4, 8, 16, 32, 64, 128, 256):
        if p > newdim:
            break
        HVnew = np.asfortranarray(rng.standard_normal((npw, p))
                                  + 1j * rng.standard_normal((npw, p)))
        Vnew = np.asfortranarray(rng.standard_normal((npw, p))
                                 + 1j * rng.standard_normal((npw, p)))
        zgemm(1.0, V, HVnew, trans_a=2)  # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            zgemm(1.0, V, HVnew, trans_a=2)
            zgemm(1.0, V, Vnew, trans_a=2)
        dt = time.perf_counter() - t0
        flops = 2 * reps * 8.0 * newdim * p * npw
        # per-gemm bytes (V + panel read, C write); AI = flops/bytes
        bytes_1 = 16.0 * (npw * newdim + npw * p + newdim * p)
        ai = (8.0 * newdim * p * npw) / bytes_1
        print(f"  p={p:4d} wall={dt:7.3f}s GFLOP/s={flops / dt / 1e9:7.1f} "
              f"AI={ai:6.2f}flop/byte  model_B/flop={2 * bytes_1 / (2 * 8.0 * newdim * p * npw):.4f}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "dims":
        run_dims()
        return
    if sys.argv[1] == "gemm":
        _, _, npw, newdim, p, reps = sys.argv
        run_gemm(int(npw), int(newdim), int(p), int(reps))
        return
    if sys.argv[1] == "sweep":
        _, _, npw, newdim, reps = sys.argv
        run_sweep(int(npw), int(newdim), int(reps))
        return
    raise SystemExit(f"unknown mode {sys.argv[1]!r}")


if __name__ == "__main__":
    main()
