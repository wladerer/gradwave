"""Quantum geometric tensor of the Si valence bands via autograd ∂/∂k.

Milestone-1 demo: converge a small Si SCF, freeze the potential into the
dense differentiable Bloch Hamiltonian H(k) (postscf.kgeometry.BlochHK), and
print the Fubini–Study metric g_μν and Berry curvature Ω_μν of the 4-band
valence group at a few k-points, plus the gauge-invariant Wannier-spread
functional Ω_I ≈ (1/N_k) Σ_k Tr g(k) on a coarse shifted mesh.

Diamond Si has inversion + time-reversal, so Ω(k) should vanish pointwise —
the FFT box is pinned to 20³ (divisible by 4) so the discretized potential
respects the bond-center inversion exactly; see tests/unit/test_kgeometry.py.

Run:  uv run python experiments/qgt/demo_si.py   (~30 s, CPU)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.postscf.kgeometry import BlochHK, metric_curvature, qgt, qgt_sos  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from tests.helpers import RY, si_fcc, si_upf  # noqa: E402

VALENCE = [0, 1, 2, 3]


def main() -> None:
    torch.set_num_threads(2)
    t0 = time.time()
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(2, 2, 2), nbands=8, use_symmetry=True,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    print(f"SCF converged in {res.n_iter} iterations ({time.time() - t0:.1f} s)\n")

    np.set_printoptions(precision=5, suppress=True)
    print("QGT of the Si valence group (4 bands) at a few generic k "
          "[fractional]; g and Ω in Å²:")
    for kf in (np.array([0.13, 0.07, -0.21]),
               np.array([0.31, -0.17, 0.09]),
               np.array([-0.08, 0.22, 0.41])):
        hk = BlochHK.from_scf(res, kf)
        q = qgt(hk.h, hk.k_cart(kf), VALENCE)
        g, omega = metric_curvature(q)
        print(f"\nk = {kf}")
        print("  g_μν =")
        for row in g.numpy():
            print("   ", row)
        print(f"  tr g       = {g.trace().item():.5f}  Å²")
        print(f"  max|Ω_μν|  = {omega.abs().max().item():.2e}  Å²  "
              "(≈ 0: PT symmetry)")

    # Wannier-spread estimate: Ω_I = (1/N_k) Σ_k Tr g(k) on a coarse mesh.
    # A generic (incommensurate) shift keeps every mesh point off the
    # symmetry lines; qgt_sos tolerates band-group-internal degeneracies
    # anyway, so it is the right evaluator for mesh sweeps.
    n = 2
    shift = np.array([0.11, 0.23, 0.37])
    t0 = time.time()
    trace_g, omega_sum = [], torch.zeros(3, 3, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                kf = (np.array([i, j, k]) + shift) / n - 0.5
                hk = BlochHK.from_scf(res, kf)
                q = qgt_sos(hk.h, hk.k_cart(kf), VALENCE)
                g, omega = metric_curvature(q)
                trace_g.append(g.trace().item())
                omega_sum += omega
    nk = n**3
    print(f"\nCoarse {n}×{n}×{n} shifted-mesh BZ averages "
          f"({time.time() - t0:.1f} s):")
    print(f"  Ω_I ≈ (1/N_k) Σ_k tr g(k) = {np.mean(trace_g):.4f}  Å²  "
          f"(gauge-invariant Wannier spread, {len(VALENCE)} bands)")
    print(f"  |(1/N_k) Σ_k Ω(k)|_max    = {(omega_sum / nk).abs().max().item():.2e}  Å²  "
          "(→ 0: trivial topology)")


if __name__ == "__main__":
    main()
