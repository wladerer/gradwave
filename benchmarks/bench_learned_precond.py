"""Learned multi-pole Kerker preconditioner: iterations-to-convergence.

Two parts, run either or both (``uv run python benchmarks/bench_learned_precond.py
[synthetic|al|both]``, default both):

1. synthetic — a diagonal response with two length scales, where a single Kerker
   pole is provably the wrong shape. Fits the multi-pole filter against the model
   response by differentiating through the unrolled linear-mixing recurrence, and
   reports the spectral radius and the implied iteration count against the best
   single-pole Kerker. This isolates the mechanism from any DFT cost.

2. al — the full loop on a real solver: a short plain-mixing PROBE run on fcc Al
   captures the SCF residual history through the `mixer_hook`, `response_from_
   residuals` estimates the per-shell response d(G), `fit_multipole` fits the
   poles, and the fitted filter is deployed as `scf(..., precond_op=...)`. Reports
   n_iter for bare Kerker vs the learned filter, and the energy difference (which
   is zero to convergence — a preconditioner cannot move the fixed point).

Iteration count, energy-gated for the metal, is the trustworthy metric for a
solver-logic question (docs/manual/performance.md, and bench_precond.py). On a
homogeneous bulk metal bare Kerker is already near-optimal (bench_precond.py finds
local_tf neutral there too), so a radial filter is expected to tie on bulk Al; its
headroom is systems whose G-space response carries more than one scale.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradwave.dtypes import RDTYPE  # noqa: E402
from gradwave.scf.learned_precond import (  # noqa: E402
    fit_multipole,
    response_from_residuals,
    spectral_radius,
)

RY = 13.605693122994
FIX = ROOT / "tests/fixtures/qe/pseudos"
torch.set_num_threads(8)


def iters_to_tol(rho: float, tol: float = 1e-8) -> float:
    """Iterations for the residual to fall by `tol` at asymptotic rate rho."""
    import math
    return math.inf if rho >= 1.0 else math.log(tol) / math.log(rho)


def run_synthetic():
    print("\n=== synthetic two-scale response ===")
    alpha = 0.7
    # finite-cell |G|² range (~16 A box → smallest |G| ~0.39 A⁻¹); two response
    # length scales 1/q1, 1/q2 that no single Kerker pole can screen together.
    g2 = torch.linspace(0.15, 40.0, 400, dtype=RDTYPE)
    q1, q2 = 0.3, 2.5
    d = 0.5 * g2 / (g2 + q1**2) + 0.5 * g2 / (g2 + q2**2)

    q_grid = torch.linspace(0.2, 4.0, 120)
    rho1, q0_best = min(
        (float(spectral_radius(g2 / (g2 + q0**2), d, alpha)), float(q0))
        for q0 in q_grid.tolist()
    )
    P, info = fit_multipole(g2, d, n_poles=3, alpha=alpha, n_unroll=40, steps=600)
    rho3 = info["rho_final"]

    print(f"  best single-pole Kerker  q0={q0_best:.2f} A⁻¹   "
          f"rho={rho1:.4f}   ~{iters_to_tol(rho1):.0f} iters")
    print(f"  learned 3-pole           rho={rho3:.4f}   "
          f"~{iters_to_tol(rho3):.0f} iters")
    print(f"  {P.summary()}")
    print(f"  speedup (iteration ratio): {iters_to_tol(rho1)/iters_to_tol(rho3):.2f}x")


def _al_system(kmesh=(6, 6, 6)):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system
    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return setup_system(cell, np.array([[0.0, 0, 0]]), [0], [al],
                        ecut=30 * RY, kmesh=kmesh, nbands=12, use_symmetry=True)


def run_al():
    print("\n=== fcc Al (PBE, gaussian 0.1 eV, 30 Ry, 6x6x6) ===")
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf
    xc = PBE()
    common = dict(smearing="gaussian", width=0.1, nspin=1, verbose=False,
                  rhotol=1e-6, etol=1e-8)
    grid = _al_system().grid
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)
    alpha_probe = 0.5

    # reference: bare Kerker
    t = time.perf_counter()
    ref = scf(_al_system(), xc, mixing_alpha=0.7, **common)
    t_ref = time.perf_counter() - t
    f_ref = float(ref.energies.free_energy)
    print(f"  kerker            {ref.n_iter:3d} iters   F={f_ref:+.6f} eV   {t_ref:.1f}s")

    # probe: plain damped mixing (history=1, no DIIS) so the residual ratio is
    # the bare response 1 − alpha·d; capture the history via mixer_hook.
    res_hist: list[torch.Tensor] = []

    def hook(it, rho_in, rho_out):
        res_hist.append((rho_out - rho_in).detach().clone())

    scf(_al_system(), xc, mixing_alpha=alpha_probe, mixing_history=1,
        kerker=False, max_iter=16, mixer_hook=hook, **common)
    g2_shell, d_shell, count = response_from_residuals(
        res_hist, g2_dens, alpha_probe, n_bins=48, skip=2)
    print(f"  probe captured {len(res_hist)} residuals; "
          f"d(G) over {len(d_shell)} shells, d in [{float(d_shell.min()):.2f}, "
          f"{float(d_shell.max()):.2f}]")

    # fit the filter to the estimated response, deploy on the density sphere
    P, info = fit_multipole(g2_shell, d_shell, n_poles=3, alpha=0.7,
                            n_unroll=40, steps=600, weight=count)
    P = P.rebind(g2_dens).detach_()
    print(f"  fitted {P.summary()}  (rho {info['rho_init']:.3f}→{info['rho_final']:.3f})")

    t = time.perf_counter()
    learned = scf(_al_system(), xc, mixing_alpha=0.7, precond_op=P, **common)
    t_l = time.perf_counter() - t
    f_l = float(learned.energies.free_energy)
    de = f_l - f_ref
    print(f"  learned filter    {learned.n_iter:3d} iters   F={f_l:+.6f} eV   {t_l:.1f}s")
    print(f"  dF vs kerker = {de:+.2e} eV  (same fixed point; only the path differs)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("synthetic", "both"):
        run_synthetic()
    if which in ("al", "both"):
        run_al()
