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


METALS = {
    # label: (upf, lattice a [Å], ecut [Ry], nbands, kmesh) — fcc metals
    "al": ("Al_ONCV_PBE-1.2.upf", 4.05, 30, 12, (6, 6, 6)),
    "cu": ("Cu_ONCV_PBE-1.2.upf", 3.615, 45, 20, (6, 6, 6)),  # 3s3p semicore d-band
}


def _fcc_system(upf_name, a, ecut_ry, nbands, kmesh):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system
    pp = parse_upf(FIX / upf_name)
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return setup_system(cell, np.array([[0.0, 0, 0]]), [0], [pp],
                        ecut=ecut_ry * RY, kmesh=kmesh, nbands=nbands,
                        use_symmetry=True)


def run_metal(label, q0=1.1):
    upf, a, ecut_ry, nbands, kmesh = METALS[label]
    kx = "x".join(map(str, kmesh))
    print(f"\n=== fcc {label} ({upf}, gaussian 0.1 eV, {ecut_ry} Ry, {kx}) ===")
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf
    xc = PBE()
    common = dict(smearing="gaussian", width=0.1, nspin=1, verbose=False,
                  rhotol=1e-6, etol=1e-8)

    def sys_():
        return _fcc_system(upf, a, ecut_ry, nbands, kmesh)

    grid = sys_().grid
    g2_dens = grid.g2.reshape(-1)[grid.dens_mask.reshape(-1)].to(RDTYPE)

    # reference: bare Kerker
    t = time.perf_counter()
    ref = scf(sys_(), xc, mixing_alpha=0.7, **common)
    f_ref = float(ref.energies.free_energy)
    dt = time.perf_counter() - t
    print(f"  kerker            {ref.n_iter:3d} iters   F={f_ref:+.6f} eV   {dt:.1f}s")

    # probe: Kerker-on plain damping (history=1, no DIIS) — stable on d-band
    # metals where plain damping alone sloshes. Divide the Kerker factor back out
    # of the residual ratios to recover the bare response d(G).
    res_hist: list[torch.Tensor] = []

    def hook(it, rho_in, rho_out):
        res_hist.append((rho_out - rho_in).detach().clone())

    alpha_probe = 0.7
    scf(sys_(), xc, mixing_alpha=alpha_probe, mixing_history=1, kerker=True,
        max_iter=14, mixer_hook=hook, **common)
    kfac = g2_dens / (g2_dens + q0**2)
    g2_shell, d_shell, count = response_from_residuals(
        res_hist, g2_dens, alpha_probe, n_bins=48, skip=2, precond_fac=kfac)
    print(f"  probe {len(res_hist)} residuals → d(G) over {len(d_shell)} shells, "
          f"d in [{float(d_shell.min()):.2f}, {float(d_shell.max()):.2f}]")

    # fit through the ACTUAL DIIS mixer the deploy run uses (history 8), so the
    # filter complements DIIS rather than duplicating its low-G work.
    P, info = fit_multipole(g2_shell, d_shell, n_poles=3, alpha=0.7,
                            mixer="diis", history=8, q0=q0, n_unroll=30,
                            steps=700, weight=count)
    P = P.rebind(g2_dens).detach_()
    print(f"  fitted {P.summary()}  (plain rho {info['rho_init']:.3f}→{info['rho_final']:.3f})")

    t = time.perf_counter()
    learned = scf(sys_(), xc, mixing_alpha=0.7, precond_op=P, **common)
    f_l = float(learned.energies.free_energy)
    dt = time.perf_counter() - t
    verdict = ("win" if learned.n_iter < ref.n_iter else
               "tie" if learned.n_iter == ref.n_iter else "loss")
    print(f"  learned filter    {learned.n_iter:3d} iters   F={f_l:+.6f} eV   "
          f"{dt:.1f}s   [{verdict}]")
    print(f"  dF vs kerker = {f_l-f_ref:+.2e} eV  (same fixed point; only the path differs)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("synthetic", "both"):
        run_synthetic()
    if which in ("al", "both"):
        run_metal("al")
    if which in ("cu", "both"):
        run_metal("cu")
