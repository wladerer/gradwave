"""The energy-metric SCF stopping rule on ferromagnetic bcc Fe.

The default SCF gate watches the density residual r = ρ_out − ρ_in. But the
quantity a user cares about — the free energy — converges quadratically in r:
its exact second-order error is ½⟨r|K_Hxc|r⟩, evaluable per iteration from the
same response kernel the mixer already uses. The opt-in energy-metric gate
(`scf.convergence: energy`, `scf.entol`) stops on that number instead.

This script runs one spin-polarized SCF on bcc Fe with the metric recorded
every iteration (threshold set to never trigger), then draws both convergence
measures with their default thresholds: the density gate keeps polishing for
several more iterations after the energy error is already orders of magnitude
below anything physical.

Fixtures: tests/fixtures/qe/pseudos/Fe_ONCV_PBE-1.2.upf (SG15 ONCV, PBE).
Runtime: ~2 min on 8 CPU threads.

Run from the repo root:
    uv run python examples/fe_energy_gate.py --outdir examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

torch.set_num_threads(8)
RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

RHOTOL = 1e-7   # the default density gate drawn in the figure
ENTOL = 1e-6    # eV, the default energy gate drawn in the figure


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE-1.2.upf")
    system = setup_system(cell, np.zeros((1, 3)), [0], [fe], ecut=60 * RY,
                          kmesh=(6, 6, 6), nbands=12)
    # entol far below reachable: the gate never fires, so the full trace of
    # both measures is recorded down to machine precision
    res = scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.4], etol=1e-12, rhotol=1e-9, max_iter=60,
              energy_metric=True, entol=1e-14, verbose=False)
    print(f"converged={res.converged} n_iter={res.n_iter} "
          f"m={float(res.mag_total):.4f} muB")

    iters = res.recorder.iters
    it = [i["it"] for i in iters]
    drho = [i["drho"] for i in iters]
    emet = [abs(i["e_metric"]) if i["e_metric"] is not None else np.nan
            for i in iters]
    stop_rho = next(i for i, d in zip(it, drho, strict=True) if d < RHOTOL)
    stop_e = next(i for i, e in zip(it, emet, strict=True) if e < ENTOL)
    print(f"density gate (rhotol={RHOTOL:g}) stops at iteration {stop_rho}, "
          f"energy gate (entol={ENTOL:g} eV) at iteration {stop_e}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axr, axe) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
    axr.semilogy(it, drho, lw=1.8, color="#2a78d6")
    axr.axhline(RHOTOL, color="#52514e", lw=1.0, ls="--", alpha=0.8)
    axr.axvline(stop_rho, color="#2a78d6", lw=1.0, ls=":", alpha=0.8)
    axr.annotate(f"rhotol → stop at {stop_rho}", (stop_rho, RHOTOL),
                 textcoords="offset points", xytext=(-8, -16), ha="right",
                 fontsize=9, color="#2a2a28")
    axr.set_xlabel("SCF iteration")
    axr.set_ylabel("‖ρ_out − ρ_in‖")
    axe.semilogy(it, emet, lw=1.8, color="#c1442e")
    axe.axhline(ENTOL, color="#52514e", lw=1.0, ls="--", alpha=0.8)
    axe.axvline(stop_e, color="#c1442e", lw=1.0, ls=":", alpha=0.8)
    axe.annotate(f"entol → stop at {stop_e}", (stop_e, ENTOL),
                 textcoords="offset points", xytext=(8, 10), fontsize=9,
                 color="#2a2a28")
    axe.set_xlabel("SCF iteration")
    axe.set_ylabel("|½⟨r|K_Hxc|r⟩|  [eV]")
    from matplotlib.ticker import MaxNLocator

    for ax in (axr, axe):
        ax.grid(True, which="both", lw=0.4, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("bcc Fe (nspin=2) — density gate vs energy-metric gate",
                 fontsize=11)
    fig.tight_layout()
    png = outdir / "fe_energy_gate.png"
    fig.savefig(png, dpi=180)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
