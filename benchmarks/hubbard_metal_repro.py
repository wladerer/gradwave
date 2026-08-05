"""Cheap probe of the large-U-on-metal DFT+U occupation flip-flop and its two
remedies (occupation-matrix damping `occ_mix` and the linear U-ramp).

Bulk fcc Pt is metallic with a partly filled 5d band at E_F. A large Dudarev U
on that manifold shifts the lagged-occupation levels by about U/2, far above the
smearing width, so the occupations at E_F flip between full and empty each SCF
iteration. The flight recorder's band-reorder count is the observable, and it
scales with U (519 at U=12, 674 at U=15, 1324 at U=18 on a 4x4x4 mesh).

This 1-atom cell shows the flip-flop SIGNATURE but does not itself diverge: the
Broyden/Pulay density mixer absorbs the oscillation because the occupation
changes are spread over the broad bulk d-band and average out in the density.
The full non-convergence of the H100 round-4 diagnosis (issue #206) needs the
Pt(111)+H slab, where a localized adsorbate state pinned at E_F flips coherently
and drives a large density swing the mixer cannot damp. No H pseudopotential
ships in the repo, so that exact system is not reproduced here. What this script
does show is that occ_mix and the U-ramp reach the SAME converged fixed point as
the raw one-step lag (energies agree to ~1e-11 eV), i.e. they are convergence
aids, not physics changes. The mechanism-level contraction proof is the unit
test tests/unit/test_hubbard_convergence.py::test_occ_mix_contracts_flip_flop.

Run: uv run python benchmarks/hubbard_metal_repro.py [U_eV] [ecut_Ry]
"""

from __future__ import annotations

import sys

import numpy as np

from gradwave.constants import RY_EV
from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

PT = "benchmarks/delta_gauge/pseudos/Pt.upf"


def build(ecut_ry: float, kmesh=(4, 4, 4)):
    a = 3.92  # fcc Pt lattice constant (Angstrom)
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pt = parse_upf(PT)
    # +U forces the full BZ (no IBZ folding of the occupation matrix)
    return setup_system(cell, np.zeros((1, 3)), [0], [pt],
                        ecut=ecut_ry * RY_EV, kmesh=kmesh, use_symmetry=False)


def run(system, u, *, occ_mix=1.0, u_ramp_iters=0, max_iter=80):
    r = scf(system, PBE(), smearing="gaussian", width=0.2,
            hubbard=[HubbardManifold(0, l=2, u=u)],
            hub_occ_mix=occ_mix, hub_u_ramp_iters=u_ramp_iters,
            max_iter=max_iter, etol=1e-7, rhotol=1e-6, verbose=False)
    reorder = sum(int(i["reorder"]) for i in r.recorder.iters) if r.recorder else 0
    return r, reorder


def plot_fig(runs: list[tuple[str, str, object]], u: float, png: str) -> None:
    """Two-panel flight-recorder view: density residual and per-iteration
    band-reorder count, raw one-step lag vs the two convergence aids."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axr, axo) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for label, color, res in runs:
        iters = res.recorder.iters
        it = [i["it"] for i in iters]
        axr.semilogy(it, [i["drho"] for i in iters], lw=1.8, color=color,
                     label=label)
        axo.plot(it, [i["reorder"] for i in iters], lw=1.8, color=color,
                 label=label)
    axr.set_xlabel("SCF iteration")
    axr.set_ylabel("‖ρ_out − ρ_in‖")
    axo.set_xlabel("SCF iteration")
    axo.set_ylabel("band reorders / iteration")
    axo.legend(frameon=False, fontsize=9)
    for ax in (axr, axo):
        ax.grid(True, which="both", lw=0.4, alpha=0.25)
    fig.suptitle(f"fcc Pt, U = {u:g} eV on 5d — the occupation flip-flop "
                 "in the SCF flight recorder", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    print(f"wrote {png}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fig_path = None
    for a in sys.argv[1:]:
        if a.startswith("--fig"):
            fig_path = a.split("=", 1)[1] if "=" in a else "hubbard_flipflop.png"
    u = float(args[0]) if len(args) > 0 else 9.0
    ecut = float(args[1]) if len(args) > 1 else 35.0
    system = build(ecut)
    print(f"fcc Pt, U={u} eV on 5d, ecut={ecut} Ry, gaussian smearing 0.2 eV\n")

    base, base_re = run(system, u)
    print(f"defaults (β=1, no ramp): converged={base.converged} "
          f"n_iter={base.n_iter} F={base.energies.free_energy:.6f} eV "
          f"band-reorders={base_re}")

    damp, damp_re = run(system, u, occ_mix=0.3)
    print(f"occ_mix=0.3            : converged={damp.converged} "
          f"n_iter={damp.n_iter} F={damp.energies.free_energy:.6f} eV "
          f"band-reorders={damp_re}")

    ramp, ramp_re = run(system, u, u_ramp_iters=15)
    print(f"u_ramp_iters=15        : converged={ramp.converged} "
          f"n_iter={ramp.n_iter} F={ramp.energies.free_energy:.6f} eV "
          f"band-reorders={ramp_re}")

    if damp.converged and ramp.converged:
        d = abs(float(damp.energies.free_energy) - float(ramp.energies.free_energy))
        print(f"\ndamped-vs-ramped |ΔF| = {d:.2e} eV (same fixed point)")

    if fig_path:
        plot_fig([("raw one-step lag (β=1)", "#c1442e", base),
                  ("occ_mix = 0.3", "#2a78d6", damp),
                  ("U-ramp, 15 iters", "#3a8f6f", ramp)], u, fig_path)


if __name__ == "__main__":
    main()
