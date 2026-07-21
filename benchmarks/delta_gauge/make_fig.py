"""Publication figure for the periodic-table Δ-gauge.

Per-element Δ vs the WIEN2k all-electron reference: gradwave against PseudoDojo's
own published Δ for the identical standard pseudopotential. Tracking of the two
bars is the claim — gradwave reproduces, element by element, the equation of
state the reference pseudopotential is designed to give.

Reads results/delta_summary.json. Agg backend (headless); 16 pt.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SP = Path(__file__).parent
Z = {"Li": 3, "Na": 11, "K": 19, "Ca": 20, "Sr": 38, "Al": 13, "Si": 14,
     "Ge": 32, "Sn": 50, "V": 23, "Nb": 41, "Ta": 73, "Mo": 42, "W": 74,
     "Cu": 29, "Ag": 47, "Au": 79, "Pd": 46, "Pt": 78, "Ir": 77, "Rh": 45,
     "Pb": 82, "Fe": 26, "Ni": 28, "Cr": 24}
BLUE, GRAY = "#2a78d6", "#b5b3ad"
plt.rcParams.update({"font.size": 16, "axes.linewidth": 0.8})


def main():
    s = json.loads((SP / "results" / "delta_summary.json").read_text())
    els = sorted(s, key=lambda e: Z[e])
    gw = np.array([s[e]["delta_wien2k"] for e in els])
    pd = np.array([s[e]["dfact_pdojo"] for e in els])
    x = np.arange(len(els))

    ycap = 3.0  # Cu (defective PseudoDojo UPF, gw=QE to 0.08 meV) clips off-scale
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(els)), 5.2))
    ax.bar(x - 0.21, np.minimum(gw, ycap), 0.42, color=BLUE, label="gradwave")
    ax.bar(x + 0.21, np.minimum(pd, ycap), 0.42, color=GRAY,
           label="PseudoDojo (ABINIT)")
    for xi, g, p in zip(x, gw, pd, strict=True):  # numeric labels on bars clipped by the cap
        if g > ycap:
            ax.text(xi - 0.21, ycap + 0.02, f"{g:.1f}", ha="center", va="bottom",
                    fontsize=12, color=BLUE)
        if p > ycap:
            ax.text(xi + 0.21, ycap + 0.02, f"{p:.1f}", ha="center", va="bottom",
                    fontsize=12, color="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(els)
    ax.set_ylabel("Δ vs WIEN2k  (meV/atom)")
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.text(len(els) - 0.5, 1.03, "reproducible (1 meV/atom)", ha="right",
            va="bottom", fontsize=12, color="0.4")
    ax.set_xlim(-0.7, len(els) - 0.3)
    ax.set_ylim(0, ycap)
    ax.legend(frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = SP / "results" / "delta_gauge.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({len(els)} elements, gw mean Δ {gw.mean():.3f} meV/atom)")


if __name__ == "__main__":
    main()
