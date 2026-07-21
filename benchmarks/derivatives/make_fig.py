"""Figure for the derivative-accuracy credential: every gradwave derivative on a
log axis by its relative agreement, coloured by whether the reference is a finite
difference of gradwave itself or the specialized QE response module. Reads
accuracy.json. Agg, 16 pt."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SP = Path(__file__).parent
BLUE, AQUA = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.size": 16, "axes.linewidth": 0.8})


def main():
    rows = json.loads((SP / "accuracy.json").read_text())
    rows = sorted(rows, key=lambda r: (r["ref_type"] == "QE", -r["rel"]))
    y = np.arange(len(rows))
    rel = np.array([r["rel"] for r in rows])
    col = [AQUA if r["ref_type"] == "QE" else BLUE for r in rows]
    labels = [f"{r['q']}  ({r['sys']})" for r in rows]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axvspan(1e-9, 1e-5, color=BLUE, alpha=0.06)
    ax.axvline(1e-5, color=BLUE, lw=1, ls="--")
    ax.axvline(1e-3, color=AQUA, lw=1, ls="--")
    ax.text(1e-5, len(rows) - 0.3, " FD floor (1st deriv)", color=BLUE, fontsize=12,
            ha="left", va="top")
    ax.text(1e-3, len(rows) - 0.3, " cross-code", color=AQUA, fontsize=12,
            ha="left", va="top")
    ax.scatter(rel, y, c=col, s=90, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(3e-9, 3e-2)
    ax.set_xlabel("relative agreement with reference")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE,
                          label="vs finite difference / gradcheck"),
               plt.Line2D([], [], marker="o", ls="", color=AQUA,
                          label="vs QE response module (ph.x / hp.x / pw.x)")]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = SP / "derivative_accuracy.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({len(rows)} derivatives)")


if __name__ == "__main__":
    main()
