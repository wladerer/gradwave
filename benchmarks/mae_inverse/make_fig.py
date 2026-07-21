"""FePt MAE vs tetragonal strain: the anisotropy landscape and the
MAE-maximizing c/a. Reads strain.json. Agg, 16 pt."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SP = Path(__file__).parent
BLUE, AQUA, GRAY = "#2a78d6", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 16, "axes.linewidth": 0.8})


def main():
    d = json.loads((SP / "strain.json").read_text())
    rows = sorted(d["rows"], key=lambda r: r["ratio"])
    r = np.array([x["ratio"] for x in rows])
    mae = np.array([x["mae"] for x in rows])
    ca0 = d["c0"] / d["a0"]

    # the landscape is non-monotonic (a dip below the reference), so a global
    # parabola is wrong: smooth-interpolate for the eye, and locate the maximum
    # from a local parabola through the three points bracketing the peak.
    from scipy.interpolate import PchipInterpolator
    rr = np.linspace(r.min(), r.max(), 400)
    curve = PchipInterpolator(r, mae)(rr)
    i = int(mae.argmax())
    lo, hi = max(0, i - 1), min(len(r), i + 2)
    p = np.polyfit(r[lo:hi], mae[lo:hi], 2)
    r_opt = -p[1] / (2 * p[0])
    m_opt = np.polyval(p, r_opt)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="0.6", lw=1)
    ax.plot(rr, curve, "-", color=GRAY, lw=1.5, zorder=1)
    ax.scatter(r, mae, s=90, color=BLUE, zorder=3, label="gradwave (force theorem)")
    ax.axvline(ca0, color="0.6", ls=":", lw=1.2)
    ax.text(ca0, ax.get_ylim()[0], " L1₀ (c/a=%.2f)" % ca0, rotation=90,
            va="bottom", ha="right", fontsize=12, color="0.4")
    if r.min() < r_opt < r.max():
        m_opt = np.polyval(p, r_opt)
        ax.scatter([r_opt], [m_opt], marker="*", s=340, color=AQUA, zorder=4,
                   label="MAE-max  c/a=%.2f" % r_opt)
    ax.set_xlabel("tetragonal ratio  c/a  (fixed volume)")
    ax.set_ylabel("MAE = E[100] − E[001]  (meV/cell)")
    ax.legend(frameon=False, loc="best", fontsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = SP / "mae_strain.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
