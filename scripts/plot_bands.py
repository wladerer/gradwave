"""Plot a band structure from a gradwave bands.json.

Usage: uv run python scripts/plot_bands.py out/bands.json [-o bands.png]
Requires matplotlib (not a package dependency — install it where needed).
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("-o", "--output", default="bands.png")
    ap.add_argument("--window", type=float, nargs=2, default=(-13, 8),
                    help="energy window around the reference [eV]")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed — `uv pip install matplotlib`") from None

    d = json.loads(open(args.json_file).read())
    x, eigs, ref = d["x"], d["eigenvalues_eV"], d["reference_eV"]

    fig, ax = plt.subplots(figsize=(5, 5))
    nb = len(eigs[0])
    for b in range(nb):
        ax.plot(x, [e[b] - ref for e in eigs], color="#2060a0", lw=1.2)
    for xt, _lab in d["labels"]:
        ax.axvline(xt, color="0.8", lw=0.6, zorder=0)
    if "irreps" in d:
        xmax = max(x)
        seen = set()
        for point in d["irreps"]:
            key = round(point["x"], 8)
            if key in seen:
                continue
            seen.add(key)
            at_end = point["x"] > 0.98 * xmax
            dx = -0.012 * xmax if at_end else 0.012 * xmax
            ha = "right" if at_end else "left"
            for cl in point["clusters"]:
                y = cl["e"] - ref
                if not (args.window[0] < y < args.window[1]):
                    continue
                ax.annotate(cl["label"], (point["x"] + dx, y), fontsize=7,
                            ha=ha, va="center", color="#a03020",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.75))
    ax.axhline(0.0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_xticks([xt for xt, _ in d["labels"]])
    ax.set_xticklabels([lab.replace("G", "Γ") for _, lab in d["labels"]])
    ax.set_ylabel("E − E_ref (eV)")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*args.window)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
