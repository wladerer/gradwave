"""Plot an E(theta) anisotropy curve from a fept_mae_map.json-style file.

Usage: uv run python scripts/plot_mae.py fept_mae_map.json [-o mae_map.png]
Requires matplotlib (not a package dependency — install it where needed).

Reads the JSON that examples/fept_mae_map.py writes: measured dF per theta,
the K1 sin^2 + K2 sin^4 fit, fold counts and provenance. Draws the measured
points, the fit evaluated on a dense theta grid, and a residual strip.
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("-o", "--output", default="mae_map.png")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed — `uv pip install matplotlib`") from None

    d = json.load(open(args.json_file))
    th = np.asarray(d["theta_deg"])
    dF = np.asarray(d["dF_meV"])
    k1, k2 = d["K1_meV"], d["K2_meV"]
    th_fine = np.linspace(th.min(), th.max(), 361)
    s2 = np.sin(np.deg2rad(th_fine)) ** 2
    fit_fine = k1 * s2 + k2 * s2**2
    resid_ueV = (dF - np.asarray(d["fit_meV"])) * 1000.0

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.4, 5.4), height_ratios=[4, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})
    ax.plot(th_fine, fit_fine, "-", lw=1.6,
            label=rf"$K_1\sin^2\theta + K_2\sin^4\theta$"
                  f"\n$K_1$={k1:+.4f}, $K_2$={k2:+.4f} meV/cell")
    ax.plot(th, dF, "o", ms=6, label="measured (one folded solve each)")
    ax.set_ylabel(r"$F(\theta) - F(0)$  [meV/cell]")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{d.get('system', '')}  {tuple(d.get('kmesh', ()))} mesh, "
                 f"{d.get('ecut_eV', 0) / 13.605693122994:.0f} Ry", fontsize=10)

    axr.axhline(0.0, color="0.6", lw=0.8)
    axr.plot(th, resid_ueV, "o", ms=4)
    axr.set_xlabel(r"$\theta$ from the reference axis  [deg]")
    axr.set_ylabel(r"resid. [$\mu$eV]")

    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
