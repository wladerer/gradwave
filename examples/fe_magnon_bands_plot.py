"""Render the bcc-Fe magnon dispersion figure from extracted per-shell exchange.

Companion to examples/fe_magnon_stiffness.py: takes the per-shell isotropic
couplings that script prints (curvature convention, meV per unit-moment pair)
and draws the LSWT dispersion along the bcc high-symmetry path, one curve per
cumulative shell set (J1, J1+J2, ...), to show shell convergence. Writes
docs/manual/img/fe_magnon_bands.png.

Usage:
    python examples/fe_magnon_bands_plot.py --j 22.4 10.9 --out fe_magnon_bands.png
"""

from __future__ import annotations

import argparse

import numpy as np

from gradwave.postscf.magnons import ExchangeBond, HeisenbergModel, magnon_bands

A_BCC = 2.87
M_MOMENT = 2.222


def _bcc_primitive(a: float) -> np.ndarray:
    return 0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)


def _shell_vectors(prim: np.ndarray):
    shell_r = {1: np.sqrt(3) / 2 * A_BCC, 2: A_BCC, 3: np.sqrt(2) * A_BCC}
    out: dict[int, list[tuple[int, int, int]]] = {1: [], 2: [], 3: []}
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                if (n1, n2, n3) == (0, 0, 0):
                    continue
                d = float(np.linalg.norm(np.array([n1, n2, n3], float) @ prim))
                for n, r in shell_r.items():
                    if abs(d - r) < 0.05:
                        out[n].append((n1, n2, n3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", type=float, nargs="+", required=True,
                    help="per-shell J curvature [meV], J1 [J2 [J3]]")
    ap.add_argument("--out", default="docs/manual/img/fe_magnon_bands.png")
    ap.add_argument("--npoints", type=int, default=300)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = _bcc_primitive(A_BCC)
    srs = _shell_vectors(prim)
    s = M_MOMENT / 2.0

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = ["#9dbfe3", "#5a92cf", "#2a78d6"]
    labels = None
    x = None
    for upto in range(1, len(args.j) + 1):
        bonds = []
        for n in range(1, upto + 1):
            jm = args.j[n - 1] * 1e-3 / s**2  # meV curvature -> eV model coupling
            for r in srs[n]:
                bonds.append(ExchangeBond(0, 0, r, jm))
        model = HeisenbergModel(cell=prim, spins=[s], bonds=bonds)
        bs = magnon_bands(model, path="GHNGP", npoints=args.npoints)
        x, labels = bs.x, bs.labels
        tag = "+".join(f"J{k}" for k in range(1, upto + 1))
        ax.plot(bs.x, bs.frequencies[:, 0], lw=1.6,
                color=colors[(upto - 1) % len(colors)], label=tag)
    assert labels is not None and x is not None
    for xt, _lab in labels:
        ax.axvline(xt, color="#52514e", lw=0.5, alpha=0.5)
    ax.axhline(0.0, color="#52514e", lw=0.5, ls="--", alpha=0.7)
    ax.set_xticks([xt for xt, _ in labels])
    ax.set_xticklabels([lab.replace("G", "Γ") for _, lab in labels])
    ax.set_ylabel("ω [meV]")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.legend(frameon=False, title="shells")
    ax.set_title("bcc Fe magnon dispersion (LSWT on extracted J)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
