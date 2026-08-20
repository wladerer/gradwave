"""Solid-state NMR from scratch: the ¹⁷O quadrupolar coupling of oxygen, all-electron FLAPW.

This is the end-to-end AutoAPW NMR pipeline. For each tetragonal distortion of an O cell it runs a
self-consistent all-electron FLAPW SCF, extracts the electric field gradient at the nucleus from
the aspherical valence density (the l=2 sphere Poisson + full-potential machinery in
``gradwave.flapw.efg``), and converts ``V_zz`` to the measured quadrupolar coupling
``C_Q = e Q V_zz / h`` for the ¹⁷O nucleus (``gradwave.flapw.nmr``).

Why oxygen: a closed-shell atom is spherical, so its EFG vanishes by symmetry. Oxygen's open 2p⁴
shell has a p-hole, and in a non-cubic field that hole is aspherical — a nonzero EFG, exactly the
quadrupolar coupling that ¹⁷O solid-state NMR measures. Compress the cell along z and the p_x,p_y
doublet fills (an axial p_z hole, asymmetry η→0); the coupling tracks the crystal-field distortion.

Run:
    uv run python examples/flapw_oxygen_nmr.py --outdir examples
Runtime: ~5-8 min (one self-consistent EFG per distortion) on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gradwave.flapw import atomic_scf, crystal_scf_multi, log_mesh
from gradwave.flapw.nmr import NUCLEAR_Q, quadrupolar_coupling

# tetragonal cells (a, a, c) in Bohr; a=6 fixed, c scanned across the cubic point.
_A = 6.0
_C_OVER_A = [0.80, 0.90, 1.00, 1.10, 1.20]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r, dx = log_mesh(1e-5, 28.0, 2500)
    at, _ = atomic_scf("O", r, dx)
    print(f"[1] atomic O (LDA): 1s {at['1s']:.1f}  2s {at['2s']:.2f}  2p {at['2p']:.2f} eV")
    q, spin = NUCLEAR_Q["17O"]
    print(f"    ¹⁷O nuclear quadrupole moment Q = {q} barn, spin I = {spin}\n")

    print("[2] tetragonal O: self-consistent FLAPW → EFG → ¹⁷O quadrupolar coupling")
    print(f"    {'c/a':>5} | {'V_zz(eV/Å²)':>12} | {'η':>5} | {'C_Q(¹⁷O)MHz':>12} | {'ν_Q MHz':>8}")
    rows = []
    for ca in _C_OVER_A:
        cell = [_A, _A, _A * ca]
        _, info = crystal_scf_multi(cell, [((0.5, 0.5, 0.5), "O")], {"O": 1.3},
                                    ecut=200.0, iters=30, efg=True, smearing=0.15)
        s = info["efg"]["a0"]
        cq = quadrupolar_coupling(s["V_zz"], s["eta"], "17O")
        rows.append((ca, s["V_zz"], s["eta"], cq["C_Q_MHz"], cq["nu_Q_MHz"]))
        print(f"    {ca:5.2f} | {s['V_zz']:13.3f} | {s['eta']:5.3f} | "
              f"{cq['C_Q_MHz']:15.3f} | {cq['nu_Q_MHz']:10.4f}")

    if args.no_plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ca = np.array([x[0] for x in rows])
    cq = np.array([abs(x[3]) for x in rows])
    eta = np.array([x[2] for x in rows])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.axvline(1.0, color="0.7", ls=":", lw=1)
    ax1.plot(ca, cq, "o-", color="tab:blue", label="|C_Q|(¹⁷O)")
    ax1.set_xlabel("tetragonal distortion  c/a")
    ax1.set_ylabel("|C_Q(¹⁷O)|  (MHz)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ca, eta, "s--", color="tab:red", label="asymmetry η")
    ax2.set_ylabel("asymmetry η", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.02, 1.0)
    ax1.set_title("¹⁷O quadrupolar coupling vs tetragonal distortion (all-electron FLAPW)")
    fig.tight_layout()
    out = outdir / "flapw_oxygen_nmr.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
