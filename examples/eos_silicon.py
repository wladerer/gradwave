"""Equation of state: bulk modulus of diamond Si and the Delta gauge.

Runs an isotropic volume scan on the 2-atom Si cell (run_eos warm-starts the SCF
from one volume to the next and pins every volume to a shared FFT grid), fits the
third-order Birch-Murnaghan equation of state (Birch, Phys. Rev. 71, 809 (1947)),
and reports V0, B0, and B0'. It then computes the Delta gauge of Lejaeghere et
al. (Science 351, aad3000 (2016)) against the WIEN2k all-electron reference, the
RMS energy difference between the two E(V) curves over a +/-6% window.

The differentiable E(V) points come from ordinary SCFs; the fit and the Delta
integral are plain post-processing in postscf.eos.

Fixtures: tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf (SG15 ONCV, PBE).
Runtime: ~3-5 min (seven warm-started SCFs) on 8 CPU threads.

Run:
    uv run python examples/eos_silicon.py --outdir examples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from gradwave.api import run_eos
from gradwave.inputs import EOSParams, Input, KPointsParams, SmearingParams
from gradwave.postscf.eos import birch_murnaghan, delta_value

RY = 13.605693122994
EV_A3_TO_GPA = 160.2176634
# WIEN2k v13.1 all-electron reference (Lejaeghere 2016): V0 [Å³/atom], B0 [GPa], B1.
WIEN2K_SI = dict(v0=20.4530, b0_GPa=88.545, b1=4.31)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = 5.47
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    atoms = Atoms("Si2", positions=np.array([[0.0, 0, 0], [a / 4] * 3]),
                  cell=cell, pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=Path("tests/fixtures/qe/pseudos"),
                pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=30 * RY, xc="pbe",
                kpoints=KPointsParams(mesh=(8, 8, 8)),
                smearing=SmearingParams(type="none"), eos=EOSParams())

    print("EOS: Si volume scan, 30 Ry, 8x8x8 k-mesh, 7 volumes ...")
    out = run_eos(inp, verbose=True)

    b0 = out["b0_GPa"]
    v0 = out["v0_ang3_per_atom"]
    b0p = out["b0_prime"]
    print(f"\n  gradwave (PBE):  V0 = {v0:.3f} Å³/atom   B0 = {b0:.1f} GPa   B0' = {b0p:.2f}")
    print(f"  WIEN2k (AE ref): V0 = {WIEN2K_SI['v0']:.3f} Å³/atom   "
          f"B0 = {WIEN2K_SI['b0_GPa']:.1f} GPa   B1 = {WIEN2K_SI['b1']:.2f}")

    ref = (0.0, WIEN2K_SI["v0"], WIEN2K_SI["b0_GPa"] / EV_A3_TO_GPA, WIEN2K_SI["b1"])
    fit = (out["e0_eV_per_atom"], v0, out["b0_eV_ang3"], b0p)
    delta = delta_value(fit, ref)
    print(f"  Δ gauge vs WIEN2k = {delta:.2f} meV/atom "
          f"(all-electron codes agree to ~1 meV; a good pseudopotential lands a few)")

    out["delta_meV_vs_wien2k"] = delta
    (outdir / "eos_silicon.json").write_text(json.dumps(out, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return

    vv = np.asarray(out["volumes_ang3_per_atom"])
    ee = np.asarray(out["energies_eV_per_atom"])
    e0 = out["e0_eV_per_atom"]
    vgrid = np.linspace(vv.min() * 0.99, vv.max() * 1.01, 200)
    ecurve = birch_murnaghan(vgrid, e0, v0, out["b0_eV_ang3"], b0p)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(vgrid, (ecurve - e0) * 1e3, "-", color="#1f6f8b", lw=2,
            label="Birch-Murnaghan fit")
    ax.plot(vv, (ee - e0) * 1e3, "o", color="#c1483a", ms=7, label="SCF points")
    ax.axvline(v0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("Volume  (Å³ / atom)")
    ax.set_ylabel("E − E₀  (meV / atom)")
    ax.set_title(f"Si equation of state — B₀ = {b0:.1f} GPa (WIEN2k {WIEN2K_SI['b0_GPa']:.1f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "eos_silicon.png", dpi=150)
    print(f"  wrote {(outdir / 'eos_silicon.png').name}, eos_silicon.json")


if __name__ == "__main__":
    main()
