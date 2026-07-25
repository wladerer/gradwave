"""End-to-end miscibility-gap phase diagram for a real binary, Cu-Ag by default.

Cu and Ag are noble metals (nonmagnetic, FCC) that barely mix, so the DFT mixing
energy is strongly positive and the solid solution unmixes. The pipeline relaxes
the volume of pure A, pure B, and the ordered L1_0 A-B by a small energy-volume
scan, forms the mixing energy, fits the regular-solution interaction Omega, and
runs the phase-4 common-tangent construction to get the miscibility gap and its
critical temperature. The mean-field T_c = Omega/(2 k_B) is compared to the
known system.

Heavy: run on asus via gwq. Semicore (19-electron) pseudos plus metals need a
high cutoff and a dense k-mesh.

  uv run python benchmarks/phase_diagram/cu_ag.py \
      --pseudo-dir benchmarks/delta_gauge/pseudos --outdir ~/cuag_pd
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.build import bulk

RY = 13.605693122994
KB = 8.617333262e-5  # eV/K


def build_structures(a_a, a_b, a_ab, sym_a, sym_b):
    """Conventional FCC A4 and B4 and the ordered L1_0 A2B2 (layers along z)."""
    a4 = bulk(sym_a, "fcc", a=a_a, cubic=True)
    b4 = bulk(sym_b, "fcc", a=a_b, cubic=True)
    ab = bulk(sym_a, "fcc", a=a_ab, cubic=True)
    frac = ab.get_scaled_positions()
    ab.set_chemical_symbols([sym_a if p[2] < 0.25 else sym_b for p in frac])
    return {"A": a4, "B": b4, "AB": ab}


def relaxed_energy(atoms, upfs, sym_to_species, ecut, kmesh, scales, kw):
    """Volume-scan energy per atom. Scale the cell isotropically, SCF each point,
    and fit a parabola in volume for the minimum."""
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf, setup_system

    species = [sym_to_species[s] for s in atoms.get_chemical_symbols()]
    nat = len(atoms)
    vols, energies = [], []
    for s in scales:
        cell = np.asarray(atoms.cell) * s
        pos = atoms.get_positions() * s
        system = setup_system(cell, pos, species, upfs, ecut=ecut, kmesh=kmesh,
                              use_symmetry=True)
        r = scf(system, PBE(), **kw)
        vols.append(float(abs(np.linalg.det(cell))) / nat)
        energies.append(float(r.energies.free_energy) / nat)
    c = np.polyfit(vols, energies, 2)  # E(V) parabola per atom
    v0 = -c[1] / (2 * c[0])
    e0 = np.polyval(c, v0)
    return {"e0": float(e0), "v0": float(v0),
            "vols": vols, "energies": energies, "converged": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--elements", nargs=2, default=["Cu", "Ag"])
    ap.add_argument("--a0", nargs=3, type=float, default=[3.63, 4.15, 3.90],
                    help="estimated FCC lattice constants for A, B, and the alloy")
    ap.add_argument("--ecut-ry", type=float, default=55.0)
    ap.add_argument("--kmesh", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--exp-tc", type=float, default=None,
                    help="experimental critical temperature [K] for comparison")
    a = ap.parse_args()

    from gradwave.pseudo.upf import parse_upf

    sym_a, sym_b = a.elements
    pdir = Path(a.pseudo_dir)
    upfs = [parse_upf(pdir / f"{sym_a}.upf"), parse_upf(pdir / f"{sym_b}.upf")]
    sym_to_species = {sym_a: 0, sym_b: 1}
    structs = build_structures(a.a0[0], a.a0[1], a.a0[2], sym_a, sym_b)
    scales = [0.97, 0.985, 1.0, 1.015, 1.03]
    kw = dict(smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6,
              max_iter=200, verbose=False)

    out = {"system": f"{sym_a}-{sym_b}", "ecut_ry": a.ecut_ry, "kmesh": a.kmesh}
    t0 = time.perf_counter()
    for key in ("A", "B", "AB"):
        res = relaxed_energy(structs[key], upfs, sym_to_species, a.ecut_ry * RY,
                             tuple(a.kmesh), scales, kw)
        out[key] = res
        print(f"{key}: e0={res['e0']:.5f} eV/atom  v0={res['v0']:.3f} A^3/atom")

    # mixing energy at x=0.5 and the regular-solution interaction
    dh_mix = out["AB"]["e0"] - 0.5 * out["A"]["e0"] - 0.5 * out["B"]["e0"]
    omega = 4.0 * dh_mix  # E_mix = Omega x(1-x), x=0.5
    tc_mf = omega / (2.0 * KB)
    out["dh_mix_eV_per_atom"] = dh_mix
    out["omega_eV"] = omega
    out["tc_meanfield_K"] = tc_mf
    print(f"\ndH_mix(x=0.5) = {dh_mix*1000:.1f} meV/atom  Omega = {omega*1000:.1f} meV")
    print(f"mean-field T_c = {tc_mf:.0f} K")

    # phase-4 common tangent on the regular solution (needs Omega > 0)
    if omega > 0:
        from gradwave.postscf.phase_diagram import binodal, critical_temperature
        xg = np.linspace(0.002, 0.998, 400)
        temps = np.linspace(100.0, max(200.0, 1.1 * tc_mf), 60)

        def g_of_x_t(x, temp):
            return omega * x * (1 - x) + KB * temp * (x * np.log(x)
                                                      + (1 - x) * np.log(1 - x))

        _, left, right = binodal(temps, xg, g_of_x_t)
        out["tc_common_tangent_K"] = critical_temperature(temps, left, right)
        out["binodal"] = {"temps": temps.tolist(), "left": left.tolist(),
                          "right": right.tolist()}
        print(f"common-tangent T_c = {out['tc_common_tangent_K']:.0f} K")
    if a.exp_tc:
        out["exp_tc_K"] = a.exp_tc
        print(f"experimental reference T_c = {a.exp_tc:.0f} K")
    out["wall_s"] = time.perf_counter() - t0

    outdir = Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k not in ("A", "B", "AB", "binodal")},
                     indent=2))


if __name__ == "__main__":
    main()
