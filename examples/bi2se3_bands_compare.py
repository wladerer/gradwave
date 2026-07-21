"""Bi2Se3 band-structure comparison: scalar-relativistic (no SOC) vs SOC.

Runs both branches along the SAME G-Z-F-G-L path with a shared k-point count
so the two band structures overlay on one linear-k axis. This is the
"more-k" companion to examples/bi2se3_inversion.py, which only probes the
band inversion at Gamma.

  no-SOC : SG15 scalar-relativistic pseudos, collinear PBE, bands_along_ase_path.
  SOC    : PseudoDojo fully-relativistic (FR) pseudos, non-magnetic spinor
           PBE, band_structure_nc (frozen-potential spinor Davidson).

Writes bi2se3_sr_bands.json, bi2se3_soc_bands.json, and the overlay figure
bi2se3_bands_overlay.png (the overlay needs matplotlib).

Usage: python examples/bi2se3_bands_compare.py [npoints] [outdir]
"""
import json
import sys
import time

import numpy as np
import torch
from ase import Atoms

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.bands import bands_along_ase_path
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import band_structure_nc, scf_noncollinear

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

A_HEX, C_HEX = 4.138, 28.64
MU, NU = 0.399, 0.206
CELL = np.array([
    [A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [-A_HEX / 2, A_HEX / (2 * np.sqrt(3)), C_HEX / 3],
    [0.0, -A_HEX / np.sqrt(3), C_HEX / 3],
])
FRAC = np.array([[0, 0, 0], [NU, NU, NU], [-NU, -NU, -NU],
                 [MU, MU, MU], [-MU, -MU, -MU]])
SPECIES = [0, 0, 0, 1, 1]  # 0 = Se, 1 = Bi
ECUT = 45 * RY


def plot_overlay(sr_json, soc_json, out_png, npoints):
    """SR (grey) vs SOC (red) overlay from the two band JSONs, VBM-referenced."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the overlay figure", flush=True)
        return
    sr = json.load(open(sr_json))
    so = json.load(open(soc_json))
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    x_sr = np.asarray(sr["x"])
    e_sr = np.asarray(sr["eigenvalues_eV"]) - sr["reference_eV"]
    for b in range(e_sr.shape[1]):
        ax.plot(x_sr, e_sr[:, b], color="0.55", lw=1.1, zorder=1,
                label="no SOC (scalar-rel.)" if b == 0 else None)
    x_so = np.asarray(so["x"])
    e_so = np.asarray(so["eigenvalues_eV"]) - so["reference_eV"]
    for b in range(e_so.shape[1]):
        ax.plot(x_so, e_so[:, b], color="#b03020", lw=1.1, zorder=2,
                label="SOC" if b == 0 else None)
    for xt, _lab in so["labels"]:
        ax.axvline(xt, color="0.85", lw=0.6, zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_xticks([xt for xt, _ in so["labels"]])
    ax.set_xticklabels([lab.replace("G", "Γ") for _, lab in so["labels"]])
    ax.set_ylabel("E − E$_{VBM}$ (eV)")
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(float(x_so[0]), float(x_so[-1]))
    ax.set_title(f"Bi$_2$Se$_3$ bands, Γ–Z–F–Γ–L ({npoints} k-points)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  wrote {out_png}", flush=True)


def main(npoints, outdir):
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    atoms = Atoms("Se3Bi2", scaled_positions=FRAC % 1.0, cell=CELL, pbc=True)
    print(f"Bi2Se3 band comparison  npoints={npoints}  device={dev}", flush=True)

    # ---------------- scalar-relativistic (no SOC) ----------------
    print("\n=== no-SOC (SG15 SR pseudos, collinear PBE) ===", flush=True)
    se = parse_upf(f"{PSE}/Se_ONCV_PBE-1.1.upf")
    bi = parse_upf(f"{PSE}/Bi_ONCV_PBE-1.0.upf")
    t0 = time.time()
    sys_sr = setup_system(CELL, FRAC @ CELL, SPECIES, [se, bi], ecut=ECUT,
                          kmesh=(2, 2, 2), nbands=30)
    if dev != "cpu":
        sys_sr = sys_sr.to(dev)
    r_sr = scf(sys_sr, PBE(), smearing="gaussian", width=0.05,
               etol=1e-7, rhotol=1e-6, verbose=False)
    if dev != "cpu":
        torch.cuda.synchronize()
    t_sr_scf = time.time() - t0
    print(f"  SCF conv={r_sr.converged} iters={r_sr.n_iter} "
          f"ne={sys_sr.n_electrons:.0f} ({t_sr_scf:.0f}s)", flush=True)

    t0 = time.time()
    bs = bands_along_ase_path(r_sr, atoms, "GZFGL", npoints=npoints, nbands=30,
                              verbose=True)
    if dev != "cpu":
        torch.cuda.synchronize()
    t_sr_bands = time.time() - t0
    json.dump({
        "x": bs.x.tolist(), "labels": bs.labels,
        "eigenvalues_eV": bs.eigenvalues.tolist(), "reference_eV": bs.reference,
        "kpts_frac": bs.kpts_frac.tolist(),
        "timings": {"scf_s": t_sr_scf, "bands_s": t_sr_bands},
    }, open(f"{outdir}/bi2se3_sr_bands.json", "w"))
    print(f"  bands ({t_sr_bands:.0f}s); wrote {outdir}/bi2se3_sr_bands.json",
          flush=True)

    # Release the SR allocator arena before the (heavier) SOC branch — otherwise
    # PyTorch's cached SR blocks fragment the pool and the FR spinor SCF, which
    # needs nearly the whole card, OOMs on a small (~6 GB) GPU.
    del sys_sr, r_sr, se, bi
    if dev != "cpu":
        torch.cuda.empty_cache()

    # ---------------- SOC (PseudoDojo FR, non-magnetic spinor) ----------------
    print("\n=== SOC (PseudoDojo FR pseudos, non-magnetic spinor PBE) ===",
          flush=True)
    se_fr = parse_upf(f"{PSE}/PD_Se_FR.upf")
    bi_fr = parse_upf(f"{PSE}/PD_Bi_FR.upf")
    t0 = time.time()
    sys_fr = setup_system(CELL, FRAC @ CELL, SPECIES, [se_fr, bi_fr], ecut=ECUT,
                          kmesh=(2, 2, 2), nbands=45, time_reversal=False)
    if dev != "cpu":
        sys_fr = sys_fr.to(dev)
    xc_nc = NoncollinearXC(SpinPBE())
    r_fr = scf_noncollinear(sys_fr, xc_nc, mag_vec_init=[[0, 0, 0]] * 5,
                            smearing="gaussian", width=0.05, etol=1e-7,
                            rhotol=1e-6, verbose=False, nonmagnetic=True)
    if dev != "cpu":
        torch.cuda.synchronize()
    t_fr_scf = time.time() - t0
    nocc = int(round(sys_fr.n_electrons))  # spinor bands hold 1 electron each
    print(f"  SCF conv={r_fr.converged} iters={r_fr.n_iter} ne={nocc} "
          f"({t_fr_scf:.0f}s)", flush=True)

    bp = atoms.cell.bandpath("GZFGL", npoints=npoints)
    t0 = time.time()
    eigs = band_structure_nc(r_fr, xc_nc, bp.kpts, nbands=88, verbose=True)
    if dev != "cpu":
        torch.cuda.synchronize()
    t_fr_bands = time.time() - t0

    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    # reference = VBM: highest occupied (index nocc-1) at Gamma
    ik_gamma = int(np.argmin(np.linalg.norm(bp.kpts, axis=1)))
    ref = float(np.sort(eigs[ik_gamma])[nocc - 1])
    json.dump({
        "x": x.tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "eigenvalues_eV": eigs.tolist(), "reference_eV": ref,
        "kpts_frac": bp.kpts.tolist(), "n_occupied": nocc,
        "timings": {"scf_s": t_fr_scf, "bands_s": t_fr_bands},
    }, open(f"{outdir}/bi2se3_soc_bands.json", "w"))
    print(f"  bands ({t_fr_bands:.0f}s); wrote {outdir}/bi2se3_soc_bands.json",
          flush=True)

    plot_overlay(f"{outdir}/bi2se3_sr_bands.json",
                 f"{outdir}/bi2se3_soc_bands.json",
                 f"{outdir}/bi2se3_bands_overlay.png", npoints)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    npoints = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    main(npoints, outdir)
