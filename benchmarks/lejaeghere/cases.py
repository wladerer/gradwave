"""Lejaeghere Δ-benchmark subset: PAW (psl kjpaw) vs the WIEN2k reference.

Protocol (Lejaeghere et al., Science 351, aad3000, 2016): E(V) at seven
volumes spanning 94-106% of the ALL-ELECTRON equilibrium volume, 3rd-order
Birch-Murnaghan fit, Δ = RMS difference of the fitted curves (each shifted
to its own minimum) over the +/-6% window around the average V0, per atom.

Reference values are WIEN2k version 13.1 (calcDelta package 3.0):
V0 in A^3/atom, B0 in GPa, B1 dimensionless. Structures are derived from
the reference V0 (cubic subset only, so V0 fixes the lattice constant).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from structures import EOS_SCALES as SCALES  # noqa: E402,F401
from structures import RY, a_from_v0, fcc_geometry  # noqa: E402,F401

# WIEN2k v13.1: V0 (A^3/atom), B0 (GPa), B1. Cubic members only — the
# reference C is graphite (V0 11.64 A^3/atom is the PBE interlayer, not
# diamond), so Ge stands in as the second group-IV semiconductor.
WIEN2K = {
    "si": (20.4530, 88.545, 4.31),
    "ge": (23.9148, 59.128, 4.99),
    "al": (16.4796, 78.077, 4.57),
    "cu": (11.9511, 141.335, 4.86),
    "ni": (10.8876, 200.368, 5.00),
}

# ecut/ecutrho at or above the psl-suggested cutoffs; gaussian width for
# metals is 0.01 Ry (QE degauss convention). Metal k-meshes are the main
# remaining convergence knob — 16^3 / 12^3 IBZ is GPU-sized, not CPU-sized.
CASES = {
    "si": dict(pseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF", elems=["Si", "Si"],
               frac=[[0, 0, 0], [0.25] * 3], ecut_ry=45, ecutrho_ry=180,
               kmesh=(8, 8, 8), smearing="none", width=0.0, nbands=None,
               nspin=1, start_mag=None, mixing_scheme="johnson"),
    # PBE Ge is a near-zero-gap semiconductor — tiny smearing keeps the
    # occupations well-defined across the volume scan
    "ge": dict(pseudo="Ge.pbe-dn-kjpaw_psl.1.0.0.UPF", elems=["Ge", "Ge"],
               frac=[[0, 0, 0], [0.25] * 3], ecut_ry=45, ecutrho_ry=240,
               kmesh=(8, 8, 8), smearing="gaussian", width=0.002 * RY,
               nbands=18, nspin=1, start_mag=None,
               mixing_scheme="johnson"),
    "al": dict(pseudo="Al.pbe-n-kjpaw_psl.1.0.0.UPF", elems=["Al"],
               frac=[[0, 0, 0]], ecut_ry=35, ecutrho_ry=160,
               kmesh=(16, 16, 16), smearing="gaussian", width=0.01 * RY,
               nbands=8, nspin=1, start_mag=None,
               mixing_scheme="johnson"),
    "cu": dict(pseudo="Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", elems=["Cu"],
               frac=[[0, 0, 0]], ecut_ry=50, ecutrho_ry=280,
               kmesh=(16, 16, 16), smearing="gaussian", width=0.01 * RY,
               nbands=14, nspin=1, start_mag=None,
               mixing_scheme="johnson"),
    # Johnson everywhere (2026-07-13): the QE-class mixer at default
    # damping, becsum unscaled (bec_step_scale 1.0 per-scheme default —
    # FM Ni reference config 16 iterations vs 118 for hand-damped pulay
    # alpha 0.3; pulay 0.7 collapses the moment, unweighted broyden
    # diverges; the measured Stoner-mode gain is -3.5, plain-step
    # stability boundary alpha 0.44). Energy criterion for Ni because
    # the residual floors at metallic occupation noise; warm-start
    # chaining across volumes holds the branch between scan points.
    "ni": dict(pseudo="Ni.pbe-spn-kjpaw_psl.1.0.0.UPF", elems=["Ni"],
               frac=[[0, 0, 0]], ecut_ry=75, ecutrho_ry=480,
               kmesh=(12, 12, 12), smearing="gaussian", width=0.01 * RY,
               nbands=14, nspin=2, start_mag=[0.8],
               mixing_scheme="johnson", criterion="energy"),
}


def geometry(case, scale):
    """(cell, cart_positions, elems) at volume factor `scale` x V0_ref."""
    cfg = CASES[case]
    a = a_from_v0(WIEN2K[case][0], len(cfg["elems"]), scale)
    return fcc_geometry(a, cfg["frac"], cfg["elems"])
