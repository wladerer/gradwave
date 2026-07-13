"""Lejaeghere Δ-benchmark subset: PAW (psl kjpaw) vs the WIEN2k reference.

Protocol (Lejaeghere et al., Science 351, aad3000, 2016): E(V) at seven
volumes spanning 94-106% of the ALL-ELECTRON equilibrium volume, 3rd-order
Birch-Murnaghan fit, Δ = RMS difference of the fitted curves (each shifted
to its own minimum) over the +/-6% window around the average V0, per atom.

Reference values are WIEN2k version 13.1 (calcDelta package 3.0):
V0 in A^3/atom, B0 in GPa, B1 dimensionless. Structures are derived from
the reference V0 (cubic subset only, so V0 fixes the lattice constant).
"""
import numpy as np

RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
# volume factors applied to the reference equilibrium volume
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

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
               nspin=1, start_mag=None),
    # PBE Ge is a near-zero-gap semiconductor — tiny smearing keeps the
    # occupations well-defined across the volume scan
    "ge": dict(pseudo="Ge.pbe-dn-kjpaw_psl.1.0.0.UPF", elems=["Ge", "Ge"],
               frac=[[0, 0, 0], [0.25] * 3], ecut_ry=45, ecutrho_ry=240,
               kmesh=(8, 8, 8), smearing="gaussian", width=0.002 * RY,
               nbands=18, nspin=1, start_mag=None),
    "al": dict(pseudo="Al.pbe-n-kjpaw_psl.1.0.0.UPF", elems=["Al"],
               frac=[[0, 0, 0]], ecut_ry=35, ecutrho_ry=160,
               kmesh=(16, 16, 16), smearing="gaussian", width=0.01 * RY,
               nbands=8, nspin=1, start_mag=None),
    "cu": dict(pseudo="Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", elems=["Cu"],
               frac=[[0, 0, 0]], ecut_ry=50, ecutrho_ry=280,
               kmesh=(16, 16, 16), smearing="gaussian", width=0.01 * RY,
               nbands=14, nspin=1, start_mag=None),
    # FM Ni needs the damped mixer (alpha 0.3 — the validated ni_paw_spin
    # config; alpha 0.7 collapses the moment to the NM branch) and accepts
    # the metallic-occupation residual plateau: converge on the energy tail
    "ni": dict(pseudo="Ni.pbe-spn-kjpaw_psl.1.0.0.UPF", elems=["Ni"],
               frac=[[0, 0, 0]], ecut_ry=75, ecutrho_ry=480,
               kmesh=(12, 12, 12), smearing="gaussian", width=0.01 * RY,
               nbands=14, nspin=2, start_mag=[0.8],
               mixing_alpha=0.3, rhotol=2e-3),
}


def geometry(case, scale):
    """(cell, cart_positions, elems) at volume factor `scale` x V0_ref."""
    cfg = CASES[case]
    v0_atom = WIEN2K[case][0]
    v_cell = len(cfg["elems"]) * v0_atom * scale
    a = (4.0 * v_cell) ** (1.0 / 3.0)  # fcc primitive: V_cell = a^3/4
    cell = a / 2.0 * FCC
    pos = np.array(cfg["frac"], dtype=np.float64) @ cell
    return cell, pos, cfg["elems"]
