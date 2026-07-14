"""Shared structure constructors for the benchmark family (refactor
stage 6). The two EOS families (delta_factor, lejaeghere) build the same
volume-scaled fcc primitive cells; the case files keep the physics
choices (cutoffs, meshes, smearing) and take geometry from here. The
bench_* performance scripts stay self-contained on purpose — they carry
committed reference energies and gain nothing from churn."""
import numpy as np

RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
# volume factors, ±6% around the reference volume (standard Δ window)
EOS_SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]


def fcc_geometry(a, frac, elems):
    """(cell, cart_positions, elems) for an fcc primitive cell at lattice
    constant `a` with fractional positions `frac`."""
    cell = a / 2.0 * FCC
    pos = np.asarray(frac, dtype=np.float64) @ cell
    return cell, pos, list(elems)


def scaled_a(a, vol_scale):
    """Lattice constant at volume factor `vol_scale` of the a-defined cell."""
    return a * vol_scale ** (1.0 / 3.0)


def a_from_v0(v0_per_atom, natoms, vol_scale):
    """fcc lattice constant from a per-atom reference volume
    (V_cell = a³/4), at volume factor `vol_scale`."""
    return (4.0 * natoms * v0_per_atom * vol_scale) ** (1.0 / 3.0)
