"""Cubic primitive-cell builders for the periodic-table Δ-gauge.

The Δ-factor reference (Lejaeghere et al., Science 351, aad3000, 2016) fixes
each element's ground-state structure and scales it isotropically. For the
cubic members the reference V0 (Å³/atom) fixes the lattice constant, so no CIF
is needed. Three Bravais families cover the whole cubic subset:

  fcc      1 atom/cell,  V_cell = a³/4   (noble/Pt-group metals, fcc AE, Al)
  bcc      1 atom/cell,  V_cell = a³/2   (alkali, group-5/6 refractory metals)
  diamond  2 atoms/cell, V_cell = a³/4   (group-IV semiconductors)

Non-cubic ground states (hcp Mg/Zn/Ti…, rhombohedral As/Bi, graphite C) are
out of scope here because V0 alone does not fix their geometry.
"""
import numpy as np

# fcc/diamond share the fcc primitive lattice; bcc uses its own.
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
BCC = np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])

# atoms per primitive cell, and V_cell = a³ / VDIV
_STRUCT = {
    "fcc":     dict(mat=FCC, basis=[[0, 0, 0]], vdiv=4.0),
    "bcc":     dict(mat=BCC, basis=[[0, 0, 0]], vdiv=2.0),
    "diamond": dict(mat=FCC, basis=[[0, 0, 0], [0.25, 0.25, 0.25]], vdiv=4.0),
}


def natoms(structure):
    return len(_STRUCT[structure]["basis"])


def a_from_v0(structure, v0_per_atom, scale=1.0):
    """Cubic lattice constant at volume factor `scale` × V0_ref (Å³/atom)."""
    nat = natoms(structure)
    v_cell = v0_per_atom * nat * scale
    return (v_cell * _STRUCT[structure]["vdiv"]) ** (1.0 / 3.0)


def geometry(structure, elem, v0_per_atom, scale=1.0):
    """(cell, cart_positions, elems) for an elemental cubic crystal at volume
    factor `scale` of its reference per-atom volume."""
    st = _STRUCT[structure]
    a = a_from_v0(structure, v0_per_atom, scale)
    cell = a / 2.0 * st["mat"]
    pos = np.asarray(st["basis"], dtype=np.float64) @ cell
    return cell, pos, [elem] * natoms(structure)
