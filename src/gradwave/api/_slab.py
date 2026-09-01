"""Density-tail vacuum auto-sizer for open-boundary (ESM) slabs.

ESM (``boundary="open_z"`` / ``"open_z_metal"``) makes the open-axis
electrostatics box-independent: the vacuum no longer has to be wide enough to
*decouple periodic images*, only wide enough to *hold the physical density
tail*. A habitual 10–20 Å/face of vacuum is then pure waste. Trimming the open
(c) axis to just the SAD density tail plus a margin cuts **both** the FFT box
length ``Nz`` and the plane-wave count ``npw`` — both scale ~linearly with the
open-axis length ``L_z`` (``grids.build_fft_grid`` sizes each dimension from the
Miller extent of the density sphere; ``build_gsphere`` keeps ``|k+G|² ≤ ecut``,
whose count is ``∝ Ω``) — so the FFT H-apply and the O(nb²·npw) Rayleigh–Ritz
shrink together.

The box is a **forward hyperparameter** — set once, before any SCF step, and
frozen for the whole solve, exactly like ``ecut``. The vacuum-normal stress is
physically meaningless for a slab, so there is no autograd interaction: no
mid-SCF basis change, no adjoint coupling. This module resolves the trimmed
``(cell, positions)`` from a superposition-of-atomic-densities (SAD) profile —
no SCF is run.

Two disciplines set the margin, keyed to the requested observable:

* **energy / forces** are *tail-limited* — the total energy stops moving once
  the density that spills into the vacuum is captured, so an aggressive
  ``ρ < ~1e-4 e/Å³`` cut with a thin margin suffices.
* **work function / dipole** are *plateau-limited* — ``postscf.work_function.
  vacuum_level`` reads ``v_eff`` on the lowest-density open-axis planes and
  needs a flat vacuum plateau there, so the margin is larger (a conservative
  default).

Gates (a slab satisfies them, a small bulk-ish cell does not): only trim when
the run is arithmetic-bound (``npw ≳ npw_gate``) and genuinely vacuum-dominated
(vacuum fraction ``≳ vacuum_fraction_gate``). Below either gate the box is left
exactly as the user set it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gradwave.grids import gmax_from_ecut

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gradwave.inputs import Input
    from gradwave.pseudo.upf import UPFData
    from gradwave.pseudo.upf_paw import PAWData

logger = logging.getLogger(__name__)


# Per-observable tail tolerance [e/Å³] and vacuum margin [Å] beyond where the
# SAD planar density has decayed below that tolerance. "energy" is tail-limited
# (aggressive); "workfunction" is plateau-limited (conservative — the default).
_TARGET_DEFAULTS = {
    "energy": (1.0e-4, 1.0),
    "workfunction": (1.0e-5, 4.0),
}


@dataclass(frozen=True)
class SlabBox:
    """The resolved slab box. ``trimmed`` is False when a gate held or the box
    was already tight (the caller then uses the original cell/positions)."""

    cell: np.ndarray  # (3,3) rows a_i [Å]
    positions: np.ndarray  # (na,3) Cartesian Å
    trimmed: bool
    axis: int | None  # detected vacuum axis (None if no slab detected)
    length_before: float  # open-axis length before [Å]
    length_after: float  # open-axis length after [Å]
    npw_estimate: int  # plane-wave count estimate at the *original* box
    reason: str  # human-readable note (why trimmed / why not)


def _axis_vacuum_gaps(cell: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Largest vacuum gap [Å] along each of the three cell axes.

    Projects atoms onto each axis' fractional coordinate, sorts, and measures the
    widest gap between consecutive occupied planes (including the periodic
    wrap-around). A slab has one axis whose widest gap is the vacuum layer; a
    bulk cell has small gaps on every axis. Taking the *largest* gap in
    fractional coordinates makes it robust to a protruding adsorbate — the
    adsorbate only shrinks the open-axis gap, it does not move detection onto a
    periodic axis. (Mirrors ``kpoints._axis_vacuum_gaps`` — the same detection
    the slab k-mesh uses.)"""
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(cell, axis=1)
    if len(pos) == 0:
        return np.zeros(3)
    frac = pos @ np.linalg.inv(cell)
    gaps = np.zeros(3)
    for i in range(3):
        f = np.sort(np.mod(frac[:, i], 1.0))
        if len(f) == 1:
            widest = 1.0
        else:
            internal = np.diff(f)
            wrap = (f[0] + 1.0) - f[-1]
            widest = float(max(internal.max(), wrap))
        gaps[i] = widest * lengths[i]
    return gaps


def _npw_estimate(cell: np.ndarray, ecut: float) -> int:
    """Plane-wave count of the ecut sphere: ``Ω·g_max³ / (6π²)`` with
    ``g_max = √(ecut / (ħ²/2m))``. The number of reciprocal-lattice points inside
    a sphere of radius g_max is its volume over the BZ cell volume ``(2π)³/Ω``."""
    vol = float(abs(np.linalg.det(np.asarray(cell, dtype=np.float64).reshape(3, 3))))
    gmax = gmax_from_ecut(ecut)
    return int(vol * gmax**3 / (6.0 * np.pi**2))


def size_box_from_planar_density(
    cell: np.ndarray,
    positions: np.ndarray,
    rho_planar: np.ndarray,
    axis: int,
    *,
    rho_tol: float,
    margin: float,
    min_vacuum: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Pure-geometry core: trim ``cell``'s ``axis`` to the density tail.

    ``rho_planar`` is the plane-averaged density [e/Å³] sampled on the FFT grid
    along ``axis`` (length ``Nz``), periodic. The occupied region
    ``{z : ρ(z) ≥ rho_tol}`` is centered (rolled so its widest sub-tolerance gap
    sits at the box edge), its extent measured, and the new open-axis length set
    to ``extent + 2·margin`` — but never below ``atom_span + 2·min_vacuum`` (the
    safety floor), and never *above* the original length (trim only). The slab is
    re-centered in the new box. Returns ``(new_cell, new_positions, info)``.

    Sizing to the occupied *density* extent (not the atom span) keys the box to
    the adsorbate face automatically: the protruding adsorbate's tail widens the
    occupied region on its side, so it gets the vacuum it needs.
    """
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3).copy()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
    rho = np.asarray(rho_planar, dtype=np.float64).reshape(-1)
    nz = rho.size
    axis_vec = cell[axis]
    length = float(np.linalg.norm(axis_vec))
    unit = axis_vec / length
    dz = length / nz

    occ = rho >= rho_tol
    info: dict[str, object] = {"axis": axis, "length_before": length}
    if not occ.any():
        info["reason"] = "no SAD density above rho_tol (empty cell?) — no trim"
        return cell, positions, info
    occ_idx = np.flatnonzero(occ)
    if occ_idx.size == nz:
        info["reason"] = "density fills the whole open axis — no vacuum to trim"
        return cell, positions, info

    # Center the occupied region: the vacuum is the widest circular gap *between*
    # occupied planes. Roll so that gap sits at the periodic seam, leaving the
    # occupied region as one contiguous run in the interior.
    circ_gaps = np.diff(occ_idx)  # gaps between consecutive occupied planes
    wrap_gap = (occ_idx[0] + nz) - occ_idx[-1]  # across the periodic boundary
    if circ_gaps.size and circ_gaps.max() >= wrap_gap:
        # widest interior gap: seam is the last occupied plane before it
        seam = int(occ_idx[int(np.argmax(circ_gaps))])
    else:
        seam = int(occ_idx[-1])  # widest gap already wraps the boundary
    shift = (nz - 1) - seam  # move that seam to index nz-1
    rho_rolled = np.roll(rho, shift)
    occ_rolled = rho_rolled >= rho_tol
    occ_where = np.flatnonzero(occ_rolled)
    if occ_where.size == 0:
        info["reason"] = "occupied region vanished after centering — no trim"
        return cell, positions, info
    lo_i, hi_i = int(occ_where[0]), int(occ_where[-1])
    # occupied density extent [Å] in the rolled frame (edge-to-edge of the run)
    z_lo = lo_i * dz
    z_hi = (hi_i + 1) * dz
    extent = z_hi - z_lo

    # atom axis-coordinates in the *rolled* frame (roll = +shift*dz along unit)
    s_atom = (positions @ unit + shift * dz) % length
    atom_span = float(s_atom.max() - s_atom.min()) if len(s_atom) else 0.0

    new_len = extent + 2.0 * margin
    floor_len = atom_span + 2.0 * min_vacuum
    hit_floor = new_len < floor_len
    if hit_floor:
        new_len = floor_len
    info["hit_floor"] = hit_floor

    if new_len >= length:
        info["reason"] = (
            f"box already tight (need {new_len:.2f} Å ≥ current {length:.2f} Å) — no trim"
        )
        info["length_after"] = length
        return cell, positions, info

    # place the occupied region centered in the new box
    offset = 0.5 * (new_len - extent) - z_lo
    new_axis_coord = s_atom + offset
    inplane = positions - np.outer(positions @ unit, unit)
    new_positions = inplane + np.outer(new_axis_coord, unit)
    new_cell = cell.copy()
    new_cell[axis] = unit * new_len

    info["length_after"] = new_len
    info["extent"] = extent
    info["atom_span"] = atom_span
    info["reason"] = (
        f"trimmed open axis {length:.2f} → {new_len:.2f} Å "
        f"(density extent {extent:.2f} Å + 2×{margin:.1f} Å margin"
        + (", floored" if hit_floor else "") + ")"
    )
    return new_cell, new_positions, info


def _sad_planar_density(
    cell: np.ndarray,
    positions: np.ndarray,
    species_of_atom: Sequence[int],
    upfs: Sequence[UPFData | PAWData],
    ecut: float,
    axis: int,
) -> np.ndarray:
    """Plane-averaged SAD density [e/Å³] along ``axis`` on a grid built from
    ``cell``/``ecut``. Superposition of neutral-atom densities — no SCF."""
    import torch

    from gradwave.grids import build_fft_grid
    from gradwave.scf.guess import sad_density

    grid = build_fft_grid(np.asarray(cell, dtype=np.float64), ecut)
    n_electrons = float(sum(float(upfs[s].z_valence) for s in species_of_atom))
    rho = sad_density(
        grid,
        torch.as_tensor(np.asarray(positions, dtype=np.float64)),
        list(species_of_atom),
        list(upfs),
        n_electrons,
    )  # (n1,n2,n3) [e/Å³]
    other = tuple(a for a in range(3) if a != axis)
    return rho.mean(dim=other).detach().cpu().numpy()


def resolve_slab_box(
    inp: Input,
    upfs: Sequence[UPFData | PAWData],
    species_of_atom: Sequence[int],
) -> SlabBox:
    """Resolve the (possibly trimmed) slab box for ``inp``.

    A no-op (returns the original cell/positions, ``trimmed=False``) unless the
    slab auto-sizer is enabled *and* the boundary is an open (ESM) one *and* both
    gates pass. Never runs an SCF — the tail comes from the SAD density."""
    cell = np.asarray(inp.atoms.cell.array, dtype=np.float64).reshape(3, 3)
    positions = np.asarray(inp.atoms.get_positions(), dtype=np.float64).reshape(-1, 3)
    slab = inp.slab
    npw = _npw_estimate(cell, inp.ecut)

    def noop(reason: str, axis: int | None = None) -> SlabBox:
        return SlabBox(cell, positions, False, axis, float(np.linalg.norm(cell[axis]))
                       if axis is not None else 0.0, 0.0, npw, reason)

    if not slab.vacuum_autosize:
        return noop("vacuum_autosize off")
    if inp.scf.boundary not in ("open_z", "open_z_metal"):
        return noop(
            f"boundary={inp.scf.boundary!r} is not an open (ESM) boundary — "
            "auto-size only trims when the open-axis electrostatics are box-"
            "independent")

    gaps = _axis_vacuum_gaps(cell, positions)
    axis = int(np.argmax(gaps))
    lengths = np.linalg.norm(cell, axis=1)
    length = float(lengths[axis])
    vac_frac = float(gaps[axis] / length) if length > 0 else 0.0

    if npw < slab.npw_gate:
        return noop(
            f"npw≈{npw} < gate {slab.npw_gate} — latency-bound, trimming pointless",
            axis)
    if vac_frac < slab.vacuum_fraction_gate:
        return noop(
            f"vacuum fraction {vac_frac:.2f} < gate {slab.vacuum_fraction_gate} — "
            "not vacuum-dominated (bulk-ish); leaving box",
            axis)

    rho_tol, margin = _TARGET_DEFAULTS[slab.vacuum_target]
    if slab.vacuum_tol is not None:
        rho_tol = slab.vacuum_tol
    if slab.vacuum_margin is not None:
        margin = slab.vacuum_margin

    rho_planar = _sad_planar_density(
        cell, positions, species_of_atom, upfs, inp.ecut, axis)
    new_cell, new_pos, info = size_box_from_planar_density(
        cell, positions, rho_planar, axis,
        rho_tol=rho_tol, margin=margin, min_vacuum=slab.min_vacuum)

    length_after = float(info.get("length_after", length))  # ty: ignore[invalid-argument-type]
    reason = str(info["reason"])
    trimmed = length_after < length - 1e-9
    if trimmed and info.get("hit_floor"):
        warnings.warn(
            "slab vacuum auto-size: the density tail wanted less vacuum than the "
            f"{slab.min_vacuum:.1f} Å/face safety floor; box floored at the safety "
            "minimum. Lower slab.min_vacuum only if you know the tail is captured.",
            stacklevel=2)
    if trimmed:
        logger.info(
            "slab vacuum auto-size (%s): open axis %d, %.2f → %.2f Å, "
            "npw≈%d (was for the original box); %s",
            slab.vacuum_target, axis, length, length_after, npw, reason)
    return SlabBox(
        new_cell if trimmed else cell,
        new_pos if trimmed else positions,
        trimmed, axis, length, length_after, npw, reason)
