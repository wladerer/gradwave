"""Charge-response fields — ∂n(r)/∂R_I, the volumetric derivative of the
electron density with respect to moving an atom.

This is the field a differentiable code can make that a conventional one cannot
show cheaply: where the electron density flows when an atom is nudged. It is the
force-constant physics made spatial. Rendered as an isosurface it separates into
a positive and a negative lobe around the displaced atom, the charge that piles
up ahead of the motion and depletes behind it.

The reference implementation here is a central finite difference: two SCFs at
±δ displacement of one atom, differenced on the shared grid. It is correct for
every formalism (collinear, noncollinear/SOC, USPP/PAW) and is the oracle the
analytic response (`postscf.uspp_position.position_density_response`, USPP
insulators) is checked against. Charge conservation makes ∫ ∂n/∂R dr = 0, which
`density_response_fd` verifies and reports.

The two displaced SCFs are independent, so a full 3N-component response set is
embarrassingly parallel.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gradwave.postscf.volumetric import _infer_fmt, write_volumetric


def _displaced_density(inp, atom: int, direction: int, shift: float, spin, verbose):
    """Converged ρ(r) [e/Å³] with `atom` moved by `shift` Å along `direction`."""
    from gradwave.api import run_scf

    atoms = inp.atoms.copy()
    pos = atoms.get_positions()
    pos[atom, direction] += shift
    atoms.set_positions(pos)
    # a displacement breaks the crystal symmetry; symmetrizing the two runs
    # against different broken IBZs would contaminate the difference.
    moved = dataclasses.replace(inp, atoms=atoms, symmetry=False)
    res = run_scf(moved, verbose=verbose)
    if spin is None:
        rho = res.rho
    else:
        rho = res.rho_spin[spin]
    return rho.detach().cpu().numpy()


def density_response_fd(
    inp, atom: int, direction: int, delta: float = 0.02,
    spin: int | None = None, verbose: bool = False,
):
    """∂n(r)/∂R_{atom,direction} [e/Å³/Å] by central finite difference.

    Runs two SCFs at ±`delta` Å displacement of `atom` along Cartesian
    `direction` (0=x, 1=y, 2=z) and central-differences the converged densities.
    `spin` selects a collinear ↑/↓ channel; the default differences the total.

    Returns (field, drift) where `field` is the (n1,n2,n3) response and `drift`
    is ∫ field dr, the charge-conservation residual — it should be ≈ 0 (a
    nonzero value flags an unconverged SCF or too large a `delta`).
    """
    rho_plus = _displaced_density(inp, atom, direction, +delta, spin, verbose)
    rho_minus = _displaced_density(inp, atom, direction, -delta, spin, verbose)
    field = (rho_plus - rho_minus) / (2.0 * delta)

    # ∫ field dr — the cell volume comes from the (unchanged) input cell
    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    dv = abs(np.linalg.det(cell)) / field.size
    drift = float(field.sum() * dv)
    return field, drift


def _atoms_from_input(inp):
    from ase import Atoms

    return Atoms(
        numbers=inp.atoms.get_atomic_numbers(),
        positions=inp.atoms.get_positions(),
        cell=inp.atoms.cell.array,
        pbc=True,
    )


def write_density_response(
    inp, path, atom: int, direction: int, delta: float = 0.02,
    spin: int | None = None, fmt: str | None = None, verbose: bool = False,
):
    """Write ∂n(r)/∂R_{atom,direction} to a .cube/.xsf file.

    Returns (path, drift); `drift` is the ∫ = 0 charge-conservation residual
    from `density_response_fd`.
    """
    fmt = fmt or _infer_fmt(path)
    field, drift = density_response_fd(inp, atom, direction, delta, spin, verbose)
    write_volumetric(path, field, _atoms_from_input(inp), fmt)
    return str(path), drift
