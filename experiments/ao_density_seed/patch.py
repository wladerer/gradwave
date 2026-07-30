"""Install/remove the d-localized spin seed by monkeypatching the seed builders.

Nothing here changes a default. `install()` replaces the module-level seed
functions with wrappers that, for a warm nspin=2 start-from-scratch, substitute
the d-localized magnetization shape and otherwise delegate to the original. The
electron partition (n_up, n_dn) and the USPP becsum spin fractions are left
exactly as the default computes them, so only the plane-wave magnetization SHAPE
differs between the two seeds. `restore()` puts the originals back.
"""

from __future__ import annotations

import gradwave.scf.loop as _loop
import gradwave.scf.uspp_loop as _uspp
from experiments.ao_density_seed.seed import d_localized_spin_densities

_ORIG = {}


def _nc_seed(system, nspin, start_from, start_mag, grid, vol):
    if start_from is not None or nspin == 1:
        return _ORIG["nc"](system, nspin, start_from, start_mag, grid, vol)
    nspecies = len(system.upfs)
    mags_at = _loop.resolve_atom_moments(
        start_mag, system.species_of_atom, nspecies, default=0.5)
    return d_localized_spin_densities(
        grid, system.positions, system.species_of_atom, system.upfs,
        system.n_electrons, mags_at, system.charges)


def _uspp_seed(system, grid, vol, dev, nspin, start_from, start_mag):
    if start_from is not None or nspin == 1:
        return _ORIG["uspp"](system, grid, vol, dev, nspin, start_from, start_mag)
    mags = _uspp._resolve_start_mag(start_mag, system.species_of_atom, len(system.paws))
    rho_s = d_localized_spin_densities(
        grid, system.positions, system.species_of_atom, system.paws,
        system.n_electrons, mags, system.charges)
    up = [(1.0 + m) / 2.0 for m in mags]
    dn = [(1.0 - m) / 2.0 for m in mags]
    return rho_s, [up, dn]


def install():
    if _ORIG:
        return
    _ORIG["nc"] = _loop._seed_density
    _ORIG["uspp"] = _uspp._seed_scf_density
    _loop._seed_density = _nc_seed
    _uspp._seed_scf_density = _uspp_seed


def restore():
    if not _ORIG:
        return
    _loop._seed_density = _ORIG["nc"]
    _uspp._seed_scf_density = _ORIG["uspp"]
    _ORIG.clear()
