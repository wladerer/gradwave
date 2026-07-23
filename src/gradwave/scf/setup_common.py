"""System-construction blocks shared by setup_system (scf/loop.py) and
setup_uspp (scf/uspp_setup.py): space-group / magnetic-group discovery, the
coupled-axes grid hint, the symmetrizer + k-reduction dispatch, the
local-potential tables, the NLCC core density, and the nbands heuristic.
The grid builds themselves stay per-caller (the USPP dual grid has its own
cutoff logic)."""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.fftbox import g_to_r_box
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.kpoints import monkhorst_pack


def _unique_shells(vals: np.ndarray):
    uniq, inverse = np.unique(np.round(vals, 9), return_inverse=True)
    return uniq, inverse


def find_symmetry_groups(cell, positions, species_of_atom, symprec, magmoms):
    """(sym, mag_sym) for the structure: the space group (None when P1 —
    nothing to gain, keep the plain path), and with magmoms the magnetic
    (Shubnikov) group of that moment configuration, whose unitary halfgroup
    replaces sym. Callers gate on their own use_symmetry conditions."""
    from gradwave.symmetry import find_spacegroup

    frac = positions @ np.linalg.inv(cell)
    sym = find_spacegroup(cell, frac, list(species_of_atom), symprec=symprec)
    mag_sym = None
    if sym.n_ops <= 1:
        sym = None
    elif magmoms is not None:
        from gradwave.symmetry import magnetic_spacegroup

        mag_sym = magnetic_spacegroup(sym, magmoms, cell)
        sym = mag_sym.unitary
    return sym, mag_sym


def coupled_axes(sym, mag_sym):
    """equal_dims hint for build_fft_grid: equalize only symmetry-COUPLED
    axes (a slab's vacuum axis stays independent of the in-plane pair — a
    blanket cubic box would blow the slab grid up by the vacuum-to-in-plane
    ratio)."""
    if sym is None:
        return False
    from gradwave.symmetry import coupled_axis_groups

    return coupled_axis_groups(mag_sym.combined() if mag_sym is not None
                               else sym)


def build_symmetrizer_and_kpoints(grid, cell, kmesh, kshift, sym, mag_sym,
                                  time_reversal):
    """(rho_symmetrizer, kfrac, kw): the magnetic group folds k into the
    magnetic IBZ (anti-unitary g·T ops as −W⁻ᵀ) with a MagneticSymmetrizer;
    a plain space group uses RhoSymmetrizer + reduce_mesh; no symmetry falls
    back to the Monkhorst-Pack mesh."""
    if mag_sym is not None:
        from gradwave.symmetry import MagneticSymmetrizer, reduce_mesh_magnetic

        rho_symmetrizer = MagneticSymmetrizer(grid.shape, mag_sym, cell,
                                              dens_mask=grid.dens_mask)
        kfrac, kw = reduce_mesh_magnetic(kmesh, kshift, mag_sym)
    elif sym is not None:
        from gradwave.symmetry import RhoSymmetrizer, reduce_mesh

        rho_symmetrizer = RhoSymmetrizer(grid.shape, sym,
                                         dens_mask=grid.dens_mask)
        kfrac, kw = reduce_mesh(kmesh, kshift, sym,
                                time_reversal=time_reversal)
    else:
        rho_symmetrizer = None
        kfrac, kw = monkhorst_pack(kmesh, kshift, time_reversal=time_reversal)
    return rho_symmetrizer, kfrac, kw


def default_nbands(n_electrons: float) -> int:
    """20% headroom over the occupied count, at least 4 extra bands."""
    nocc = int(np.ceil(n_electrons / 2.0))
    return max(int(np.ceil(nocc * 1.2)), nocc + 4)


def build_vloc_tables(pseudos, uniq, inverse, shape, *, guard_single_shell):
    """Per-species local-potential tables on the dense box [eV·Å³], G=0 =
    alpha-Z. guard_single_shell: the NC setup skips the vloc_of_g call when
    the grid has a single |G| shell (empty slice); the USPP setup
    historically dropped that guard — each caller keeps its exact behavior."""
    from gradwave.pseudo.local import alpha_z, vloc_of_g

    tabs = []
    for p in pseudos:
        tab = np.empty_like(uniq)
        tab[0] = alpha_z(p)
        if not guard_single_shell or len(uniq) > 1:
            tab[1:] = vloc_of_g(p, uniq[1:])
        tabs.append(tab[inverse].reshape(shape))
    return torch.as_tensor(np.stack(tabs), dtype=RDTYPE)


def build_core_density(pseudos, species_of_atom, positions, grid, uniq,
                       inverse):
    """NLCC core density on the dense grid [e/Å³] (frozen; enters XC only),
    or None when no species carries one. NO clamp on the Gibbs oscillations
    of the sphere-truncated core: QE keeps them (its XC floors ρ pointwise,
    as does ours via to_au) and clamping shifts E_xc by several meV for
    sharp 3d cores."""
    if not any(p.core_rho is not None for p in pseudos):
        return None
    from gradwave.core.structure import structure_factors
    from gradwave.pseudo.atomic import core_density_of_q

    core_g = torch.zeros(grid.n_points, dtype=CDTYPE)
    pos_t = torch.as_tensor(np.asarray(positions), dtype=RDTYPE)
    for sp_i, p in enumerate(pseudos):
        tab = torch.as_tensor(core_density_of_q(p, uniq), dtype=RDTYPE)
        shell = tab[torch.as_tensor(inverse)]
        atoms = [a for a, sa in enumerate(species_of_atom) if sa == sp_i]
        if not atoms:
            continue
        sfac = structure_factors(pos_t[atoms], grid.g_cart).sum(dim=0).reshape(-1)
        core_g += sfac * shell.to(CDTYPE) / grid.volume
    core_g = torch.where(grid.dens_mask.reshape(-1), core_g,
                         torch.zeros_like(core_g))
    return g_to_r_box(core_g.reshape(grid.shape), real=True)
