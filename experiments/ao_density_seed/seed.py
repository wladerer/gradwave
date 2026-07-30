"""Spin-resolved atomic density seed (experiment).

The default nspin=2 seed (scf/loop.py::_seed_density, scf/uspp_loop.py::
_seed_scf_density) splits the total superposition-of-atomic-densities into up
and down by a uniform per-atom factor (1+/-m)/2. The magnetization density it
produces is therefore m(r) = m_atom * rho_atom(r), proportional to the FULL
atomic valence density, including the diffuse s/p tail.

The physical spin density of a 3d transition metal is localized in the d shell.
This module builds an alternative seed whose magnetization is shaped by the
PSWFC d-orbital density |R_3d(r)|^2 instead, integrated to the SAME per-atom
moment m_atom * Z_val so that only the SHAPE differs from the default. The
total charge density is left as the default SAD.

The builder is installed by monkeypatching, never by changing a default.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.fftbox import g_to_r_box
from gradwave.dtypes import CDTYPE
from gradwave.pseudo.local import _msh
from gradwave.pseudo.radial import sbt


def _orbitals(upf):
    """AtomicOrbital PSWFC tuple (has .l/.occupation/.rchi) across NC and PAW.
    On PAW .pswfc holds partial waves without occupation, so the AtomicOrbital
    set lives in .chi; on NC it is .pswfc."""
    chi = getattr(upf, "chi", ())
    if chi:
        return chi
    return getattr(upf, "pswfc", ())


def orbital_density_of_q(upf, l: int, q: np.ndarray) -> np.ndarray | None:
    """l=0 form factor of the (2l+1)-summed density of the PSWFC shell with
    angular momentum `l`, normalized so rho_hat(0) = 1. None if absent.

    rchi = r*R_nl is stored with int (r R)^2 dr = 1, and the spherically
    summed 3D orbital density integrates the radial charge 4*pi*r^2 n(r) =
    (r R)^2 to 1, so the l=0 transform of (r R)^2 is exactly what is needed."""
    orbs = [o for o in _orbitals(upf) if o.l == l]
    if not orbs:
        return None
    # highest-occupation shell of this l (the valence d, not a semicore repeat)
    o = max(orbs, key=lambda w: w.occupation)
    n = _msh(upf)
    g = np.asarray(o.rchi[:n], dtype=np.float64) ** 2
    return sbt(0, g, upf.r[:n], upf.rab[:n], np.asarray(q, dtype=np.float64))


def _mag_density(grid, positions, species_of_atom, upfs, per_atom_moment, l=2):
    """m(r) [muB/A^3] on the dense grid: per-atom d-shell orbital density scaled
    to per_atom_moment[a]. Not rescaled or clamped (a magnetization, signed)."""
    device = grid.g2.device
    g = torch.sqrt(grid.g2).reshape(-1).cpu().numpy()
    uniq, inverse = np.unique(np.round(g, 9), return_inverse=True)
    vol = grid.volume
    pos = positions.detach()
    m_g = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
    inv_t = torch.as_tensor(inverse, device=device)
    gvec = grid.g_cart.reshape(-1, 3)
    for s, upf in enumerate(upfs):
        tab = orbital_density_of_q(upf, l, uniq)
        if tab is None:
            continue  # no d shell -> no localized magnetization for this species
        shell = torch.as_tensor(tab, dtype=torch.float64, device=device)[inv_t]
        atoms = [a for a, sa in enumerate(species_of_atom) if sa == s]
        if not atoms:
            continue
        phase = gvec @ pos[atoms].T
        sfac_a = torch.exp(torch.complex(torch.zeros_like(phase), -phase))
        w = torch.tensor([float(per_atom_moment[a]) for a in atoms],
                         dtype=sfac_a.real.dtype, device=device)
        sfac = (sfac_a * w[None, :]).sum(dim=1)
        m_g += sfac * shell.to(CDTYPE) / vol
    m_g = torch.where(grid.dens_mask.reshape(-1), m_g, torch.zeros_like(m_g))
    return g_to_r_box(m_g.reshape(grid.shape), real=True)


def d_localized_spin_densities(grid, positions, species_of_atom, upfs,
                               n_electrons, mags_at, charges):
    """[rho_up, rho_dn] with total = default SAD and magnetization shaped by the
    d shell, integrated per atom to mags_at[a]*charges[a] (the default's own
    seeded per-atom moment), so only the magnetization SHAPE differs.

    Falls back element-wise to the uniform split for species without a d shell
    (the mag density simply omits their contribution)."""
    from gradwave.scf.guess import sad_density

    rho_tot = sad_density(grid, positions, species_of_atom, upfs, n_electrons)
    per_atom_moment = [mags_at[a] * float(charges[a]) for a in range(len(species_of_atom))]
    m = _mag_density(grid, positions, species_of_atom, upfs, per_atom_moment, l=2)
    up = 0.5 * (rho_tot + m)
    dn = 0.5 * (rho_tot - m)
    up = torch.clamp(up, min=1e-12)
    dn = torch.clamp(dn, min=1e-12)
    return [up, dn]
