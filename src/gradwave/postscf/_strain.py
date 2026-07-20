"""Shared strain-parameterization scaffolding for the stress modules.

``postscf/stress.py`` (norm-conserving) and ``postscf/paw_stress.py``
(USPP/PAW) build the same ε-differentiable geometry: the strained cell and
reciprocal basis, the density-sphere G-vectors rebuilt from integer Miller
labels, the strained local-pseudopotential and NLCC-core assemblies, and the
per-k strained projector columns. Those shared pieces live here; the
genuinely different physics (augmentation charges, one-center terms, the
S-orthogonality constraint) stays in the per-variant modules.

Note on floating-point identity: each helper reproduces the operation
sequence of the former inline copies, so single-species results are
bit-identical; ``local_pp_energy`` accumulates the per-species terms into
its own scalar before the caller adds it to the total, which reassociates
the sum at ~1 ulp for multi-species cells.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.constants import MINUS_I_POW as _MINUS_I_POW
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE


def box_millers(shape, device) -> torch.Tensor:
    """(N, 3) float64 integer Miller labels of the dense FFT box."""
    axes = [np.fft.fftfreq(n, d=1.0 / n).astype(np.float64) for n in shape]
    m = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return torch.as_tensor(m, dtype=torch.float64, device=device)


def strain_cell(grid, positions: torch.Tensor, eps: torch.Tensor):
    """The strained-cell geometry: (f_map, a_e, b_e, omega0, omega, pos_e).

    r → (1+ε)r, a_i → (1+ε)a_i, τ → (1+ε)τ, with Ω(ε) = |det a_e| kept
    positive and the ε=0 volume returned alongside.
    """
    dev = positions.device
    f_map = torch.eye(3, dtype=RDTYPE, device=dev) + eps
    a0 = torch.as_tensor(grid.cell, dtype=RDTYPE, device=dev)
    a_e = a0 @ f_map.T  # rows a_i → (1+ε) a_i
    b_e = 2.0 * math.pi * torch.linalg.inv(a_e).T
    omega = torch.linalg.det(a_e)
    omega = omega * torch.sign(omega.detach())
    pos_e = positions.detach() @ f_map.T
    return f_map, a_e, b_e, grid.volume, omega, pos_e


def strained_dens_sphere(grid, b_e: torch.Tensor, device):
    """Density-sphere G-vectors rebuilt from integer Miller labels.

    Returns (mask, m_box, g_sph, g2_sph, is_g0, q_sph, inv_g2): the flat
    density-sphere mask, the dense-box Miller labels, the strained sphere
    vectors/moduli, the G=0 flag and the G=0-safe |G| and 1/G².
    """
    mask = grid.dens_mask.reshape(-1)
    m_box = box_millers(grid.shape, device)
    m_sph = m_box[mask]
    g_sph = m_sph @ b_e
    g2_sph = (g_sph**2).sum(-1)
    is_g0 = g2_sph.detach() < 1e-12
    q_sph = torch.sqrt(torch.where(is_g0, torch.ones_like(g2_sph), g2_sph))
    q_sph = torch.where(is_g0, torch.zeros_like(q_sph), q_sph)
    g2_safe = torch.where(is_g0, torch.ones_like(g2_sph), g2_sph)
    inv_g2 = torch.where(is_g0, torch.zeros_like(g2_sph), 1.0 / g2_safe)
    return mask, m_box, g_sph, g2_sph, is_g0, q_sph, inv_g2


def strained_phases(g_sph: torch.Tensor, pos_e: torch.Tensor) -> torch.Tensor:
    """Structure-factor phases e^{−iG·τ} on the sphere, (nGm, na)."""
    phase_arg = g_sph @ pos_e.T
    return torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))


def strained_kpg(sph, b_e: torch.Tensor):
    """Strained (k+G) vectors of one k-sphere from integer Miller + k_frac."""
    kfrac = torch.as_tensor(sph.k_frac, dtype=RDTYPE, device=b_e.device)
    kpg = (sph.miller.to(RDTYPE) + kfrac) @ b_e  # (npw, 3)
    return kpg, (kpg**2).sum(-1)


def kinetic_band(c: torch.Tensor, kpg2: torch.Tensor) -> torch.Tensor:
    """Per-band kinetic energies Σ_G |c|² T_G on strained (k+G)², (nb,)."""
    return torch.einsum("bg,g->b", (c.real**2 + c.imag**2), HBAR2_2M * kpg2)


def local_pp_energy(tabs, species_of_atom, phases, rho_sph, q_sph, is_g0):
    """Strained local-pseudopotential energy Σ_G ρ*(G) S_sp(G) v_sp(G).

    ``rho_sph`` is the density on the sphere in the 1/Ω(ε) normalization
    (so no extra volume factor appears here); G=0 carries the alpha-Z term.
    """
    e_loc = torch.zeros((), dtype=q_sph.dtype, device=q_sph.device)
    for sp, tab in enumerate(tabs):
        atoms = [a for a, sa in enumerate(species_of_atom) if sa == sp]
        if not atoms:
            continue
        s_sp = phases[:, atoms].sum(dim=1)  # (nGm,)
        v = torch.zeros_like(q_sph)  # fresh buffer: index-assign is autograd-safe here
        v[~is_g0] = tab.vloc_of_g(q_sph[~is_g0])
        v[is_g0] = tab.alpha
        e_loc = e_loc + (rho_sph.conj() * s_sp * v.to(rho_sph.dtype)).sum().real
    return e_loc


def nlcc_core_strained(tabs, species_of_atom, phases, q_sph, omega, grid,
                       scatter) -> torch.Tensor:
    """Strained NLCC core density on the real grid.

    ``scatter`` indexes the flat FFT box for the sphere entries (a boolean
    mask or an index tensor — the two call sites' historical conventions,
    which address the same points in the same order).
    """
    dev = q_sph.device
    core = torch.zeros(q_sph.shape[0], dtype=CDTYPE, device=dev)
    for sp, tab in enumerate(tabs):
        if tab.core_g is None:
            continue
        atoms = [a for a, sa in enumerate(species_of_atom) if sa == sp]
        if not atoms:
            continue
        f_core = tab.core_of_g(q_sph)
        core = core + phases[:, atoms].sum(dim=1) * f_core.to(CDTYPE) / omega.to(CDTYPE)
    core_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
    core_box[scatter] = core
    return torch.fft.ifftn(core_box.reshape(grid.shape) * grid.n_points,
                           dim=(-3, -2, -1)).real


def strained_projector_cols(tabs, species_of_atom, atom_index, lmax,
                            kpg, kpg2, omega, pos_e) -> torch.Tensor:
    """Strained KB/USPP projector matrix (nproj_tot, npw) at one k.

    Radial form factors through the differentiable SBT at the strained
    |k+G|, Ylm at the strained directions, the 1/√Ω(ε) normalization, and
    the e^{−i(k+G)·τ(ε)} phases; column ordering matches ProjectorData.
    """
    q_k = torch.sqrt(kpg2.clamp_min(1e-30))
    q_k = torch.where(kpg2.detach() < 1e-24, torch.zeros_like(q_k), q_k)
    y = ylm_all(lmax, kpg)
    pref = 4.0 * math.pi / torch.sqrt(omega)
    cols = []
    for sp in species_of_atom:
        tab = tabs[sp]
        for i, ell in enumerate(tab.beta_l):
            f = tab.beta_of_g(i, q_k)
            for m_col in range(2 * ell + 1):
                cols.append(
                    (pref * f * y[:, ell * ell + m_col]).to(CDTYPE)
                    * _MINUS_I_POW[ell]
                )
    p = torch.stack(cols, dim=0)  # (nproj_tot, npw), matches pd ordering
    parg = kpg @ pos_e.T  # (npw, na)
    ph = torch.exp(torch.complex(torch.zeros_like(parg), -parg))
    return p * ph[:, atom_index].T


def ewald_strained(pos_e, charges, a_e, b_e, omega, cell0) -> torch.Tensor:
    """ewald_energy with the cell on the autograd graph. η and the integer
    image/G-vector sets come from the unstrained cell (the excluded boundary
    terms are erfc(8)-suppressed, so their ε-derivative is negligible)."""
    from gradwave.core.energies.ewald import _ACC, _g_vectors, _image_vectors

    cell0 = np.asarray(cell0, dtype=np.float64)
    omega0 = abs(np.linalg.det(cell0))
    eta = (math.pi / omega0 ** (1.0 / 3.0)) ** 2
    sqrt_eta = math.sqrt(eta)
    rcut = _ACC / sqrt_eta
    gcut = 2.0 * sqrt_eta * _ACC

    dev = pos_e.device
    rdt = torch.float64
    # integer labels of the ε=0 sets
    n_img = np.round(_image_vectors(cell0, rcut) @ np.linalg.inv(cell0)).astype(np.int64)
    b0 = 2.0 * math.pi * np.linalg.inv(cell0).T
    m_g = np.round(_g_vectors(cell0, gcut) @ np.linalg.inv(b0)).astype(np.int64)

    images = torch.as_tensor(n_img, dtype=rdt, device=dev) @ a_e
    gvecs = torch.as_tensor(m_g, dtype=rdt, device=dev) @ b_e
    z = charges.to(rdt)

    d = pos_e[:, None, None, :] - pos_e[None, :, None, :] + images[None, None, :, :]
    r = torch.linalg.norm(d, dim=-1)
    na = r.shape[0]
    img0 = torch.as_tensor((np.abs(n_img).sum(axis=1) == 0), device=dev)
    self_pair = torch.eye(na, dtype=torch.bool, device=dev)[:, :, None] & img0[None, None, :]
    r_safe = torch.where(self_pair, torch.ones_like(r), r)
    pair = torch.erfc(sqrt_eta * r_safe) / r_safe
    pair = torch.where(self_pair, torch.zeros_like(pair), pair)
    e_real = 0.5 * E2 * torch.einsum("a,b,abr->", z, z, pair)

    g2 = (gvecs**2).sum(-1)
    phase = pos_e @ gvecs.T
    s_re = (z[:, None] * torch.cos(phase)).sum(0)
    s_im = (z[:, None] * torch.sin(phase)).sum(0)
    e_recip = (2.0 * math.pi * E2 / omega) * (
        (s_re**2 + s_im**2) * torch.exp(-g2 / (4.0 * eta)) / g2
    ).sum()

    e_self = -E2 * sqrt_eta / math.sqrt(math.pi) * (z**2).sum()
    e_bg = -math.pi * E2 / (2.0 * eta * omega) * z.sum() ** 2
    return e_real + e_recip + e_self + e_bg
