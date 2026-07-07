"""Spin-orbit (j-resolved) Kleinman–Bylander projectors for spinor SCF.

Fully-relativistic UPFs carry projectors per (l, j = l ± ½). Each expands
into 2j+1 spinor projectors built from spin spherical harmonics:

    |l, j=l+½, mj⟩ = √((l+mj+½)/(2l+1)) Y_l^{mj−½} χ↑ + √((l−mj+½)/(2l+1)) Y_l^{mj+½} χ↓
    |l, j=l−½, mj⟩ = −√((l−mj+½)/(2l+1)) Y_l^{mj−½} χ↑ + √((l+mj+½)/(2l+1)) Y_l^{mj+½} χ↓

(global signs per channel cancel in the D-contraction; the RELATIVE ↑/↓
sign is the physics). Complex harmonics come from our verified real ones:

    Y_l^0 = Z_{l0};  m>0:  Y_l^m = (−1)^m (Z_{l,m} + i Z_{l,−m})/√2,
                     Y_l^{−m} =        (Z_{l,m} − i Z_{l,−m})/√2

The assembled projectors live directly on the DOUBLED plane-wave axis of
the spinor code: q (nk, nproj_so, 2·npw_max); the nonlocal apply/energy is
then structurally identical to the scalar case. D_ij couples channels with
equal (l, j) at equal mj.
"""

from __future__ import annotations

import math

import torch

from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE

_MINUS_I_POW = [1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j]


def complex_ylm(lmax: int, g: torch.Tensor) -> torch.Tensor:
    """Y_l^m (Condon–Shortley) for l ≤ lmax; index l² + (m + l). (..., (lmax+1)²)."""
    z = ylm_all(lmax, g)  # real, ordered (l,0),(l,+1),(l,−1),(l,+2),(l,−2)...
    out = torch.zeros(*z.shape, dtype=CDTYPE, device=z.device)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for l in range(lmax + 1):
        base_r = l * l
        base_c = l * l + l  # complex index of m = 0
        out[..., base_c] = z[..., base_r].to(CDTYPE)
        for m in range(1, l + 1):
            zc = z[..., base_r + 2 * m - 1]
            zs = z[..., base_r + 2 * m]
            out[..., base_c + m] = ((-1.0) ** m) * inv_sqrt2 * torch.complex(zc, zs)
            out[..., base_c - m] = inv_sqrt2 * torch.complex(zc, -zs)
    return out


def _cg(l: int, j: float, mj: float):
    """(c_up, m_up, c_dn, m_dn) Clebsch–Gordan factors; None m if |m| > l."""
    up_m, dn_m = mj - 0.5, mj + 0.5
    a = math.sqrt((l + mj + 0.5) / (2 * l + 1))
    b = math.sqrt((l - mj + 0.5) / (2 * l + 1))
    if abs(j - (l + 0.5)) < 1e-8:
        c_up, c_dn = a, b
    else:
        c_up, c_dn = -b, a
    m_up = int(round(up_m)) if abs(up_m) <= l else None
    m_dn = int(round(dn_m)) if abs(dn_m) <= l else None
    return c_up, m_up, c_dn, m_dn


def build_so_projectors(bk, system, so_tables=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(q (nk, nproj_so, 2·npw_max), dij_so) for fully-relativistic pseudos.

    so_tables: per-species (nk, nchan, npw_max) F_i(|k+G|); defaults to
    system.so_beta_tables (the SCF mesh). Pass fresh tables for band paths.
    """
    if so_tables is None:
        so_tables = system.so_beta_tables
    device = bk.mask.device
    nk, m_pw = bk.nk, bk.npw_max
    lmax = max(b.l for u in system.upfs for b in u.betas)
    pos = system.positions

    # phases per atom: e^{−i(k+G)·τ_a}, (nk, npw, na)
    phase_arg = torch.einsum("kgi,ai->kga", bk.kpg, pos)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

    ylm_c = complex_ylm(lmax, bk.kpg)  # (nk, npw, (lmax+1)²)

    cols, col_meta = [], []
    for a, sp in enumerate(system.species_of_atom):
        upf = system.upfs[sp]
        for i, beta in enumerate(upf.betas):
            l, j = beta.l, beta.j
            f_tab = so_tables[sp][:, i, :]  # (nk, npw_max)
            pref = (4.0 * math.pi) * _MINUS_I_POW[l]
            n_mj = int(round(2 * j + 1))
            for imj in range(n_mj):
                mj = -j + imj
                c_up, m_up, c_dn, m_dn = _cg(l, j, mj)
                base = pref * f_tab.to(CDTYPE) * phases[:, :, a]
                qu = torch.zeros(nk, m_pw, dtype=CDTYPE, device=device)
                qd = torch.zeros(nk, m_pw, dtype=CDTYPE, device=device)
                if m_up is not None:
                    qu = base * (c_up * ylm_c[..., l * l + l + m_up])
                if m_dn is not None:
                    qd = base * (c_dn * ylm_c[..., l * l + l + m_dn])
                cols.append(torch.cat([qu, qd], dim=-1))
                col_meta.append((a, sp, i, l, j, mj))

    vol_norm = 1.0 / math.sqrt(system.grid.volume)
    q = torch.stack(cols, dim=1) * vol_norm  # (nk, nproj_so, 2npw)

    # D matrix: couples equal (atom, l, j, mj) across radial channels i, i′
    n = len(col_meta)
    dij_so = torch.zeros(n, n, dtype=torch.float64, device=device)
    for p_i, (a1, sp1, i1, l1, j1, mj1) in enumerate(col_meta):
        for p_j, (a2, _sp2, i2, l2, j2, mj2) in enumerate(col_meta):
            if a1 == a2 and l1 == l2 and abs(j1 - j2) < 1e-8 and abs(mj1 - mj2) < 1e-8:
                dij_so[p_i, p_j] = float(system.upfs[sp1].dij[i1, i2])
    return q, dij_so
