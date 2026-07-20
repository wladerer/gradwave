"""Kleinman–Bylander projectors and the Hamiltonian apply (Layer A).

Projector for atom a, UPF channel i (angular momentum l_i), magnetic m:

    p_{a,i,m}(k+G) = (4π/√Ω) · (−i)^{l_i} · Y_{l_i m}(k+G^) · F_i(|k+G|) · e^{−i(k+G)·τ_a}

F_i tables come from pseudo/kb.py (setup layer, frozen per geometry's |k+G|
values — NOT differentiable in the cell; positions enter only through the
phase, which is built here differentiably).

    H|ψ⟩ = T|ψ⟩ + FFT⁻¹[V_eff(r)·ψ(r)] + Σ p (D ⟨p|ψ⟩)

Phase/conjugation convention: becp_n,i = ⟨p_i|ψ_n⟩ = Σ_G p_i*(G) c_n(G);
E_NL = Σ D_ij becp*_i becp_j. The (−i)^l and e^{−i(k+G)τ} phases cancel in
same-atom same-l contractions but their RELATIVE phases across G are what
make forces right — tested on low-symmetry geometries (M2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gradwave.constants import HBAR2_2M
from gradwave.constants import MINUS_I_POW as _MINUS_I_POW
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE


@dataclass
class ProjectorData:
    """Per-k frozen radial/angular data (setup layer product; no positions)."""

    # per projector column: which atom, which species-channel, l, m
    atom_index: torch.Tensor  # (nproj_tot,) int64
    f_ylm_phase_free: torch.Tensor  # (nproj_tot, npw) complex: (4π/√Ω)(−i)^l Y_lm F(|k+G|)
    kpg: torch.Tensor  # (npw, 3)
    dij_full: torch.Tensor  # (nproj_tot, nproj_tot) real [eV], block-diag over atoms


def build_projector_data(
    sphere,
    species_of_atom: list[int],
    beta_tables: list[torch.Tensor],  # per species: (nchan, npw) F_i(|k+G|)
    beta_ls: list[list[int]],  # per species: l of each channel
    dij_species: list[torch.Tensor],  # per species: (nchan, nchan) [eV]
    volume: float,
) -> ProjectorData:
    """Assemble the position-independent projector factors for one k-point."""
    lmax = max((max(ls) for ls in beta_ls if ls), default=0)
    y = ylm_all(lmax, sphere.kpg)  # (npw, (lmax+1)²)

    def lm_index(l: int, m_col: int) -> int:
        return l * l + m_col  # our ylm ordering is dense in l²..(l+1)²-1

    cols = []
    atom_idx = []
    blocks = []
    for a, s in enumerate(species_of_atom):
        ls = beta_ls[s]
        # m-expanded D block for this atom: D_(i,m),(j,m') = D_ij δ_ll' δ_mm'
        nchan = len(ls)
        cols_a = []
        for i in range(nchan):
            l = ls[i]
            for m_col in range(2 * l + 1):
                f = beta_tables[s][i]  # (npw,)
                yl = y[:, lm_index(l, m_col)]
                pref = (4.0 * math.pi / math.sqrt(volume)) * _MINUS_I_POW[l]
                cols_a.append(pref * (f * yl).to(CDTYPE))
                atom_idx.append(a)
        cols += cols_a
        # expanded block
        nb_a = len(cols_a)
        block = torch.zeros((nb_a, nb_a), dtype=RDTYPE, device=y.device)
        row = 0
        offs = []
        for i in range(nchan):
            offs.append(row)
            row += 2 * ls[i] + 1
        for i in range(nchan):
            for j in range(nchan):
                if ls[i] != ls[j]:
                    continue
                d = dij_species[s][i, j]
                for m_col in range(2 * ls[i] + 1):
                    block[offs[i] + m_col, offs[j] + m_col] = d
        blocks.append(block)

    dij_full = (
        torch.block_diag(*blocks) if blocks
        else torch.zeros((0, 0), dtype=RDTYPE, device=y.device)
    )
    return ProjectorData(
        atom_index=torch.tensor(atom_idx, dtype=torch.int64, device=y.device),
        f_ylm_phase_free=torch.stack(cols, dim=0) if cols
        else torch.zeros((0, sphere.npw), dtype=CDTYPE, device=y.device),
        kpg=sphere.kpg,
        dij_full=dij_full,
    )


def projectors(pd: ProjectorData, positions: torch.Tensor) -> torch.Tensor:
    """Full projectors p (nproj_tot, npw), differentiable in positions."""
    phase_arg = pd.kpg @ positions.T  # (npw, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))  # e^{−i(k+G)·τ}
    return pd.f_ylm_phase_free * phases[:, pd.atom_index].T


def becp(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """⟨p_i|ψ_n⟩ = Σ_G p_i*(G) c_n(G). p (np, npw), c (nb, npw) → (nb, np)."""
    return c @ p.conj().T


class HamiltonianK:
    """H apply at one k-point for fixed V_eff(r) and fixed positions (solver use).

    Everything here is plain tensor math (works under no_grad for Davidson,
    or traced for M4 Sternheimer solves).
    """

    def __init__(self, sphere, shape, v_eff_r: torch.Tensor, pd: ProjectorData, p: torch.Tensor):
        self.sphere = sphere
        self.shape = shape
        self.v_eff_r = v_eff_r  # (n1,n2,n3) real [eV]
        self.pd = pd
        self.p = p  # (nproj_tot, npw)
        self.t = HBAR2_2M * sphere.kpg2  # (npw,)

    def apply(self, c: torch.Tensor) -> torch.Tensor:
        """H c for a block c (nb, npw)."""
        out = self.t * c
        psi = g_to_r(c, self.sphere.flat_idx, self.shape)
        v_psi = psi * self.v_eff_r
        out = out + box_to_sphere(r_to_g(v_psi), self.sphere.flat_idx)
        if self.p.shape[0]:
            b = becp(self.p, c)  # (nb, np)
            out = out + (b.to(self.p.dtype) @ self.pd.dij_full.to(self.p.dtype)) @ self.p
        return out
