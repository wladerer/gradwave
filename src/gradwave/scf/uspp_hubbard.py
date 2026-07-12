"""DFT+U for USPP/PAW — S-metric occupation matrices and V_U (Dudarev).

With ultrasoft/PAW the atomic-orbital projections carry the S operator
(QE's U_projection_type='atomic' for USPP): every ⟨φ|ψ⟩ becomes ⟨φ|S|ψ⟩.
Since S|φ_m⟩ = |φ_m⟩ + Σ_ij |β_i⟩ q_ij ⟨β_j|φ_m⟩ can be built ONCE per SCF
(positions fixed), the whole correction reduces to the NC structure with
S-dressed projectors:

    n^{Iσ}_{mm'} = Σ_{kv} f ⟨Sφ_m|ψ⟩⟨ψ|Sφ_{m'}⟩
    E_U = Σ_{Iσ} (U−J)/2 Tr[n(1−n)]
    V_U = Σ |Sφ_m⟩ (U−J)(½δ−n)_{mm'} ⟨Sφ_{m'}|

so occupation_matrices / hubbard_energy / hubbard_dmatrix from core/hubbard
are reused verbatim on the padded batched (nk, nprojU, npw_max) layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.hubbard import _MINUS_I_POW, HubbardManifold
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.radial import sbt

__all__ = ["HubbardManifold", "build_uspp_hubbard"]


@dataclass
class USPPHubbardData:
    """S-dressed atomic-orbital projectors, padded over k (setup product)."""

    sphi: torch.Tensor  # (nk, nprojU, npw_max) — S|φ⟩, phased at positions
    sites: list  # per correlated atom: {atom, l, u, j, start, dim}
    nproj: int


def hubbard_sites(system, manifolds: list[HubbardManifold]) -> list:
    """Per-correlated-atom site descriptors {atom, l, u, j, start, dim}."""
    man_by_sp = {m.species: m for m in manifolds}
    correlated = [(a, s) for a, s in enumerate(system.species_of_atom)
                  if s in man_by_sp]
    if not correlated:
        raise ValueError("no atoms match the requested Hubbard manifolds")
    sites, col = [], 0
    for a, s in correlated:
        m = man_by_sp[s]
        sites.append({"atom": a, "l": m.l, "u": m.u, "j": m.j,
                      "start": col, "dim": 2 * m.l + 1})
        col += 2 * m.l + 1
    return sites


@torch.no_grad()
def phi_free_per_k(system, sites: list) -> list[torch.Tensor]:
    """Per-k PHASE-FREE atomic-orbital projector factors (nprojU, npw_k) —
    the position-independent product; multiply by e^{−i(k+G)·τ} per column
    atom for the full projector (the same split as the KB ProjectorData).

    Conventions (each ~100 meV when wrong, both from QE): RAW PP_PSWFC
    orbitals (a PAW pseudo-orbital's PLAIN norm is deliberately ≠ 1 — Ni 3d:
    0.588 — the S metric supplies the rest; renormalizing corrupts the
    amplitude), and radial integrals truncated at msh = 10 bohr
    (init_tab_atwfc; psl meshes run to 53 Å and the oscillating SBT tail
    pollutes the form factors at finite q)."""
    device = system.positions.device
    vol = system.grid.volume
    site_by_atom = {s["atom"]: s for s in sites}
    species_used = sorted({system.species_of_atom[a] for a in site_by_atom})
    l_by_sp = {sp: next(s["l"] for s in sites
                        if system.species_of_atom[s["atom"]] == sp)
               for sp in species_used}

    def _raw_rchi(sp):
        orbs = system.paws[sp].hubbard_orbitals(l_by_sp[sp])
        if not orbs:
            raise ValueError(
                f"species {sp}: no PP_PSWFC orbital with l={l_by_sp[sp]}")
        return orbs[0].rchi

    rchi_by_sp = {sp: _raw_rchi(sp) for sp in species_used}
    l_max = max(s["l"] for s in sites)
    nproj = sum(s["dim"] for s in sites)

    out = []
    for sph in system.spheres:
        qmag = np.sqrt(sph.kpg2.cpu().numpy())
        y = ylm_all(l_max, sph.kpg)
        f_by_sp = {}
        for sp, rchi in rchi_by_sp.items():
            p = system.paws[sp]
            n = p.msh
            f_by_sp[sp] = torch.as_tensor(
                sbt(l_by_sp[sp], (rchi * p.r)[:n], p.r[:n], p.rab[:n], qmag),
                dtype=RDTYPE, device=device)
        phi_k = torch.zeros(nproj, sph.npw, dtype=CDTYPE, device=device)
        for site in sites:
            ll = site["l"]
            sp = system.species_of_atom[site["atom"]]
            pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[ll]
            for mm in range(2 * ll + 1):
                phi_k[site["start"] + mm] = pref * (
                    f_by_sp[sp] * y[:, ll * ll + mm]).to(CDTYPE)
        out.append(phi_k)
    return out


def atom_of_col(sites: list) -> torch.Tensor:
    return torch.tensor([s["atom"] for s in sites for _ in range(s["dim"])],
                        dtype=torch.int64)


@torch.no_grad()
def build_uspp_hubbard(system, manifolds: list[HubbardManifold], bk,
                       p_b: torch.Tensor) -> USPPHubbardData:
    """Assemble S|φ⟩ projectors on the padded batch.

    bk: core.batch.BatchedK for the system's spheres; p_b: phased KB
    projectors (nk, nprojβ, npw_max) at the current (fixed) positions."""
    device = system.positions.device
    sites = hubbard_sites(system, manifolds)
    nproj = sum(s["dim"] for s in sites)
    phi_free = phi_free_per_k(system, sites)
    acol = atom_of_col(sites).to(device)

    phi = torch.zeros(len(system.spheres), nproj, bk.npw_max, dtype=CDTYPE,
                      device=device)
    for ik, sph in enumerate(system.spheres):
        phase_arg = sph.kpg @ system.positions.T  # (npw, na)
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg),
                                         -phase_arg))
        phi[ik, :, :sph.npw] = phi_free[ik] * phases[:, acol].T

    # S-dress: S|φ⟩ = |φ⟩ + Σ_ij |β_i⟩ q_ij ⟨β_j|φ⟩ (padded slots stay zero
    # because both φ and the β projectors are masked)
    bphi = torch.einsum("kjg,kmg->kmj", p_b.conj(), phi)  # ⟨β_j|φ_m⟩
    q = system.q_full.to(CDTYPE)
    sphi = phi + torch.einsum("kmj,ij,kig->kmg", bphi, q, p_b)
    return USPPHubbardData(sphi=sphi, sites=sites, nproj=nproj)


def hubbard_e_channel(sites, phi_free, q_full, pos, spheres, projs, coeffs,
                      becps, occ_row, kweights, occ_scale: float):
    """E_U of ONE spin channel as an in-graph function of positions.

    n(τ) is built from ⟨Sφ(τ)|ψ⟩ with the given (detached) coefficients —
    autograd then carries BOTH position chains: the φ phases and the β
    phases inside the S-dressing. occ_scale = 0.5 for nspin=1 (the [0,2]
    channel splits in two); the caller doubles the returned energy there."""
    acol = atom_of_col(sites).to(pos.device)
    q = q_full.to(CDTYPE)
    nproj = sum(s["dim"] for s in sites)
    n_full = torch.zeros(nproj, nproj, dtype=CDTYPE, device=pos.device)
    for ik, sph in enumerate(spheres):
        pharg = sph.kpg @ pos.T  # (npw, na)
        ph = torch.exp(torch.complex(torch.zeros_like(pharg), -pharg))
        phik = phi_free[ik] * ph[:, acol].T  # (nprojU, npw)
        povl = torch.einsum("bg,mg->bm", coeffs[ik], phik.conj())
        bphi = torch.einsum("mg,ig->im", phik.conj(), projs[ik])  # ⟨φ_m|β_i⟩
        sproj = povl + torch.einsum("im,ij,bj->bm", bphi, q, becps[ik])
        w = (kweights[ik] * occ_row[ik] * occ_scale).to(CDTYPE)
        n_full = n_full + torch.einsum("b,bm,bn->mn", w, sproj, sproj.conj())
    e = torch.zeros((), dtype=RDTYPE, device=pos.device)
    for s in sites:
        st, dim = s["start"], s["dim"]
        blk = n_full[st:st + dim, st:st + dim]
        blk = 0.5 * (blk + blk.conj().T)
        uj = s["u"] - s["j"]
        e = e + 0.5 * uj * (torch.diagonal(blk).sum()
                            - torch.diagonal(blk @ blk).sum()).real
    return e
