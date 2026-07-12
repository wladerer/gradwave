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


@torch.no_grad()
def build_uspp_hubbard(system, manifolds: list[HubbardManifold], bk,
                       p_b: torch.Tensor) -> USPPHubbardData:
    """Assemble S|φ⟩ projectors on the padded batch.

    bk: core.batch.BatchedK for the system's spheres; p_b: phased KB
    projectors (nk, nprojβ, npw_max) at the current (fixed) positions."""
    device = system.positions.device
    vol = system.grid.volume
    man_by_sp = {m.species: m for m in manifolds}
    correlated = [(a, s) for a, s in enumerate(system.species_of_atom)
                  if s in man_by_sp]
    if not correlated:
        raise ValueError("no atoms match the requested Hubbard manifolds")
    # RAW PP_PSWFC orbitals (QE 'atomic' convention): a PAW pseudo-orbital's
    # PLAIN norm is deliberately ≠ 1 (Ni 3d: 0.588) — the S metric supplies
    # the rest (⟨φ|S|φ⟩ ≈ 1). Renormalizing the plain norm (the NC
    # manifold_radial convention) corrupts the projector amplitude.
    def _raw_rchi(sp):
        orbs = system.paws[sp].hubbard_orbitals(man_by_sp[sp].l)
        if not orbs:
            raise ValueError(
                f"species {sp}: no PP_PSWFC orbital with l={man_by_sp[sp].l}")
        return orbs[0].rchi

    rchi_by_sp = {sp: _raw_rchi(sp) for sp in man_by_sp}
    l_max = max(m.l for m in manifolds)

    sites, col = [], 0
    for a, s in correlated:
        m = man_by_sp[s]
        sites.append({"atom": a, "l": m.l, "u": m.u, "j": m.j,
                      "start": col, "dim": 2 * m.l + 1})
        col += 2 * m.l + 1
    nproj = col

    phi = torch.zeros(len(system.spheres), nproj, bk.npw_max, dtype=CDTYPE,
                      device=device)
    for ik, sph in enumerate(system.spheres):
        npw = sph.npw
        kpg = sph.kpg
        qmag = np.sqrt(sph.kpg2.cpu().numpy())
        y = ylm_all(l_max, kpg)
        f_by_sp = {}
        for sp, rchi in rchi_by_sp.items():
            p = system.paws[sp]
            n = p.msh  # QE truncates atomic-wfc integrals at msh (10 bohr),
            # like every local-channel radial table (init_tab_atwfc)
            f_by_sp[sp] = torch.as_tensor(
                sbt(man_by_sp[sp].l, (rchi * p.r)[:n], p.r[:n], p.rab[:n], qmag),
                dtype=RDTYPE, device=device)
        phase_arg = kpg @ system.positions.T  # (npw, na)
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg),
                                         -phase_arg))
        for a, _s in correlated:
            site = next(x for x in sites if x["atom"] == a)
            ll = site["l"]
            pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[ll]
            for mm in range(2 * ll + 1):
                phi[ik, site["start"] + mm, :npw] = (
                    pref * (f_by_sp[system.species_of_atom[a]]
                            * y[:, ll * ll + mm]).to(CDTYPE) * phases[:, a])

    # S-dress: S|φ⟩ = |φ⟩ + Σ_ij |β_i⟩ q_ij ⟨β_j|φ⟩ (padded slots stay zero
    # because both φ and the β projectors are masked)
    bphi = torch.einsum("kjg,kmg->kmj", p_b.conj(), phi)  # ⟨β_j|φ_m⟩
    q = system.q_full.to(CDTYPE)
    sphi = phi + torch.einsum("kmj,ij,kig->kmg", bphi, q, p_b)
    return USPPHubbardData(sphi=sphi, sites=sites, nproj=nproj)
