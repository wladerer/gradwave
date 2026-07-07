"""DFT+U (Dudarev, rotationally invariant) — projectors, occupation matrix,
and the corrective energy/potential.

The correlated manifold of atom I (angular momentum l) is spanned by the
pseudo-atomic orbitals φ^I_m from PP_PSWFC. Their plane-wave projectors reuse
the exact Kleinman–Bylander structure (core/hamiltonian.py):

    φ_{I,m}(k+G) = (4π/√Ω)·(−i)^l·Y_{lm}(k+G^)·F(|k+G|)·e^{−i(k+G)·τ_I}

with F(q) = ∫ R_nl(r) j_l(qr) r² dr = sbt(l, (r·R)·r, r, rab, q).

Occupation matrix (per spin σ):  n^{Iσ}_{mm'} = Σ_{kv} f_{kvσ} ⟨φ_m|ψ⟩⟨ψ|φ_{m'}⟩.

Dudarev energy/potential (U_eff = U − J):
    E_U = Σ_{I,σ} (U_eff/2) Tr[ n^{Iσ}(1 − n^{Iσ}) ]
    V_U = Σ_{I,m,m'} |φ_m⟩ (U_eff)(½δ_{mm'} − n^{Iσ}_{mm'}) ⟨φ_{m'}|

V_U is a nonlocal projector operator with a density-dependent D-matrix — it
plugs into the same becp/D-contraction the KB nonlocal term already uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.radial import sbt

_MINUS_I_POW = [1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j]  # (−i)^l


@dataclass(frozen=True)
class HubbardManifold:
    """A +U correction on every atom of `species` in the (l) manifold."""

    species: int
    l: int
    u: float  # Hubbard U [eV]
    j: float = 0.0  # Hund J [eV]; enters Dudarev only as U_eff = U − J


@dataclass
class HubbardData:
    """Position-independent Hubbard projector factors (setup product).

    The atomic-orbital projectors factor as q_free·e^{−i(k+G)·τ}; keeping the
    phase separate lets positions flow through autograd for +U forces, exactly
    like the KB ProjectorData/projectors() split."""

    q_free: torch.Tensor  # (nk, nproj, npw_max) — (4π/√Ω)(−i)^l Y_lm F, no phase
    atom_of_col: torch.Tensor  # (nproj,) int64 — atom index per projector column
    kpg: torch.Tensor  # (nk, npw_max, 3) k+G for the phase
    sites: list  # per correlated atom: {atom, l, u, j, start, dim}
    nproj: int

    @property
    def n_sites(self) -> int:
        return len(self.sites)


def hubbard_projectors(hub: HubbardData, positions: torch.Tensor) -> torch.Tensor:
    """Phased projectors q (nk, nproj, npw_max), differentiable in positions."""
    phase_arg = torch.einsum("kgd,ad->kga", hub.kpg, positions)  # (nk, npw, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))
    return hub.q_free * phases[:, :, hub.atom_of_col].permute(0, 2, 1)


def manifold_radial(upf, l: int) -> np.ndarray:
    """Effective scalar r·R_l(r) for the l manifold, renormalized to ∫=1.
    Fully-relativistic pseudos split l into j = l ± ½; combine them by the
    (2j+1) degeneracy weight to recover the scalar-relativistic orbital."""
    orbs = upf.hubbard_orbitals(l)
    if not orbs:
        raise ValueError(f"{upf.element}: no PP_PSWFC orbital with l={l}")
    if len(orbs) == 1 or orbs[0].j is None:
        rchi = orbs[0].rchi.copy()
    else:
        w = np.array([2 * o.j + 1 for o in orbs], dtype=np.float64)
        w /= w.sum()
        rchi = sum(wi * o.rchi for wi, o in zip(w, orbs, strict=True))
    norm = np.sqrt(np.sum(rchi**2 * upf.rab))
    return rchi / norm


@torch.no_grad()
def build_hubbard_projectors(system, manifolds: list[HubbardManifold]) -> HubbardData:
    """Assemble batched atomic-orbital projectors (phases at system.positions)."""
    device = system.positions.device
    bk = system.batch
    npw_max = bk.npw_max
    vol = system.grid.volume
    man_by_sp = {m.species: m for m in manifolds}
    correlated = [(a, s) for a, s in enumerate(system.species_of_atom) if s in man_by_sp]
    if not correlated:
        raise ValueError("no atoms match the requested Hubbard manifolds")
    rchi_by_sp = {sp: manifold_radial(system.upfs[sp], man_by_sp[sp].l) for sp in man_by_sp}
    l_max = max(m.l for m in manifolds)

    sites, col = [], 0
    for a, s in correlated:
        m = man_by_sp[s]
        sites.append({"atom": a, "l": m.l, "u": m.u, "j": m.j,
                      "start": col, "dim": 2 * m.l + 1})
        col += 2 * m.l + 1
    nproj = col

    q_free = torch.zeros(len(system.spheres), nproj, npw_max, dtype=CDTYPE, device=device)
    for ik, sph in enumerate(system.spheres):
        npw = sph.npw
        kpg = sph.kpg.to(device)
        qmag = np.sqrt(sph.kpg2.cpu().numpy())
        y = ylm_all(l_max, kpg)  # (npw, (l_max+1)²)
        f_by_sp = {}
        for sp, rchi in rchi_by_sp.items():
            u = system.upfs[sp]
            f_by_sp[sp] = torch.as_tensor(
                sbt(man_by_sp[sp].l, rchi * u.r, u.r, u.rab, qmag),
                dtype=RDTYPE, device=device)
        for a, s in correlated:
            site = next(x for x in sites if x["atom"] == a)
            l = site["l"]
            pref = (4.0 * math.pi / math.sqrt(vol)) * _MINUS_I_POW[l]
            for mm in range(2 * l + 1):
                yl = y[:, l * l + mm]
                q_free[ik, site["start"] + mm, :npw] = pref * (f_by_sp[s] * yl).to(CDTYPE)
    return HubbardData(q_free=q_free,
                       atom_of_col=torch.tensor([x["atom"] for x in sites
                                                 for _ in range(x["dim"])],
                                                dtype=torch.int64, device=device),
                       kpg=bk.kpg, sites=sites, nproj=nproj)


def occupation_matrices(q: torch.Tensor, coeffs: torch.Tensor, occ: torch.Tensor,
                        kweights: torch.Tensor, sites: list) -> list[torch.Tensor]:
    """Per-site occupation matrices n^I_{mm'} (Hermitian) for one spin channel.

    q (nk, nproj, npw_max) phased projectors, coeffs (nk, nb, npw_max), occ
    (nk, nb) in the channel's electron units. On-site blocks (Dudarev)."""
    becp = torch.einsum("kpg,kbg->kbp", q.conj(), coeffs)  # (nk, nb, nproj)
    w = (kweights[:, None] * occ).to(RDTYPE)
    n_full = torch.einsum("kb,kbp,kbq->pq", w, becp, becp.conj())  # (nproj, nproj)
    return [n_full[s["start"]:s["start"] + s["dim"],
                   s["start"]:s["start"] + s["dim"]] for s in sites]


def hubbard_energy(mats: list[torch.Tensor], sites: list) -> torch.Tensor:
    """Dudarev E_U for one spin channel: Σ_I (U−J)/2 Tr[n(1−n)]."""
    e = torch.zeros((), dtype=RDTYPE, device=mats[0].device)
    for n, s in zip(mats, sites, strict=True):
        uj = s["u"] - s["j"]
        e = e + 0.5 * uj * (torch.trace(n).real - torch.trace(n @ n).real)
    return e


def hubbard_dmatrix(mats: list[torch.Tensor], sites: list, nproj: int,
                    device) -> torch.Tensor:
    """Block-diagonal D^I_{mm'} = (U−J)(½δ − n^I) over all correlated sites,
    shape (nproj, nproj) complex Hermitian (one spin channel)."""
    d = torch.zeros(nproj, nproj, dtype=CDTYPE, device=device)
    for n, s in zip(mats, sites, strict=True):
        uj = s["u"] - s["j"]
        dim, st = s["dim"], s["start"]
        eye = torch.eye(dim, dtype=CDTYPE, device=device)
        d[st:st + dim, st:st + dim] = uj * (0.5 * eye - n)
    return d
