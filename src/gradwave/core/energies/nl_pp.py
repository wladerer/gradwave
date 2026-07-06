"""Kleinman–Bylander nonlocal energy (Layer A).

With projector coefficients p_i(k+G) built by core/hamiltonian.py
(β form factor × Ylm × (−i)^l × structure phase × 4π/√Ω) and overlaps
b_i,nk = Σ_G p_i*(k+G) c_nk(G):

    E_NL = Σ_k w_k Σ_n f_nk Σ_a Σ_ij D^a_ij · b*_ai,nk b_aj,nk   [eV]

D_ij couples projectors of the same atom (and same l, m by construction of
the UPF). The dij tensor passed here is block-diagonal over atoms with the
m-expanded UPF blocks on the diagonal.
"""

from __future__ import annotations

import torch


def nonlocal_energy(
    becp_per_k: list[torch.Tensor],  # [(nb, nproj_tot) complex] overlaps ⟨p|ψ⟩
    dij: torch.Tensor,  # (nproj_tot, nproj_tot) real [eV]
    occ: torch.Tensor,  # (nk, nb)
    kweights: torch.Tensor,  # (nk,)
) -> torch.Tensor:
    e = None
    for ik, b in enumerate(becp_per_k):
        # Σ_ij D_ij b*_i b_j per band (D real symmetric ⇒ result real)
        quad = torch.einsum("bi,ij,bj->b", b.conj(), dij.to(b.dtype), b).real
        term = (kweights[ik] * occ[ik, : b.shape[0]] * quad).sum()
        e = term if e is None else e + term
    return e
