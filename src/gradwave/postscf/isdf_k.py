"""ISDF-K: interpolative separable density fitting for multi-k exchange (Layer C).

``exchange_multik.multik_exchange_energy`` is the *direct* multi-k Fock build —
O(N_k²·N_occ²) co-density FFTs. ISDF-K compresses it the way ``isdf.py``
compresses the Γ build. The periodic orbital parts u_{ik}(r) across all
occupied bands and all k share one interpolation set: pick points {r_μ} and fit
one family of interpolation vectors ζ_μ(r) so that every co-density factorizes,

    u*_{ik}(r) u_{jk′}(r) ≈ Σ_μ ζ_μ(r) u*_{ik}(r_μ) u_{jk′}(r_μ).

(The co-density is cell-periodic — both u's are periodic — so a single periodic
ζ set serves all k-pairs; the crystal momentum q = k′−k enters only through the
Coulomb kernel.) The exchange energy then contracts without per-pair FFTs:

    E_x = −(1/2Ω) Σ_{k,k′} w_k w_{k′} Σ_{μν} V^q_{μν} · [Ā_k ⊙ A_{k′}]_{μν},
    V^q_{μν} = Σ_G ζ̃_μ(G) K(q+G) ζ̃_ν*(G),   A_k[μ,ν] = Σ_i u_{ik}(r_μ) u*_{ik}(r_ν),

with ζ̃_μ built once (N_μ FFTs total, versus N_k²·N_occ² for the direct build).
Only V^q — a dense O(N_μ²·N_G) contraction — is per k-pair, and it is the object
a future distinct-q reuse or ACE-ISDF wiring would optimize.

Conventions match ``exchange_multik``: periodic orbitals u_{ik} = g_to_r(c_{ik})
on the dense grid, full BZ (unreduced mesh), the range-separated kernel from
``coulomb_kernel``, differentiable in ω. At a single k-point (Γ) this reduces to
the ``isdf.py`` Γ build; away from saturation the interpolation rank N_μ is the
accuracy knob, exactly as at Γ.
"""

from __future__ import annotations

import torch

from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE
from gradwave.postscf.coulomb_kernel import coulomb_kernel
from gradwave.postscf.isdf import build_isdf, select_interpolation_points


def build_isdf_k(u_per_k, n_mu, *, generator=None, sketch=None):
    """Shared interpolation points + vectors for the whole occupied k-set.

    u_per_k: list of (n_occ_k, N_r) periodic orbitals per k. Stacks them into one
    orbital set, selects ≤ n_mu points by pivoted QR on the combined pair space,
    and fits ζ. Returns (points (≤n_mu,), zeta (N_r, n_mu))."""
    u_all = torch.cat(list(u_per_k), dim=0)  # (N_tot, N_r)
    points = select_interpolation_points(u_all, n_mu, generator=generator, sketch=sketch)
    zeta = build_isdf(u_all, points)
    return points, zeta


def isdf_k_exchange_energy(
    u_per_k, kcart_per_k, kweights, points, zeta, g_cart, volume, *,
    mode: str = "full", omega=None,
) -> torch.Tensor:
    """Compressed multi-k exchange through the ISDF-K factorization.

    u_per_k, kcart_per_k, kweights, g_cart, volume as in
    ``exchange_multik.multik_exchange_energy``; points, zeta from
    ``build_isdf_k``. Differentiable in ω. Returns a real scalar [eV]. Matches
    the direct build once N_μ saturates the pair rank."""
    shape = tuple(g_cart.shape[:3])
    n_mu = int(points.shape[0])
    n_k = len(u_per_k)

    # Coulomb-side: interpolation vectors in G-space, ζ̃_μ(G), built once.
    zg = r_to_g(zeta.transpose(0, 1).to(CDTYPE).reshape(n_mu, *shape)).reshape(n_mu, -1)

    # Orbital-side: per-k point-Gram A_k[μ,ν] = Σ_i u_{ik}(r_μ) u*_{ik}(r_ν).
    a_per_k = []
    for u in u_per_k:
        u_mu = u[:, points]                                  # (n_occ_k, n_mu)
        a_per_k.append(u_mu.transpose(0, 1) @ u_mu.conj())   # (n_mu, n_mu)

    g_flat = g_cart.reshape(-1, 3)
    total = zeta.new_zeros((), dtype=CDTYPE)
    for ka in range(n_k):
        for kb in range(n_k):
            q = kcart_per_k[kb] - kcart_per_k[ka]
            qg2 = ((g_flat + q) ** 2).sum(dim=-1)            # (N_G,)
            kern = coulomb_kernel(qg2, mode, omega)          # (N_G,)
            vq = (zg * kern) @ zg.conj().transpose(0, 1)     # (n_mu, n_mu)
            t = a_per_k[ka].conj() * a_per_k[kb]             # Ā_k ⊙ A_{k′}
            total = total + kweights[ka] * kweights[kb] * (vq * t).sum()
    return -0.5 * total.real / volume
