"""Multi-k Fock exchange and the learnable-hybrid parameter slot (Layer C).

Generalizes the Γ exchange in ``exchange.py`` to a full k-mesh. Exchange couples
every pair of k-points through the co-density crystal momentum q = k′ − k:

    ρ_{ik,jk′}(r) = ψ*_{ik}(r) ψ_{jk′}(r) = (1/Ω) e^{iq·r} u*_{ik}(r) u_{jk′}(r),

whose periodic part u*_{ik} u_{jk′} carries momentum q, so its Coulomb self-
energy is evaluated with the kernel K(q+G) (``coulomb_kernel``) at the shifted
grid |q+G|². The exact-exchange energy is

    E_x = −½ Σ_{k,k′} w_k w_{k′} Σ_{i∈occ(k)} Σ_{j∈occ(k′)}
              (1/Ω) Σ_G |ũ*ũ(G)|² K(q+G).

At a single k-point (Γ, q=0, full kernel) this is exactly the ``exchange.py`` /
``isdf.py`` build, and it is the reference the ISDF-K acceleration will target.

Two requirements the caller must meet, both physical:

- **Full BZ, no symmetry folding.** Exchange needs every k as a genuine BZ point
  (build the system with ``use_symmetry=False, time_reversal=False``); the
  density's IBZ reduction does not carry to the k,k′ double sum without
  unfolding. ``occupied_periodic_orbitals`` reads whatever mesh the result
  carries, so pass an unreduced one.
- **Dense grid.** The co-density support reaches 2·G_max(wfc), so the transforms
  run on the dense (ecutrho ≥ 4·ecut) grid — ``system.grid`` — to avoid aliasing.

The occupied-orbital *set* at each k (occupation above ``occ_tol``, each counted
once, k-weighted) is used, matching the Γ convention exactly; fractional /
metallic occupation weighting is a documented extension, not applied here.
"""

from __future__ import annotations

import math

import torch

from gradwave.core.fftbox import g_to_r, g_to_r_box, r_to_g
from gradwave.postscf.coulomb_kernel import coulomb_kernel


def occupied_periodic_orbitals(res, system, occ_tol: float = 1e-6):
    """Periodic parts u_{ik}(r) = Σ_G c_{ik}(G) e^{iG·r} of the occupied orbitals.

    Returns (u_per_k, kcart_per_k, kweights): u_per_k[ik] is (n_occ_k, N_r) on
    the dense grid, kcart_per_k[ik] the Cartesian k [Å⁻¹]. Reads whatever k-mesh
    the SCF result carries — pass an unreduced (full-BZ) system for exchange."""
    shape = system.grid.shape
    u_per_k, kcart = [], []
    for ik, sph in enumerate(system.spheres):
        occ = res.occupations[ik] > occ_tol
        u = g_to_r(res.coeffs[ik][occ], sph.flat_idx, shape).reshape(int(occ.sum()), -1)
        u_per_k.append(u)
        kcart.append(sph.k_cart)
    return u_per_k, kcart, system.kweights


def multik_exchange_energy(
    u_per_k, kcart_per_k, kweights, g_cart, volume, *,
    mode: str = "full", omega=None,
) -> torch.Tensor:
    """E_x over a full-BZ k-mesh with the range-separated kernel ``mode``.

    u_per_k: list of (n_occ_k, N_r) periodic orbitals; kcart_per_k: list of (3,);
    kweights: (n_k,) summing to 1; g_cart: (n1,n2,n3,3) dense reciprocal grid;
    volume: Ω. Differentiable in ``omega``. Returns a real scalar [eV]."""
    shape = tuple(g_cart.shape[:3])
    n_k = len(u_per_k)
    total = g_cart.new_zeros(())
    for ka in range(n_k):
        ua = u_per_k[ka]
        n_ia = ua.shape[0]
        for kb in range(n_k):
            ub = u_per_k[kb]
            n_jb = ub.shape[0]
            q = kcart_per_k[kb] - kcart_per_k[ka]                    # (3,)
            qg2 = ((g_cart + q) ** 2).sum(dim=-1)                    # (n1,n2,n3)
            kern = coulomb_kernel(qg2, mode, omega)                  # (n1,n2,n3)
            p = ua.conj()[:, None, :] * ub[None, :, :]              # (n_ia,n_jb,N_r)
            p_g = r_to_g(p.reshape(n_ia * n_jb, *shape))            # (·, n1,n2,n3)
            contrib = ((p_g.abs() ** 2) * kern).sum(dim=(-3, -2, -1))
            total = total + kweights[ka] * kweights[kb] * contrib.sum()
    return -0.5 * total / volume


def coulomb_potential_q(sigma_r, q, g_cart, mode: str = "full", omega=None):
    """Coulomb potential of a co-density of crystal momentum q, periodic part only.

    ``sigma_r`` (..., N_r) is the *periodic* part of a co-density whose crystal
    momentum q has been factored out (ρ(r) = e^{iq·r} σ(r)); the returned field is
    likewise the periodic part ṽ(r) of its potential, ∫ρ(r′)K(r−r′)dr′ = e^{iq·r}ṽ(r):

        ṽ(r) = Σ_G K(q+G) σ̃(G) e^{iG·r},   K from ``coulomb_kernel``.

    Same FFT normalization as ``exchange.coulomb_potential`` (its q=0, full-kernel
    special case), so the operator built from it is consistent with
    ``multik_exchange_energy``. Differentiable in ``omega``."""
    shape = tuple(g_cart.shape[:3])
    qg2 = ((g_cart + q) ** 2).sum(dim=-1)                 # (n1,n2,n3)
    kern = coulomb_kernel(qg2, mode, omega)               # (n1,n2,n3)
    batch = sigma_r.shape[:-1]
    sigma_g = r_to_g(sigma_r.reshape(*batch, *shape))
    v_g = sigma_g * kern
    v_r = g_to_r_box(v_g)
    return v_r.reshape(*batch, -1)


def multik_exchange_operator(
    psi_per_k, kcart_per_k, kweights, g_cart, volume, *,
    mode: str = "full", omega=None,
):
    """Direct multi-k Fock operator applied to each k's occupied set.

    ``psi_per_k[k]`` is (n_occ_k, N_r), the *physical* periodic orbital parts
    ψ̂_{ik} = u_{ik}/√Ω (``exchange.physical_orbitals``). Returns ``w_per_k``, a
    list of (n_occ_k, N_r): W_{tk} = (V_x ψ̂_{tk}) with V_x summing over the whole
    BZ,

        W_{tk}(r) = −Σ_{k′} w_{k′} Σ_{j∈occ(k′)} ψ̂_{jk′}(r) ṽ_{jk′,tk}(r),

    where ṽ is the periodic potential (``coulomb_potential_q``) of the co-density
    ψ̂*_{jk′}ψ̂_{tk} at momentum q = k − k′. At a single k-point (q=0, full kernel)
    this is exactly ``exchange.exchange_operator_direct``; its energy trace
    ½ Σ_k w_k Σ_t ⟨ψ̂_{tk}|W_{tk}⟩ matches ``multik_exchange_energy``. This is the
    O(N_k²·N_occ²) reference the per-k ACE compresses. Differentiable in ``omega``."""
    n_k = len(psi_per_k)
    w_per_k = [torch.zeros_like(p) for p in psi_per_k]
    for ka in range(n_k):
        pa = psi_per_k[ka]                                 # test set at k
        if pa.shape[0] == 0:
            continue
        wa = w_per_k[ka]
        for kb in range(n_k):
            pb = psi_per_k[kb]                             # occupied at k′
            if pb.shape[0] == 0:
                continue
            q = kcart_per_k[ka] - kcart_per_k[kb]          # k − k′
            for t in range(pa.shape[0]):
                sigma = pb.conj() * pa[t][None, :]         # (n_occ_kb, N_r) ψ̂*_{jk′}ψ̂_{tk}
                v = coulomb_potential_q(sigma, q, g_cart, mode, omega)
                wa[t] = wa[t] - kweights[kb] * (pb * v).sum(dim=0)
    return w_per_k


def physical_periodic_orbitals(res, system, occ_tol: float = 1e-6):
    """Like ``occupied_periodic_orbitals`` but returns the *physical* ψ̂ = u/√Ω.

    Convenience for the operator build, which needs the normalized orbitals
    (⟨ψ̂|ψ̂⟩ = 1) rather than the bare periodic parts u the energy build uses."""
    u_per_k, kcart, kw = occupied_periodic_orbitals(res, system, occ_tol)
    s = math.sqrt(system.grid.volume)
    return [u / s for u in u_per_k], kcart, kw


class HybridExchangeParams(torch.nn.Module):
    """Learnable hybrid exchange parameters: mixing fraction α and screening ω.

    Mirrors ``core/xc/learnable.py``: the raw parameters are unconstrained and
    reparameterized (sigmoid for α ∈ (0,1), softplus for ω > 0) so training stays
    in a physical range, and the defaults reproduce a standard screened hybrid
    (HSE-like: α = 0.25 short-range). The exchange energy is differentiable in
    both, so a learned hybrid trains the mixing and range end to end — the free
    dE/dθ at SCF convergence is the same argument the learnable-XC slot uses.
    """

    def __init__(self, alpha: float = 0.25, omega: float = 0.2, mode: str = "short_range"):
        super().__init__()
        if mode not in ("full", "short_range", "long_range"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.raw_alpha = torch.nn.Parameter(_inv_sigmoid(alpha))
        self.raw_omega = torch.nn.Parameter(_inv_softplus(omega))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    @property
    def omega(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_omega)


def hybrid_exchange_energy(u_per_k, kcart_per_k, kweights, g_cart, volume,
                           params: HybridExchangeParams) -> torch.Tensor:
    """α · E_x[K_mode(ω)] — the hybrid Fock-exchange contribution, differentiable
    in the ``params`` (α, ω). The full hybrid XC energy adds this to the scaled
    semilocal exchange and the correlation; that assembly is the SCF-wiring step
    still to come."""
    omega = params.omega if params.mode != "full" else None
    e_x = multik_exchange_energy(u_per_k, kcart_per_k, kweights, g_cart, volume,
                                 mode=params.mode, omega=omega)
    return params.alpha * e_x


def _inv_sigmoid(y: float) -> torch.Tensor:
    y = torch.tensor(float(y), dtype=torch.float64)
    return torch.log(y) - torch.log1p(-y)


def _inv_softplus(y: float) -> torch.Tensor:
    y = torch.tensor(float(y), dtype=torch.float64)
    return y + torch.log(-torch.expm1(-y))
