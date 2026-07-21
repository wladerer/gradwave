"""Self-consistent hybrid functionals: exact exchange in the SCF (Layer C, Γ).

Ties the ISDF/ACE exchange machinery into the SCF loop so gradwave can *solve* a
hybrid, not just evaluate exchange on a fixed density. A global hybrid (PBE0
form) is

    E_xc = (1−α) E_x^PBE + α E_x^Fock + E_c^PBE,

so two pieces cooperate: ``ScaledExchangePBE`` scales the semilocal exchange by
(1−α) on the functional side, and ``GammaFockExchange`` supplies the α E_x^Fock —
its operator through the SCF ``fock`` hook (added to ``BatchedHamiltonian.apply``
each Davidson step) and its energy as the ``EnergyBreakdown.fock`` term. The Fock
operator is orbital-dependent, so it is rebuilt from the current orbitals each
SCF iteration and lags one step, exactly like the DFT+U occupation matrices; ACE
(``exchange.build_ace``) makes the frozen operator cheap to re-apply.

Two operators live here. ``GammaFockExchange`` is the single-k (Γ) build. For a
k-mesh, ``MultiKFockExchange`` compresses a *per-k* ACE operator whose exchange at
each k sums over the whole BZ through the co-density momentum q = k−k′ and the
range-separated kernel — completing global (PBE0) and screened (HSE) hybrids on a
mesh. At a single Γ point (q=0, full kernel) it reduces to ``GammaFockExchange``.

Spin factor. At nspin=1 each spatial orbital holds two electrons, so the physical
exchange is twice the value ``ACEExchange.energy`` returns for the spatial-orbital
set (2/nspin); the *operator* carries no such factor (the occupation 2 divides
out of the eigenvalue equation), so energy and operator stay derivative-consistent.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.batch import box_to_sphere_b, g_to_r_b
from gradwave.core.density import sigma_from_rho
from gradwave.core.xc._pbe_kernels import pbe_enhancement, pbe_h
from gradwave.core.xc.base import to_au
from gradwave.core.xc.lda_pw92 import eps_c_pw92, eps_x_lda
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import RDTYPE
from gradwave.postscf.exchange import (
    ACEExchange,
    build_ace,
    exchange_operator_direct,
    physical_orbitals,
)
from gradwave.postscf.exchange_multik import (
    HybridExchangeParams,
    multik_exchange_energy,
    multik_exchange_operator,
    occupied_periodic_orbitals,
)
from gradwave.scf.loop import scf


class ScaledExchangePBE(PBE):
    """PBE with the semilocal exchange scaled by (1−exx_fraction).

    The omitted fraction is supplied as exact Fock exchange by the SCF ``fock``
    hook, giving a PBE0-form global hybrid. At exx_fraction = 0 this is plain PBE
    (a reduction gate); correlation is untouched."""

    def __init__(self, exx_fraction: float = 0.25):
        super().__init__()
        if not 0.0 <= exx_fraction <= 1.0:
            raise ValueError("exx_fraction must be in [0, 1]")
        self.exx_fraction = float(exx_fraction)

    def energy_density(
        self, rho: torch.Tensor, sigma: torch.Tensor | None = None, tau=None
    ) -> torch.Tensor:
        if sigma is None:
            raise ValueError("PBE requires sigma = |grad rho|^2")
        rho_au = to_au(rho)
        sigma_au = torch.clamp(sigma * BOHR_ANG ** 8, min=0.0)
        grad_au = torch.sqrt(sigma_au + 1e-30)
        kf = (3.0 * math.pi ** 2 * rho_au) ** (1.0 / 3.0)
        s = grad_au / (2.0 * kf * rho_au)
        eps_x = eps_x_lda(rho_au) * pbe_enhancement(s * s, self.kappa, self.mu)
        eps_c_lda = eps_c_pw92(rho_au)
        ks = torch.sqrt(4.0 * kf / math.pi)
        t = grad_au / (2.0 * ks * rho_au)
        eps_c = eps_c_lda + pbe_h(t * t, eps_c_lda)
        return rho * ((1.0 - self.exx_fraction) * eps_x + eps_c) * HARTREE_EV


class GammaFockExchange:
    """Hybrid Fock-exchange operator at Γ, as an SCF ``fock`` hook.

    ``rebuild`` builds the ACE exchange operator from the current occupied
    orbitals (per spin) and returns (apply_delta_s, e_fock): a per-spin callable
    that applies α·V_x to batched sphere coefficients, and the exchange energy
    α·E_x added to the total. Built to be passed as ``scf(..., fock=...)``."""

    def __init__(self, alpha: float = 0.25, occ_tol: float = 1e-6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.occ_tol = occ_tol

    def rebuild(self, coeffs_b_s, occ_s, system):
        nspin = len(coeffs_b_s)
        spin_factor = 2.0 / nspin  # nspin=1: two electrons per spatial orbital
        flat_idx = system.spheres[0].flat_idx
        shape, vol, g2 = system.grid.shape, system.grid.volume, system.grid.g2
        bk = system.batch

        apply_s, e_fock = [], torch.zeros((), dtype=RDTYPE, device=coeffs_b_s[0].device)
        for sp in range(nspin):
            occ = occ_s[sp][0] > self.occ_tol            # (nb,) occupied at Γ
            psi = physical_orbitals(coeffs_b_s[sp][0][occ], flat_idx, shape, vol)
            vx = exchange_operator_direct(psi, psi, shape, g2)
            ace = build_ace(psi, vx, vol)
            e_fock = e_fock + spin_factor * ace.energy(psi).to(RDTYPE)
            apply_s.append(self._apply_for(ace, bk, shape))
        return apply_s, self.alpha * e_fock

    def _apply_for(self, ace, bk, shape):
        alpha = self.alpha

        def apply_delta(c: torch.Tensor) -> torch.Tensor:
            nk, nb = c.shape[0], c.shape[1]
            f = g_to_r_b(c, bk, shape).reshape(nk * nb, -1)  # periodic field Σ_G c e^{iGr}
            wf = ace.apply(f).reshape(nk, nb, *shape)        # V_x f (linear; = √Ω·V_x ψ)
            return alpha * box_to_sphere_b(wf, bk)

        return apply_delta


class MultiKFockExchange:
    """Hybrid Fock exchange on a full-BZ k-mesh, as an SCF ``fock`` hook.

    The k-mesh generalization of ``GammaFockExchange``. ``rebuild`` extracts the
    occupied orbitals at every k, builds the direct multi-k Fock operator (each
    k's action summing over the whole BZ via the co-density momentum q = k−k′ and
    the ``mode``/``omega`` range-separated kernel), and ACE-compresses it *per k*.
    It returns a per-spin callable that applies α·V_x block-by-block over the k
    batch, plus the exchange energy α·E_x.

    ``mode`` selects the kernel: ``"full"`` (PBE0), ``"short_range"``,
    ``"long_range"``. Requires a full-BZ system (built with
    ``use_symmetry=False, time_reversal=False``) on the dense grid, exactly as
    ``exchange_multik``. The spin factor matches ``GammaFockExchange``: 2/nspin in
    the energy, none in the operator.

    Screened caveat. The screened Fock *operator* (short_range) is exact and
    energy-consistent, but a complete HSE also range-separates the *semilocal*
    exchange it replaces (keep the long-range PBE exchange, remove only the
    short-range fraction). ``ScaledExchangePBE`` scales the whole PBE exchange by
    (1−α) — correct for full-range PBE0, but for a screened hybrid it double-counts
    the long-range exchange. A proper HSE needs the range-separated (wPBE)
    enhancement on the semilocal side, which is not implemented here; use
    ``mode="full"`` for a physically complete SCF (PBE0) until then.

    At a single Γ point with ``mode="full"`` this reproduces ``GammaFockExchange``
    to machine precision (the reduction gate)."""

    def __init__(self, alpha: float = 0.25, mode: str = "full", omega=None,
                 occ_tol: float = 1e-6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if mode not in ("full", "short_range", "long_range"):
            raise ValueError(f"unknown Coulomb-kernel mode {mode!r}")
        if mode != "full" and omega is None:
            raise ValueError(f"mode {mode!r} needs omega (the range-separation length)")
        self.alpha = float(alpha)
        self.mode = mode
        self.omega = omega
        self.occ_tol = occ_tol

    def rebuild(self, coeffs_b_s, occ_s, system):
        nspin = len(coeffs_b_s)
        spin_factor = 2.0 / nspin  # nspin=1: two electrons per spatial orbital
        shape, vol = system.grid.shape, system.grid.volume
        g_cart, bk = system.grid.g_cart, system.batch
        kweights = system.kweights
        kcart = [sph.k_cart for sph in system.spheres]
        n_r = int(shape[0] * shape[1] * shape[2])
        device = coeffs_b_s[0].device

        apply_s, e_fock = [], torch.zeros((), dtype=RDTYPE, device=device)
        for sp in range(nspin):
            psi_per_k = [
                physical_orbitals(
                    coeffs_b_s[sp][ik][occ_s[sp][ik] > self.occ_tol, : sph.npw],
                    sph.flat_idx, shape, vol)
                for ik, sph in enumerate(system.spheres)
            ]
            w_per_k = multik_exchange_operator(
                psi_per_k, kcart, kweights, g_cart, vol,
                mode=self.mode, omega=self.omega)
            ace_per_k, e_sp = [], torch.zeros((), dtype=RDTYPE, device=device)
            for ik in range(len(psi_per_k)):
                if psi_per_k[ik].shape[0] == 0:
                    empty = psi_per_k[ik].new_zeros((n_r, 0))
                    ace_per_k.append(ACEExchange(xi=empty, volume=vol, n_r=n_r))
                    continue
                ace = build_ace(psi_per_k[ik], w_per_k[ik], vol)
                ace_per_k.append(ace)
                e_sp = e_sp + kweights[ik] * ace.energy(psi_per_k[ik]).to(RDTYPE)
            e_fock = e_fock + spin_factor * e_sp
            apply_s.append(self._apply_for(ace_per_k, bk, shape))
        return apply_s, self.alpha * e_fock

    def _apply_for(self, ace_per_k, bk, shape):
        alpha = self.alpha

        def apply_delta(c: torch.Tensor) -> torch.Tensor:
            nk, nb = c.shape[0], c.shape[1]
            f = g_to_r_b(c, bk, shape)                    # (nk, nb, *shape) periodic parts
            wf = torch.empty_like(f)
            for ik in range(nk):
                fi = f[ik].reshape(nb, -1)                # (nb, N_r) trial parts at k
                wf[ik] = ace_per_k[ik].apply(fi).reshape(nb, *shape)
            return alpha * box_to_sphere_b(wf, bk)

        return apply_delta


def hybrid_scf(system, alpha: float = 0.25, *, mode: str = "full", omega=None,
               params: HybridExchangeParams | None = None, **scf_kwargs):
    """Self-consistent PBE0-form / screened global hybrid, exchange fraction ``alpha``.

    Runs the standard SCF with the semilocal exchange scaled by (1−α) and α·Fock
    exchange added through the ``fock`` hook, via the k-mesh ``MultiKFockExchange``
    (which reduces to the Γ build at a single k-point). ``mode="full"`` is a
    physically complete PBE0; the screened modes supply an exact screened Fock
    operator but not yet the matching range-separated semilocal exchange (see
    ``MultiKFockExchange`` — the semilocal side would double-count long-range
    exchange), so treat them as the operator half of a screened hybrid. At α = 0
    this is exactly a PBE SCF. Extra keyword arguments pass through to ``scf``.

    Pass a ``HybridExchangeParams`` as ``params`` to solve at its current (α, ω)
    values — the SCF itself runs under ``no_grad``; gradients for a *learned*
    hybrid come from ``differentiable_hybrid_energy`` on the converged result."""
    if params is not None:
        alpha = float(params.alpha)
        mode = params.mode
        omega = float(params.omega) if params.mode != "full" else None
    xc = ScaledExchangePBE(alpha)
    fock = (MultiKFockExchange(alpha, mode=mode, omega=omega)
            if alpha > 0.0 else None)
    return scf(system, xc, fock=fock, **scf_kwargs)


def _pbe_exchange_energy(res) -> float:
    """∫ρ ε_x^PBE on the converged density [eV] — the exchange ``ScaledExchangePBE``
    scales by (1−α). Computed as (full PBE XC) − (PBE with exchange removed), so it
    matches the SCF's semilocal-exchange bookkeeping exactly. θ-independent."""
    system = res.system
    vol = system.grid.volume
    rho = res.rho if res.system.rho_core is None else res.rho + res.system.rho_core
    sigma = sigma_from_rho(rho, system.grid.g_cart)
    e_full = PBE().energy(rho, vol, sigma)
    e_no_x = ScaledExchangePBE(1.0).energy(rho, vol, sigma)
    return float(e_full - e_no_x)


def differentiable_hybrid_energy(res, params: HybridExchangeParams, *,
                                 occ_tol: float = 1e-6) -> torch.Tensor:
    """Converged hybrid total energy as a differentiable function of (α, ω).

    Turns a converged hybrid SCF into a trainable objective. At self-consistency
    the orbitals/density are stationary, so by the Hellmann–Feynman theorem
    dE_total/dθ = ∂E_total/∂θ — only the *explicit* θ-dependence of the exchange
    terms survives, evaluated on the *frozen* converged orbitals and density. The
    returned scalar equals ``res.energies.total`` in value and carries the exact
    dE_total/dα, dE_total/dω into ``params`` on ``.backward()``; build any loss on
    it (e.g. matching a reference gap or energy) and step an optimizer over the
    ``params`` to train a learned hybrid. Single spin (nspin=1).

    The frozen pieces, matching the SCF's (1−α)E_x^PBE + α E_x^Fock split:
      E_x^Fock(ω) = (2/nspin)·``multik_exchange_energy``(ω)  — differentiable in ω,
      E_x^PBE     = ∫ρ ε_x^PBE                               — a θ-independent constant,
    and the θ-dependent energy α·(E_x^Fock(ω) − E_x^PBE) has α-derivative
    E_x^Fock − E_x^PBE (the exact exchange-mixing gradient) and ω-derivative
    α·∂E_x^Fock/∂ω."""
    if getattr(res, "nspin", 1) != 1:
        raise ValueError("differentiable_hybrid_energy supports nspin=1")
    system = res.system
    vol = system.grid.volume
    u, kc, kw = occupied_periodic_orbitals(res, system, occ_tol)
    u = [x.detach() for x in u]                          # frozen occupied orbitals
    omega = params.omega if params.mode != "full" else None
    e_fock = 2.0 * multik_exchange_energy(u, kc, kw, system.grid.g_cart, vol,
                                          mode=params.mode, omega=omega)
    e_x_pbe = _pbe_exchange_energy(res)                  # θ-independent constant
    e_theta = params.alpha * (e_fock - e_x_pbe)
    e_const = float(res.energies.total) - float(e_theta.detach())
    return e_const + e_theta


def hybrid_energy_gradient(res, params: HybridExchangeParams, *,
                           occ_tol: float = 1e-6):
    """Exact stationary (dE_total/dα, dE_total/dω) at the converged hybrid [eV].

    Convenience/verification wrapper over ``differentiable_hybrid_energy``: runs
    one backward pass and chain-rules the raw-parameter gradients back to the
    physical (α, ω). The ω-gradient is ``None`` for ``mode="full"``."""
    e = differentiable_hybrid_energy(res, params, occ_tol=occ_tol)
    params.zero_grad(set_to_none=True)
    e.backward()
    alpha = float(params.alpha.detach())
    d_alpha = float(params.raw_alpha.grad) / (alpha * (1.0 - alpha))
    if params.mode == "full":
        return d_alpha, None
    omega = float(params.omega.detach())
    d_omega = float(params.raw_omega.grad) / (1.0 - math.exp(-omega))
    return d_alpha, d_omega
