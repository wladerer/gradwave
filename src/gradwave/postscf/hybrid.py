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

Single k-point (Γ) — the ACE operator is Γ-only. A k-mesh hybrid needs the
per-k / ``exchange_multik`` generalization (see docs/ideas.md).

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
from gradwave.core.xc._pbe_kernels import pbe_enhancement, pbe_h
from gradwave.core.xc.base import to_au
from gradwave.core.xc.lda_pw92 import eps_c_pw92, eps_x_lda
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import RDTYPE
from gradwave.postscf.exchange import build_ace, exchange_operator_direct, physical_orbitals
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

    def energy_density(self, rho: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
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


def hybrid_scf(system, alpha: float = 0.25, **scf_kwargs):
    """Self-consistent PBE0-form global hybrid at Γ, exchange fraction ``alpha``.

    Runs the standard SCF with the semilocal exchange scaled by (1−α) and α·Fock
    exchange added through the ``fock`` hook. At α = 0 this is exactly a PBE SCF.
    Extra keyword arguments pass through to ``scf`` (smearing, etol, ...)."""
    xc = ScaledExchangePBE(alpha)
    fock = GammaFockExchange(alpha) if alpha > 0.0 else None
    return scf(system, xc, fock=fock, **scf_kwargs)
