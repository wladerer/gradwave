"""Hartree energy and potential in reciprocal space (Layer A).

With ρ(G) the Fourier-series coefficients of the electron density [e/Å³]
(fftbox convention), for the periodic neutralized system:

    E_H = (Ω/2) Σ_{G≠0} 4π e² |ρ(G)|² / G²
    v_H(G) = 4π e² ρ(G)/G²,  v_H(G=0) ≡ 0

The divergent G=0 term is EXCLUDED here — its cancellation against the
local-pseudopotential tail and the Ewald background is documented in
energies/total.py.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2

# |G|² [Å⁻²] below this counts as the G=0 (Coulomb-divergent) component, whose
# 1/G² factor is excluded here — its cancellation is handled in energies/total.
G2_ZERO_TOL = 1e-12


def _inv_g2_masked(g2: torch.Tensor) -> torch.Tensor:
    """1/G² [Å²] with the G=0 term set to 0 (v_H(0) ≡ 0)."""
    return torch.where(g2 > G2_ZERO_TOL, 1.0 / torch.clamp(g2, min=G2_ZERO_TOL),
                       torch.zeros_like(g2))


def hartree_energy(rho_g: torch.Tensor, g2: torch.Tensor, volume: float) -> torch.Tensor:
    """E_H [eV]. rho_g, g2: dense-box tensors (fftbox layout)."""
    inv_g2 = _inv_g2_masked(g2)
    return 0.5 * volume * 4.0 * math.pi * E2 * ((rho_g.abs() ** 2) * inv_g2).sum()


def hartree_potential_g(rho_g: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """v_H(G) [eV] on the dense box, v_H(0) = 0."""
    inv_g2 = _inv_g2_masked(g2)
    return 4.0 * math.pi * E2 * rho_g * inv_g2


def hartree_potential_r(rho_r: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Real v_H(r) [eV] straight from the real density ρ(r), via a real FFT.

    Fuses ``g_to_r_box(hartree_potential_g(r_to_g(ρ), g2), real=True)`` into one
    ``rfftn``/``irfftn`` pair — the density and Hartree potential are both real,
    so only the n3//2+1 half of the last box axis carries independent data and
    the transform does ~half the work of the full-complex round trip.

    Bit-exact to the full-complex path (not an approximation). That path takes
    Re[·] of the G→r transform; for a real ρ (Hermitian ρ(G)), Re[ifftn(K·ρ(G))]
    equals ifftn of ρ(G) times the G→−G *symmetrized* kernel. 1/G² is only
    reflection-symmetric on the box when |G|² is — which fails on the Nyquist
    planes of a non-orthogonal cell — so we symmetrize inv_g2 here before the
    half-box multiply. irfftn then reconstructs exactly the same field the
    ``real=True`` full path returns (verified to ~1e-14).
    """
    shape = tuple(int(s) for s in rho_r.shape[-3:])
    nh3 = shape[-1] // 2 + 1
    inv_g2 = _inv_g2_masked(g2)
    # G→−G reflection on the box is index (n−k) mod n per axis = roll(flip, 1).
    inv_g2_sym = 0.5 * (
        inv_g2
        + torch.roll(torch.flip(inv_g2, (-3, -2, -1)), (1, 1, 1), (-3, -2, -1))
    )
    rho_g = torch.fft.rfftn(rho_r, dim=(-3, -2, -1))
    v_g = 4.0 * math.pi * E2 * rho_g * inv_g2_sym[..., :nh3]
    return torch.fft.irfftn(v_g, s=shape, dim=(-3, -2, -1))
