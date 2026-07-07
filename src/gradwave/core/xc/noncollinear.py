"""Non-collinear XC in the locally-collinear approximation (Layer A).

Standard treatment (Kübler): at each grid point, project onto the local
quantization axis — the XC energy depends only on (ρ, |m|):

    E_xc[ρ, m⃗] = E_xc^collinear(ρ↑, ρ↓),   ρ± = (ρ ± |m⃗|)/2

Because this is written differentiably in (ρ, mx, my, mz), autograd yields
BOTH potentials at once:

    v_xc = ∂e/∂ρ,      B⃗_xc = ∂e/∂m⃗ = (∂e/∂|m|)·m̂

i.e. the exchange field is automatically parallel to the local moment with
the correct magnitude — no hand-coded projection anywhere. The |m⃗| → 0
limit is regularized smoothly (ε in the norm); B_xc → 0 there as it must.

This module is the XC leg of the non-collinear phase; the spinor
Hamiltonian/SCF (see docs/noncollinear.md) consumes (v_xc, B⃗_xc) as the
2×2 potential  V = v_xc·1 + B⃗_xc·σ⃗.
"""

from __future__ import annotations

import torch

from gradwave.core.xc.spin import SpinXC


class NoncollinearXC(torch.nn.Module):
    """Wraps any collinear SpinXC into a (ρ, m⃗) functional."""

    def __init__(self, collinear: SpinXC, m_eps: float = 1e-24):
        super().__init__()
        self.collinear = collinear
        self.m_eps = m_eps

    @property
    def needs_gradient(self) -> bool:
        return self.collinear.needs_gradient

    def energy(self, rho, m_vec, volume, sigma_uu=None, sigma_dd=None, sigma_tot=None):
        """E_xc [eV]. rho (grid), m_vec (3, grid). GGA σ's are those of the
        locally-collinear channels (caller builds them from ρ± spectra)."""
        m_norm = torch.sqrt((m_vec**2).sum(dim=0) + self.m_eps)
        rho_up = 0.5 * (rho + m_norm)
        rho_dn = 0.5 * (rho - m_norm)
        return self.collinear.energy(rho_up, rho_dn, volume, sigma_uu, sigma_dd, sigma_tot)


def vxc_and_bxc(nc_xc: NoncollinearXC, rho, m_vec, grid):
    """(v_xc(r), B⃗_xc(r), E_xc) via one autograd call (LDA-level fields)."""
    r = rho.detach().clone().requires_grad_(True)
    m = m_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        e = nc_xc.energy(r, m, grid.volume)
        vr, vm = torch.autograd.grad(e, (r, m))
    scale = grid.n_points / grid.volume
    return vr * scale, vm * scale, e.detach()
