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
Hamiltonian/SCF consumes (v_xc, B⃗_xc) as the
2×2 potential  V = v_xc·1 + B⃗_xc·σ⃗.
"""

from __future__ import annotations

import torch

from gradwave.core.xc.spin import SpinXC


class NoncollinearXC(torch.nn.Module):
    """Wraps any collinear SpinXC into a (ρ, m⃗) functional."""

    def __init__(self, collinear: SpinXC, m_eps: float = 1e-24):
        super().__init__()
        if getattr(collinear, "needs_tau", False):
            raise NotImplementedError(
                "meta-GGA (needs_tau) is not supported on the non-collinear/SOC "
                "spinor path yet — the spinor SCF does not build τ. Use a "
                "collinear (nspin=1/2) calculation for r2SCAN.")
        self.collinear = collinear
        self.m_eps = m_eps

    @property
    def needs_gradient(self) -> bool:
        return self.collinear.needs_gradient

    def energy(self, rho, m_vec, volume, sigma_uu=None, sigma_dd=None, sigma_tot=None,
               rho_core=None):
        """E_xc [eV]. rho (grid), m_vec (3, grid). GGA σ's are those of the
        locally-collinear channels (caller builds them from ρ± spectra).
        rho_core (NLCC) shifts ρ only — same semantics as energy_with_grid, so
        the two entry points share the ρ± projection and cannot disagree."""
        r_up, r_dn = _rho_pm(self, rho, m_vec, rho_core)
        return self.collinear.energy(r_up, r_dn, volume, sigma_uu, sigma_dd, sigma_tot)


def _rho_pm(nc_xc: NoncollinearXC, rho, m_vec, rho_core):
    """Locally-collinear channels ρ± = (ρ ± |m⃗|)/2, with the NLCC core (which
    carries no magnetization) folded into ρ before the split. Shared by both
    energy entry points so the projection is defined in exactly one place."""
    if rho_core is not None:
        rho = rho + rho_core
    m_norm = torch.sqrt((m_vec**2).sum(dim=0) + nc_xc.m_eps)
    return 0.5 * (rho + m_norm), 0.5 * (rho - m_norm)


def energy_with_grid(nc_xc: NoncollinearXC, rho, m_vec, grid, rho_core=None):
    """E_xc with locally-collinear GGA σ's computed spectrally (in-graph).
    rho_core (NLCC) shifts ρ only — the core carries no magnetization."""
    from gradwave.core.density import sigma_from_rho

    r_up, r_dn = _rho_pm(nc_xc, rho, m_vec, rho_core)
    if nc_xc.collinear.needs_gradient:
        rho_tot = rho if rho_core is None else rho + rho_core
        s_uu = sigma_from_rho(r_up, grid.g_cart)
        s_dd = sigma_from_rho(r_dn, grid.g_cart)
        s_tt = sigma_from_rho(rho_tot, grid.g_cart)
    else:
        s_uu = s_dd = s_tt = None
    return nc_xc.collinear.energy(r_up, r_dn, grid.volume, s_uu, s_dd, s_tt)


def vxc_and_bxc(nc_xc: NoncollinearXC, rho, m_vec, grid, rho_core=None):
    """(v_xc(r), B⃗_xc(r), E_xc) via one autograd call — GGA terms included."""
    r = rho.detach().clone().requires_grad_(True)
    m = m_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        e = energy_with_grid(nc_xc, r, m, grid, rho_core=rho_core)
        vr, vm = torch.autograd.grad(e, (r, m))
    scale = grid.n_points / grid.volume
    return vr * scale, vm * scale, e.detach()
