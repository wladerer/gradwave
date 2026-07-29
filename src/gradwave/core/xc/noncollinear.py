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

from types import SimpleNamespace

import torch

from gradwave.core.xc.spin import SpinXC
from gradwave.grids import FFTGrid


class NoncollinearXC(torch.nn.Module):
    """Wraps any collinear SpinXC into a (ρ, m⃗) functional."""

    def __init__(self, collinear: SpinXC, m_eps: float = 1e-24) -> None:
        super().__init__()
        self.collinear = collinear
        self.m_eps = m_eps

    @property
    def needs_gradient(self) -> bool:
        return self.collinear.needs_gradient

    @property
    def needs_tau(self) -> bool:
        """Meta-GGA on the spinor path: the SCF must build τ from the two-
        component orbitals and apply the 2×2 v_τ operator (see the spinor SCF
        and core.metagga.spinor_metagga_tau_operator)."""
        return getattr(self.collinear, "needs_tau", False)

    def energy(
        self,
        rho: torch.Tensor,
        m_vec: torch.Tensor,
        volume: float,
        sigma_uu: torch.Tensor | None = None,
        sigma_dd: torch.Tensor | None = None,
        sigma_tot: torch.Tensor | None = None,
        rho_core: torch.Tensor | None = None,
        tau_up: torch.Tensor | None = None,
        tau_dn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """E_xc [eV]. rho (grid), m_vec (3, grid). GGA σ's are those of the
        locally-collinear channels (caller builds them from ρ± spectra).
        rho_core (NLCC) shifts ρ only — same semantics as energy_with_grid, so
        the two entry points share the ρ± projection and cannot disagree.
        tau_up/tau_dn (meta-GGA) are the local-frame per-spin kinetic-energy
        densities the caller forms with `local_frame_tau`."""
        r_up, r_dn = _rho_pm(self, rho, m_vec, rho_core)
        return self.collinear.energy(r_up, r_dn, volume, sigma_uu, sigma_dd,
                                     sigma_tot, tau_up, tau_dn)


def _rho_pm(
    nc_xc: NoncollinearXC, rho: torch.Tensor, m_vec: torch.Tensor, rho_core: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locally-collinear channels ρ± = (ρ ± |m⃗|)/2, with the NLCC core (which
    carries no magnetization) folded into ρ before the split. Shared by both
    energy entry points so the projection is defined in exactly one place."""
    if rho_core is not None:
        rho = rho + rho_core
    m_norm = torch.sqrt((m_vec**2).sum(dim=0) + nc_xc.m_eps)
    return 0.5 * (rho + m_norm), 0.5 * (rho - m_norm)


def energy_with_grid(
    nc_xc: NoncollinearXC,
    rho: torch.Tensor,
    m_vec: torch.Tensor,
    grid: FFTGrid | SimpleNamespace,
    rho_core: torch.Tensor | None = None,
    tau_up: torch.Tensor | None = None,
    tau_dn: torch.Tensor | None = None,
) -> torch.Tensor:
    """E_xc with locally-collinear GGA σ's computed spectrally (in-graph).
    rho_core (NLCC) shifts ρ only — the core carries no magnetization.
    tau_up/tau_dn (meta-GGA) are the local-frame per-spin τ, passed through as
    held-fixed orbital fields exactly as the collinear path passes τ↑/τ↓."""
    from gradwave.core.density import sigma_from_rho

    r_up, r_dn = _rho_pm(nc_xc, rho, m_vec, rho_core)
    if nc_xc.collinear.needs_gradient:
        rho_tot = rho if rho_core is None else rho + rho_core
        s_uu = sigma_from_rho(r_up, grid.g_cart)
        s_dd = sigma_from_rho(r_dn, grid.g_cart)
        s_tt = sigma_from_rho(rho_tot, grid.g_cart)
    else:
        s_uu = s_dd = s_tt = None
    return nc_xc.collinear.energy(r_up, r_dn, grid.volume, s_uu, s_dd, s_tt,
                                  tau_up, tau_dn)


def vxc_and_bxc(
    nc_xc: NoncollinearXC,
    rho: torch.Tensor,
    m_vec: torch.Tensor,
    grid: FFTGrid,
    rho_core: torch.Tensor | None = None,
    tau_up: torch.Tensor | None = None,
    tau_dn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(v_xc(r), B⃗_xc(r), E_xc) via one autograd call — GGA terms included.

    For a meta-GGA τ_± are held FIXED (detached) during the (ρ, m⃗) autograd, so
    v_xc = ∂e/∂ρ and B⃗_xc = ∂e/∂m⃗ are the multiplicative parts; the τ response
    is the separate 2×2 generalized-KS operator (see vtau_up_dn /
    core.metagga.spinor_metagga_tau_operator), mirroring the collinear split."""
    r = rho.detach().clone().requires_grad_(True)
    m = m_vec.detach().clone().requires_grad_(True)
    tu = None if tau_up is None else tau_up.detach()
    td = None if tau_dn is None else tau_dn.detach()
    with torch.enable_grad():
        e = energy_with_grid(nc_xc, r, m, grid, rho_core=rho_core,
                             tau_up=tu, tau_dn=td)
        vr, vm = torch.autograd.grad(e, (r, m))
    scale = grid.n_points / grid.volume
    return vr * scale, vm * scale, e.detach()


def local_frame_tau(
    m_vec: torch.Tensor, tau_scalar: torch.Tensor, tau_vec: torch.Tensor, m_eps: float = 1e-24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local-frame per-spin τ_± = (τ_0 ± τ⃗·m̂)/2 from the KE-density-matrix
    Pauli components (τ_0, τ⃗) built by core.metagga.spinor_tau_matrix_b.

    m̂ = m⃗/|m⃗| is the SAME density magnetization axis that fixes
    ρ± = (ρ ± |m|)/2, so τ_± is the diagonal of the 2×2 KE-density matrix
    rotated into the frame that diagonalizes the spin density — the meta-GGA
    analogue of ρ±. With every moment along ẑ, τ⃗·m̂ = τ_z = τ_↑↑ − τ_↓↓ and
    (τ_+, τ_-) = (τ_↑↑, τ_↓↓), the collinear per-spin τ."""
    m_norm = torch.sqrt((m_vec ** 2).sum(dim=0) + m_eps)
    m_hat = m_vec / m_norm
    tau_par = (tau_vec * m_hat).sum(dim=0)
    return 0.5 * (tau_scalar + tau_par), 0.5 * (tau_scalar - tau_par)


def tau_operator_fields(
    v_tau_up: torch.Tensor, v_tau_dn: torch.Tensor, m_vec: torch.Tensor, m_eps: float = 1e-24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the local-frame τ potentials (v_τ↑, v_τ↓) back to the 2×2
    operator fields (v_τ0, v_τ⃗) that core.metagga.spinor_metagga_tau_operator
    consumes:

        v_τ0 = (v_τ↑ + v_τ↓)/2,   v_τ⃗ = (v_τ↑ − v_τ↓)/2 · m̂,

    the τ analogue of V = v_xc·𝟙 + B⃗_xc·σ⃗ with the same m̂ used for τ_±. With
    m⃗ ∥ ẑ this makes the operator diagonal, diag(v_τ↑, v_τ↓)."""
    m_norm = torch.sqrt((m_vec ** 2).sum(dim=0) + m_eps)
    m_hat = m_vec / m_norm
    v0 = 0.5 * (v_tau_up + v_tau_dn)
    vvec = 0.5 * (v_tau_up - v_tau_dn) * m_hat
    return v0, vvec


def vtau_up_dn(nc_xc: NoncollinearXC, rho, m_vec, grid, tau_up, tau_dn,
               rho_core=None):
    """(v_τ↑, v_τ↓) = ∂E_xc/∂τ_± at fixed (ρ, m⃗, σ), scaled to the pointwise
    energy-density derivative the τ operator consumes. τ_± are the leaves; ρ, m⃗
    are held fixed — the locally-collinear meta-GGA treats τ as an independent
    orbital field, exactly like the collinear vtau_spin_potential. A τ-flat
    functional (∂e/∂τ = 0) yields inert zero operators."""
    r = rho.detach()
    m = m_vec.detach()
    tu = tau_up.detach().clone().requires_grad_(True)
    td = tau_dn.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        e = energy_with_grid(nc_xc, r, m, grid, rho_core=rho_core,
                             tau_up=tu, tau_dn=td)
        if not e.requires_grad:
            z = torch.zeros_like(tu)
            return z, z.clone()
        vu, vd = torch.autograd.grad(e, (tu, td), allow_unused=True)
    scale = grid.n_points / grid.volume
    vu = torch.zeros_like(tu) if vu is None else vu * scale
    vd = torch.zeros_like(td) if vd is None else vd * scale
    return vu, vd
