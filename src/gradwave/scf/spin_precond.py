"""Stoner preconditioner for the magnetization channel (Layer B).

Near a ferromagnetic solution the SCF map's magnetization mode carries a
Stoner-enhanced gain (the dielectric operator 1 − K χ₀ has a near-zero or
negative eigenvalue along it), and history-based mixing alone either
diverges, collapses the moment, or plateaus — measured extensively on fcc
Ni. Following arXiv:2606.26693, approximate the susceptibility by its
occupation-response diagonal (the Stoner model: rigid band shifts, no
orbital relaxation):

    χ₀^diag[δv](r) = Σ_α c_α ρ_α(r) ⟨ρ_α, δv⟩,   c_α = w_k f'(ε_α),

with ρ_α the band codensities |ψ_α|²/Ω of the Fermi-window states of BOTH
spin channels (both enter the m-m block with + sign), and take the local
LDA-diagonal of the spin kernel K_mm(r) = ∂v_m/∂m. The preconditioned
residual is the Newton model step

    M⁻¹ r,   M = I − χ₀^diag K_mm,

inverted EXACTLY by the Woodbury identity — χ₀^diag is rank-(n_FS bands),
so M⁻¹ = I + U (D⁻¹ − W†U)⁻¹ W† with an n_FS × n_FS dense solve. The same
f'/codensity machinery validated in the Fermi-surface adjoint
(postscf/uspp_implicit.py) supplies every ingredient.

Model approximations (fine for a preconditioner): smooth densities only
(no augmentation cross term), no δμ Fermi-shift coupling, sphere-truncated
pairings, kernel diagonal from the current (ρ, m). The paper's finding
holds here too: outside its regime the operator is close to identity and
does no harm.
"""

from __future__ import annotations

import torch

from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import CDTYPE


class StonerSpinPrecond:
    """(I − χ₀^diag K_mm)⁻¹ on the m-channel of the mixing vector."""

    def __init__(self, u_g: torch.Tensor, w_g: torch.Tensor,
                 cvals: torch.Tensor, volume: float) -> None:
        """u_g: (r, ng) codensities ρ̂_α(G) on the density sphere.
        w_g: (r, ng) kernel-weighted codensities (K_mm ρ_α)^(G).
        cvals: (r,) w_k f'(ε_α) (negative). volume: Ω (the ⟨,⟩ factor)."""
        d_inv = torch.diag(1.0 / (volume * cvals.to(CDTYPE)))
        # A = D⁻¹ − W†U  (r × r); pairing ⟨a,b⟩ = Ω Σ_G â* b̂ carries Ω into D
        a = d_inv - torch.einsum("ag,bg->ab", w_g.conj(), u_g)
        self._u = u_g
        self._w = w_g
        self.cvals = cvals
        self.volume = volume
        self._a_lu = torch.linalg.lu_factor(a)

    def apply(self, r_m: torch.Tensor) -> torch.Tensor:
        """M⁻¹ r on one m-channel sphere vector."""
        proj = torch.einsum("ag,g->a", self._w.conj(), r_m)
        sol = torch.linalg.lu_solve(*self._a_lu, proj[:, None])[:, 0]
        return r_m + torch.einsum("ag,a->g", self._u, sol)


def build_stoner_precond(system, coeffs_s, eigs_s, mu, scheme,
                         width, rho_tot, m_r, xc, fp_cut: float=1e-8,
                         max_bands: int=96):
    """Assemble the preconditioner from the current SCF iteration's state.

    coeffs_s/eigs_s: per-spin lists as in the scf_uspp loop. Returns None
    when no state carries Fermi-surface weight (insulating limit — the
    operator would be the identity)."""
    from gradwave.core.fftbox import g_to_r, r_to_g

    grid = system.grid
    shape, vol = grid.shape, grid.volume
    mask_flat = grid.dens_mask.reshape(-1)

    # f' per spin channel by autograd through the smearing scheme
    picks = []  # (isp, ik, band index, c = w_k f')
    for isp in (0, 1):
        eigs = eigs_s[isp]
        x = ((eigs - mu) / width).detach().requires_grad_(True)
        with torch.enable_grad():
            f = scheme.occupation(x)
            (dfdx,) = torch.autograd.grad(f.sum(), x)
        fp = dfdx / width  # (nk, nb), ≤ 0
        for ik in range(eigs.shape[0]):
            wk = float(system.kweights[ik])
            for n in torch.nonzero(fp[ik].abs() > fp_cut).flatten().tolist():
                picks.append((isp, ik, n, wk * float(fp[ik][n])))
    if not picks:
        return None
    # keep the largest |c| columns (cost control on dense meshes)
    picks.sort(key=lambda t: -abs(t[3]))
    picks = picks[:max_bands]

    # local m-m kernel diagonal K_mm(r) = ∂v_m/∂m at the current (ρ, m)
    fmm = _stoner_kernel_diag(rho_tot, m_r, xc, grid, vol, system.rho_core)

    u_rows, w_rows, cvals = [], [], []
    for isp, ik, n, c in picks:
        sph = system.spheres[ik]
        psi_r = g_to_r(coeffs_s[isp][ik][n], sph.flat_idx, shape)
        dens = (psi_r.conj() * psi_r).real / vol  # ∫ρ_α = 1
        u_rows.append(r_to_g(dens.to(CDTYPE)).reshape(-1)[mask_flat])
        w_rows.append(r_to_g((fmm * dens).to(CDTYPE)).reshape(-1)[mask_flat])
        cvals.append(c)
    return StonerSpinPrecond(
        torch.stack(u_rows), torch.stack(w_rows),
        torch.tensor(cvals, dtype=torch.float64, device=u_rows[0].device),
        float(vol))


def _stoner_kernel_diag(rho_tot, m_r, xc, grid, vol, rho_core):
    """Local m-m kernel diagonal K_mm(r) = ∂v_m/∂m at the current (ρ, m).

    For a local (LDA/GGA-diagonal) kernel, (K·1)(r) IS the diagonal — one
    double backward through E_xc^collinear(ρ±m). ``xc`` is a collinear SpinXC
    (the non-collinear path passes NoncollinearXC.collinear and m_r = |m⃗|,
    the locally-collinear longitudinal magnetization)."""
    from gradwave.scf.common import spin_sigmas

    rho_leaf = rho_tot.detach().clone()
    m_leaf = m_r.detach().clone().requires_grad_(True)
    # Double backward through E_xc for the Stoner kernel, so force eager.
    with torch.enable_grad(), xc_eager():
        up = 0.5 * (rho_leaf + m_leaf)
        dn = 0.5 * (rho_leaf - m_leaf)
        if rho_core is not None:
            up = up + 0.5 * rho_core
            dn = dn + 0.5 * rho_core
        s_uu, s_dd, s_tot = spin_sigmas(up, dn, xc, grid.g_cart)
        e_xc = xc.energy(up, dn, vol, s_uu, s_dd, s_tot)
        (v_m,) = torch.autograd.grad(e_xc, m_leaf, create_graph=True)
        (fmm,) = torch.autograd.grad(v_m.sum(), m_leaf)
    # grid second gradient Σ_j' ∂²E/∂m_j∂m_j' = (Ω/N)·(K·1)(r_j)
    return fmm * (grid.n_points / vol)


def build_stoner_precond_nc(system, coeffs, eigs, mu, scheme, width,
                            rho_tot, m_vec, nc_xc, m_pw, fp_cut: float=1e-8,
                            max_bands: int=96):
    """Stoner preconditioner for the NON-COLLINEAR moment channel.

    The non-collinear analogue of ``build_stoner_precond``. Near a
    ferromagnetic solution the LONGITUDINAL magnetization mode (m ∥ the local
    moment axis) carries the same Stoner-enhanced gain that stalls the
    collinear FM metal — here it is the m_z channel of the (ρ, m⃗) mixing
    vector (measured on fcc Ni + SOC: |m_z| residual floors at ~1e-4 while ρ
    and the transverse m_x/m_y run three orders lower). In the locally-
    collinear XC (``NoncollinearXC``, ρ± = (ρ ± |m⃗|)/2) the longitudinal
    kernel K_mm IS the collinear one evaluated at m = |m⃗|, so the same
    Woodbury-inverted Newton model (I − χ₀^diag K_mm)⁻¹ applies.

    Differences from the collinear builder: the susceptibility codensities come
    from SPINOR bands (one Fermi-degeneracy g=1 list, ρ_α = |ψ_α↑|² + |ψ_α↓|²
    summed over both spinor components) and the kernel uses the locally-
    collinear longitudinal magnitude m = |m⃗|. The returned operator is applied
    to each Cartesian m-channel; near a collinear FM state the transverse
    channels are in the identity regime of the operator and it does no harm.

    coeffs: (nk, nbands, 2·m_pw) spinor coefficients (up = [:m_pw], down =
    [m_pw:]). eigs: (nk, nbands). Returns None in the insulating limit (no
    Fermi-surface weight)."""
    from gradwave.core.fftbox import g_to_r, r_to_g

    grid = system.grid
    shape, vol = grid.shape, grid.volume
    mask_flat = grid.dens_mask.reshape(-1)
    device = coeffs.device

    # f' per spinor band by autograd through the smearing scheme (g=1)
    x = ((eigs - mu) / width).detach().requires_grad_(True)
    with torch.enable_grad():
        f = scheme.occupation(x)
        (dfdx,) = torch.autograd.grad(f.sum(), x)
    fp = dfdx / width  # (nk, nb), ≤ 0
    picks = []  # (ik, band index, c = w_k f')
    for ik in range(eigs.shape[0]):
        wk = float(system.kweights[ik])
        for n in torch.nonzero(fp[ik].abs() > fp_cut).flatten().tolist():
            picks.append((ik, n, wk * float(fp[ik][n])))
    if not picks:
        return None
    picks.sort(key=lambda t: -abs(t[2]))
    picks = picks[:max_bands]

    # longitudinal kernel diagonal from the locally-collinear magnitude |m⃗|
    m_mag = torch.sqrt((m_vec.detach() ** 2).sum(dim=0) + nc_xc.m_eps)
    fmm = _stoner_kernel_diag(rho_tot, m_mag, nc_xc.collinear, grid, vol,
                              system.rho_core)

    u_rows, w_rows, cvals = [], [], []
    for ik, n, c in picks:
        sph = system.spheres[ik]
        npw = sph.npw
        psi_up = g_to_r(coeffs[ik, n, :npw], sph.flat_idx, shape)
        psi_dn = g_to_r(coeffs[ik, n, m_pw:m_pw + npw], sph.flat_idx, shape)
        dens = ((psi_up.conj() * psi_up).real
                + (psi_dn.conj() * psi_dn).real) / vol  # ∫ρ_α = 1 (spinor norm)
        u_rows.append(r_to_g(dens.to(CDTYPE)).reshape(-1)[mask_flat])
        w_rows.append(r_to_g((fmm * dens).to(CDTYPE)).reshape(-1)[mask_flat])
        cvals.append(c)
    return StonerSpinPrecond(
        torch.stack(u_rows), torch.stack(w_rows),
        torch.tensor(cvals, dtype=torch.float64, device=device),
        float(vol))
