"""Differentiable band gap of a converged hybrid, as a function of (α, ω).

The total-energy gradient dE/dθ is exact at self-consistency (stationarity). An
eigenvalue is not stationary, so the gap gradient here is the *frozen-orbital*
Hellmann-Feynman one: hold the converged orbitals and density fixed and vary only
the explicit θ in the Hamiltonian. For orbital i,

    ε_i(α,ω) = ε_i^conv + α·Δ_i(ω) − α_conv·Δ_i(ω_conv),
    Δ_i(ω)  = ⟨i|V_x^Fock(ω)|i⟩ − ⟨i|v_x^PBE|i⟩,

mirroring ``differentiable_hybrid_energy``: the value equals the converged
ε_c − ε_v and the scalar carries dgap/dα, dgap/dω. The frozen-orbital derivative
omits the SCF response of the eigenvalues (the density re-relaxing with θ); its
size is measured against finite difference in ``validate.py``.

Convention note: the diagonal Fock element reuses the exact inner loop of
``multik_exchange_energy`` (same periodic-orbital normalization), and
``exx_energy_from_diagonal`` cross-checks that ½ Σ_i w_ki ⟨i|V_x|i⟩ reproduces it.
"""
import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.postscf.coulomb_kernel import coulomb_kernel
from gradwave.postscf.exchange_multik import occupied_periodic_orbitals
from gradwave.postscf.hybrid import ScaledExchangePBE


def _orbital_r(res, ik, ib):
    """Periodic part u_{ik}(r) of one band, same normalization as
    occupied_periodic_orbitals (g_to_r on the plane-wave coefficients)."""
    sph = res.system.spheres[ik]
    return g_to_r(res.coeffs[ik][ib:ib + 1], sph.flat_idx,
                  res.system.grid.shape).reshape(-1)


def diagonal_fock(u_i, k_i, u_occ, kcart, kweights, g_cart, vol, *,
                  mode="full", omega=None):
    """⟨i|V_x^Fock(ω)|i⟩ for orbital i (periodic part u_i at k_i) against the
    occupied manifold. Sum over the BZ of occupied (kb, j): −Σ w_kb Σ_G
    |FT[u_i* u_j](G)|² K(k_b−k_i+G) / Ω. Differentiable in ω."""
    shape = tuple(g_cart.shape[:3])
    total = g_cart.new_zeros(())
    for kb, ub in enumerate(u_occ):
        q = kcart[kb] - k_i
        qg2 = ((g_cart + q) ** 2).sum(dim=-1)
        kern = coulomb_kernel(qg2, mode, omega)
        p = u_i.conj()[None, :] * ub                       # (n_jb, N_r)
        p_g = r_to_g(p.reshape(p.shape[0], *shape))
        contrib = ((p_g.abs() ** 2) * kern).sum(dim=(-3, -2, -1))
        total = total + kweights[kb] * contrib.sum()
    return -total / vol


def exx_energy_from_diagonal(res, *, mode="full", omega=None):
    """½ Σ_ik w_k ⟨i|V_x|i⟩ over occupied bands — must equal
    multik_exchange_energy (a factor/convention self-check)."""
    system = res.system
    vol, g_cart = system.grid.volume, system.grid.g_cart
    u_occ, kcart, kw = occupied_periodic_orbitals(res, system)
    e = g_cart.new_zeros(())
    for ka, ua in enumerate(u_occ):
        for ib in range(ua.shape[0]):
            d = diagonal_fock(ua[ib], kcart[ka], u_occ, kcart, kw, g_cart, vol,
                              mode=mode, omega=omega)
            e = e + kw[ka] * d
    return 0.5 * e


def vbm_cbm(res, occ_tol=1e-6):
    """Fundamental-gap edges over the mesh: (ik, ib, eps) for VBM and CBM."""
    occ, eig = res.occupations, res.eigenvalues
    vbm = (-1, -1, -1e30)
    cbm = (-1, -1, +1e30)
    for ik in range(eig.shape[0]):
        for ib in range(eig.shape[1]):
            e = float(eig[ik, ib])
            if float(occ[ik, ib]) > occ_tol:
                if e > vbm[2]:
                    vbm = (ik, ib, e)
            elif e < cbm[2]:
                cbm = (ik, ib, e)
    return vbm, cbm


def _pbe_vx_r(res):
    """PBE exchange potential on the grid [eV], v_x = δ(∫ρ ε_x^PBE)/δρ, by autograd
    (θ-independent — evaluated on the frozen converged density)."""
    system = res.system
    vol = system.grid.volume
    rho = res.rho if system.rho_core is None else res.rho + system.rho_core
    rho = rho.detach().clone().requires_grad_(True)
    sigma = sigma_from_rho(rho, system.grid.g_cart)
    e = ScaledExchangePBE(1.0).energy(rho, vol, sigma)  # ∫ρ ε_x^PBE
    (g,) = torch.autograd.grad(e, rho)
    n = rho.numel()
    return (g * n / vol).detach()                        # δE/δρ → v_x(r)


def _pbe_diag(u_i, vx_r, vol):
    """⟨i|v_x^PBE|i⟩ = ∫|ψ_i|² v_x dr, ψ_i = u_i/√Ω (periodic part u_i)."""
    n = u_i.numel()
    # ∫|ψ|²v = (1/N)Σ_r |u_r|² v_r when (1/N)Σ|u_r|² = ⟨ψ|ψ⟩ = 1
    return ((u_i.abs() ** 2) * vx_r.reshape(-1)).sum().real / n


def differentiable_hybrid_gap(res, params, occ_tol=1e-6):
    """Converged fundamental gap ε_c − ε_v as a differentiable function of
    (α, ω) via the frozen-orbital derivative. Value equals the converged gap;
    ``.backward()`` carries dgap/dα, dgap/dω into ``params``."""
    system = res.system
    vol, g_cart = system.grid.volume, system.grid.g_cart
    u_occ, kcart, kw = occupied_periodic_orbitals(res, system, occ_tol)
    u_occ = [x.detach() for x in u_occ]
    (ikv, ibv, ev), (ikc, ibc, ec) = vbm_cbm(res, occ_tol)
    uv = _orbital_r(res, ikv, ibv).detach()
    uc = _orbital_r(res, ikc, ibc).detach()
    omega = params.omega if params.mode != "full" else None
    vx_r = _pbe_vx_r(res)
    # Δ_i(ω) = ⟨i|V_x^Fock(ω)|i⟩ − ⟨i|v_x^PBE|i⟩
    dv = (diagonal_fock(uv, kcart[ikv], u_occ, kcart, kw, g_cart, vol,
                        mode=params.mode, omega=omega) - _pbe_diag(uv, vx_r, vol))
    dc = (diagonal_fock(uc, kcart[ikc], u_occ, kcart, kw, g_cart, vol,
                        mode=params.mode, omega=omega) - _pbe_diag(uc, vx_r, vol))
    gap_theta = params.alpha * (dc - dv)
    gap_const = (ec - ev) - float(gap_theta.detach())
    return gap_const + gap_theta
