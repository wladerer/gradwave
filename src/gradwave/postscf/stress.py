"""Stress tensor via autograd through a strain parameterization (Layer A entry).

At SCF convergence the free energy is stationary in (ψ, f), so the fixed-basis
(Nielsen–Martin) stress is the partial derivative of the energy expression at
fixed plane-wave coefficients under a homogeneous strain ε:

    r → (1+ε) r,   a_i → (1+ε) a_i,   τ → (1+ε) τ,
    G_m = m·B → m·B(ε)  (integer Miller labels m fixed),
    Ω → det(1+ε) Ω,     ρ(G) → ρ(G)·Ω₀/Ω(ε)   (fixed coefficients c),

    σ_αβ = (1/Ω) ∂E/∂ε_αβ            (P = −tr σ / 3)

QE's analytic stress uses the same fixed-basis convention, so the two agree
at identical ecut/k/pseudopotentials (the shared basis-set incompleteness —
"Pulay stress" — is not corrected by either code).

Every ε-dependent quantity is rebuilt from integers (Miller indices, k_frac,
image counts) on the autograd graph; radial form factors go through the
differentiable spherical Bessel transforms in pseudo/radial_torch.py. The
smearing entropy has no explicit ε-dependence at fixed occupations and is
omitted. The ε = 0 energy of this expression must reproduce the SCF
breakdown to ~1e-9 eV — tested, and worth asserting when debugging.

Sign convention: σ as returned is +(1/Ω)∂E/∂ε (tension positive); QE prints
the negative of the pressure-like part the same way, so the comparison in the
tests is direct. Units: eV/Å³; stress_kbar() converts.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.constants import MINUS_I_POW as _MINUS_I_POW
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.radial_torch import RadialTables

EV_A3_TO_KBAR = 1602.176634  # 1 eV/Å³ = 160.2176634 GPa


def stress_kbar(sigma: torch.Tensor) -> torch.Tensor:
    return sigma * EV_A3_TO_KBAR


def stress(res, xc, symmetrize: bool = True) -> torch.Tensor:
    """σ_αβ = (1/Ω) ∂E/∂ε_αβ at the converged SCF point, (3,3) [eV/Å³].

    nspin=1, scalar-relativistic, no +U (the Hubbard strain term is not
    implemented). NLCC is handled (the core density is rebuilt on the graph).
    """
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("stress for nspin=2 not implemented yet")
    system = res.system
    if getattr(system, "is_fr", False):
        raise NotImplementedError("stress for fully-relativistic pseudos not implemented yet")
    if getattr(res, "hub_occ", None) is not None:
        raise NotImplementedError("stress with DFT+U not implemented yet")

    dev = system.positions.device
    eps = torch.zeros(3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    e = _energy_strained(res, xc, eps)
    (grad,) = torch.autograd.grad(e, eps)

    omega0 = system.grid.volume
    sigma = 0.5 * (grad + grad.T) / omega0
    if symmetrize and system.sym is not None:
        sigma = symmetrize_stress(sigma, system.sym, system.grid.cell)
    return sigma


def symmetrize_stress(sigma: torch.Tensor, sg, cell: np.ndarray) -> torch.Tensor:
    """σ ← (1/N) Σ_op S σ Sᵀ with S the Cartesian rotation of each op."""
    a_t = np.asarray(cell, dtype=float).T
    acc = torch.zeros_like(sigma)
    for w_mat in sg.rotations:
        s = torch.as_tensor(
            a_t @ w_mat @ np.linalg.inv(a_t), dtype=sigma.dtype, device=sigma.device
        )
        acc = acc + s @ sigma @ s.T
    return acc / sg.n_ops


def _energy_strained(
    res, xc, eps: torch.Tensor, *, rho=None, coeffs=None, spheres=None
) -> torch.Tensor:
    """The KS energy as a function of strain at fixed coefficients/occupations.

    Also usable with a plain (non-leaf) eps for finite-difference checks.

    The optional ``rho`` (grid density), ``coeffs`` (per-k, on ``spheres``) and
    ``spheres`` override the converged, detached electronic state. They are used
    WITHOUT detaching, so a caller can carry an extra autograd graph through them
    (e.g. a density-matrix perturbation for a discretization-error estimate).
    When omitted the converged detached state is used, i.e. the plain stress.
    """
    system = res.system
    grid = system.grid
    dev = system.positions.device
    rdt = RDTYPE

    rho = res.rho.detach() if rho is None else rho
    coeffs = [c.detach() for c in res.coeffs] if coeffs is None else coeffs
    spheres = system.spheres if spheres is None else spheres

    f_map = torch.eye(3, dtype=rdt, device=dev) + eps
    a0 = torch.as_tensor(grid.cell, dtype=rdt, device=dev)
    a_e = a0 @ f_map.T  # rows a_i → (1+ε) a_i
    b_e = 2.0 * math.pi * torch.linalg.inv(a_e).T
    omega0 = grid.volume
    omega = torch.linalg.det(a_e)
    omega = omega * torch.sign(omega.detach())
    pos_e = system.positions.detach() @ f_map.T

    # dense-box Miller indices, restricted to the density sphere
    shape = grid.shape
    mask = grid.dens_mask.reshape(-1)
    m_box = _box_millers(shape, dev)  # (N, 3) float64
    m_sph = m_box[mask]
    g_sph = m_sph @ b_e  # (nGm, 3)
    g2_sph = (g_sph**2).sum(-1)
    is_g0 = g2_sph.detach() < 1e-12
    q_sph = torch.sqrt(torch.where(is_g0, torch.ones_like(g2_sph), g2_sph))
    q_sph = torch.where(is_g0, torch.zeros_like(q_sph), q_sph)

    # fixed density coefficients: ρ̃(G) = ρ(G)·Ω₀  [e]
    from gradwave.core.fftbox import r_to_g

    rho_t = (r_to_g(rho.to(torch.complex128)) * omega0).reshape(-1)[mask]
    rho_g = rho_t / omega.to(rho_t.dtype)

    # ---- Hartree (G=0 excluded)
    g2_safe = torch.where(is_g0, torch.ones_like(g2_sph), g2_sph)
    inv_g2 = torch.where(is_g0, torch.zeros_like(g2_sph), 1.0 / g2_safe)
    e_h = 0.5 * 4.0 * math.pi * E2 / omega * ((rho_t.abs() ** 2) * inv_g2).sum()

    # ---- local pseudopotential (G=0 carries alpha-Z)
    tabs = [RadialTables(u, device=dev) for u in system.upfs]
    phase_arg = g_sph @ pos_e.T  # (nGm, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))
    e_loc = torch.zeros((), dtype=rdt, device=dev)
    for sp, tab in enumerate(tabs):
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        if not atoms:
            continue
        s_sp = phases[:, atoms].sum(dim=1)  # (nGm,)
        v = torch.zeros_like(q_sph)  # fresh buffer: index-assign is autograd-safe here
        v[~is_g0] = tab.vloc_of_g(q_sph[~is_g0])
        v[is_g0] = tab.alpha
        e_loc = e_loc + (rho_g.conj() * s_sp * v.to(rho_g.dtype)).sum().real

    # ---- XC (density values scale as 1/detJ; NLCC core rebuilt on the graph)
    rho_e = rho * (omega0 / omega)
    rho_xc = rho_e
    if system.rho_core is not None:
        core = torch.zeros(g2_sph.shape[0], dtype=torch.complex128, device=dev)
        for sp, tab in enumerate(tabs):
            if tab.core_g is None:
                continue
            atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
            if not atoms:
                continue
            f_core = tab.core_of_g(q_sph)
            core = core + phases[:, atoms].sum(dim=1) * f_core.to(torch.complex128) / omega.to(
                torch.complex128
            )
        n_pts = grid.n_points
        core_box = torch.zeros(n_pts, dtype=torch.complex128, device=dev)
        core_box[mask] = core
        rho_core_e = torch.fft.ifftn(
            core_box.reshape(shape) * n_pts, dim=(-3, -2, -1)
        ).real
        rho_xc = rho_e + rho_core_e
    sigma_xc = None
    if xc.needs_gradient:
        from gradwave.core.density import sigma_from_rho

        g_box = (m_box @ b_e).reshape(*shape, 3)
        sigma_xc = sigma_from_rho(rho_xc, g_box)
    e_xc = xc.energy(rho_xc, omega, sigma_xc)

    # ---- kinetic + nonlocal, per k (strained k+G from integer Miller + k_frac)
    occ = res.occupations.detach()
    kw = system.kweights
    e_kin = torch.zeros((), dtype=rdt, device=dev)
    e_nl = torch.zeros((), dtype=rdt, device=dev)
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    for ik, sph in enumerate(spheres):
        kfrac = torch.as_tensor(sph.k_frac, dtype=rdt, device=dev)
        kpg = (sph.miller.to(rdt) + kfrac) @ b_e  # (npw, 3)
        kpg2 = (kpg**2).sum(-1)
        c = coeffs[ik]
        band = torch.einsum("bg,g->b", (c.real**2 + c.imag**2), HBAR2_2M * kpg2)
        e_kin = e_kin + (kw[ik] * occ[ik, : c.shape[0]] * band).sum()

        pd = system.proj_data[ik]
        if pd.dij_full.shape[0] == 0:
            continue
        q_k = torch.sqrt(kpg2.clamp_min(1e-30))
        q_k = torch.where(kpg2.detach() < 1e-24, torch.zeros_like(q_k), q_k)
        y = ylm_all(lmax, kpg)
        pref = 4.0 * math.pi / torch.sqrt(omega)
        cols = []
        for sp in system.species_of_atom:
            tab = tabs[sp]
            for i, l in enumerate(tab.beta_l):
                f = tab.beta_of_g(i, q_k)
                for m_col in range(2 * l + 1):
                    cols.append(
                        (pref * f * y[:, l * l + m_col]).to(torch.complex128)
                        * _MINUS_I_POW[l]
                    )
        p = torch.stack(cols, dim=0)  # (nproj_tot, npw), matches pd ordering
        parg = kpg @ pos_e.T  # (npw, na)
        ph = torch.exp(torch.complex(torch.zeros_like(parg), -parg))
        p = p * ph[:, pd.atom_index].T
        b_ovl = c @ p.conj().T  # (nb, nproj_tot)
        quad = torch.einsum(
            "bi,ij,bj->b", b_ovl.conj(), pd.dij_full.to(b_ovl.dtype), b_ovl
        ).real
        e_nl = e_nl + (kw[ik] * occ[ik, : c.shape[0]] * quad).sum()

    # ---- Ewald (integer image/G sets fixed at ε=0, vectors strained)
    e_ew = _ewald_strained(pos_e, system.charges, a_e, b_e, omega, grid.cell)

    return e_kin + e_h + e_xc + e_loc + e_nl + e_ew


def _box_millers(shape, device) -> torch.Tensor:
    axes = [np.fft.fftfreq(n, d=1.0 / n).astype(np.float64) for n in shape]
    m = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return torch.as_tensor(m, dtype=torch.float64, device=device)


def _ewald_strained(pos_e, charges, a_e, b_e, omega, cell0) -> torch.Tensor:
    """ewald_energy with the cell on the autograd graph. η and the integer
    image/G-vector sets come from the unstrained cell (the excluded boundary
    terms are erfc(8)-suppressed, so their ε-derivative is negligible)."""
    from gradwave.core.energies.ewald import _ACC, _g_vectors, _image_vectors

    cell0 = np.asarray(cell0, dtype=np.float64)
    omega0 = abs(np.linalg.det(cell0))
    eta = (math.pi / omega0 ** (1.0 / 3.0)) ** 2
    sqrt_eta = math.sqrt(eta)
    rcut = _ACC / sqrt_eta
    gcut = 2.0 * sqrt_eta * _ACC

    dev = pos_e.device
    rdt = torch.float64
    # integer labels of the ε=0 sets
    n_img = np.round(_image_vectors(cell0, rcut) @ np.linalg.inv(cell0)).astype(np.int64)
    b0 = 2.0 * math.pi * np.linalg.inv(cell0).T
    m_g = np.round(_g_vectors(cell0, gcut) @ np.linalg.inv(b0)).astype(np.int64)

    images = torch.as_tensor(n_img, dtype=rdt, device=dev) @ a_e
    gvecs = torch.as_tensor(m_g, dtype=rdt, device=dev) @ b_e
    z = charges.to(rdt)

    d = pos_e[:, None, None, :] - pos_e[None, :, None, :] + images[None, None, :, :]
    r = torch.linalg.norm(d, dim=-1)
    na = r.shape[0]
    img0 = torch.as_tensor((np.abs(n_img).sum(axis=1) == 0), device=dev)
    self_pair = torch.eye(na, dtype=torch.bool, device=dev)[:, :, None] & img0[None, None, :]
    r_safe = torch.where(self_pair, torch.ones_like(r), r)
    pair = torch.erfc(sqrt_eta * r_safe) / r_safe
    pair = torch.where(self_pair, torch.zeros_like(pair), pair)
    e_real = 0.5 * E2 * torch.einsum("a,b,abr->", z, z, pair)

    g2 = (gvecs**2).sum(-1)
    phase = pos_e @ gvecs.T
    s_re = (z[:, None] * torch.cos(phase)).sum(0)
    s_im = (z[:, None] * torch.sin(phase)).sum(0)
    e_recip = (2.0 * math.pi * E2 / omega) * (
        (s_re**2 + s_im**2) * torch.exp(-g2 / (4.0 * eta)) / g2
    ).sum()

    e_self = -E2 * sqrt_eta / math.sqrt(math.pi) * (z**2).sum()
    e_bg = -math.pi * E2 / (2.0 * eta * omega) * z.sum() ** 2
    return e_real + e_recip + e_self + e_bg
