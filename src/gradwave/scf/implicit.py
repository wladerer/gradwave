"""Implicit differentiation through the SCF fixed point (M4) — insulators.

For a loss L(ρ) on the CONVERGED density, gradients w.r.t. functional
parameters θ require the density response (unlike energy losses, which get
dE/dθ free by stationarity). The adjoint formulation solves ONE
self-consistent linear problem regardless of the number of parameters:

    u = v̄ + K_Hxc[χ₀ u],        v̄(r) = ∂L/∂ρ(r)
    dL/dθ = ⟨χ₀ u, ∂v_xc/∂θ⟩_grid + ∂L/∂θ|_explicit

χ₀ w:  independent-particle response of the density to a local potential w.
       Insulator: one conduction-projected Sternheimer solve (H − ε_n +
       αP_occ)|δψ_n⟩ = −P_c w|ψ_n⟩ per occupied band per k (CG; positive
       definite on the conduction space thanks to the gap). Metal (any
       fractional occupation): the window scheme (``_chi0_channel_metal``) —
       an above-window Sternheimer (project out the *whole* computed window,
       positive definite through the Fermi level), plus the in-window band
       pairs with divided-difference occupation weights, plus the rank-one
       δμ Fermi-level-shift term that conserves particle number. It reduces
       smoothly to the insulator response as the Fermi-surface weight occ'→0.
K_Hxc: Hartree kernel 4πe²/G² + f_xc, with f_xc·w obtained as an autograd
       Hessian-vector product of E_xc — any twice-differentiable functional
       works automatically, including learnable ones.

Degeneracy note: P_c = 1 − Σ_occ |ψ⟩⟨ψ| projects out the ENTIRE occupied
subspace, so degenerate valence tops (Si Γ) are handled correctly.

nspin=2 (collinear): the density becomes a per-channel pair (ρ↑, ρ↓) and
every structure above becomes a two-element list. χ₀ is block-diagonal over
spin — each channel has its own occupied bands, its own v_eff^σ and its own
Sternheimer solves, with the single-band degeneracy weight g = 1 instead of
2. K_Hxc keeps its cross-spin blocks: the Hartree kernel acts on the TOTAL
δρ = δρ↑ + δρ↓ and enters both channels, while f_xc^{σσ'} is the spin HVP of
the grid E_xc (fxc_hvp_spin, NLCC core split half/half). The loss stays a
functional of the total density, so v̄ = ∂L/∂ρ seeds both channels equally.
This mirrors the USPP/PAW twin in postscf/uspp_implicit.py, minus the
augmentation/one-center blocks (norm-conserving has none). The metallic
Fermi-surface channel above applies per spin channel independently, so a
collinear magnet with a partially-filled spin channel is handled.

This module is the mathematical core that a torch.autograd.Function wrapper
(and torch.func Hessians) will build on; the direct API here is
`density_loss_param_grads`.
"""

from __future__ import annotations

from typing import cast

import torch

# Cycle-free import direction: postscf._response depends only on core/, and
# gradwave.postscf's __init__ is empty, so scf modules may pull the shared
# response kernels from there.
from gradwave.core._anderson import AndersonMixer
from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.core.occupations import SCHEMES
from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import RDTYPE
from gradwave.postscf._response import (
    divided_difference_weights,
    fxc_hvp,
    fxc_hvp_spin,
    hartree_kernel,
    is_insulating,
    occupation_derivative,
    spin_sigma_triple,
    sternheimer_shift,
)
from gradwave.scf.common import symmetrize_rho, symmetrize_rho_pair
from gradwave.scf.loop import SCFResult
from gradwave.solvers.precond import teter
from gradwave.symmetry import CollinearMagneticSymmetrizer, RhoSymmetrizer


def _check_no_symmetry(res: SCFResult):
    if getattr(res.system, "sym", None) is not None:
        raise NotImplementedError(
            "implicit SCF backward requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full "
            "(TR-reduced) k-mesh"
        )


def _sym_fold(symmetrizer, field, grid, nspin):
    """Fold a totally-symmetric real field into the symmetric subspace, matching
    the forward SCF's ``_output_density`` dispatch (scf/loop.py).

    nspin=1: scalar ``RhoSymmetrizer`` round-trip. nspin=2 (collinear): dispatch
    on the symmetrizer class exactly as the forward density does — a
    ``CollinearMagneticSymmetrizer`` folds the (ρ↑, ρ↓) pair jointly via the
    (n=ρ↑+ρ↓, m=ρ↑−ρ↓) representation (the −1 anti-unitary sign on m, channel
    swap), a plain ``RhoSymmetrizer`` (uniform-moment / FM, chemical group
    unbroken) folds each spin channel independently. ``field`` is (2, *grid) for
    nspin=2."""
    if nspin == 1:
        return symmetrize_rho(symmetrizer, field, grid)
    if isinstance(symmetrizer, CollinearMagneticSymmetrizer):
        up, dn = symmetrize_rho_pair(symmetrizer, field[0], field[1], grid)
        return torch.stack([up, dn])
    if isinstance(symmetrizer, RhoSymmetrizer):
        return torch.stack([symmetrize_rho(symmetrizer, field[isp], grid)
                            for isp in range(nspin)])
    raise NotImplementedError(
        "symmetric χ₀ fold (nspin=2) supports RhoSymmetrizer (uniform-moment) "
        f"and CollinearMagneticSymmetrizer, not {type(symmetrizer).__name__}")


def _occupied(res: SCFResult, isp: int, ik: int):
    """Occupied block (coeffs, ε) of spin channel ``isp`` at k-point ``ik``.

    Insulators only: every occupied band must be filled to ``f_full`` (2 for
    nspin=1, 1 per channel for nspin=2), the rest empty. Each spin channel is
    checked independently, so a collinear magnet with N↑ ≠ N↓ integer-filled
    channels is supported."""
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        occ = res.occupations[ik]
        coeffs, eps = res.coeffs[ik], res.eigenvalues[ik]
        f_full = 2.0
    else:
        occ = res.occupations[isp][ik]
        coeffs, eps = res.coeffs[isp][ik], res.eigenvalues[isp][ik]
        f_full = 1.0
    n_occ = int((occ > 1e-8).sum())
    if not torch.all((occ[:n_occ] - f_full).abs() < 1e-8):
        raise NotImplementedError(
            f"implicit SCF backward supports insulators only (occ = {f_full})")
    return coeffs[:n_occ], eps[:n_occ]


def _hamiltonians(res: SCFResult) -> list[HamiltonianK] | list[list[HamiltonianK]]:
    """Per-k HamiltonianK (nspin=1) or per-spin, per-k list-of-lists (nspin=2).

    For nspin=2 each channel uses its own v_eff^σ; the plane-wave sphere and
    the KB projectors are spin-independent, so they are built once per k."""
    system = res.system
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        hs = []
        for ik, sph in enumerate(system.spheres):
            p = projectors(system.proj_data[ik], system.positions)
            hs.append(HamiltonianK(sph, system.grid.shape, res.v_eff,
                                   system.proj_data[ik], p))
        return hs
    hs = [[] for _ in range(nspin)]
    for ik, sph in enumerate(system.spheres):
        p = projectors(system.proj_data[ik], system.positions)
        for isp in range(nspin):
            hs[isp].append(HamiltonianK(sph, system.grid.shape, res.v_eff[isp],
                                        system.proj_data[ik], p))
    return hs


def projected_cg(a_apply, precond, x, r, tol: float, max_iter: int):
    """Projected preconditioned CG with a per-band breakdown guard.

    Solves ``A x = rhs`` in batched (per-band) form given the initial guess
    ``x`` and residual ``r = rhs − A x``. ``a_apply`` is the SPD operator on the
    conduction space; ``precond(r)`` applies the preconditioner — the USPP twin
    folds the S-projector into it, the norm-conserving path passes a plain Teter
    preconditioner. Bands whose curvature goes non-positive or non-finite are
    frozen (their search direction is zeroed); the operator is positive definite
    on the conduction space, so this only fires at the round-off floor after a
    band has already converged, where an unguarded ``pap ≤ 0`` would give a
    1e300 step → Inf → NaN.

    No autograd path — the callers are the no-grad inner solves of implicit
    differentiation. Returns ``x``; any final projection is the caller's.
    """
    z = precond(r)
    p = z
    rz = torch.einsum("bg,bg->b", r.conj(), z).real
    for _ in range(max_iter):
        ap = a_apply(p)
        pap = torch.einsum("bg,bg->b", p.conj(), ap).real
        p2 = torch.einsum("bg,bg->b", p.conj(), p).real
        ok = torch.isfinite(pap) & (pap > 1e-30 * p2.clamp_min(1e-300))
        if not bool(ok.any()):
            break
        a_cg = torch.where(ok, rz / pap.clamp_min(1e-300), torch.zeros_like(rz))
        x = x + a_cg[:, None] * p
        r = r - a_cg[:, None] * ap
        if float(torch.linalg.norm(r, dim=1).max()) < tol:
            break
        z = precond(r)
        rz_new = torch.einsum("bg,bg->b", r.conj(), z).real
        beta = torch.where(ok, rz_new / rz.clamp_min(1e-300), torch.zeros_like(rz))
        p = torch.where(ok[:, None], z + beta[:, None] * p, torch.zeros_like(p))
        rz = rz_new
    return x


def _sternheimer(h: HamiltonianK, c_occ, eps_occ, w_r, alpha: float, tol: float, max_iter: int):
    """Solve (H − ε_n + α P_occ) δψ_n = −P_c w ψ_n for all occupied n at one k.

    Returns δψ (n_occ, npw), entirely in the conduction space.
    """

    def p_c(x):
        return x - (x @ c_occ.conj().T) @ c_occ

    # RHS: −P_c (w ψ_n)
    psi_r = g_to_r(c_occ, h.sphere.flat_idx, h.shape)
    w_psi = box_to_sphere(r_to_g(psi_r * w_r), h.sphere.flat_idx)
    rhs = -p_c(w_psi)

    def a_apply(x):
        hx = h.apply(x) - eps_occ[:, None] * x
        return p_c(hx) + alpha * ((x @ c_occ.conj().T) @ c_occ)

    x = torch.zeros_like(rhs)
    r = rhs - a_apply(x)
    t_g = h.t
    t_band = torch.clamp(
        torch.einsum("bg,g,bg->b", c_occ.conj(), t_g.to(c_occ.dtype), c_occ).real, min=1e-6
    )
    x = projected_cg(a_apply, lambda rr: teter(rr, t_g, t_band), x, r, tol, max_iter)
    return p_c(x)


def _all_bands(res: SCFResult, isp: int, ik: int):
    """All computed (coeffs, ε, occ) of spin channel ``isp`` at k-point ``ik``.

    The full window the metallic χ₀ projects out of (Sternheimer) and sums band
    pairs over — occupied, partially occupied, and the empty buffer bands. No
    insulator check: the metallic path handles any occupation."""
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        return res.coeffs[ik], res.eigenvalues[ik], res.occupations[ik]
    return res.coeffs[isp][ik], res.eigenvalues[isp][ik], res.occupations[isp][ik]


def _sternheimer_above(h: HamiltonianK, c_win, c_solve, eps_solve, w_r, alpha: float,
                       tol: float, max_iter: int):
    """(H − ε_n + α P_win) δψ_n = −P_⊥ w ψ_n for the ``c_solve`` bands, δψ living
    ABOVE the entire computed window (P_win = c_win c_win†, P_⊥ = 1 − P_win).

    The metallic counterpart of ``_sternheimer``: projecting out the *whole*
    window (not just the occupied bands) keeps H − ε_n positive definite on the
    complement even for a band n sitting at the Fermi level, so CG never sees the
    indefinite Fermi-surface block. Returns δψ (n_solve, npw), all above-window.
    """

    def p_win(x):
        return (x @ c_win.conj().T) @ c_win

    psi_r = g_to_r(c_solve, h.sphere.flat_idx, h.shape)
    w_psi = box_to_sphere(r_to_g(psi_r * w_r), h.sphere.flat_idx)
    rhs = -(w_psi - p_win(w_psi))

    def a_apply(x):
        hx = h.apply(x) - eps_solve[:, None] * x
        return (hx - p_win(hx)) + alpha * p_win(x)

    x = torch.zeros_like(rhs)
    r = rhs - a_apply(x)
    t_g = h.t
    t_band = torch.clamp(
        torch.einsum("bg,g,bg->b", c_solve.conj(), t_g.to(c_solve.dtype), c_solve).real,
        min=1e-6,
    )
    x = projected_cg(a_apply, lambda rr: teter(rr, t_g, t_band), x, r, tol, max_iter)
    return x - p_win(x)


def _chi0_channel_metal(res: SCFResult, hs_k: list[HamiltonianK], isp: int,
                        w_r: torch.Tensor, f_full: float, mu: float, scheme,
                        width: float, tol: float, max_iter: int) -> torch.Tensor:
    """δρ_σ(r) = χ₀^σ w for one spin channel WITH partial occupations (metal).

    The window scheme (wisdom.md, "Response, adjoints, and autograd"): split the
    Adler–Wiser response into three physically disjoint pieces, summed over the
    computed band window at each k —

      1. window → above-window transitions, via a Sternheimer solve that projects
         out the *entire* window (``_sternheimer_above``), weighted by the band's
         own occupation occ_n;
      2. in-window band pairs (n, m), summed explicitly with the divided-
         difference weight β_nm = (occ_n−occ_m)/(ε_n−ε_m), which stays finite
         through the Fermi surface (the analytic derivative limit on the
         diagonal and near-degenerate pairs);
      3. the rank-one Fermi-level-shift −δμ Σ_n occ'_n |ψ_n|² that keeps the
         electron number fixed, δμ = ⟨occ'·V_nn⟩ / ⟨occ'⟩ summed over the whole
         Brillouin zone.

    In the insulating limit occ'_n → 0, so pieces (2)-diagonal and (3) vanish and
    the pair sum reduces to the occupied→empty response — the same physics as the
    ``_chi0_channel`` insulator path (verified to agree in the tests). δμ is a
    global scalar, so a first pass accumulates its numerator/denominator before
    the second pass assembles the density.
    """
    system = res.system
    grid = system.grid
    ng = grid.shape
    dr = torch.zeros(ng, dtype=RDTYPE, device=grid.g2.device)

    stash = []
    num = 0.0
    den = 0.0
    for ik, h in enumerate(hs_k):
        c_all, eps, occ = _all_bands(res, isp, ik)
        occ_d = occupation_derivative(eps, mu, scheme, width, f_full)
        psi_r = g_to_r(c_all, h.sphere.flat_idx, ng)
        wpsi_sph = box_to_sphere(r_to_g(psi_r * w_r), h.sphere.flat_idx)
        v_mat = c_all.conj() @ wpsi_sph.transpose(-1, -2)  # V[m,n]=⟨ψ_m|w|ψ_n⟩
        vnn = torch.diagonal(v_mat).real
        kw = float(system.kweights[ik])
        num += kw * float((occ_d * vnn).sum())
        den += kw * float(occ_d.sum())
        stash.append((h, c_all, eps, occ, occ_d, psi_r, v_mat, kw))
    dmu = num / den if abs(den) > 1e-30 else 0.0

    for h, c_all, eps, occ, occ_d, psi_r, v_mat, kw in stash:
        pr = psi_r.reshape(c_all.shape[0], -1)  # (nb, ngrid)
        beta = divided_difference_weights(eps, occ, occ_d)
        a_mat = beta * v_mat.transpose(-1, -2)  # A[n,m] = β[n,m]·V[m,n]
        inwin = torch.einsum("nm,ng,mg->g", a_mat, pr.conj(), pr).real
        occ_resp = torch.einsum("n,ng->g", occ_d, (pr.conj() * pr).real)
        contrib = kw * (inwin - dmu * occ_resp)

        mask = occ > 1e-6 * f_full
        if bool(mask.any()):
            c_solve, eps_solve, occ_solve = c_all[mask], eps[mask], occ[mask]
            dpsi = _sternheimer_above(h, c_all, c_solve, eps_solve, w_r,
                                      sternheimer_shift(eps), tol, max_iter)
            dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, ng).reshape(occ_solve.shape[0], -1)
            above = 2.0 * torch.einsum("n,ng->g", occ_solve,
                                       (pr[mask].conj() * dpsi_r).real)
            contrib = contrib + kw * above
        dr += contrib.reshape(ng)
    return dr / grid.volume


def _chi0_channel(res: SCFResult, hs_k: list[HamiltonianK], isp: int, w_r: torch.Tensor,
                  f_full: float, tol: float, max_iter: int) -> torch.Tensor:
    """δρ_σ(r) = χ₀^σ w for one spin channel (block-diagonal in spin)."""
    system = res.system
    grid = system.grid
    dr = torch.zeros(grid.shape, dtype=RDTYPE, device=grid.g2.device)
    for ik, h in enumerate(hs_k):
        c_occ, eps_occ = _occupied(res, isp, ik)
        gap_shift = sternheimer_shift(eps_occ)
        dpsi = _sternheimer(h, c_occ, eps_occ, w_r, alpha=gap_shift,
                            tol=tol, max_iter=max_iter)
        psi_r = g_to_r(c_occ, h.sphere.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, grid.shape)
        # per-band occupation f_full, plus factor 2 from the c.c. pair
        # (ψ*δψ + δψ*ψ): nspin=1 gives 4 (f=2), nspin=2 gives 2 (f=1).
        contrib = (2.0 * f_full * float(system.kweights[ik])
                   * (psi_r.conj() * dpsi_r).real.sum(dim=0))
        dr += contrib
    return dr / grid.volume


def _res_is_insulating(res: SCFResult) -> bool:
    """Every spin channel integer-filled → the exact insulator χ₀ path applies."""
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        return is_insulating(res.occupations, 2.0)
    return all(is_insulating(res.occupations[isp], 1.0) for isp in range(nspin))


@torch.no_grad()
def apply_chi0(res: SCFResult, w_r: torch.Tensor, tol: float = 1e-8,
               max_iter: int = 200, *, assume_totally_symmetric: bool = False) -> torch.Tensor:
    """δρ = χ₀ w for a real local field w — insulator or metal.

    nspin=1: ``w_r`` is a grid field, returns a grid field. nspin=2: ``w_r`` is
    the stacked per-spin field (2, *grid.shape) and the return is the stacked
    per-spin density response — χ₀ is block-diagonal over spin.

    Dispatch on the occupations: an integer-filled result uses the exact
    fully-occupied Sternheimer path (``_chi0_channel``); any fractional
    occupation (a smeared metal, or a magnet with partial spin fillings) uses
    the partial-occupation window path (``_chi0_channel_metal``), which reduces
    smoothly to the insulator response as the Fermi-surface weight occ' → 0.

    Symmetry (``assume_totally_symmetric``): the SCF backward normally requires
    ``use_symmetry=False`` — a perturbation breaks the crystal group, so folded
    IBZ representatives no longer suffice. The ONE exception is a *totally
    symmetric* perturbation (isotropic strain / EOS, a symmetry-preserving
    composition or XC-parameter change): there the response δρ = χ₀ w is itself
    totally symmetric, so the IBZ k-sum (``res.kweights`` are already the star
    multiplicities) folded by the scalar ``RhoSymmetrizer`` reproduces the
    full-BZ response — the same identity that makes the forward symmetrized SCF
    bit-exact. Pass ``assume_totally_symmetric=True`` to opt into that fold on a
    ``use_symmetry=True`` system; the caller certifies w is totally symmetric (it
    is projected on input to drop asymmetric noise). Without the flag a symmetric
    system still raises, so no existing caller silently changes.

    nspin=2 (collinear) is supported: the per-spin response pair is folded by the
    same dispatch the forward density uses (``_sym_fold`` → the (n,m) magnetic
    fold for a ``CollinearMagneticSymmetrizer``, else independent per-channel for
    a uniform-moment ``RhoSymmetrizer``). The perturbation must be totally
    symmetric under the system's (magnetic) group."""
    sym = getattr(res.system, "sym", None)
    symmetrizer = getattr(res.system, "rho_symmetrizer", None)
    nspin = getattr(res, "nspin", 1)
    if sym is not None:
        if not assume_totally_symmetric:
            _check_no_symmetry(res)          # raises: the safe default
        w_r = _sym_fold(symmetrizer, w_r, res.system.grid, nspin)  # project input
    hs = _hamiltonians(res)
    if _res_is_insulating(res):
        if nspin == 1:
            out = _chi0_channel(res, cast("list[HamiltonianK]", hs), 0, w_r, 2.0,
                                tol, max_iter)
        else:
            hs_spin = cast("list[list[HamiltonianK]]", hs)
            out = torch.stack([_chi0_channel(res, hs_spin[isp], isp, w_r[isp], 1.0,
                                             tol, max_iter) for isp in range(nspin)])
    else:
        scheme = SCHEMES[res.smearing]
        mu = float(res.fermi)
        if nspin == 1:
            out = _chi0_channel_metal(res, cast("list[HamiltonianK]", hs), 0, w_r, 2.0,
                                      mu, scheme, res.width, tol, max_iter)
        else:
            hs_spin = cast("list[list[HamiltonianK]]", hs)
            out = torch.stack([
                _chi0_channel_metal(res, hs_spin[isp], isp, w_r[isp], 1.0, mu, scheme,
                                    res.width, tol, max_iter) for isp in range(nspin)])
    if sym is not None:
        out = _sym_fold(symmetrizer, out, res.system.grid, nspin)  # fold the star
    return out


def apply_k_hxc(res: SCFResult, xc, w_r: torch.Tensor) -> torch.Tensor:
    """(K_Hxc w)(r) = Hartree kernel + f_xc·w, via autograd HVP for the XC part.

    Both kernels are the shared response primitives in postscf._response
    (``fxc_hvp`` carries the per-grid-cell → physical n_points/Ω conversion).

    f_xc = ∂v_xc/∂ρ is evaluated at the SCF density INCLUDING the NLCC partial
    core, ρ + ρ_core, because the SCF builds v_xc at exactly that point
    (scf/loop.py: ``rho_for_xc = rho + rho_core``). Omitting the core would
    evaluate the response kernel at the wrong density (~1% off for NLCC
    pseudos), making the implicit-diff SCF adjoint subtly wrong. Matches the
    collinear ``_k_hxc_spin`` (which adds 0.5·ρ_core per channel) and the USPP
    ``k_hxc_grid``. For a valence-only pseudo (rho_core is None) this reduces
    exactly to the plain valence kernel — byte-identical.

    nspin=2: ``w_r`` is the stacked per-spin field (2, *grid.shape). The
    Hartree kernel acts on the TOTAL δρ = δρ↑ + δρ↓ and enters both channels;
    f_xc becomes the spin HVP f_xc^{σσ'} (fxc_hvp_spin) at the converged
    per-channel densities, NLCC core split half/half — matching the SCF's
    ``effective_potentials`` spin assembly and the USPP twin's ``k_hxc_grid``.
    """
    grid = res.system.grid
    core = res.system.rho_core
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        rho_xc = res.rho if core is None else res.rho + core
        return hartree_kernel(grid, w_r) + fxc_hvp(xc, rho_xc, grid, w_r)
    kh = hartree_kernel(grid, w_r[0] + w_r[1])
    c2 = 0.0 if core is None else 0.5 * core
    assert res.rho_spin is not None  # nspin=2 always carries per-spin densities
    fu, fd = fxc_hvp_spin(xc, res.rho_spin[0] + c2, res.rho_spin[1] + c2,
                          grid, w_r[0], w_r[1])
    return torch.stack([kh + fu, kh + fd])


@torch.no_grad()
def solve_adjoint(res: SCFResult, xc, vbar_r: torch.Tensor, beta: float = 0.4,
                  tol: float = 1e-9, max_iter: int = 100,
                  history: int = 8, *, assume_totally_symmetric: bool = False) -> torch.Tensor:
    """Solve u = v̄ + K_Hxc[χ₀ u] by Anderson-accelerated fixed-point iteration.

    ``assume_totally_symmetric`` threads to ``apply_chi0`` to allow a
    ``use_symmetry=True`` (nspin=1) system when v̄ (and hence u) is totally
    symmetric — the IBZ χ₀ is folded by the scalar RhoSymmetrizer, and K_Hxc
    (local Hartree + f_xc) preserves the totally-symmetric subspace, so the whole
    fixed point stays in it. See :func:`apply_chi0`.

    For nspin=2 ``vbar_r`` (and hence ``u``) is the stacked per-spin pair
    (2, *grid.shape); the loop runs on the flattened tensor — every operation
    is per-channel or total-density-coupled inside the two kernels.

    Anderson mixing (not plain damping) because the screened response
    u = v̄ + Kχ₀u has gain > 1 modes near a spin instability where damping
    diverges — the same NiO lesson the USPP twin (uspp_implicit) and the
    dielectric/Hubbard adjoints all hit. For a nonmagnetic insulator the
    fixed point is contractive and Anderson reduces to (and converges like)
    the former damped loop, landing on the identical fixed point."""
    shape = vbar_r.shape

    def g(u):
        return vbar_r + apply_k_hxc(res, xc, apply_chi0(
            res, u, assume_totally_symmetric=assume_totally_symmetric))

    u = vbar_r.clone()
    mixer = AndersonMixer(history, beta)
    step = float("inf")
    for _ in range(max_iter):
        r = g(u) - u
        step = float(torch.linalg.norm(r)) / max(
            1.0, float(torch.linalg.norm(u)))
        if step < tol:
            return u
        u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(shape)
    raise RuntimeError(
        f"adjoint fixed point not converged ({step:.2e} after {max_iter} iters)")


def density_loss_param_grads(
    res: SCFResult, xc, loss_fn, *, assume_totally_symmetric: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gradients dL/dθ of a density-dependent loss through the SCF fixed point.

    loss_fn: rho(grid tensor of the TOTAL density) -> scalar torch tensor
    (pure, differentiable). For nspin=2 the loss stays a functional of ρ_tot,
    so its gradient v̄ = ∂L/∂ρ seeds both spin channels equally.
    Returns (L, {param_name: grad}).

    ``assume_totally_symmetric`` (use_symmetry=True nspin=1 only): run the adjoint
    on the IBZ for a symmetry-invariant loss (its v̄ = ∂L/∂ρ is totally symmetric),
    a ~7x cheaper backward with an identical gradient. See :func:`apply_chi0`.
    """
    grid = res.system.grid
    nspin = getattr(res, "nspin", 1)
    rho = res.rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = loss_fn(rho)
        (vbar,) = torch.autograd.grad(loss, rho)
    # v̄ as physical δL/δρ(r): loss_fn works on grid values directly, so vbar
    # already is ∂L/∂ρ_j — the grid-sum adjoint field. nspin=2: the loss is a
    # functional of ρ_tot, so v̄ enters both channels equally (stacked).
    vbar_seed = vbar if nspin == 1 else torch.stack([vbar] * nspin)
    u = solve_adjoint(res, xc, vbar_seed, assume_totally_symmetric=assume_totally_symmetric)
    chi0_u = apply_chi0(res, u, assume_totally_symmetric=assume_totally_symmetric)

    # dL/dθ = Σ_σ ⟨χ₀u_σ, ∂v_xc^σ/∂θ⟩, differentiate ⟨χ₀u, v_xc(ρ; θ)⟩ w.r.t. θ
    # at fixed ρ. Double backward through E_xc, so force eager with xc_eager().
    # v_xc (hence ∂v_xc/∂θ) is evaluated at ρ + ρ_core, matching the SCF and
    # apply_k_hxc above; for valence-only pseudos (rho_core is None) this is
    # byte-identical to res.rho.
    core = res.system.rho_core
    scale = grid.n_points / grid.volume
    params = list(xc.parameters())
    with torch.enable_grad(), xc_eager():
        if nspin == 1:
            rho_xc = res.rho if core is None else res.rho + core
            rho_fixed = rho_xc.detach().clone().requires_grad_(True)
            sigma = (sigma_from_rho(rho_fixed, grid.g_cart)
                     if xc.needs_gradient else None)
            e_xc = xc.energy(rho_fixed, grid.volume, sigma)
            (v_xc,) = torch.autograd.grad(e_xc, rho_fixed, create_graph=True)
            inner = (v_xc * chi0_u.detach()).sum() * scale
        else:
            c2 = 0.0 if core is None else 0.5 * core
            assert res.rho_spin is not None  # nspin=2 always carries per-spin densities
            ru = (res.rho_spin[0] + c2).detach().clone().requires_grad_(True)
            rd = (res.rho_spin[1] + c2).detach().clone().requires_grad_(True)
            s_uu, s_dd, s_tt = spin_sigma_triple(xc, ru, rd, grid.g_cart)
            e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tt)
            vu, vd = torch.autograd.grad(e_xc, (ru, rd), create_graph=True)
            inner = ((vu * chi0_u[0].detach()).sum()
                     + (vd * chi0_u[1].detach()).sum()) * scale
        grads = torch.autograd.grad(inner, params, allow_unused=True)
    named = {
        name: (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(xc.named_parameters(), grads, strict=True)
    }
    return loss.detach(), named
