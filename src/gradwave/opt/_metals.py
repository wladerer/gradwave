"""Variational occupations for the joint free-energy descent (metals, #122).

The insulator prototype (``opt/joint.py``) carries only the ``N_e/2`` occupied
bands at fixed ``f = 2`` and minimises the KS total energy ``E``. Metals have a
partially filled band at the Fermi surface, so the object to minimise is the
Mermin free energy ``F = E − σS`` with occupations that respond to the orbital
state. This module supplies those occupations by REUSING the SCF's smearing /
Fermi-level machinery (``core.occupations``) on the eigenvalues of the cheap
subspace Hamiltonian.

Per closure, given the (orthonormal) orbitals ``C_k`` (now ``n_bands ≥ N_e/2``
per k) at the current density:

    H_sub[k] = C_k^† Ĥ[ρ] C_k        (n_bands×n_bands, one per k)

is assembled term by term from the SAME ingredients ``joint_energy`` already
builds (per-band kinetic on (k+G)², the real-space ψ grids for the local
V_eff matrix elements, the projector overlaps for the nonlocal block) — so it
adds NO extra sphere↔grid FFTs beyond the ψ grids the density build already
pays for. Diagonalising H_sub gives Ritz values ``λ_i``; the shared Fermi
search turns them into occupations ``f_i = 2·f((λ_i−μ)/σ)`` and an entropy
``−σS``, exactly as the SCF loop does.

Circularity (ρ → V_eff → H_sub → λ → f → ρ) is closed by a short DETACHED
fixed-point inner loop at the base cell: the orbitals are held fixed and only
the occupation weights / potential iterate, so it is cheap and carries no
autograd graph. Because the whole occupation solve is detached, the notorious
``torch.linalg.eigh`` backward — whose ``1/(λ_i−λ_j)`` factors blow up at the
degenerate Fermi surface — never runs. The occupations enter the LIVE free
energy only through (a) the diagonal weights ``f`` (detached) and (b) the
subspace rotation ``R`` that maps the coefficient block onto the Ritz basis
(also detached). Freezing ``R``/``f`` per closure is the block-coordinate
(alternating orbital / occupation) resolution; the occupation block is
re-solved every closure so both blocks reach stationarity together. This is
exact at the SCF fixed point — there ``C`` already diagonalises H_sub (``R=I``)
and ``f`` is the self-consistent Fermi filling — so the ε=0 free energy and its
strain derivative reproduce the SCF free energy and ``stress()`` (fixed
occupations have no explicit ε dependence; see ``postscf/stress.py``).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import g_to_r, g_to_r_box
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._strain import strained_kpg, strained_projector_cols
from gradwave.scf.loop import (
    System,
    effective_potentials,
    local_potential_r,
)


class RobustEigh(torch.autograd.Function):
    """``torch.linalg.eigh`` with a degeneracy-robust backward.

    Forward is the EXACT Hermitian eigendecomposition ``A = V diag(λ) Vᴴ``. The
    backward replaces the eigenvector-response denominators ``1/(λ_i−λ_j)`` —
    which diverge as two levels approach and are NaN at exact degeneracy — with
    the Lorentzian-broadened

        F_ij = (λ_i − λ_j) / ((λ_i − λ_j)² + ε²),   F_ii = 0,

    which stays finite (→ 0 at exact degeneracy) and reduces to the exact
    ``1/(λ_i−λ_j)`` when ``|λ_i−λ_j| ≫ ε``. Tying ``ε`` to the smearing width σ
    is the physically motivated choice: two levels within σ of each other have
    near-equal Fermi occupations, so their mutual ensemble is smeared out and
    the broadened (finite) gradient is exactly the smeared-ensemble limit rather
    than an artefact.

    Gauge note: for complex-Hermitian ``A`` the eigenvectors carry an arbitrary
    per-column phase, so a bare cotangent ``gV`` is gauge dependent; callers
    must contract ``V`` only into gauge-invariant quantities (occupations depend
    on λ alone; the Ritz density ``Σ f_i |Σ_a V_{ai}ψ_a|²`` is phase-invariant),
    which the free-energy descent does.
    """

    @staticmethod
    def forward(ctx, a, eps):
        w, v = torch.linalg.eigh(a)
        ctx.save_for_backward(w, v)
        ctx.eps = float(eps)
        return w, v

    @staticmethod
    def backward(ctx, gw, gv):
        w, v = ctx.saved_tensors
        eps = ctx.eps
        vh = v.conj().transpose(-2, -1)
        # F_{ij} = (λ_j − λ_i)/((λ_i − λ_j)² + ε²); the diagonal (λ diff 0) is 0.
        # (Sign matches torch.linalg.eigh's backward convention for gV; verified
        # against autograd in tests/unit/test_metals_eigh.py.)
        wd = w.unsqueeze(-2) - w.unsqueeze(-1)  # (..., i, j) = λ_j − λ_i
        f_broad = wd / (wd * wd + eps * eps)
        inner = torch.zeros_like(v)
        if gv is not None:
            inner = f_broad.to(v.dtype) * (vh @ gv)
        if gw is not None:
            inner = inner + torch.diag_embed(gw.to(v.dtype))
        ga = v @ inner @ vh
        ga = 0.5 * (ga + ga.conj().transpose(-2, -1))
        return ga, None


def robust_eigh(a, eps):
    """Degeneracy-robust Hermitian eigendecomposition (see ``RobustEigh``)."""
    return RobustEigh.apply(a, eps)


def _static_blocks(system, tabs, coeffs, lmax):
    """Per-k (H_kin, H_nl, ψ-grids, projector cols) that do not depend on ρ.

    Built once at the base (unstrained) cell; reused across the detached
    occupation fixed-point. All detached — the occupation solve carries no
    graph. Returns lists indexed by k plus the (Ω, N_grid) normalisation.
    """
    grid = system.grid
    shape = grid.shape
    n_grid = shape[0] * shape[1] * shape[2]
    dev = coeffs[0].device
    a0 = torch.as_tensor(grid.cell, dtype=RDTYPE, device=dev)
    b_e = 2.0 * math.pi * torch.linalg.inv(a0).T
    omega = torch.linalg.det(a0)
    hkin, hnl, psis = [], [], []
    for ik, sph in enumerate(system.spheres):
        c = coeffs[ik].detach().to(CDTYPE)
        kpg2 = sph.kpg2.to(RDTYPE)
        # kinetic block H[a,b] = ⟨c_a|T|c_b⟩ = Σ_G c_a*(G) T_G c_b(G). The first
        # (bra) index is conjugated — the SAME convention hloc and hnl below use,
        # so H_sub is a consistent Hermitian matrix. (Conjugating only the ket,
        # as a per-band Rayleigh quotient invites, leaves the real diagonal
        # unchanged but transposes-conjugates the off-diagonal — a silent bug
        # that survives every energy test yet corrupts every eigenvalue.)
        hkin.append((c.conj() * (HBAR2_2M * kpg2)[None, :]) @ c.T)
        # nonlocal block ⟨c_a|V_nl|c_b⟩ = becp_a^† D becp_b, becp_bi = ⟨p_i|c_b⟩
        pd = system.proj_data[ik]
        if pd.dij_full.shape[0] == 0:
            hnl.append(torch.zeros(c.shape[0], c.shape[0], dtype=CDTYPE, device=dev))
        else:
            kpg_vec, kpg2_v = strained_kpg(sph, b_e)
            p = strained_projector_cols(
                tabs, system.species_of_atom, pd.atom_index, lmax,
                kpg_vec, kpg2_v, omega, system.positions.to(dev))
            bov = c @ p.conj().T
            hnl.append(bov.conj() @ pd.dij_full.to(CDTYPE) @ bov.T)
        psis.append(g_to_r(c, sph.flat_idx, shape))  # (nb, n1,n2,n3)
    return hkin, hnl, psis, omega, n_grid


def diagonal_occupations(
    system, xc, tabs, coeffs, smearing, width, n_electrons, *, lmax, n_inner=5,
    occ_init=None,
):
    """Detached DIAGONAL-ensemble occupations for a block-coordinate metal step.

    Assigns each (fixed) orbital ψ_i the occupation ``f_i = 2·f((e_i−μ)/σ)`` from
    its OWN Rayleigh quotient ``e_i = ⟨ψ_i|Ĥ[ρ]|ψ_i⟩`` — the diagonal of the
    subspace Hamiltonian, which needs no eigendecomposition and no rotation. The
    density ρ (hence Ĥ) is closed self-consistently in a short detached inner
    loop. Frozen per L-BFGS chunk this gives a smooth E(orbitals, cell) with a
    matching gradient (unlike the full subspace scheme, whose per-chunk rotation
    jumps stalled strong-Wolfe on a metal). At the minimum the orbitals are
    eigenstates, so the diagonal quotients are the true eigenvalues and this
    coincides with the Fermi filling. Returns ``(occ (nk,nb) detached, entropy
    scalar detached, mu, e (nk,nb) Rayleigh quotients detached)``.
    """
    kw = system.kweights
    dev = coeffs[0].device
    nk = len(coeffs)
    nb = coeffs[0].shape[0]
    scheme = SCHEMES[smearing]
    vloc_r = local_potential_r(system)
    hkin, hnl, psis, omega, n_grid = _static_blocks(system, tabs, coeffs, lmax)

    if occ_init is not None:
        occ = occ_init.detach().clone()
    else:
        occ = torch.full((nk, nb), n_electrons / nb, dtype=RDTYPE,
                         device=dev).clamp(max=2.0)
    e = torch.zeros(nk, nb, dtype=RDTYPE, device=dev)
    mu = 0.0
    entropy = torch.zeros((), dtype=RDTYPE, device=dev)
    for _ in range(n_inner):
        rho = None
        for ik in range(nk):
            w = (kw[ik] * occ[ik]).to(RDTYPE)
            contrib = torch.einsum("b,bxyz->xyz", w,
                                   psis[ik].real**2 + psis[ik].imag**2)
            rho = contrib if rho is None else rho + contrib
        rho = rho / omega
        veff = effective_potentials(system, xc, [rho], vloc_r)[0]
        e_rows = []
        for ik in range(nk):
            psi = psis[ik]
            hloc = torch.einsum("bxyz,bxyz->b", psi.conj(),
                                psi * veff[None]).real / n_grid
            e_rows.append(torch.diagonal(hkin[ik] + hnl[ik]).real + hloc)
        e = torch.stack(e_rows)
        mu = float(find_fermi(e, kw, scheme, width, n_electrons, degeneracy=2.0))
        mut = torch.tensor(mu, dtype=RDTYPE, device=dev)
        x = (e - mut) / width
        occ = 2.0 * scheme.occupation(x)
        entropy = -width * (2.0 * kw[:, None] * scheme.entropy(x)).sum()
    return occ.detach(), entropy.detach(), mu, e.detach()


def mv_occupations(eta, kweights, smearing, width, n_electrons):
    """Marzari–Vanderbilt occupations from occupation-LEVEL variables ``eta``.

    ``eta`` (nk, n_bands) are unconstrained real leaves — one auxiliary "energy"
    per orbital. Occupations ``f = 2·f((η−μ)/σ)`` and the entropy ``−σS`` are
    LIVE in ``eta`` (the smearing functions are the differentiable ones the SCF
    uses), so occupations enter the free energy as explicit L-BFGS variables:
    no subspace diagonalisation, no ``eigh`` on the graph, and the gradient the
    optimizer sees is exactly the gradient of the objective it evaluates (unlike
    the frozen-subspace scheme, whose per-chunk rotation/occupation jumps stall
    strong-Wolfe on a metal). ``μ`` is the Fermi level from a detached bisection
    (a Lagrange multiplier enforcing Σ f = N_e); its ``η``-derivative is the
    flat gauge mode and is correctly ~zero. Returns ``(f, entropy, mu)``.
    """
    scheme = SCHEMES[smearing]
    mu = float(find_fermi(eta.detach(), kweights, scheme, width, n_electrons,
                          degeneracy=2.0))
    mut = torch.tensor(mu, dtype=eta.dtype, device=eta.device)
    x = (eta - mut) / width
    f = 2.0 * scheme.occupation(x)
    entropy = -width * (2.0 * kweights[:, None] * scheme.entropy(x)).sum()
    return f, entropy, mu


def subspace_occupations(
    system: System,
    xc,
    tabs,
    coeffs,
    smearing: str,
    width: float,
    n_electrons: float,
    *,
    lmax: int,
    n_inner: int = 5,
):
    """Detached subspace-diagonalisation occupations for the free-energy descent.

    Returns ``(rot, occ, entropy, eigs, mu)``:

    - ``rot``: per-k unitary (n_bands×n_bands) mapping the coefficient block onto
      the Ritz basis; rotate live orbitals with ``C' = rotᵀ · C`` (still
      orthonormal). Detached.
    - ``occ``: (nk, n_bands) occupations in [0, 2] (spin-restricted). Detached.
    - ``entropy``: scalar ``−σS`` (add to E for F). Detached.
    - ``eigs``: (nk, n_bands) Ritz values [eV]. Detached.
    - ``mu``: Fermi level [eV].
    """
    kw = system.kweights
    dev = coeffs[0].device
    nk = len(coeffs)
    nb = coeffs[0].shape[0]
    scheme = SCHEMES[smearing]
    vloc_r = local_potential_r(system)

    hkin, hnl, psis, omega, n_grid = _static_blocks(system, tabs, coeffs, lmax)

    # occupation seed: fill uniformly to roughly the right electron count
    occ = torch.full((nk, nb), n_electrons / nb, dtype=RDTYPE, device=dev).clamp(max=2.0)
    rot = [torch.eye(nb, dtype=CDTYPE, device=dev) for _ in range(nk)]
    eigs = torch.zeros(nk, nb, dtype=RDTYPE, device=dev)
    mu = 0.0
    entropy = torch.zeros((), dtype=RDTYPE, device=dev)

    for _ in range(n_inner):
        # density from the current Ritz orbitals ψ'_i = Σ_a rot[a,i] ψ_a
        rho = None
        for ik in range(nk):
            psir = torch.einsum("ai,axyz->ixyz", rot[ik], psis[ik])
            w = (kw[ik] * occ[ik]).to(RDTYPE)
            contrib = torch.einsum("b,bxyz->xyz", w, psir.real**2 + psir.imag**2)
            rho = contrib if rho is None else rho + contrib
        rho = rho / omega
        veff = effective_potentials(system, xc, [rho], vloc_r)[0]  # (n1,n2,n3) eV
        # subspace Hamiltonian in the ORIGINAL coefficient basis, per k
        new_eigs = []
        for ik in range(nk):
            psi = psis[ik]
            hloc = torch.einsum("bxyz,cxyz->bc", psi.conj(), psi * veff[None]) / n_grid
            h = hkin[ik] + hloc + hnl[ik]
            h = 0.5 * (h + h.conj().T)
            lam, vec = torch.linalg.eigh(h)
            new_eigs.append(lam)
            rot[ik] = vec
        eigs = torch.stack(new_eigs)
        mu = float(find_fermi(eigs, kw, scheme, width, n_electrons, degeneracy=2.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=dev)
        occ_rows, ent = [], torch.zeros((), dtype=RDTYPE, device=dev)
        for ik in range(nk):
            o, s = occupations_and_entropy(eigs[ik], mu_t, scheme, width, degeneracy=2.0)
            occ_rows.append(o)
            ent = ent - width * (2.0 * kw[ik] * s).sum()
        occ = torch.stack(occ_rows)
        entropy = ent
    return rot, occ.detach(), entropy.detach(), eigs.detach(), mu


def robust_subspace_occupations(
    system: System,
    xc,
    tabs,
    coeffs,
    frac,
    smearing: str,
    width: float,
    n_electrons: float,
    *,
    lmax: int,
    n_inner: int = 5,
    broaden: float | None = None,
):
    """Subspace occupations with a LIVE, degeneracy-robust subspace rotation.

    Closes the frozen-occupation force bias (#129). Two stages:

    1. A short DETACHED self-consistent inner loop (identical to
       ``subspace_occupations``) converges (ρ, V_eff, μ) at the current orbitals
       — cheap, no autograd graph.

    2. ONE LIVE pass rebuilds the subspace Hamiltonian ``H_sub`` on the graph
       from the live ``coeffs`` and live fractional positions ``frac``,
       diagonalises it with the Lorentzian-broadened :class:`RobustEigh`, and
       returns the resulting rotation, Fermi occupations and entropy — all
       LIVE. Rotating the live orbitals onto this Ritz basis (``C' = rotᵀ·C``)
       makes the subspace rotation and occupation response part of the
       objective's gradient, so the force is the full Mermin free-energy force
       instead of the biased frozen-occupation one.

    Consistency is what makes the gradient trustworthy: every channel of the
    λ-response carries the prefactor ``(λ_i − ε_i^true)``, where ``ε_i^true`` is
    the band derivative ``∂E/∂f_i`` of the LIVE ``joint_energy``. If ``H_sub``
    were built with the frozen inner-loop potential, that prefactor would
    vanish only AT the inner fixed point and enter linearly in the orbital
    error away from it — drowning the ~0.3 eV/Å physical force in noise along a
    descent trajectory (observed on Al). So stage 2 rebuilds the ENTIRE local
    potential LIVE:

    - ``v_H + v_xc`` from the live density of the stage-1-rotated orbitals
      (the XC functional derivative is taken with ``create_graph`` so the
      kernel response rides the graph too),
    - the local pseudopotential and the nonlocal projectors from the live
      ``frac``,

    making ``λ_i`` and ``ε_i`` refer to the same Hamiltonian up to one
    (converged) occupation update — the prefactor is at the inner-loop residual
    level at EVERY point of the trajectory, not just the fixed point. The inner
    loop iterates to an occupation tolerance (1e-10, capped at
    ``max(n_inner, 30)`` sweeps) for the same reason.

    The strain ε is deliberately NOT threaded into ``H_sub`` (it is built on the
    base-cell reciprocal lattice): at fixed occupations dF/dε has no explicit
    strain term (matching ``postscf/stress.py``), and the occupation/rotation
    response to ε vanishes at self-consistency by the envelope theorem — so this
    keeps ``dF/dε = Ω·stress`` bit-identical to the detached scheme while adding
    the position/orbital response that the force needs. ``ε`` (``broaden``)
    defaults to the smearing width ``width``.

    Returns ``(rot, occ, entropy, eigs, mu)`` with ``rot``/``occ``/``entropy``
    LIVE (``eigs`` detached, for reporting).
    """
    kw = system.kweights
    dev = coeffs[0].device
    nk = len(coeffs)
    nb = coeffs[0].shape[0]
    scheme = SCHEMES[smearing]
    if broaden is None:
        broaden = width
    grid = system.grid
    shape = grid.shape
    n_grid = shape[0] * shape[1] * shape[2]
    a0 = torch.as_tensor(grid.cell, dtype=RDTYPE, device=dev)
    b_e = 2.0 * math.pi * torch.linalg.inv(a0).T  # base-cell reciprocal lattice
    omega = torch.linalg.det(a0)
    pos_live = frac @ a0  # live Cartesian positions on the base cell

    # ---- stage 1: detached self-consistent (ρ, V_eff, μ, occ) fixed point.
    # v_loc is built at the LIVE (detached) positions so the frozen screening
    # v_H + v_xc and the live-in-frac stage-2 v_loc add up to the potential this
    # loop actually converged.
    with torch.no_grad():
        vloc_g_d = local_potential_g(
            frac.detach() @ a0, system.species_index, system.vloc_tables,
            system.grid.g_cart, float(omega), vloc_atom=system.vloc_atom)
        vloc_r = g_to_r_box(vloc_g_d, real=True)
    hkin_d, hnl_d, psis_d, _, _ = _static_blocks(system, tabs, coeffs, lmax)
    occ = torch.full((nk, nb), n_electrons / nb, dtype=RDTYPE, device=dev).clamp(max=2.0)
    rot = [torch.eye(nb, dtype=CDTYPE, device=dev) for _ in range(nk)]
    mu = 0.0
    veff = None
    # Damped fixed point: the raw occupation/potential iteration charge-sloshes
    # (oscillates without converging) when the Fermi shell is near-degenerate —
    # observed on 2×2×2 Al, where the undamped loop stalls at |Δocc| ~ 7e-3 and
    # the leftover (λ−ε) prefactor puts a ~0.07 eV/Å bias on the live force.
    # Linear mixing (β = 0.5) makes it contract; the residual gate is on the RAW
    # update so the reported convergence is of the true fixed point.
    beta = 0.5
    n_sweeps = max(n_inner, 60)
    for _ in range(n_sweeps):
        occ_prev = occ
        rho = None
        for ik in range(nk):
            psir = torch.einsum("ai,axyz->ixyz", rot[ik], psis_d[ik])
            w = (kw[ik] * occ[ik]).to(RDTYPE)
            contrib = torch.einsum("b,bxyz->xyz", w, psir.real**2 + psir.imag**2)
            rho = contrib if rho is None else rho + contrib
        rho = rho / omega
        veff = effective_potentials(system, xc, [rho], vloc_r)[0]
        new_eigs = []
        for ik in range(nk):
            psi = psis_d[ik]
            hloc = torch.einsum("bxyz,cxyz->bc", psi.conj(), psi * veff[None]) / n_grid
            h = hkin_d[ik] + hloc + hnl_d[ik]
            h = 0.5 * (h + h.conj().T)
            lam, vec = torch.linalg.eigh(h)
            new_eigs.append(lam)
            rot[ik] = vec
        eigs = torch.stack(new_eigs)
        mu = float(find_fermi(eigs, kw, scheme, width, n_electrons, degeneracy=2.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=dev)
        occ_raw = torch.stack([
            occupations_and_entropy(eigs[ik], mu_t, scheme, width, degeneracy=2.0)[0]
            for ik in range(nk)])
        resid = float((occ_raw - occ_prev).abs().max())
        occ = occ_prev + beta * (occ_raw - occ_prev)
        if resid < 1e-10:
            occ = occ_raw
            break
    mu_t = torch.tensor(mu, dtype=RDTYPE, device=dev)

    # ---- stage 2: one LIVE pass — rebuild H_sub on the graph, robust eigh.
    # The whole local potential is LIVE (see docstring): v_H + v_xc from the
    # live density at the converged stage-1 rotation/occupations (detached
    # weights — their response channels carry the ~zero (λ−ε) prefactor), and
    # v_loc from the live positions.
    psis_live = [g_to_r(coeffs[ik].to(CDTYPE), sph.flat_idx, shape)
                 for ik, sph in enumerate(system.spheres)]
    rho_live = None
    for ik in range(nk):
        psir = torch.einsum("ai,axyz->ixyz", rot[ik], psis_live[ik])
        w = (kw[ik] * occ[ik]).to(RDTYPE)
        contrib = torch.einsum("b,bxyz->xyz", w, psir.real**2 + psir.imag**2)
        rho_live = contrib if rho_live is None else rho_live + contrib
    rho_live = rho_live / omega
    live_graph = rho_live.requires_grad
    if not live_graph:  # detached callers (anchor tests) still get a potential
        rho_live = rho_live.detach().requires_grad_(True)
    v_h = hartree_potential_r(rho_live, grid.g2)
    rho_xc = rho_live if system.rho_core is None else rho_live + system.rho_core
    with torch.enable_grad():
        sig = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_xc, grid.volume, sig, None)
        (v_xc,) = torch.autograd.grad(e_xc, rho_live, create_graph=live_graph)
    v_xc = v_xc * (grid.n_points / grid.volume)
    vloc_g_live = local_potential_g(
        pos_live, system.species_index, system.vloc_tables,
        system.grid.g_cart, float(omega), vloc_atom=system.vloc_atom)
    v_stage2 = v_h + v_xc + g_to_r_box(vloc_g_live, real=True)  # fully LIVE
    rot_live, occ_live, eigs_out = [], [], []
    entropy = torch.zeros((), dtype=RDTYPE, device=dev)
    for ik, sph in enumerate(system.spheres):
        c = coeffs[ik].to(CDTYPE)  # LIVE orbital coefficients
        kpg2 = sph.kpg2.to(RDTYPE)
        hkin = (c.conj() * (HBAR2_2M * kpg2)[None, :]) @ c.T
        psi = psis_live[ik]
        hloc = torch.einsum("bxyz,cxyz->bc", psi.conj(),
                            psi * v_stage2[None]) / n_grid
        pd = system.proj_data[ik]
        if pd.dij_full.shape[0] == 0:
            hnl = torch.zeros(nb, nb, dtype=CDTYPE, device=dev)
        else:
            kpg_vec, kpg2_v = strained_kpg(sph, b_e)
            p = strained_projector_cols(
                tabs, system.species_of_atom, pd.atom_index, lmax,
                kpg_vec, kpg2_v, omega, pos_live)  # LIVE in frac
            bov = c @ p.conj().T
            hnl = bov.conj() @ pd.dij_full.to(CDTYPE) @ bov.T
        h = hkin + hloc + hnl
        h = 0.5 * (h + h.conj().T)
        lam, vec = robust_eigh(h, broaden)  # LIVE, degeneracy-robust backward
        rot_live.append(vec)
        eigs_out.append(lam)
        x = (lam - mu_t) / width
        occ_live.append(2.0 * scheme.occupation(x))
        entropy = entropy - width * (2.0 * kw[ik] * scheme.entropy(x)).sum()
    occ_out = torch.stack(occ_live)
    if not live_graph:  # detached callers get detached outputs back
        rot_live = [r.detach() for r in rot_live]
        occ_out = occ_out.detach()
        entropy = entropy.detach()
    return (rot_live, occ_out, entropy,
            torch.stack(eigs_out).detach(), mu)
