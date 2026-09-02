"""Independent-particle / RPA optical dielectric function ε(ω) and absorption.

For a converged insulating SCF this evaluates the frequency-dependent macroscopic
dielectric function at the independent-particle (RPA-without-local-fields) level and
the derived optical constants. The interband (Adler–Wiser, q→0) imaginary part is

    ε₂(ω) = (4π² e²/Ω) Σ_k w_k Σ_{v∈occ, c∈unocc}
                |V_cv|² / (ε_c − ε_v)² · L_η(ε_c − ε_v − ω),

with the velocity operator V = ∂H/∂k. Its local (kinetic) part in gradwave's eV/Å
units is 2·(ħ²/2m)·(k+G) — the same convention as the static ε∞ build in
``postscf.dielectric``. The nonlocal ``[V_nl, r]`` commutator is a documented
refinement (dielectric.py obtains it by a finite difference of the KB tables in k);
velocity="full" includes it. Directions are averaged isotropically. ε₁(ω) follows by
Kramers–Kronig; the refractive index n, extinction κ and absorption α(ω) from ε(ω).

Norm-conserving, insulators. Spin: nspin=1 (spin-paired), nspin=2 (collinear, two
independent channels summed) and noncollinear/spinor (two-component states, one
electron per band) are all handled — the spin-degeneracy fold is bookkept so a
non-magnetic nspin=2 or spinor run reproduces the nspin=1 result. The BZ sum reuses
the SCF k-mesh + weights; the SCF orbitals are re-diagonalized with extra conduction
bands (band_structure keeps only eigenvalues, so the eigenvectors the matrix elements
need are rebuilt) — per spin channel for collinear, on the doubled spinor axis for
noncollinear.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2, HARTREE_EV, HBAR2_2M
from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE
from gradwave.solvers.davidson import davidson_batched_ms

_HBARC = 1973.269804  # eV·Å  (ħc), for α(ω) = 2ω κ / ħc


def _nocc_collinear(res: Any, sp: int, nspin: int) -> int:
    """Occupied-band count for collinear spin channel ``sp``: nelec//2 for the
    spin-paired case, else the integer electron count in that channel (insulator,
    g=1 occupations)."""
    if nspin == 1:
        system = res.system
        nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                              for i in range(len(system.species_of_atom)))))
        return nelec // 2
    return int(round(float(res.occupations[sp].sum(-1).float().mean())))


def _velocity_scalar(system: Any, evec: torch.Tensor, velocity: str, dk: float):
    """Scalar (non-spinor) velocity operator applied to every band:
    [(∂H/∂k_α)|ψ⟩]_α. ``full`` includes the nonlocal [V_nl,r] commutator; ``local``
    keeps only the kinetic part 2·(ħ²/2m)·(k+G)."""
    bk = system.batch
    if velocity == "full":
        from gradwave.postscf.dielectric import _dhdk_psi
        return [_dhdk_psi(system, evec, a, dk) for a in range(3)]
    if velocity == "local":
        mask = bk.mask.to(evec.real.dtype)
        return [(2.0 * HBAR2_2M) * (bk.kpg[..., a] * mask)[:, None, :] * evec
                for a in range(3)]
    raise ValueError(f"velocity must be 'full' or 'local', got {velocity!r}")


def _vmat_from_dhv(evec: torch.Tensor, dhv: list[torch.Tensor]) -> torch.Tensor:
    """Band-basis velocity matrix V^α_{cv}=⟨c|∂H/∂k_α|v⟩, stacked over α. The G
    contraction spans the full coefficient axis (for a spinor that sums the up and
    down inner products, as it must)."""
    return torch.stack([torch.einsum("kcg,kvg->kcv", evec.conj(), d) for d in dhv])


def _collinear_blocks(res: Any, n_extra_bands: int, velocity: str, dk: float,
                      diago_tol: float):
    """Per-spin-channel (eig, vmat, nocc) for a collinear SCFResult (nspin 1 or 2),
    plus the spin-degeneracy occupancy g=2/nspin. Re-diagonalizes the SCF mesh with
    extra conduction bands, keeping eigenvectors, using the channel's own v_eff."""
    system = res.system
    bk, grid = system.batch, system.grid
    device = res.v_eff.device
    nspin = getattr(res, "nspin", 1)
    v_eff_s = res.v_eff if nspin == 2 else res.v_eff[None]  # leading spin axis
    p_b = projectors_b(bk, system.positions)                # spin-independent
    blocks = []
    for sp in range(nspin):
        nocc = _nocc_collinear(res, sp, nspin)
        nbands = nocc + int(n_extra_bands)
        h = BatchedHamiltonian(bk, grid.shape, v_eff_s[sp], p_b)
        c0 = torch.zeros(bk.nk, nbands, bk.npw_max, dtype=CDTYPE, device=device)
        diag = torch.arange(nbands, device=device)
        c0[:, diag, diag] = 1.0
        out = davidson_batched_ms(h.apply, c0, bk.t, bk.mask, tol=diago_tol,
                                  max_iter=80, mixed_precision=False)
        dhv = _velocity_scalar(system, out.eigenvectors, velocity, dk)
        blocks.append((out.eigenvalues, _vmat_from_dhv(out.eigenvectors, dhv), nocc))
    return blocks, 2.0 / nspin


def _spinor_block(res: Any, xc: Any, n_extra_bands: int, velocity: str, dk: float,
                  diago_tol: float):
    """Single (eig, vmat, nocc) block for a noncollinear/spinor NCResult, occupancy
    g=1. Rebuilds the converged (v_r, b_xc) from (ρ, m⃗) via ``xc`` (NCResult carries
    no v_eff — mirrors scf.noncollinear.band_structure_nc), re-diagonalizes the
    SpinorHamiltonian on the doubled axis with extra bands, and forms the spinor
    velocity matrix elements."""
    from gradwave.core.energies.hartree import hartree_potential_g
    from gradwave.core.energies.local_pp import local_potential_g
    from gradwave.core.fftbox import g_to_r_box, r_to_g
    from gradwave.core.xc.noncollinear import vxc_and_bxc
    from gradwave.scf.noncollinear import SpinorHamiltonian

    if xc is None:
        raise ValueError("noncollinear optics needs the XC functional to rebuild the "
                         "spinor potential — pass xc= (a NoncollinearXC)")
    system = res.system
    grid = system.grid
    bk = system.batch
    device = res.rho.device
    if getattr(xc, "needs_tau", False):
        raise NotImplementedError(
            "noncollinear optics: meta-GGA (needs_tau) velocity/rebuild not wired")

    nonmagnetic = float(res.m.abs().max()) < 1e-12
    rho_g_box = r_to_g(res.rho.to(CDTYPE))
    v_h = g_to_r_box(hartree_potential_g(rho_g_box, grid.g2), real=True)
    v_xc, b_xc, _ = vxc_and_bxc(xc, res.rho, res.m, grid, rho_core=system.rho_core)
    if nonmagnetic:
        b_xc = torch.zeros_like(b_xc)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)
    vloc_r = g_to_r_box(vloc_g, real=True)
    v_r = v_h + v_xc + vloc_r

    is_fr = bool(getattr(system, "is_fr", False))
    p_b = projectors_b(bk, system.positions)
    q_so = dij_so = None
    if is_fr:
        from gradwave.core.spinor_proj import build_so_projectors
        q_so, dij_so = build_so_projectors(bk, system)
    h = SpinorHamiltonian(bk, grid.shape, v_r, b_xc, p_b, q=q_so, dij_so=dij_so)

    nocc = int(round(float(res.occupations.sum(-1).float().mean())))
    nbands = nocc + int(n_extra_bands)
    c0 = torch.zeros(bk.nk, nbands, 2 * bk.npw_max, dtype=CDTYPE, device=device)
    for b_i in range(nbands):                               # seed distinct spinors
        c0[:, b_i, (b_i // 2) + (b_i % 2) * bk.npw_max] = 1.0
    t2 = torch.cat([bk.t, bk.t], dim=-1)
    mask2 = torch.cat([bk.mask, bk.mask], dim=-1)
    out = davidson_batched_ms(h.apply, c0, t2, mask2, tol=diago_tol, max_iter=100,
                              mixed_precision=False)
    dhv = _spinor_velocity(system, out.eigenvectors, velocity, dk, is_fr)
    return out.eigenvalues, _vmat_from_dhv(out.eigenvectors, dhv), nocc


def _spinor_velocity(system: Any, evec: torch.Tensor, velocity: str, dk: float,
                     is_fr: bool):
    """Velocity operator on the doubled spinor axis. ``full``: the exact ∂H/∂k —
    ``_dhdk_psi_soc`` (kinetic on each component + j-resolved SOC nonlocal) for a
    fully-relativistic pseudo, else the scalar ``_dhdk_psi`` applied to each spinor
    component independently (the scalar-relativistic KB nonlocal is spin-diagonal).
    ``local``: kinetic only, per component."""
    bk = system.batch
    m = bk.npw_max
    if velocity == "full":
        if is_fr:
            from gradwave.core.spinor_proj import build_so_projectors, so_projector_channels
            from gradwave.postscf.dielectric import _dhdk_psi_soc
            from gradwave.pseudo.radial_torch import RadialTables
            _, dij_so = build_so_projectors(bk, system)
            dij_c = dij_so.to(CDTYPE)
            col_meta, lmax = so_projector_channels(system)
            tabs = [RadialTables(u, device=evec.device) for u in system.upfs]
            return [_dhdk_psi_soc(system, tabs, col_meta, lmax, dij_c, evec, a, dk)
                    for a in range(3)]
        from gradwave.postscf.dielectric import _dhdk_psi
        return [torch.cat([_dhdk_psi(system, evec[..., :m], a, dk),
                           _dhdk_psi(system, evec[..., m:], a, dk)], dim=-1)
                for a in range(3)]
    if velocity == "local":
        mask = bk.mask.to(evec.real.dtype)
        out = []
        for a in range(3):
            fac = (2.0 * HBAR2_2M) * (bk.kpg[..., a] * mask)[:, None, :]
            out.append(torch.cat([fac * evec[..., :m], fac * evec[..., m:]], dim=-1))
        return out
    raise ValueError(f"velocity must be 'full' or 'local', got {velocity!r}")


@torch.no_grad()
def _optical_lfe(res, om, eta, n_extra_bands, verbose,
                 scissor=0.0, q0=0.05, n_lfe=27):
    """Macroscopic ε(ω) with RPA local fields via the finite-q Dyson ε = 1 − vχ₀.

    Builds the microscopic χ₀_{GG'}(q,ω) at a small finite q along a reciprocal
    axis (plane-wave density matrix elements — no q→0 head/wing bookkeeping),
    forms ε = δ − v_G χ₀, and inverts: ε_M = 1/[ε⁻¹]₀₀. Reuses the SCF k-mesh,
    re-diagonalizing at k and k+q. For nspin=2 the two collinear channels are
    summed (each g=1). IBZ weights (approximate at finite q). Returns (ε₁, ε₂,
    ε₁_head, ε₂_head) on the same ω grid — the last two are the same-convention
    macroscopic head with no local fields.
    """
    from gradwave.core.fftbox import g_to_r, r_to_g
    from gradwave.core.hamiltonian import HamiltonianK, projectors
    from gradwave.grids import build_gsphere, reciprocal_cell
    from gradwave.postscf._kb import projector_data_at_k, species_projector_tables

    system = res.system
    grid = system.grid
    dev = res.v_eff.device
    shape = grid.shape
    omega_vol_au = float(grid.volume) / BOHR_ANG**3
    nspin = getattr(res, "nspin", 1)
    v_eff_s = res.v_eff if nspin == 2 else res.v_eff[None]   # leading spin axis
    g_occ = 2.0 / nspin                                       # spin-degeneracy fold
    kfracs = np.array([np.asarray(s.k_frac, float) for s in system.spheres])
    kw = np.asarray(system.kweights, float)
    kw = kw / kw.sum()
    q_frac = np.array([q0, 0.0, 0.0])

    # local-field G-set (smallest |q+G|) + Coulomb v_G = 4π/|q+G|²
    recip = reciprocal_cell(grid.cell)
    q_cart = q_frac @ recip
    ms = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4)
                   for k in range(-3, 4)])
    order = np.argsort(np.sum((q_cart + ms @ recip) ** 2, axis=1))[:n_lfe]
    miller = ms[order]
    qg2 = np.sum((q_cart + miller @ recip) ** 2, axis=1) * BOHR_ANG**2   # bohr⁻²
    vG = torch.as_tensor(4.0 * np.pi / qg2, dtype=CDTYPE, device=dev)
    n1, n2, n3 = shape
    lfe_flat = torch.as_tensor(
        (miller[:, 0] % n1) * (n2 * n3) + (miller[:, 1] % n2) * n3
        + (miller[:, 2] % n3), device=dev)

    beta_ls, dij = species_projector_tables(system.upfs, dev)

    def _diag_at(kf, take, v_eff):
        sph = build_gsphere(grid, system.ecut, np.asarray(kf, float), device=dev)
        pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                                 beta_ls, dij, grid.volume, dev)
        p = projectors(pd, system.positions)
        hk = HamiltonianK(sph, shape, v_eff, pd, p)
        hd = hk.apply(torch.eye(sph.npw, dtype=CDTYPE, device=dev)).transpose(0, 1)
        hd = 0.5 * (hd + hd.conj().t())
        ev, vecs = torch.linalg.eigh(hd)
        return ev[:take], vecs[:, :take], sph

    om_au = torch.as_tensor(om / HARTREE_EV, device=dev)
    eta_au = eta / HARTREE_EV
    chi = torch.zeros((len(om), n_lfe, n_lfe), dtype=CDTYPE, device=dev)
    for sp in range(nspin):
        v_eff = v_eff_s[sp]
        nocc = _nocc_collinear(res, sp, nspin)
        nb = nocc + int(n_extra_bands)
        for ik, kf in enumerate(kfracs):
            ev_k, v_k, sph_k = _diag_at(kf, nocc, v_eff)
            ev_q, v_q, sph_q = _diag_at(kf + q_frac, nb, v_eff)
            uv = g_to_r(v_k.t(), sph_k.flat_idx, shape).reshape(nocc, -1)
            uc = g_to_r(v_q[:, nocc:nb].t(), sph_q.flat_idx, shape).reshape(nb - nocc, -1)
            ev = (ev_k[:nocc] / HARTREE_EV)
            ec = (ev_q[nocc:nb] / HARTREE_EV)
            m = torch.empty((nocc, nb - nocc, n_lfe), dtype=CDTYPE, device=dev)
            for v in range(nocc):
                prod = (uv[v].conj()[None] * uc).reshape(nb - nocc, *shape)
                m[v] = r_to_g(prod).reshape(nb - nocc, -1)[:, lfe_flat]
            mf = m.reshape(nocc * (nb - nocc), n_lfe)
            de = (ec[None, :] - ev[:, None]).reshape(-1)
            gd = de > 1e-5
            mf, de = mf[gd], de[gd]
            de = de + scissor / HARTREE_EV   # rigid conduction-band shift (a.u.)
            denom = (1.0 / (om_au[:, None] - de[None, :] + 1j * eta_au)
                     - 1.0 / (om_au[:, None] + de[None, :] + 1j * eta_au))
            chi = chi + (g_occ * float(kw[ik])) * torch.einsum(
                "wv,vg,vh->wgh", denom.to(CDTYPE), mf, mf.conj())
    chi = chi / omega_vol_au

    eye = torch.eye(n_lfe, dtype=CDTYPE, device=dev)
    eps_m = torch.empty(len(om), dtype=CDTYPE, device=dev)
    eps_h = torch.empty(len(om), dtype=CDTYPE, device=dev)
    for w in range(len(om)):
        eps_mat = eye - vG[:, None] * chi[w]
        eps_h[w] = eps_mat[0, 0]                           # head only (no local fields)
        eps_m[w] = 1.0 / torch.linalg.inv(eps_mat)[0, 0]   # macroscopic, with LFE
    em, eh = eps_m.cpu().numpy(), eps_h.cpu().numpy()
    if verbose:
        print(f"[optics/LFE] finite-q={q0}, n_LFE={n_lfe}, nspin={nspin}: "
              f"ε₁(0) {eh.real[0]:.2f} (no LFE) → {em.real[0]:.2f} (LFE)")
    return em.real, em.imag, eh.real, eh.imag


def _kramers_kronig(omega: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    """ε₁(ω) = 1 + (2/π) P∫₀^∞ ω' ε₂(ω') / (ω'² − ω²) dω'  (discrete, PV)."""
    dw = omega[1] - omega[0]
    w2 = omega**2
    eps1 = np.ones_like(omega)
    for i in range(len(omega)):
        d = w2 - w2[i]
        d[i] = np.inf  # principal value: drop the singular point
        eps1[i] += (2.0 / np.pi) * np.sum(omega * eps2 / d) * dw
    return eps1


@torch.no_grad()
def optical_epsilon(
    res: Any, *, xc: Any = None, omega_max: float = 20.0, n_omega: int = 600,
    eta: float = 0.1, n_extra_bands: int = 8, velocity: str = "full",
    local_fields: bool = False, scissor: float = 0.0, dk: float = 1e-3,
    diago_tol: float = 1e-9, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(ω, ε₁, ε₂, α_cm, info) for a converged insulating result ``res``.

    ω [eV], ε₁/ε₂ the isotropic (trace) average [dimensionless], α in cm⁻¹.

    Handles nspin=1, collinear nspin=2 (two channels summed) and noncollinear/spinor
    results. The spinor path needs the XC functional to rebuild the potential (NCResult
    carries no v_eff) — pass ``xc`` (a NoncollinearXC).

    velocity: "full" uses the exact velocity operator ∂H/∂k (kinetic + the nonlocal
      [V_nl,r] commutator, via dielectric._dhdk_psi / _dhdk_psi_soc); "local" keeps
      only the kinetic part 2·(ħ²/2m)·(k+G).
    local_fields: add RPA local-field effects via the Dyson ε=1−vχ₀ (a small
      finite-q microscopic dielectric, inverted) — then ε₁/ε₂ are the
      local-field-corrected macroscopic dielectric and info carries the IP values.
      Not implemented for the spinor path.
    scissor: rigid conduction-band shift [eV] to correct the DFT gap (Del
      Sole–Girlanda: the oscillator strengths are preserved and only the
      transition energies shift, so the ε₂ peaks blue-shift by the scissor).
    info also carries the diagonal ε tensor components (eps{1,2}_tensor: xx,yy,zz).
    """
    system = res.system
    formalism = getattr(res, "formalism", "nc")
    if formalism in ("uspp", "uspp_noncollinear"):
        raise NotImplementedError(
            "task: optics is norm-conserving only (the USPP/PAW velocity operator "
            "and the ultrasoft overlap S are not wired)")
    spinor = formalism == "noncollinear"
    nspin = getattr(res, "nspin", 1)
    grid = system.grid
    bk = system.batch
    if bk is None:
        raise ValueError("optics needs system.batch (the k-batched geometry)")
    if local_fields and spinor:
        raise NotImplementedError(
            "local fields for the noncollinear/spinor path are not implemented "
            "(the spinor χ₀ density matrix elements are a further generalization)")
    device = (res.rho if spinor else res.v_eff).device
    omega_vol = float(grid.volume)

    # re-diagonalize the SCF mesh KEEPING eigenvectors, per spin block, and form the
    # band-basis velocity matrix elements V^α_{cv}=⟨c|∂H/∂k_α|v⟩ (band_structure keeps
    # only eigenvalues). Collinear: one block per channel; spinor: one doubled block.
    if spinor:
        blocks = [_spinor_block(res, xc, n_extra_bands, velocity, dk, diago_tol)]
        g_occ = 1.0
    else:
        blocks, g_occ = _collinear_blocks(res, n_extra_bands, velocity, dk, diago_tol)

    rdtype = blocks[0][0].dtype
    kw = torch.as_tensor(system.kweights, dtype=rdtype, device=device)
    kw = kw / kw.sum()
    omega = torch.linspace(1e-3, omega_max, n_omega, device=device, dtype=rdtype)
    # velocity-gauge IP prefactor 4π²e²/Ω, calibrated (nspin=1, g=2) against LDA Si
    # ε₁(0)≈15 / ε₂ peak ≈70. Per block it scales by g_occ/2 so a single collinear
    # channel or spinor band-set (g=1) carries half the fold — a non-magnetic nspin=2
    # or spinor run then reproduces the nspin=1 result exactly.
    prefac = 4.0 * np.pi**2 * E2 / omega_vol
    blk_prefac = prefac * g_occ / 2.0

    eps2_t = torch.zeros((3, 3, n_omega), device=device, dtype=rdtype)
    for eig, vmat, nocc in blocks:
        nb = eig.shape[1]
        for ik in range(bk.nk):
            e = eig[ik]
            de = e[nocc:nb, None] - e[None, :nocc]         # (ncond, nocc)
            good = de > 1e-4
            de_f = de[good]                                 # (npair,)
            vcv = vmat[:, ik, nocc:nb, :nocc][:, good]      # (3, npair)
            # ε₂ tensor: Σ Re(V^a V^b*)/Δ² · Lorentzian. scissor (Del Sole–Girlanda):
            # the oscillator strength |V|²/Δ² is preserved (DFT Δ), only the transition
            # energy in the Lorentzian shifts by `scissor`.
            mab = torch.einsum("ap,bp->abp", vcv, vcv.conj()).real / de_f**2
            de_q = de_f + scissor
            lor = (eta / np.pi) / ((omega[:, None] - de_q[None, :]) ** 2 + eta**2)
            eps2_t = eps2_t + blk_prefac * float(kw[ik]) * torch.einsum(
                "wp,abp->abw", lor, mab)

    om = omega.cpu().numpy()
    eps2_t = eps2_t.cpu().numpy()                         # (3, 3, nω)
    # symmetrize the tensor over the crystal point group: the IBZ sum gives the
    # correct trace (rotation-invariant) but not the individual Cartesian
    # components — each IBZ k stands for its whole star.
    from gradwave.postscf.irreps import _cartesian_rotation
    from gradwave.symmetry import find_spacegroup
    frac = system.positions.cpu().numpy() @ np.linalg.inv(grid.cell)
    sg = find_spacegroup(grid.cell, frac, system.species_of_atom)
    rots = [_cartesian_rotation(np.asarray(w, float), grid.cell) for w in sg.rotations]
    eps2_t = sum(np.einsum("ai,bj,ijw->abw", s, s, eps2_t) for s in rots) / len(rots)
    e2 = (eps2_t[0, 0] + eps2_t[1, 1] + eps2_t[2, 2]) / 3.0
    e1 = _kramers_kronig(om, e2)
    eps2_diag = np.stack([eps2_t[0, 0], eps2_t[1, 1], eps2_t[2, 2]])   # (3, nω)
    eps1_diag = np.stack([_kramers_kronig(om, eps2_diag[i]) for i in range(3)])

    # n_occ / n_bands describe a representative (first) spin block, so the report's
    # "occ + cond" band count reads correctly per channel; the spin descriptor
    # (nspin / formalism) carries whether a second channel is summed underneath.
    n_occ_report = int(blocks[0][2])
    info: dict[str, Any] = {
        "n_bands": int(blocks[0][0].shape[1]), "n_occ": n_occ_report,
        "nspin": int(nspin), "formalism": formalism,
        "eps_static": float(e1[0]),
        "eps1_ip": e1.tolist(), "eps2_ip": e2.tolist(),
        "eps2_tensor": eps2_diag.tolist(), "eps1_tensor": eps1_diag.tolist(),
        "velocity": velocity, "local_fields": bool(local_fields),
        "scissor_eV": float(scissor),
    }

    if local_fields:
        e1, e2, e1_nolfe, e2_nolfe = _optical_lfe(
            res, om, eta, n_extra_bands, verbose, scissor=scissor)
        info["eps_static"] = float(e1[0])
        info["eps1_nolfe"] = e1_nolfe.tolist()  # same convention, no local fields
        info["eps2_nolfe"] = e2_nolfe.tolist()

    modulus = np.sqrt(e1**2 + e2**2)
    kappa = np.sqrt(np.maximum((modulus - e1) / 2.0, 0.0))
    alpha_cm = (2.0 * om / _HBARC) * kappa * 1e8         # Å⁻¹ → cm⁻¹
    if verbose:
        spin = ("noncollinear" if spinor else f"nspin={nspin}")
        print(f"[optics] eps1(0)={e1[0]:.2f}, eps2 peak {om[int(np.argmax(e2))]:.2f} eV, "
              f"{bk.nk} k, {n_occ_report} occ, velocity={velocity}, {spin}"
              f"{', +LFE' if local_fields else ''}")
    return om, e1, e2, alpha_cm, info
