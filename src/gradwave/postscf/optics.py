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
this first cut uses the local part, which is exact for a purely local potential.
Directions are averaged isotropically. ε₁(ω) follows by Kramers–Kronig; the
refractive index n, extinction κ and absorption coefficient α(ω) from ε(ω).

Norm-conserving, nspin=1, insulators. The BZ sum reuses the SCF k-mesh + weights;
the SCF orbitals are re-diagonalized with extra conduction bands (band_structure
keeps only eigenvalues, so the eigenvectors the matrix elements need are rebuilt).
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


@torch.no_grad()
def _optical_lfe(res, om, eta, nocc, n_extra_bands, diago_tol, verbose,
                 q0=0.05, n_lfe=27):
    """Macroscopic ε(ω) with RPA local fields via the finite-q Dyson ε = 1 − vχ₀.

    Builds the microscopic χ₀_{GG'}(q,ω) at a small finite q along a reciprocal
    axis (plane-wave density matrix elements — no q→0 head/wing bookkeeping),
    forms ε = δ − v_G χ₀, and inverts: ε_M = 1/[ε⁻¹]₀₀. Reuses the SCF k-mesh,
    re-diagonalizing at k and k+q. IBZ weights (approximate at finite q). Returns
    (ε₁, ε₂) on the same ω grid.
    """
    from gradwave.core.fftbox import g_to_r, r_to_g
    from gradwave.core.hamiltonian import HamiltonianK, projectors
    from gradwave.grids import build_gsphere, reciprocal_cell
    from gradwave.postscf._kb import projector_data_at_k, species_projector_tables

    system = res.system
    grid = system.grid
    v_eff = res.v_eff
    dev = v_eff.device
    shape = grid.shape
    omega_vol_au = float(grid.volume) / BOHR_ANG**3
    nb = nocc + int(n_extra_bands)
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

    def _diag_at(kf, take):
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
    for ik, kf in enumerate(kfracs):
        ev_k, v_k, sph_k = _diag_at(kf, nocc)
        ev_q, v_q, sph_q = _diag_at(kf + q_frac, nb)
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
        denom = (1.0 / (om_au[:, None] - de[None, :] + 1j * eta_au)
                 - 1.0 / (om_au[:, None] + de[None, :] + 1j * eta_au))
        chi = chi + float(kw[ik]) * torch.einsum("wv,vg,vh->wgh",
                                                 denom.to(CDTYPE), mf, mf.conj())
    chi = chi * (2.0 / omega_vol_au)

    eye = torch.eye(n_lfe, dtype=CDTYPE, device=dev)
    eps_m = torch.empty(len(om), dtype=CDTYPE, device=dev)
    eps_h = torch.empty(len(om), dtype=CDTYPE, device=dev)
    for w in range(len(om)):
        eps_mat = eye - vG[:, None] * chi[w]
        eps_h[w] = eps_mat[0, 0]                           # head only (no local fields)
        eps_m[w] = 1.0 / torch.linalg.inv(eps_mat)[0, 0]   # macroscopic, with LFE
    em, eh = eps_m.cpu().numpy(), eps_h.cpu().numpy()
    if verbose:
        print(f"[optics/LFE] finite-q={q0}, n_LFE={n_lfe}: "
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
    res: Any, *, omega_max: float = 20.0, n_omega: int = 600, eta: float = 0.1,
    n_extra_bands: int = 8, velocity: str = "full", local_fields: bool = False,
    dk: float = 1e-3, diago_tol: float = 1e-9, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(ω, ε₁, ε₂, α_cm, info) for a converged insulating SCFResult ``res``.

    ω [eV], ε₁/ε₂ the isotropic (trace) average [dimensionless], α in cm⁻¹.

    velocity: "full" uses the exact velocity operator ∂H/∂k (kinetic + the
      nonlocal [V_nl,r] commutator, via dielectric._dhdk_psi); "local" keeps only
      the kinetic part 2·(ħ²/2m)·(k+G).
    local_fields: add RPA local-field effects via the Dyson ε=1−vχ₀ (a small
      finite-q microscopic dielectric, inverted) — then ε₁/ε₂ are the
      local-field-corrected macroscopic dielectric and info carries the IP values.
    info also carries the diagonal ε tensor components (eps{1,2}_tensor: xx,yy,zz).
    """
    system = res.system
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("task: optics currently supports nspin=1 only")
    bk, grid = system.batch, system.grid
    if bk is None:
        raise ValueError("optics needs system.batch (the k-batched geometry)")
    device = res.v_eff.device
    omega_vol = float(grid.volume)

    nelec = int(round(sum(float(system.upfs[system.species_of_atom[i]].z_valence)
                          for i in range(len(system.species_of_atom)))))
    nocc = nelec // 2
    nbands = nocc + int(n_extra_bands)

    # re-diagonalize the SCF mesh, KEEPING eigenvectors (band_structure discards them)
    p_b = projectors_b(bk, system.positions)
    h = BatchedHamiltonian(bk, grid.shape, res.v_eff, p_b)
    c0 = torch.zeros(bk.nk, nbands, bk.npw_max, dtype=CDTYPE, device=device)
    diag = torch.arange(nbands, device=device)
    c0[:, diag, diag] = 1.0
    out = davidson_batched_ms(h.apply, c0, bk.t, bk.mask, tol=diago_tol,
                              max_iter=80, mixed_precision=False)
    eig = out.eigenvalues                 # (nk, nbands) [eV]
    evec = out.eigenvectors               # (nk, nbands, npw_max) complex (padded 0)

    kpg = bk.kpg                           # (nk, npw_max, 3) [Å⁻¹]
    mask = bk.mask.to(eig.dtype)           # (nk, npw_max)
    kw = torch.as_tensor(system.kweights, dtype=eig.dtype, device=device)
    kw = kw / kw.sum()

    omega = torch.linspace(1e-3, omega_max, n_omega, device=device, dtype=eig.dtype)
    # velocity-gauge IP prefactor 4π²e²/Ω (spin folded for nspin=1); calibrated
    # against LDA Si ε₁(0)≈15 / ε₂ peak ≈ 70.
    prefac = 4.0 * np.pi**2 * E2 / omega_vol

    # velocity matrix elements V^α_{cv} = ⟨c|∂H/∂k_α|v⟩
    if velocity == "full":
        from gradwave.postscf.dielectric import _dhdk_psi
        dhv = [_dhdk_psi(system, evec, a, dk) for a in range(3)]     # each (nk,nb,npw)
    elif velocity == "local":
        dhv = [(2.0 * HBAR2_2M) * (kpg[..., a] * mask)[:, None, :] * evec
               for a in range(3)]
    else:
        raise ValueError(f"velocity must be 'full' or 'local', got {velocity!r}")
    vmat = torch.stack([torch.einsum("kcg,kvg->kcv", evec.conj(), d) for d in dhv])

    eps2_t = torch.zeros((3, 3, n_omega), device=device, dtype=eig.dtype)
    for ik in range(bk.nk):
        e = eig[ik]
        de = e[nocc:nbands, None] - e[None, :nocc]       # (ncond, nocc)
        good = de > 1e-4
        de_f = de[good]                                   # (npair,)
        vcv = vmat[:, ik, nocc:nbands, :nocc][:, good]    # (3, npair)
        # ε₂ tensor: Σ Re(V^a V^b*)/Δ² · Lorentzian
        mab = torch.einsum("ap,bp->abp", vcv, vcv.conj()).real / de_f**2
        lor = (eta / np.pi) / ((omega[:, None] - de_f[None, :]) ** 2 + eta**2)
        eps2_t = eps2_t + prefac * float(kw[ik]) * torch.einsum("wp,abp->abw", lor, mab)

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

    info: dict[str, Any] = {
        "n_bands": int(nbands), "n_occ": int(nocc), "eps_static": float(e1[0]),
        "eps1_ip": e1.tolist(), "eps2_ip": e2.tolist(),
        "eps2_tensor": eps2_diag.tolist(), "eps1_tensor": eps1_diag.tolist(),
        "velocity": velocity, "local_fields": bool(local_fields),
    }

    if local_fields:
        e1, e2, e1_nolfe, e2_nolfe = _optical_lfe(
            res, om, eta, nocc, n_extra_bands, diago_tol, verbose)
        info["eps_static"] = float(e1[0])
        info["eps1_nolfe"] = e1_nolfe.tolist()  # same convention, no local fields
        info["eps2_nolfe"] = e2_nolfe.tolist()

    modulus = np.sqrt(e1**2 + e2**2)
    kappa = np.sqrt(np.maximum((modulus - e1) / 2.0, 0.0))
    alpha_cm = (2.0 * om / _HBARC) * kappa * 1e8         # Å⁻¹ → cm⁻¹
    if verbose:
        print(f"[optics] eps1(0)={e1[0]:.2f}, eps2 peak {om[int(np.argmax(e2))]:.2f} eV, "
              f"{bk.nk} k, {nocc}+{n_extra_bands} bands, velocity={velocity}"
              f"{', +LFE' if local_fields else ''}")
    return om, e1, e2, alpha_cm, info
