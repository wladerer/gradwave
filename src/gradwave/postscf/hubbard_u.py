"""Hubbard U as a determinable quantity, not just an input.

Three capabilities:

1. `energy_derivative_u` — the exact analytic dE_total/dU. At SCF convergence
   the KS energy is stationary in the density, so by Hellmann–Feynman the total
   derivative w.r.t. the parameter U equals the *partial*:
       dE/dU = Σ_{I,σ} ½ Tr[ n^{Iσ}(1 − n^{Iσ}) ]   (per manifold, U_eff=U−J).
   This makes U a first-class differentiable parameter — the gradient a learning
   loop would backprop — with no finite differences or SCF re-runs.

2. `linear_response_u` — the code computes its OWN U from occupation response
   (Cococcioni–de Gironcoli). A rigid probe α_J·Σ_m|φ^J_m⟩⟨φ^J_m| is added to
   manifold J and the on-site occupation N_I = Tr[n^I] is measured:
       χ_{IJ}  = dN_I/dα_J  (interacting: SCF re-converged)
       χ0_{IJ} = dN_I/dα_J  (bare: one non-self-consistent diagonalization)
       U = (χ0^{-1} − χ^{-1})_II
   The χ^{-1} background subtraction removes the rigid-shift (delocalized)
   response, leaving the local Hubbard interaction.

3. `linear_response_u_autodiff` — the same U with NO finite differences and NO
   probe SCF re-runs: analytic dN_I/dα_J from ONE converged ground state.
   The occupied orbitals' first-order response to the projector probe is a
   conduction-projected Sternheimer solve per spin (exact infinitesimal
   response, insulators); the interacting χ additionally screens the probe
   self-consistently through the spin Hxc kernel, obtained as an autograd
   Hessian-vector product of E_Hxc — DFPT-like response without hand-coding
   f_xc, so it works for any twice-differentiable (including learnable) spin
   functional. χ0 is the bare (u=0) first iteration; χ is the fixed point of
       u^σ = K_Hxc^{σσ'}[ Δρ^{σ'}(P_probe + u) ].
"""

from __future__ import annotations

from typing import Any, cast

import torch

from gradwave.core._anderson import AndersonMixer
from gradwave.core.hubbard import (
    HubbardData,
    HubbardManifold,
    build_hubbard_projectors,
    hubbard_projectors,
    occupation_matrices,
)
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE, RDTYPE

# The batched Sternheimer CG and the coefficient padding moved to
# postscf._response under public names; the private aliases stay importable
# from here for existing callers (postscf.dielectric, postscf.forces, ...).
from gradwave.postscf._response import (
    cg_sternheimer as _cg_sternheimer_b,
)
from gradwave.postscf._response import (
    fxc_hvp,
    fxc_hvp_spin,
    hartree_kernel,
    insulator_window,
    sternheimer_shift,
)
from gradwave.postscf._response import (
    pad_coeffs as _pad,
)
from gradwave.postscf._strain import _HubbardSite
from gradwave.scf.loop import SCFResult, System, scf


def energy_derivative_u(res, manifolds: list[HubbardManifold]) -> float:
    """Exact dE_total/dU [dimensionless] at the converged +U point (HF)."""
    if res.hub_occ is None:
        raise ValueError("res has no Hubbard occupation matrices (run scf with hubbard=...)")
    de = 0.0
    for sp_mats in res.hub_occ:  # per spin
        for n in sp_mats:  # per site
            de += 0.5 * float((torch.trace(n) - torch.trace(n @ n)).real)
    return de


def _site_occupations(res, hub, hub_q) -> torch.Tensor:
    """Per-site total occupation N_I = Σ_σ Tr[n^{Iσ}] from a converged result."""
    n = torch.zeros(hub.n_sites, dtype=RDTYPE, device=hub_q.device)
    for sp in range(res.nspin):
        occ_sp = res.occupations if res.nspin == 1 else res.occupations[sp]
        w = 0.5 * occ_sp if res.nspin == 1 else occ_sp
        cpad = _pad(res.coeffs if res.nspin == 1 else res.coeffs[sp],
                    hub.q_free.shape[-1])
        mats = occupation_matrices(hub_q, cpad, w, res.system.kweights, hub.sites)
        for i, m in enumerate(mats):
            n[i] += torch.trace(m).real
    return n * (2.0 if res.nspin == 1 else 1.0)


@torch.no_grad()
def _bare_response_occ(system, base_res, hub, hub_q, alpha_vec, smearing, width):
    """One non-self-consistent diagonalization at frozen (converged) v_eff plus
    the rigid probe α, returning per-site total occupations N_I."""
    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.core.occupations import (
        SCHEMES,
        find_fermi,
        occupations_and_entropy,
    )
    from gradwave.solvers.davidson import davidson_batched

    bk, grid = system.batch, system.grid
    nspin = base_res.nspin
    g_spin = 2.0 / nspin
    projs_b = projectors_b(bk, system.positions)
    veff = base_res.v_eff if nspin == 2 else base_res.v_eff[None]

    eigs_s, coeffs_s = [], []
    for sp in range(nspin):
        dij = torch.zeros(hub.nproj, hub.nproj, dtype=CDTYPE, device=hub_q.device)
        for si, s in enumerate(hub.sites):
            st, dim = s["start"], s["dim"]
            dij[st:st + dim, st:st + dim] += alpha_vec[si] * torch.eye(
                dim, dtype=CDTYPE, device=hub_q.device)
        h = BatchedHamiltonian(bk, grid.shape, veff[sp], projs_b,
                               hub_q=hub_q, hub_dij=dij.conj())
        c0 = base_res.coeffs[sp] if nspin == 2 else base_res.coeffs
        c0 = _pad(c0, bk.npw_max)
        dav = davidson_batched(h.apply, c0, bk.t, bk.mask, tol=1e-9)
        eigs_s.append(dav.eigenvalues.to(RDTYPE))
        coeffs_s.append(dav.eigenvectors.to(CDTYPE))

    scheme = SCHEMES[smearing]
    eigs_cat = torch.cat(eigs_s, dim=0)
    kw_cat = torch.cat([system.kweights] * nspin)
    mu = torch.as_tensor(find_fermi(eigs_cat, kw_cat, scheme, width,
                                    system.n_electrons, degeneracy=g_spin)).to(RDTYPE)
    n = torch.zeros(hub.n_sites, dtype=RDTYPE, device=hub_q.device)
    for sp in range(nspin):
        # occupations_and_entropy returns g·f (g = g_spin = 2/nspin), but
        # occupation_matrices weights becp·becp* linearly and expects a PER-SPIN
        # weight f∈[0,1]. Drop the degeneracy for nspin=1 (0.5·w = f), matching
        # _site_occupations' `0.5 * occ_sp`; nspin=2 already has g=1 so w = f.
        # The trailing ×2.0 (nspin=1) then supplies the spin degeneracy, keeping
        # this χ0 path on the SAME occupation scale as the interacting χ path.
        w, _ = occupations_and_entropy(eigs_s[sp], mu, scheme, width, degeneracy=g_spin)
        w = 0.5 * w if nspin == 1 else w
        mats = occupation_matrices(hub_q, coeffs_s[sp], w, system.kweights, hub.sites)
        for i, m in enumerate(mats):
            n[i] += torch.trace(m).real
    return n * (2.0 if nspin == 1 else 1.0)


def _fd_response_column(system, xc, base, hub, hub_q, j, alpha, man,
                        smearing, width, scf_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference response column (χ_col, χ0_col) [1/eV] from perturbing
    the single site `j`: dN_I/dα_j for all sites I (interacting and bare)."""
    ns = hub.n_sites
    chi_cols, chi0_cols = [], []
    for sgn in (+1.0, -1.0):
        av = [0.0] * ns
        av[j] = sgn * alpha
        # interacting: full SCF re-converged with the probe
        r = scf(system, xc, smearing=smearing, width=width, hubbard=man,
                hub_alpha=av, **scf_kwargs)
        chi_cols.append(_site_occupations(r, hub, hub_q))
        # bare: one diagonalization at the base self-consistent potential
        chi0_cols.append(_bare_response_occ(system, base, hub, hub_q,
                                            torch.tensor(av), smearing, width))
    chi_col = (chi_cols[0] - chi_cols[1]) / (2 * alpha)   # (ns,)
    chi0_col = (chi0_cols[0] - chi0_cols[1]) / (2 * alpha)
    return chi_col, chi0_col


def linear_response_u(system: System, xc: XCFunctional | SpinXC, l: int, species: int, *,
                      site: int = 0, alpha: float = 0.1, smearing: str="gaussian",
                      width: float=0.05, scf_kwargs=None) -> dict[str, Any]:
    """Compute the linear-response Hubbard U [eV] for a manifold.

    Measures the on-site occupation response to a rigid projector probe and
    returns χ0, χ, and U = (χ0^{-1} − χ^{-1})_{site,site}. For a lone site or
    two symmetry-equivalent sites the symmetric [[a,b],[b,a]] matrix is known
    from perturbing `site` alone (one probe direction, cheapest). For
    inequivalent sites (different species or l) or ≥3 sites χ_IJ is genuinely
    asymmetric, so each site is perturbed independently to build the full
    response matrix, then inverted as (χ0^{-1} − χ^{-1}) (Cococcioni–de
    Gironcoli, general case)."""
    scf_kwargs = dict(scf_kwargs or {})
    man = [HubbardManifold(species=species, l=l, u=0.0, j=0.0)]  # U computed at U=0
    hub = build_hubbard_projectors(system, man)
    hub_q = hubbard_projectors(hub, system.positions)
    ns = hub.n_sites

    # scf()'s own `xc: XCFunctional` annotation doesn't reflect its real
    # runtime contract -- api.py's nspin=2 collinear path already calls it
    # with a genuine SpinXC instance under an identical `cast` (run_scf's
    # `scf(system, cast("XCFunctional", xc), ...)`), so this isn't a new
    # widening, just the same established escape hatch at another call site.
    base = scf(system, cast("XCFunctional", xc), smearing=smearing, width=width,
              hubbard=man, **scf_kwargs)

    if _use_full_matrix(hub.sites, system.species_of_atom):
        chi_cols, chi0_cols = [], []  # column j = response to perturbing site j
        for j in range(ns):
            cj, c0j = _fd_response_column(system, xc, base, hub, hub_q, j, alpha,
                                          man, smearing, width, scf_kwargs)
            chi_cols.append(cj)
            chi0_cols.append(c0j)
        chi_mat = torch.stack(chi_cols, dim=1)   # χ_IJ = dN_I/dα_J
        chi0_mat = torch.stack(chi0_cols, dim=1)
        return _assemble_u_matrix(chi_mat, chi0_mat, site)

    chi_col, chi0_col = _fd_response_column(system, xc, base, hub, hub_q, site,
                                            alpha, man, smearing, width, scf_kwargs)
    return _assemble_u(chi_col, chi0_col, site, hub.sites, system.species_of_atom)


def _all_sites_equivalent(sites: list[_HubbardSite], species_of_atom: list[int]) -> bool:
    """True if every Hubbard site shares the first site's species and l — the
    symmetry-equivalent case where the symmetric single-column shortcut holds."""
    if len(sites) <= 1:
        return True
    sp0, l0 = species_of_atom[sites[0]["atom"]], sites[0]["l"]
    return all(species_of_atom[s["atom"]] == sp0 and s["l"] == l0 for s in sites)


def _use_full_matrix(sites: list[_HubbardSite], species_of_atom: list[int]) -> bool:
    """Whether the general per-site response matrix is needed. The cheap
    single-column shortcut only covers a lone site (scalar) or exactly two
    symmetry-equivalent sites ([[a,b],[b,a]]); anything else — an inequivalent
    pair or ≥3 sites — needs the full χ_IJ built by perturbing each site."""
    ns = len(sites)
    if ns == 1:
        return False
    return not (ns == 2 and _all_sites_equivalent(sites, species_of_atom))


def _assemble_u(chi_col: torch.Tensor, chi0_col: torch.Tensor, site: int,
                sites: list[_HubbardSite], species_of_atom: list[int]) -> dict[str, Any]:
    """U = (χ0^{-1} − χ^{-1})_II from one response column.

    The cheap single-perturbed-site path: a lone site gives the scalar estimate;
    two symmetry-equivalent sites give the symmetric [[a,b],[b,a]] matrix. A
    single column cannot reconstruct an inequivalent pair (χ_IJ ≠ χ_JI there),
    so that case raises — the drivers route it to the full-matrix path
    (`_assemble_u_matrix`) instead of calling this."""
    ns = chi_col.shape[0]
    if ns == 2:
        s0, s1 = sites[site], sites[1 - site]
        if (species_of_atom[s0["atom"]] != species_of_atom[s1["atom"]]
                or s0["l"] != s1["l"]):
            raise NotImplementedError(
                "single-column linear-response U for two Hubbard sites of "
                "different species or l is not defined: the [[a,b],[b,a]] "
                "symmetric reconstruction from one perturbed site assumes the "
                "sites are symmetry-equivalent — perturb each site for the full "
                "response matrix (linear_response_u does this automatically)")
        chi = torch.tensor([[chi_col[site], chi_col[1 - site]],
                            [chi_col[1 - site], chi_col[site]]])
        chi0 = torch.tensor([[chi0_col[site], chi0_col[1 - site]],
                             [chi0_col[1 - site], chi0_col[site]]])
        u = float((torch.linalg.inv(chi0) - torch.linalg.inv(chi))[site, site])
    else:  # single-site scalar estimate
        u = float(1.0 / chi0_col[site] - 1.0 / chi_col[site])
    return {"U_eV": u, "chi": chi_col[site].item(), "chi0": chi0_col[site].item(),
            "chi_col": chi_col.tolist(), "chi0_col": chi0_col.tolist()}


def _assemble_u_matrix(chi_mat: torch.Tensor, chi0_mat: torch.Tensor,
                       site: int) -> dict[str, Any]:
    """U = (χ0^{-1} − χ^{-1})_{site,site} from the full response matrix χ_IJ
    (column J = response to perturbing site J), built by perturbing every
    correlated site independently — the general inequivalent/multi-site path."""
    inv = torch.linalg.inv(chi0_mat) - torch.linalg.inv(chi_mat)
    u = float(inv[site, site])
    return {"U_eV": u, "chi": float(chi_mat[site, site]),
            "chi0": float(chi0_mat[site, site]),
            "chi_col": chi_mat[:, site].tolist(),
            "chi0_col": chi0_mat[:, site].tolist(),
            "chi_mat": chi_mat.tolist(), "chi0_mat": chi0_mat.tolist()}


def _k_hxc_spin(
    res: SCFResult, xc: SpinXC, dru: torch.Tensor, drd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Δv↑, Δv↓) = K_Hxc^{σσ'} Δρ^{σ'}: Hartree kernel on Δρ_tot (G=0 excluded)
    plus f_xc^{σσ'} as an autograd HVP of E_xc at the SCF density (NLCC core
    split half/half per channel, exactly as the SCF potential was built).
    Both kernels are the shared postscf._response primitives."""
    core = res.system.rho_core
    cu2 = 0.0 if core is None else 0.5 * core
    kh = hartree_kernel(res.system.grid, dru + drd)
    # _k_hxc_spin is only ever called from _k_hxc_channels's `nspin == 2`
    # branch, and the NC SCF always sets rho_spin then (see results.py).
    assert res.rho_spin is not None
    fu, fd = fxc_hvp_spin(xc, res.rho_spin[0] + cu2, res.rho_spin[1] + cu2,
                          res.system.grid, dru, drd)
    return kh + fu, kh + fd


def _k_hxc_channels(
    res: SCFResult, xc: XCFunctional | SpinXC, drho: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Per-spin-channel Hxc potential response [Δv^σ] from the per-channel
    density response [Δρ^σ] (one entry per computed spin channel).

    nspin=2: the collinear spin kernel `_k_hxc_spin` on (Δρ↑, Δρ↓), with `xc` a
    spin functional. nspin=1: the spin-restricted limit — the projector probe is
    spin-symmetric on a closed-shell ground state, so Δρ↑=Δρ↓=drho[0], Δv↑=Δv↓
    and only one channel is returned. `xc` is then a non-spin functional (as for
    the nspin=1 SCF), so the closed-shell f_xc·Δρ_tot is the non-spin `fxc_hvp`
    at the full ρ+ρ_core, exactly matching the nspin=1 dielectric kernel."""
    if res.nspin == 2:
        assert isinstance(xc, SpinXC)
        du, dd = _k_hxc_spin(res, xc, drho[0], drho[1])
        return [du, dd]
    assert isinstance(xc, XCFunctional)
    core = res.system.rho_core
    rho_xc = res.rho if core is None else res.rho + core
    drho_tot = 2.0 * drho[0]  # Δρ_tot = Δρ↑ + Δρ↓, spin degeneracy of the one channel
    grid = res.system.grid
    return [hartree_kernel(grid, drho_tot) + fxc_hvp(xc, rho_xc, grid, drho_tot)]


@torch.no_grad()
def _response_columns(
    res: SCFResult,
    xc: XCFunctional | SpinXC,
    hub: HubbardData,
    hub_q: torch.Tensor,
    site: int,
    *,
    beta: float = 0.2,
    outer_tol: float = 1e-6,
    max_outer: int = 200,
    cg_tol: float = 1e-8,
    history: int = 8,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(χ0_col, χ_col, n_outer): analytic dN_I/dα_site by Sternheimer response.

    Bare column = first pass (frozen Hxc potential); interacting column = the
    damped fixed point of the response potential u^σ = K_Hxc Δρ(P + u).

    Both nspin: nspin=2 solves one Sternheimer channel per spin; nspin=1 solves
    the single spin-restricted channel and folds the ×2 spin degeneracy into the
    occupation response and the total Δρ that drives the Hxc kernel (the probe
    is spin-symmetric, so u↑=u↓ and only one channel is tracked)."""
    from gradwave.core.batch import (
        BatchedHamiltonian,
        box_to_sphere_b,
        g_to_r_b,
        projectors_b,
    )

    system = res.system
    bk, grid = system.batch, system.grid
    # the batched Sternheimer path below needs system.batch built.
    assert bk is not None
    kw = system.kweights
    nspin = res.nspin
    # g = spin degeneracy folded into each computed channel and the full band
    # filling per channel: 2 for nspin=1 (doubly-occupied spatial orbitals),
    # 1 for nspin=2 (one electron per spin channel).
    g = 2.0 / nspin
    ns = hub.n_sites
    st, dim = hub.sites[site]["start"], hub.sites[site]["dim"]
    qj = hub_q[:, st:st + dim, :]
    projs_b = projectors_b(bk, system.positions)

    # Per-spin-channel views of the (possibly spin-agnostic, nspin=1) result.
    def _ch(x, sp):
        return x if nspin == 1 else x[sp]

    c_occ, eps_occ, hs, shifts, probe_psi = [], [], [], [], []
    for sp in range(nspin):
        nocc = insulator_window(
            _ch(res.occupations, sp), g,
            "Sternheimer response needs insulating occupations (gap ≫ width)")
        c = _pad(_ch(res.coeffs, sp), bk.npw_max)[:, :nocc]
        e = _ch(res.eigenvalues, sp)[:, :nocc].to(RDTYPE)
        c_occ.append(c)
        eps_occ.append(e)
        hs.append(BatchedHamiltonian(bk, grid.shape, _ch(res.v_eff, sp), projs_b))
        shifts.append(sternheimer_shift(e))
        b = torch.einsum("kpg,kbg->kbp", qj.conj(), c)
        probe_psi.append(torch.einsum("kbp,kpg->kbg", b, qj))

    def n_response(dpsi_s):
        col = torch.zeros(ns, dtype=RDTYPE)
        for i, s in enumerate(hub.sites):
            qi = hub_q[:, s["start"]:s["start"] + s["dim"], :]
            for sp in range(nspin):
                b_c = torch.einsum("kpg,kbg->kbp", qi.conj(), c_occ[sp])
                b_d = torch.einsum("kpg,kbg->kbp", qi.conj(), dpsi_s[sp])
                # dN_I^σ = 2 Re Σ_n ⟨ψ_n|φ^I⟩⟨φ^I|dψ_n⟩; ×g folds the spin
                # degeneracy of the single nspin=1 channel (no-op for nspin=2).
                col[i] += g * 2.0 * float(
                    (kw[:, None, None] * (b_c.conj() * b_d).real).sum())
        return col

    # Anderson-accelerated fixed point on u = G(u) := K_Hxc[Δρ(P + u)].
    # Plain damping diverges here: the magnetization channel of K·χ0 can have
    # eigenvalues well below −1 (NiO: ≈ −6) — an antisymmetric Δm mode on the
    # two Ni that plain Richardson iteration amplifies.
    n_pts = grid.n_points
    u_flat = torch.zeros(nspin * n_pts, dtype=RDTYPE, device=hub_q.device)
    mixer = AndersonMixer(history, beta)
    dpsi = [torch.zeros_like(c_occ[sp]) for sp in range(nspin)]
    chi0_col, chi_prev = None, None
    for it in range(1, max_outer + 1):
        u_r = [u_flat[sp * n_pts:(sp + 1) * n_pts].reshape(grid.shape)
               for sp in range(nspin)]
        drho = []
        for sp in range(nspin):
            psi_r = g_to_r_b(c_occ[sp], bk, grid.shape)
            rhs = probe_psi[sp]
            if it > 1:
                rhs = rhs + box_to_sphere_b(psi_r * u_r[sp].to(psi_r.dtype), bk)
            ov = torch.einsum("kng,kbg->kbn", c_occ[sp].conj(), rhs)
            rhs = -(rhs - torch.einsum("kbn,kng->kbg", ov, c_occ[sp]))
            dpsi[sp] = _cg_sternheimer_b(hs[sp], bk, c_occ[sp], eps_occ[sp], rhs,
                                         dpsi[sp], shifts[sp], tol=cg_tol)
            dpsi_r = g_to_r_b(dpsi[sp], bk, grid.shape)
            # per-channel Δρ^σ = 2 Re Σ_n ψ_n* dψ_n (one electron per orbital);
            # _k_hxc_channels folds the nspin=1 spin factor into the total.
            dr = 2.0 * (kw[:, None, None, None, None]
                        * (psi_r.conj() * dpsi_r).real).sum(dim=(0, 1)) / grid.volume
            drho.append(dr)
        chi_col = n_response(dpsi)
        if it == 1:
            chi0_col = chi_col.clone()
        if verbose:
            print(f"  response it {it:3d}: chi_col = {chi_col.tolist()}")
        if chi_prev is not None and float((chi_col - chi_prev).abs().max()) < outer_tol:
            # chi_prev is only real from it==2 onward, and chi0_col is always
            # set on it==1, so it's real whenever this return is reached.
            assert chi0_col is not None
            return chi0_col, chi_col, it
        chi_prev = chi_col
        dv = _k_hxc_channels(res, xc, drho)
        r_vec = torch.cat([dv[sp].reshape(-1) for sp in range(nspin)]) - u_flat
        u_flat = mixer.step(u_flat, r_vec)
    raise RuntimeError(f"response fixed point not converged in {max_outer} iterations")


def linear_response_u_autodiff(
    system: System,
    xc: XCFunctional | SpinXC,
    l: int,
    species: int,
    *,
    site: int = 0,
    smearing: str = "gaussian",
    width: float = 0.05,
    scf_kwargs=None,
    beta: float = 0.2,
    outer_tol: float = 1e-6,
    max_outer: int = 200,
    cg_tol: float = 1e-8,
    history: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
    """Linear-response Hubbard U [eV] with analytic (Sternheimer) response —
    no finite differences, no probe SCF re-runs; ONE ground-state SCF total.

    The Hxc screening kernel comes from an autograd HVP of E_Hxc, so any
    twice-differentiable spin functional (including learnable ones) works
    without hand-coded f_xc. Insulators only (conduction-projected CG)."""
    scf_kwargs = dict(scf_kwargs or {})
    man = [HubbardManifold(species=species, l=l, u=0.0, j=0.0)]  # U computed at U=0
    hub = build_hubbard_projectors(system, man)
    hub_q = hubbard_projectors(hub, system.positions)

    # see linear_response_u's identical cast for why: scf()'s own
    # `xc: XCFunctional` annotation doesn't reflect its real nspin=2 contract.
    base = scf(system, cast("XCFunctional", xc), smearing=smearing, width=width,
              hubbard=man, **scf_kwargs)
    if not base.converged:
        raise RuntimeError("base SCF did not converge")

    def _column(j):
        return _response_columns(
            base, xc, hub, hub_q, j, beta=beta, outer_tol=outer_tol,
            max_outer=max_outer, cg_tol=cg_tol, history=history, verbose=verbose)

    if _use_full_matrix(hub.sites, system.species_of_atom):
        chi_cols, chi0_cols, n_outers = [], [], []
        for j in range(hub.n_sites):
            c0j, cj, no = _column(j)
            chi_cols.append(cj)
            chi0_cols.append(c0j)
            n_outers.append(no)
        chi_mat = torch.stack(chi_cols, dim=1)   # χ_IJ = dN_I/dα_J
        chi0_mat = torch.stack(chi0_cols, dim=1)
        out = _assemble_u_matrix(chi_mat, chi0_mat, site)
        out["n_outer"] = max(n_outers)
        return out

    chi0_col, chi_col, n_outer = _column(site)
    out = _assemble_u(chi_col, chi0_col, site, hub.sites, system.species_of_atom)
    out["n_outer"] = n_outer
    return out
