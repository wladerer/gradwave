"""Hubbard U as a determinable quantity, not just an input.

Two capabilities:

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

import math

import torch

from gradwave.constants import E2
from gradwave.core.density import sigma_from_rho
from gradwave.core.xc.base import xc_eager
from gradwave.core.fftbox import r_to_g
from gradwave.core.hubbard import (
    HubbardManifold,
    build_hubbard_projectors,
    hubbard_projectors,
    occupation_matrices,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.loop import scf


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


def _pad(coeffs_per_k, npw_max):
    nk = len(coeffs_per_k)
    nb = coeffs_per_k[0].shape[0]
    out = torch.zeros(nk, nb, npw_max, dtype=CDTYPE, device=coeffs_per_k[0].device)
    for ik, c in enumerate(coeffs_per_k):
        out[ik, :, : c.shape[1]] = c.detach()
    return out


@torch.no_grad()
def _bare_response_occ(system, base_res, hub, hub_q, alpha_vec, xc, smearing, width):
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
        occ, _ = occupations_and_entropy(eigs_s[sp], mu, scheme, width, degeneracy=g_spin)
        w = g_spin * occ if nspin == 1 else occ  # to electron units per spin
        w = 0.5 * w if nspin == 1 else w
        mats = occupation_matrices(hub_q, coeffs_s[sp], w, system.kweights, hub.sites)
        for i, m in enumerate(mats):
            n[i] += torch.trace(m).real
    return n * (2.0 if nspin == 1 else 1.0)


def linear_response_u(system, xc, l: int, species: int, *, site: int = 0,
                      alpha: float = 0.1, smearing="gaussian", width=0.05,
                      scf_kwargs=None) -> dict:
    """Compute the linear-response Hubbard U [eV] for a manifold.

    Perturbs one correlated `site`, measures the on-site occupation response,
    and returns χ0, χ, and U = (χ0^{-1} − χ^{-1})_{site,site}. Runs one base +
    two perturbed SCFs (interacting χ) and cheap one-shot solves (bare χ0)."""
    scf_kwargs = dict(scf_kwargs or {})
    man = [HubbardManifold(species=species, l=l, u=0.0, j=0.0)]  # U computed at U=0
    hub = build_hubbard_projectors(system, man)
    hub_q = hubbard_projectors(hub, system.positions)
    ns = hub.n_sites

    base = scf(system, xc, smearing=smearing, width=width, hubbard=man, **scf_kwargs)

    chi_cols, chi0_cols = [], []  # response of all sites to perturbing `site`
    for sgn in (+1.0, -1.0):
        av = [0.0] * ns
        av[site] = sgn * alpha
        # interacting: full SCF re-converged with the probe
        r = scf(system, xc, smearing=smearing, width=width, hubbard=man,
                hub_alpha=av, **scf_kwargs)
        chi_cols.append(_site_occupations(r, hub, hub_q))
        # bare: one diagonalization at the base self-consistent potential
        chi0_cols.append(_bare_response_occ(system, base, hub, hub_q,
                                            torch.tensor(av), xc, smearing, width))

    # central difference dN_I/dα_site
    chi_col = (chi_cols[0] - chi_cols[1]) / (2 * alpha)   # (ns,)
    chi0_col = (chi0_cols[0] - chi0_cols[1]) / (2 * alpha)
    return _assemble_u(chi_col, chi0_col, site)


def _assemble_u(chi_col: torch.Tensor, chi0_col: torch.Tensor, site: int) -> dict:
    """U = (χ0^{-1} − χ^{-1})_II from one response column.

    Two equivalent sites: the symmetric [[a,b],[b,a]] matrix is known from
    perturbing one site; otherwise the single-site scalar estimate."""
    ns = chi_col.shape[0]
    if ns == 2:
        chi = torch.tensor([[chi_col[site], chi_col[1 - site]],
                            [chi_col[1 - site], chi_col[site]]])
        chi0 = torch.tensor([[chi0_col[site], chi0_col[1 - site]],
                             [chi0_col[1 - site], chi0_col[site]]])
        u = float((torch.linalg.inv(chi0) - torch.linalg.inv(chi))[site, site])
    else:  # single-site scalar estimate
        u = float(1.0 / chi0_col[site] - 1.0 / chi_col[site])
    return {"U_eV": u, "chi": chi_col[site].item(), "chi0": chi0_col[site].item(),
            "chi_col": chi_col.tolist(), "chi0_col": chi0_col.tolist()}


def _cg_sternheimer_b(h, bk, c_occ, eps_occ, rhs, x0, shift, tol=1e-8, max_iter=400):
    """Batched conduction-projected Sternheimer: (H − ε_n + s·P_occ)δψ = rhs,
    for all occupied bands of all k at once ((nk, nocc, npw_max), masked).
    rhs must already lie in the conduction space; positive definite there
    thanks to the gap — insulators only."""
    from gradwave.solvers.precond import teter_b

    def p_occ(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return torch.einsum("kbn,kng->kbg", ov, c_occ)

    def a_apply(x):
        hx = h.apply(x) - eps_occ[..., None] * x
        return hx - p_occ(hx) + shift * p_occ(x)

    t_band = torch.clamp(
        torch.einsum("kbg,kg,kbg->kb", c_occ.conj(), bk.t.to(c_occ.dtype), c_occ).real,
        min=1e-6,
    )
    x = x0.clone()
    r = rhs - a_apply(x)
    z = teter_b(r, bk.t, t_band)
    p = z
    rz = torch.einsum("kbg,kbg->kb", r.conj(), z).real
    for _ in range(max_iter):
        ap = a_apply(p)
        pap = torch.einsum("kbg,kbg->kb", p.conj(), ap).real
        a_cg = rz / torch.clamp(pap, min=1e-300)
        x = x + a_cg[..., None] * p
        r = r - a_cg[..., None] * ap
        if float(torch.linalg.norm(r, dim=-1).max()) < tol:
            break
        z = teter_b(r, bk.t, t_band)
        rz_new = torch.einsum("kbg,kbg->kb", r.conj(), z).real
        p = z + (rz_new / torch.clamp(rz, min=1e-300))[..., None] * p
        rz = rz_new
    return x - p_occ(x)


def _k_hxc_spin(res, xc, dru, drd):
    """(Δv↑, Δv↓) = K_Hxc^{σσ'} Δρ^{σ'}: Hartree kernel on Δρ_tot (G=0 excluded)
    plus f_xc^{σσ'} as an autograd HVP of E_xc at the SCF density (NLCC core
    split half/half per channel, exactly as the SCF potential was built)."""
    grid = res.system.grid
    w_g = r_to_g((dru + drd).to(CDTYPE))
    inv_g2 = torch.where(
        grid.g2 > 1e-12, 1.0 / torch.clamp(grid.g2, min=1e-12), torch.zeros_like(grid.g2)
    )
    kh = (torch.fft.ifftn(4.0 * math.pi * E2 * w_g * inv_g2, dim=(-3, -2, -1))
          * grid.n_points).real

    core = res.system.rho_core
    cu2 = 0.0 if core is None else 0.5 * core
    ru = (res.rho_spin[0] + cu2).detach().clone().requires_grad_(True)
    rd = (res.rho_spin[1] + cu2).detach().clone().requires_grad_(True)
    # Double backward through E_xc for the f_xc kernel, so force eager.
    with torch.enable_grad(), xc_eager():
        if xc.needs_gradient:
            s_uu = sigma_from_rho(ru, grid.g_cart)
            s_dd = sigma_from_rho(rd, grid.g_cart)
            s_tot = sigma_from_rho(ru + rd, grid.g_cart)
        else:
            s_uu = s_dd = s_tot = None
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)
        vu, vd = torch.autograd.grad(e_xc, (ru, rd), create_graph=True)
        inner = (vu * dru.detach()).sum() + (vd * drd.detach()).sum()
        fu, fd = torch.autograd.grad(inner, (ru, rd))
    scale = grid.n_points / grid.volume
    return kh + fu * scale, kh + fd * scale


@torch.no_grad()
def _response_columns(res, xc, hub, hub_q, site, *, beta=0.2, outer_tol=1e-6,
                      max_outer=200, cg_tol=1e-8, history=8, verbose=False):
    """(χ0_col, χ_col, n_outer): analytic dN_I/dα_site by Sternheimer response.

    Bare column = first pass (frozen Hxc potential); interacting column = the
    damped fixed point of the response potential u^σ = K_Hxc Δρ(P + u)."""
    from gradwave.core.batch import (
        BatchedHamiltonian,
        box_to_sphere_b,
        g_to_r_b,
        projectors_b,
    )

    if res.nspin != 2:
        raise NotImplementedError("Sternheimer linear-response U: nspin=2 only for now")
    system = res.system
    bk, grid = system.batch, system.grid
    kw = system.kweights
    ns = hub.n_sites
    st, dim = hub.sites[site]["start"], hub.sites[site]["dim"]
    qj = hub_q[:, st:st + dim, :]
    projs_b = projectors_b(bk, system.positions)

    c_occ, eps_occ, hs, shifts, probe_psi = [], [], [], [], []
    for sp in range(2):
        occ = res.occupations[sp]
        nocc = int((occ[0] > 0.5).sum())
        ins = (occ[:, :nocc] > 1.0 - 1e-6).all() and (
            occ[:, nocc:] < 1e-6).all() if occ.shape[1] > nocc else (occ > 1.0 - 1e-6).all()
        if not ins:
            raise NotImplementedError(
                "Sternheimer response needs insulating occupations (gap ≫ width)")
        c = _pad(res.coeffs[sp], bk.npw_max)[:, :nocc]
        e = res.eigenvalues[sp][:, :nocc].to(RDTYPE)
        c_occ.append(c)
        eps_occ.append(e)
        hs.append(BatchedHamiltonian(bk, grid.shape, res.v_eff[sp], projs_b))
        shifts.append(2.0 * float(e.max() - e.min()) + 10.0)
        b = torch.einsum("kpg,kbg->kbp", qj.conj(), c)
        probe_psi.append(torch.einsum("kbp,kpg->kbg", b, qj))

    def n_response(dpsi_s):
        col = torch.zeros(ns, dtype=RDTYPE)
        for i, s in enumerate(hub.sites):
            qi = hub_q[:, s["start"]:s["start"] + s["dim"], :]
            for sp in range(2):
                b_c = torch.einsum("kpg,kbg->kbp", qi.conj(), c_occ[sp])
                b_d = torch.einsum("kpg,kbg->kbp", qi.conj(), dpsi_s[sp])
                col[i] += 2.0 * float(
                    (kw[:, None, None] * (b_c.conj() * b_d).real).sum())
        return col

    # Anderson-accelerated fixed point on u = G(u) := K_Hxc[Δρ(P + u)].
    # Plain damping diverges here: the magnetization channel of K·χ0 can have
    # eigenvalues well below −1 (NiO: ≈ −6) — an antisymmetric Δm mode on the
    # two Ni that plain Richardson iteration amplifies.
    n_pts = grid.n_points
    u_flat = torch.zeros(2 * n_pts, dtype=RDTYPE, device=hub_q.device)
    prev_u, prev_r, hist_du, hist_dr = None, None, [], []
    dpsi = [torch.zeros_like(c_occ[sp]) for sp in range(2)]
    chi0_col, chi_prev = None, None
    for it in range(1, max_outer + 1):
        u_r = [u_flat[:n_pts].reshape(grid.shape), u_flat[n_pts:].reshape(grid.shape)]
        drho = []
        for sp in range(2):
            psi_r = g_to_r_b(c_occ[sp], bk, grid.shape)
            rhs = probe_psi[sp]
            if it > 1:
                rhs = rhs + box_to_sphere_b(psi_r * u_r[sp].to(psi_r.dtype), bk)
            ov = torch.einsum("kng,kbg->kbn", c_occ[sp].conj(), rhs)
            rhs = -(rhs - torch.einsum("kbn,kng->kbg", ov, c_occ[sp]))
            dpsi[sp] = _cg_sternheimer_b(hs[sp], bk, c_occ[sp], eps_occ[sp], rhs,
                                         dpsi[sp], shifts[sp], tol=cg_tol)
            dpsi_r = g_to_r_b(dpsi[sp], bk, grid.shape)
            dr = 2.0 * (kw[:, None, None, None, None]
                        * (psi_r.conj() * dpsi_r).real).sum(dim=(0, 1)) / grid.volume
            drho.append(dr)
        chi_col = n_response(dpsi)
        if it == 1:
            chi0_col = chi_col.clone()
        if verbose:
            print(f"  response it {it:3d}: chi_col = {chi_col.tolist()}")
        if chi_prev is not None and float((chi_col - chi_prev).abs().max()) < outer_tol:
            return chi0_col, chi_col, it
        chi_prev = chi_col
        du, dd = _k_hxc_spin(res, xc, drho[0], drho[1])
        r_vec = torch.cat([du.reshape(-1), dd.reshape(-1)]) - u_flat
        if prev_r is not None:
            hist_du.append(u_flat - prev_u)
            hist_dr.append(r_vec - prev_r)
            if len(hist_dr) > history:
                hist_du.pop(0)
                hist_dr.pop(0)
        prev_u, prev_r = u_flat, r_vec
        if hist_dr:
            dr_m = torch.stack(hist_dr, dim=1)  # (2n, h)
            du_m = torch.stack(hist_du, dim=1)
            gamma = torch.linalg.lstsq(dr_m, r_vec[:, None]).solution[:, 0]
            u_flat = u_flat + beta * r_vec - (du_m + beta * dr_m) @ gamma
        else:
            u_flat = u_flat + beta * r_vec
    raise RuntimeError(f"response fixed point not converged in {max_outer} iterations")


def linear_response_u_autodiff(system, xc, l: int, species: int, *, site: int = 0,
                               smearing="gaussian", width=0.05, scf_kwargs=None,
                               beta=0.2, outer_tol=1e-6, max_outer=200,
                               cg_tol=1e-8, history=8, verbose=False) -> dict:
    """Linear-response Hubbard U [eV] with analytic (Sternheimer) response —
    no finite differences, no probe SCF re-runs; ONE ground-state SCF total.

    The Hxc screening kernel comes from an autograd HVP of E_Hxc, so any
    twice-differentiable spin functional (including learnable ones) works
    without hand-coded f_xc. Insulators only (conduction-projected CG)."""
    scf_kwargs = dict(scf_kwargs or {})
    man = [HubbardManifold(species=species, l=l, u=0.0, j=0.0)]  # U computed at U=0
    hub = build_hubbard_projectors(system, man)
    hub_q = hubbard_projectors(hub, system.positions)

    base = scf(system, xc, smearing=smearing, width=width, hubbard=man, **scf_kwargs)
    if not base.converged:
        raise RuntimeError("base SCF did not converge")
    chi0_col, chi_col, n_outer = _response_columns(
        base, xc, hub, hub_q, site, beta=beta, outer_tol=outer_tol,
        max_outer=max_outer, cg_tol=cg_tol, history=history, verbose=verbose)
    out = _assemble_u(chi_col, chi0_col, site)
    out["n_outer"] = n_outer
    return out
