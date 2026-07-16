"""Position response through the USPP/PAW SCF (analytic, Γ-point).

The SCF map F(x; τ) takes the composite state x = (ρ, becsum) to its
output at atom positions τ. Its analytic τ-derivative at fixed x has a
solver part — the orbitals respond to the bare perturbation of H and S —
and an explicit part from the augmentation phases and the moving
projectors in the becsum assembly. The self-consistent state response
then follows from the same forward fixed point the Newton finisher uses,

    δx = δx_bare + χ̃ K δx,

with χ̃K applied by the adjoint machinery (postscf/uspp_implicit). The
bare perturbation of the generalized eigenproblem carries the S motion:
first order in the S-metric gives, for window states,

    c_mn = ⟨ψ_m| δH − ε_n δS |ψ_n⟩ / (ε_n − ε_m)   (m ≠ n)
    c_nn = −½ ⟨ψ_n| δS |ψ_n⟩,

which restores S-orthonormality automatically, and a conduction-
complement Sternheimer solve with the same right-hand side. Fixed
occupations only (insulators): the divided-difference occupation
weights and the δμ channel of the θ-response do not appear here.

Every phase derivative is analytic: KB/atomic projectors carry
e^{−i(k+G)·τ} (∂ = −i(k+G)_α on the atom's columns), the augmentation
pairing carries e^{+iG·τ} (∂ = +iG_α), and the augmentation density
carries the conjugate (∂ = −iG_α). δv_loc comes from a jvp through the
τ-differentiable local potential. Coverage: nspin=1, no +U, Γ-phonon
scope (q = 0).
"""

from __future__ import annotations

import torch

from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import becp
from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._anderson import AndersonMixer
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.newton import _pack, _unpack


def _dvloc_r(system, a: int, alpha: int) -> torch.Tensor:
    """∂v_loc(r)/∂τ_{aα} by a jvp through the τ-differentiable builder."""
    grid = system.grid
    dev = system.positions.device
    tang = torch.zeros_like(system.positions)
    tang[a, alpha] = 1.0

    def f(pos):
        vg = local_potential_g(pos,
                               torch.tensor(system.species_of_atom,
                                            device=dev),
                               system.vloc_tables, grid.g_cart, grid.volume)
        return (torch.fft.ifftn(vg, dim=(-3, -2, -1)) * grid.n_points).real

    _, dv = torch.func.jvp(f, (system.positions,), (tang,))
    return dv


def _drho_core_r(system, a: int, alpha: int) -> torch.Tensor:
    """∂ρ_core(r)/∂τ_{aα} — the NLCC core density is atom-centered and
    moves with its atom (analytic phase derivative of the setup product).
    Returns None when the species carries no core."""
    if system.rho_core is None:
        return None
    sp_a = system.species_of_atom[a]
    paw = system.paws[sp_a]
    if paw.core_rho is None:
        return None
    import numpy as np

    from gradwave.pseudo.atomic import core_density_of_q
    from gradwave.scf.loop import _unique_shells

    grid = system.grid
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    tab = torch.as_tensor(core_density_of_q(paw, uniq), dtype=RDTYPE)
    shell = tab[torch.as_tensor(inverse)]
    gc = grid.g_cart.reshape(-1, 3)
    phase = torch.exp(torch.complex(
        torch.zeros(gc.shape[0], dtype=RDTYPE),
        -(gc @ system.positions[a].to(RDTYPE))))
    dcore_g = (-1j * gc[:, alpha].to(CDTYPE)) * phase         * shell.to(CDTYPE) / grid.volume
    mask = grid.dens_mask.reshape(-1)
    dcore_g = torch.where(mask, dcore_g, torch.zeros_like(dcore_g))
    return torch.fft.ifftn(dcore_g.reshape(grid.shape) * grid.n_points,
                           dim=(-3, -2, -1)).real


def _fxc_apply(cs, w: torch.Tensor) -> torch.Tensor:
    """f_xc·w at the converged density — the XC half of k_hxc_grid without
    the Hartree kernel (the NLCC core carries no Hartree)."""
    from gradwave.core.density import sigma_from_rho

    grid = cs.grid
    rho = cs.rho_xc.detach().clone().requires_grad_(True)
    # Double backward through E_xc for the position f_xc, so force eager.
    with torch.enable_grad(), xc_eager():
        sigma = (sigma_from_rho(rho, grid.g_cart)
                 if cs.xc.needs_gradient else None)
        e_xc = cs.xc.energy(rho, cs.vol, sigma)
        (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
        inner = (v_xc * w.detach()).sum()
        (fxc_w,) = torch.autograd.grad(inner, rho)
    return fxc_w * (grid.n_points / cs.vol)


class PositionPerturbation:
    """Frozen ingredients of ∂F/∂τ_{aα} at one displacement."""

    def __init__(self, cs: _ConvergedUSPP, a: int, alpha: int):
        system = cs.system
        self.cs, self.a, self.alpha = cs, a, alpha
        self.s0, self.s1 = system.atom_slices[a]
        self.dv_r = _dvloc_r(system, a, alpha)
        # NLCC: the atom-centered core density moves too, and its motion
        # perturbs v_xc as f_xc·∂ρ_core/∂τ — a bare LOCAL term (no
        # Hartree; the core only enters the XC functional)
        dcore = _drho_core_r(system, a, alpha)
        if dcore is not None:
            self.dv_r = self.dv_r + _fxc_apply(cs, dcore)
        # ∂(∫v_eff Q_a)/∂τ_α: the dscr pairing carries e^{+iG·τ_a}
        v_g = r_to_g(cs.veff_sp[0].to(CDTYPE)).reshape(-1)[cs.mask_flat]
        g_a = system.g_sphere[:, alpha]
        contr = torch.einsum(
            "ijg,g->ij",
            system.aug[system.species_of_atom[a]].q_g[
                :, :, :].conj(),
            v_g * (1j * g_a.to(CDTYPE)) * cs.phase_pos[:, a])
        self.d_dscr = torch.zeros_like(system.q_full)
        self.d_dscr[self.s0:self.s1, self.s0:self.s1] = \
            (0.5 * (contr + contr.conj().T)).real
        # ∫δv_loc Q for EVERY atom (the local motion also re-screens D)
        self.d_dscr = self.d_dscr + cs._aug_dmat(self.dv_r)

    def dproj(self, hk, sph):
        """∂p rows for atom a's columns: −i(k+G)_α ⊙ p (zero elsewhere)."""
        kpga = sph.kpg[:, self.alpha].to(CDTYPE)
        dp = torch.zeros_like(hk.p)
        dp[self.s0:self.s1] = -1j * kpga[None, :] * hk.p[self.s0:self.s1]
        return dp

    def dh_ds_psi(self, isp: int, ik: int, w_extra=None, d_extra=None):
        """(δH|ψ⟩, δS|ψ⟩) over the window at k. Bare perturbation by
        default (local δv_loc, D re-screening, moving projectors); pass
        w_extra (grid) and d_extra (full D matrix) to add the converged
        self-consistent potential change for total-δψ reconstruction."""
        cs = self.cs
        hk = cs.hks[isp][ik]
        c = cs.c_win[isp][ik]
        sph = cs.system.spheres[ik]
        b = cs.b_win[isp][ik]
        dp = self.dproj(hk, sph)
        db = becp(dp, c)
        dscr = hk.dscr
        qf = hk.q
        dv = self.dv_r if w_extra is None else self.dv_r + w_extra
        dd = self.d_dscr if d_extra is None else self.d_dscr + d_extra
        psi_r = g_to_r(c, sph.flat_idx, cs.shape)
        dh = box_to_sphere(r_to_g(psi_r * dv), sph.flat_idx)
        dh = dh + (b @ dd.to(CDTYPE)) @ hk.p
        dh = dh + (b @ dscr) @ dp + (db @ dscr) @ hk.p
        ds = (b @ qf) @ dp + (db @ qf) @ hk.p
        return dh, ds, dp, db

    def bare_map_derivative(self, dpsi_warm, cg_tol=1e-10, cg_max_iter=400):
        """∂F/∂τ_{aα} at fixed input x: (δρ_out, δbec_out per atom).

        Window part from S-metric perturbation theory, complement part
        from the projected Sternheimer solve, plus the explicit motion of
        the becsum projectors and the augmentation phases."""
        cs = self.cs
        system, grid = cs.system, cs.grid
        kw = system.kweights
        isp = 0  # nspin=1 scope
        drho_sm = torch.zeros(cs.shape, dtype=RDTYPE)
        dbec = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE)
                for (s0, s1) in system.atom_slices]
        for ik, sph in enumerate(system.spheres):
            hk, c = cs.hks[isp][ik], cs.c_win[isp][ik]
            ns = cs.n_solve[isp][ik]
            f, eps = cs.f_win[isp][ik], cs.eps_win[isp][ik]
            wk = float(kw[ik])
            dh, ds, dp, db = self.dh_ds_psi(isp, ik)

            # window coefficients c_mn (m any window state, n occupied)
            hmat = torch.einsum("mg,ng->mn", c.conj(), dh)
            smat = torch.einsum("mg,ng->mn", c.conj(), ds)
            de = (eps[None, :] - eps[:, None]).to(CDTYPE)  # ε_n − ε_m
            num = hmat - smat * eps[None, :].to(CDTYPE)
            safe = de.abs() > 1e-8
            de_safe = torch.where(safe, de, torch.ones_like(de))
            # m ≠ n: c_mn = ⟨m|δH − ε_n δS|n⟩/(ε_n − ε_m); degenerate and
            # diagonal entries take the S-orthonormality gauge −½⟨m|δS|n⟩
            # (equal-f degenerate rotations cancel in every invariant sum)
            cmn = torch.where(safe, num / de_safe, -0.5 * smat)
            dpsi_win = cmn.mT @ c  # δψ_n(win) = Σ_m c_mn ψ_m, rows n

            # complement: (H − ε_n S) δψ⊥ = −P_c†(δH − ε_n δS)|ψ_n⟩, occ n
            rhs = dh[:ns] - eps[:ns, None].to(CDTYPE) * ds[:ns]
            rhs = -(rhs - (rhs @ c.conj().T) @ cs.s_win[isp][ik])
            dperp = cs._sternheimer_k(isp, ik, rhs, dpsi_warm[ik],
                                      cg_tol, cg_max_iter)
            dpsi_warm[ik] = dperp
            dpsi = dpsi_win[:ns] + dperp

            fw = f[:ns]
            psi_r = g_to_r(c[:ns], sph.flat_idx, cs.shape)
            dpsi_r = g_to_r(dpsi, sph.flat_idx, cs.shape)
            drho_sm += 2.0 * wk * torch.einsum(
                "b,bxyz->xyz", fw, (psi_r.conj() * dpsi_r).real)

            # becsum: orbital response + explicit projector motion
            b = cs.b_win[isp][ik]
            b_d = becp(hk.p, dpsi) + db[:ns]
            for at, (s0, s1) in enumerate(system.atom_slices):
                bo, bd = b[:ns, s0:s1], b_d[:, s0:s1]
                m1 = torch.einsum("b,bi,bj->ij", fw.to(CDTYPE),
                                  bd.conj(), bo)
                dbec[at] += wk * (m1 + m1.conj().T)
        dbec = [0.5 * (m + m.conj().T) for m in dbec]

        # augmentation density: response becsum at fixed phases plus the
        # explicit phase motion of atom a with the map-output becsum
        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for at, sp in enumerate(system.species_of_atom):
            aug_sph = aug_sph + cs.phase_pos[:, at].conj() * torch.einsum(
                "ij,ijg->g", dbec[at], system.aug[sp].q_g)
        g_a = system.g_sphere[:, self.alpha].to(CDTYPE)
        sp_a = system.species_of_atom[self.a]
        aug_sph = aug_sph + (-1j * g_a) * cs.phase_pos[:, self.a].conj() \
            * torch.einsum("ij,ijg->g",
                           cs.rho_ij_sp[isp][self.a].to(CDTYPE),
                           system.aug[sp_a].q_g)
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        aug_box[system.sphere_idx] = aug_sph / cs.vol
        drho_aug = torch.fft.ifftn(aug_box.reshape(cs.shape) * grid.n_points,
                                   dim=(-3, -2, -1)).real
        return drho_sm / cs.vol + drho_aug, dbec


def bare_position_derivative(res: dict, xc, a: int, alpha: int,
                             cg_tol: float = 1e-10,
                             cg_max_iter: int = 400):
    """∂F_map/∂τ_{aα} at the converged state (fixed input x*). Returns
    (δρ_out, δbec_out per atom). Insulators, nspin=1, no +U."""
    _check_supported(res)
    if res.get("nspin", 1) != 1 or "hub_occ" in res:
        raise NotImplementedError("position response: nspin=1, no +U")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        return pert.bare_map_derivative(warm, cg_tol, cg_max_iter)


def _self_consistent_response(cs, bare_rho, bare_bec, *, beta=0.3,
                              history=12, inner_tol=1e-9, max_inner=80,
                              cg_tol=1e-10, cg_max_iter=300, verbose=False):
    """δx = (1 − χ̃K)⁻¹ δx_bare — the forward fixed point of the Newton
    finisher with the bare position derivative as the source. Returns the
    self-consistent (δρ*, δbec*, w_total) where w_total = K δx is the
    self-consistent potential change (needed to rebuild δψ later)."""
    system = cs.system
    shape, n_pts = tuple(cs.shape), cs.grid.n_points
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    r_vec = _pack(bare_rho.to(RDTYPE),
                  [m.real.to(RDTYPE) for m in bare_bec])
    dpsi_warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                  zip(cs.c_win[0], cs.n_solve[0], strict=True)]]
    d = r_vec.clone()
    mixer = AndersonMixer(history, beta)
    w_sp = None
    for it in range(1, max_inner + 1):
        d_rho, d_bec = _unpack(d, shape, n_pts, nbec)
        w_sp = cs.k_hxc_grid([d_rho])
        d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in d_bec]])
        chi_rho, chi_bec, _ = cs.apply_chi0(w_sp, d_ddd, dpsi_warm,
                                            cg_tol, cg_max_iter)
        g_vec = r_vec + _pack(chi_rho[0].to(RDTYPE),
                              [m.real.to(RDTYPE) for m in chi_bec[0]])
        g_res = g_vec - d
        gn = float(torch.linalg.norm(g_res)) / max(
            1.0, float(torch.linalg.norm(d)))
        if verbose:
            print(f"  pos-response it {it:3d}: |r|/|d| = {gn:.3e}")
        if gn < inner_tol:
            d = g_vec
            break
        d = mixer.step(d, g_res)
    else:
        raise RuntimeError(f"position response not converged ({gn:.2e} "
                           f"after {max_inner} iterations)")
    d_rho, d_bec = _unpack(d, shape, n_pts, nbec)
    w_sp = cs.k_hxc_grid([d_rho])
    d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in d_bec]])
    return d_rho, d_bec, (w_sp[0], d_ddd[0])


def position_density_response(res: dict, xc, a: int, alpha: int, *,
                              beta: float = 0.3, history: int = 12,
                              inner_tol: float = 1e-9, max_inner: int = 80,
                              cg_tol: float = 1e-10, cg_max_iter: int = 300,
                              verbose: bool = False):
    """Self-consistent dρ*/dτ_{aα} and dbecsum*/dτ_{aα} at the converged
    SCF point (analytic — no SCF re-runs). Insulators, nspin=1, no +U."""
    _check_supported(res)
    if res.get("nspin", 1) != 1 or "hub_occ" in res:
        raise NotImplementedError("position response: nspin=1, no +U")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        bare_rho, bare_bec = pert.bare_map_derivative(warm, cg_tol,
                                                      cg_max_iter)
        d_rho, d_bec, w_tot = _self_consistent_response(
            cs, bare_rho, bare_bec, beta=beta, history=history,
            inner_tol=inner_tol, max_inner=max_inner, cg_tol=cg_tol,
            cg_max_iter=cg_max_iter, verbose=verbose)
    return d_rho, d_bec, w_tot


def _total_orbital_response(cs, pert, w_grid, d_ddd, cg_tol=1e-10,
                            cg_max_iter=400):
    """δψ_n, δε_n, and the SMOOTH density derivative at the TOTAL
    perturbation (bare position motion + converged self-consistent
    potential change). One window-PT + Sternheimer pass per k."""
    system = cs.system
    isp = 0
    d_extra = cs._aug_dmat(w_grid)
    for a, (s0, s1) in enumerate(system.atom_slices):
        d_extra[s0:s1, s0:s1] += d_ddd[a].real
    dpsi_all, deps_all = [], []
    drho_sm = torch.zeros(cs.shape, dtype=RDTYPE)
    warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
            zip(cs.c_win[0], cs.n_solve[0], strict=True)]
    for ik, sph in enumerate(system.spheres):
        c = cs.c_win[isp][ik]
        ns = cs.n_solve[isp][ik]
        f, eps = cs.f_win[isp][ik], cs.eps_win[isp][ik]
        wk = float(system.kweights[ik])
        dh, ds, dp, db = pert.dh_ds_psi(isp, ik, w_extra=w_grid,
                                        d_extra=d_extra)
        hmat = torch.einsum("mg,ng->mn", c.conj(), dh)
        smat = torch.einsum("mg,ng->mn", c.conj(), ds)
        de = (eps[None, :] - eps[:, None]).to(CDTYPE)
        num = hmat - smat * eps[None, :].to(CDTYPE)
        safe = de.abs() > 1e-8
        de_safe = torch.where(safe, de, torch.ones_like(de))
        cmn = torch.where(safe, num / de_safe, -0.5 * smat)
        dpsi_win = cmn.mT @ c
        rhs = dh[:ns] - eps[:ns, None].to(CDTYPE) * ds[:ns]
        rhs = -(rhs - (rhs @ c.conj().T) @ cs.s_win[isp][ik])
        dperp = cs._sternheimer_k(isp, ik, rhs, warm[ik], cg_tol,
                                  cg_max_iter)
        dpsi = dpsi_win[:ns] + dperp
        deps = (hmat.diagonal().real - eps * smat.diagonal().real)[:ns]
        dpsi_all.append(dpsi)
        deps_all.append(deps)
        fw = f[:ns]
        psi_r = g_to_r(c[:ns], sph.flat_idx, cs.shape)
        dpsi_r = g_to_r(dpsi, sph.flat_idx, cs.shape)
        drho_sm += 2.0 * wk * torch.einsum(
            "b,bxyz->xyz", fw, (psi_r.conj() * dpsi_r).real)
    return dpsi_all, deps_all, drho_sm / cs.vol


def hessian_column(res: dict, xc, a: int, alpha: int, *,
                   response_kw=None, verbose: bool = False):
    """d²E/dτ dτ_{aα} — one analytic Hessian column (na, 3), no SCF
    re-runs. The τ-graph of the force expression is differentiated once
    more along the direction (e_{aα}, δstate/δτ_{aα}): the scalar
    s = dE'/dλ (one create_graph backward plus real pairings with the
    state response) has ∂s/∂τ equal to the full mixed column, explicit
    ∂²E/∂τ∂τ' included. Insulators, nspin=1, no +U."""
    from gradwave.core.density import sigma_from_rho
    from gradwave.core.energies.ewald import ewald_energy
    from gradwave.core.energies.hartree import hartree_energy
    from gradwave.core.energies.local_pp import local_energy
    from gradwave.core.energies.nl_pp import nonlocal_energy
    from gradwave.core.hamiltonian import projectors
    from gradwave.postscf.paw_forces import _aug_at_fixed, _aug_from_becsum

    _check_supported(res)
    if res.get("nspin", 1) != 1 or "hub_occ" in res:
        raise NotImplementedError("hessian_column: nspin=1, no +U")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("hessian_column: insulators only")
    system = res["system"]
    grid = system.grid
    vol, shape = grid.volume, grid.shape
    kw = system.kweights
    rkw = dict(response_kw or {})
    rkw.setdefault("verbose", verbose)

    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        bare_rho, bare_bec = pert.bare_map_derivative(warm)
        d_rho, d_bec, (w_grid, d_ddd) = _self_consistent_response(
            cs, bare_rho, bare_bec, **rkw)
        dpsi_all, deps_all, drho_sm = _total_orbital_response(
            cs, pert, w_grid, d_ddd)
        dbec_tot = d_bec  # self-consistent becsum response (Hermitian)
        # ddd response at the converged becsum response
        dddd = cs.hvp_onecenter([[m.to(torch.complex128)
                                  for m in dbec_tot]])[0]

    # rebuild the force-energy graph with STATE LEAVES
    pos = system.positions.detach().clone().requires_grad_(True)
    coeffs0 = res["coeffs"]
    occ = res["occupations"].detach()
    eigs0 = res["eigenvalues"].detach()
    is_paw = any(p.is_paw for p in system.paws)
    ddd_leaves = []
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for at, sp in enumerate(system.species_of_atom):
            _, ddd = onec[sp].energy_and_ddd(res["rho_ij_atoms"][at].detach())
            ddd_leaves.append(ddd.detach().clone().requires_grad_(True))
    ns_k = cs.n_solve[0]
    c_leaves = [res["coeffs"][ik][:ns_k[ik]].detach().clone()
                .requires_grad_(True) for ik in range(len(coeffs0))]
    eps_leaf = eigs0.clone().requires_grad_(True)
    rho_s_fixed = (res["rho"].detach()
                   - _aug_at_fixed(res, system, None)).detach()
    rho_s_leaf = rho_s_fixed.clone().requires_grad_(True)

    projs = [projectors(pd, pos) for pd in system.proj_data]
    phase_arg = system.g_sphere @ pos.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg),
                                     phase_arg))
    q = system.q_full.to(CDTYPE)
    e = ewald_energy(pos, system.charges, grid.cell)
    becps_full, rho_ij = [], [
        torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=pos.device)
        for (s0, s1) in system.atom_slices]
    for ik in range(len(c_leaves)):
        nsk = ns_k[ik]
        b = becp(projs[ik], c_leaves[ik])
        becps_full.append(b)
        w = (kw[ik] * occ[ik][:nsk]).to(CDTYPE)
        for at, (s0, s1) in enumerate(system.atom_slices):
            ba = b[:, s0:s1]
            rho_ij[at] = rho_ij[at] + torch.einsum("b,bi,bj->ij", w,
                                                   ba.conj(), ba)
    rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]
    rho_aug = _aug_from_becsum(system, rho_ij, phases)
    rho_tot = rho_s_leaf + rho_aug
    ns0 = ns_k[0]
    assert all(n == ns0 for n in ns_k), "insulator: uniform occupied count"
    e = e + nonlocal_energy(becps_full, system.proj_data[0].dij_full,
                            occ[:, :ns0], kw)
    for ik, b in enumerate(becps_full):
        nsk = ns_k[ik]
        quad = torch.einsum("bi,ij,bj->b", b.conj(), q, b).real
        e = e - (kw[ik] * occ[ik][:nsk] * eps_leaf[ik][:nsk] * quad).sum()
    if is_paw:
        for at in range(len(system.atom_slices)):
            e = e + (ddd_leaves[at].to(CDTYPE) * rho_ij[at]).sum().real
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    rho_core = None
    if system.rho_core is not None:
        from gradwave.pseudo.radial_torch import RadialTables

        q_sph = torch.linalg.norm(system.g_sphere, dim=1)
        core = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for sp in set(system.species_of_atom):
            paw = system.paws[sp]
            if paw.core_rho is None:
                continue
            tab = RadialTables(paw, device=pos.device)
            with torch.no_grad():
                f_core = tab.core_of_g(q_sph)
            atoms = [at for at, sa in enumerate(system.species_of_atom)
                     if sa == sp]
            core = core + phases[:, atoms].conj().sum(dim=1) \
                * f_core.to(CDTYPE) / vol
        core_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        core_box[system.sphere_idx] = core
        rho_core = torch.fft.ifftn(core_box.reshape(shape) * grid.n_points,
                                   dim=(-3, -2, -1)).real
    rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    # grads below take create_graph=True (a second backward follows), so keep
    # this E_xc eager, compiled aot_autograd cannot double-backward.
    with xc_eager():
        e = e + xc.energy(rho_xc, vol, sigma)
    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)
    e = e + hartree_energy(rho_g, grid.g2, vol) + local_energy(rho_g,
                                                               vloc_g, vol)

    leaves = [pos, rho_s_leaf, eps_leaf] + c_leaves + ddd_leaves
    grads = torch.autograd.grad(e, leaves, create_graph=True)
    g_pos, g_rho, g_eps = grads[0], grads[1], grads[2]
    g_c = grads[3:3 + len(c_leaves)]
    g_ddd = grads[3 + len(c_leaves):]

    # directional derivative de'/dλ along (e_{aα}, δstate); torch's
    # complex grads are conjugate-Wirtinger, so the real pairing for a
    # complex leaf is Re⟨g, δz⟩ summed over both Wirtinger halves —
    # (g.conj()*δz).real reproduces d/dt e(z + t δz)
    s = g_pos[a, alpha]
    s = s + (g_rho * drho_sm).sum()
    for ik in range(len(c_leaves)):
        s = s + (g_c[ik].conj() * dpsi_all[ik]).real.sum()
    for ik in range(len(c_leaves)):
        nsk = ns_k[ik]
        s = s + (g_eps[ik][:nsk] * deps_all[ik]).sum()
    for at in range(len(ddd_leaves)):
        s = s + (g_ddd[at] * dddd[at].real).sum()
    (col,) = torch.autograd.grad(s, pos)
    return col.detach()
