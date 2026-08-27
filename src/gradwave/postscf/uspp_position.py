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
τ-differentiable local potential.

DFT+U (Dudarev): the S-dressed atomic-orbital projectors Sφ move with their
atom too, so the position perturbation gains a bare V_U motion
δH_U = |∂(Sφ)⟩D_U⟨Sφ| + h.c. (∂(Sφ) built by a jvp through the same
build_uspp_hubbard S-dressing the SCF uses), the SCF map's composite state
gains the occupation-matrix channel n^{Iσ}, and the self-consistent response
threads its Dudarev kernel δD_U = −(U−J)·herm(δn) through the same
apply_chi0/k_hub machinery the density-loss adjoint already carries.

nspin=2: every packing layer here threads the spin axis the way the frozen
operators (_ConvergedUSPP.{apply_chi0, k_hxc_grid, hvp_onecenter, k_hub})
already do — per-spin bare map derivative, the composite fixed point on the
per-spin (δρ^σ, δbec^σ) + per-channel δn vector (newton._pack/_unpack layout),
and the spin-resolved force-energy graph in hessian_column (SpinXC on the total
density, mirroring paw_forces.forces_uspp). Coverage: nspin 1 or 2, ±U,
insulators, Γ-phonon scope (q = 0).
"""

from __future__ import annotations

import torch

from gradwave.core._anderson import AndersonMixer
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import box_to_sphere, g_to_r, g_to_r_box, r_to_g
from gradwave.core.hamiltonian import becp
from gradwave.core.xc.base import XCFunctional, xc_eager
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import fxc_hvp, fxc_hvp_spin
from gradwave.postscf.newton import _pack, _unpack
from gradwave.postscf.uspp_frozen import aug_density_from_becsum
from gradwave.postscf.uspp_implicit import _check_supported, _ConvergedUSPP
from gradwave.scf.results import USPPResult


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
        return g_to_r_box(vg, real=True)

    # `torch.func.jvp`'s stub return type is `tuple[Any, Any] |
    # tuple[Any, Any, Any]` (the 3-tuple variant is only real when
    # has_aux=True, not passed here) -- ty can't narrow the arity from the
    # default `bool` parameter type, so index instead of destructuring.
    jvp_out = torch.func.jvp(f, (system.positions,), (tang,))
    return jvp_out[1]


def _drho_core_r(system, a: int, alpha: int) -> torch.Tensor | None:
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
    dev = system.positions.device
    # the shell table is built through scipy/numpy (CPU), then moved back to
    # the working device so the response stays on-device
    g_flat = np.sqrt(grid.g2.reshape(-1).cpu().numpy())
    uniq, inverse = _unique_shells(g_flat)
    tab = torch.as_tensor(core_density_of_q(paw, uniq), dtype=RDTYPE, device=dev)
    shell = tab[torch.as_tensor(inverse, device=dev)]
    gc = grid.g_cart.reshape(-1, 3).to(dev)
    phase = torch.exp(torch.complex(
        torch.zeros(gc.shape[0], dtype=RDTYPE, device=dev),
        -(gc @ system.positions[a].to(RDTYPE))))
    dcore_g = (-1j * gc[:, alpha].to(CDTYPE)) * phase         * shell.to(CDTYPE) / grid.volume
    mask = grid.dens_mask.reshape(-1).to(dev)
    dcore_g = torch.where(mask, dcore_g, torch.zeros_like(dcore_g))
    return g_to_r_box(dcore_g.reshape(grid.shape), real=True)


def _fxc_core_apply(cs, dcore: torch.Tensor) -> list[torch.Tensor]:
    """δv_xc^σ from the NLCC core motion δρ_core — the XC half of k_hxc_grid
    without the Hartree kernel (the NLCC core carries no Hartree). Per-spin
    list. The core splits half/half over spin (ρ_xc^σ = ρ^σ + ρ_core/nspin),
    so for nspin=2 the perturbation δρ_core/2 enters BOTH channels and the spin
    kernel f_xc^{σσ'} couples them. Shared HVPs from postscf._response."""
    if cs.nspin == 1:
        # nspin=1 is always paired with a collinear XCFunctional (same
        # convention as scf/uspp_loop.py's own xc dispatch).
        assert isinstance(cs.xc, XCFunctional)
        return [fxc_hvp(cs.xc, cs.rho_xc, cs.grid, dcore)]
    assert isinstance(cs.xc, SpinXC)
    core = cs.system.rho_core
    c2 = 0.0 if core is None else 0.5 * core
    half = 0.5 * dcore
    fu, fd = fxc_hvp_spin(cs.xc, cs.rho_sp[0] + c2, cs.rho_sp[1] + c2,
                          cs.grid, half, half)
    return [fu, fd]


class PositionPerturbation:
    """Frozen ingredients of ∂F/∂τ_{aα} at one displacement."""

    def __init__(self, cs: _ConvergedUSPP, a: int, alpha: int) -> None:
        system = cs.system
        self.cs, self.a, self.alpha = cs, a, alpha
        self.s0, self.s1 = system.atom_slices[a]
        # window off-diagonal degenerate gauge (see ``window_response``).
        # False (default) → −½⟨m|δS|n⟩, the S-normalization limit that
        # reproduces the density/becsum response. True → the full metric
        # coupling −⟨m|δS|n⟩, the continuous ε_n→ε_m limit of the
        # non-degenerate coefficient that keeps the SECOND-derivative
        # assembly (``hessian_column``) continuous across a band degeneracy.
        self.deg_full = False
        nsp = cs.nspin
        # δv_loc is the spin-independent local-pseudopotential motion; the
        # per-spin δv_eff^σ (dv_r) adds the NLCC core term below
        dv_loc = _dvloc_r(system, a, alpha)
        self.dv_r = [dv_loc for _ in range(nsp)]
        # NLCC: the atom-centered core density moves too, and its motion
        # perturbs v_xc as f_xc·∂ρ_core/∂τ — a bare LOCAL term (no Hartree; the
        # core only enters the XC functional). For nspin=2 f_xc^{σσ'} makes the
        # per-spin δv_xc^σ differ, so dv_r is per spin.
        dcore = _drho_core_r(system, a, alpha)
        if dcore is not None:
            core_v = _fxc_core_apply(cs, dcore)
            self.dv_r = [self.dv_r[isp] + core_v[isp] for isp in range(nsp)]
        # ∂(∫v_eff^σ Q_a)/∂τ_α per spin: the dscr pairing carries e^{+iG·τ_a},
        # and ∫δv_eff^σ Q re-screens D for EVERY atom (the local + core motion)
        g_a = system.g_sphere[:, alpha]
        q_g_a = system.aug[system.species_of_atom[a]].q_g.conj()
        self.d_dscr = []
        for isp in range(nsp):
            v_g = r_to_g(cs.veff_sp[isp].to(CDTYPE)).reshape(-1)[cs.mask_flat]
            contr = torch.einsum(
                "ijg,g->ij", q_g_a,
                v_g * (1j * g_a.to(CDTYPE)) * cs.phase_pos[:, a])
            dd = torch.zeros_like(system.q_full)
            dd[self.s0:self.s1, self.s0:self.s1] = \
                (0.5 * (contr + contr.conj().T)).real
            dd = dd + cs._aug_dmat(self.dv_r[isp])
            self.d_dscr.append(dd)

        # DFT+U: ∂(Sφ)/∂τ_{aα}, the moving S-dressed atomic-orbital projectors
        # (both the φ phase and the β phases inside S carry atom a's motion).
        # Built by a jvp through the same build_uspp_hubbard S-dressing the SCF
        # froze into hk.hub_sphi, so the bare V_U motion stays convention-exact.
        self.dsphi = None
        if cs.hub is not None:
            self.dsphi = self._build_dsphi()

    def _build_dsphi(self):
        """Per-k ∂(Sφ)/∂τ_{aα} (nprojU, npw) for the +U projectors."""
        from gradwave.core.hamiltonian import projectors
        from gradwave.scf.uspp_hubbard import atom_of_col, phi_free_per_k

        cs = self.cs
        system = cs.system
        # only ever called from __init__ under `if cs.hub is not None:`.
        assert cs.hub is not None
        sites = cs.hub.sites
        phi_free = phi_free_per_k(system, sites)  # per-k phase-free φ
        acol = atom_of_col(sites).to(system.positions.device)
        q = system.q_full.to(CDTYPE)
        tang = torch.zeros_like(system.positions)
        tang[self.a, self.alpha] = 1.0
        out = []
        for ik, sph in enumerate(system.spheres):
            pf = phi_free[ik]
            pd = system.proj_data[ik]

            def build(pos, pf=pf, sph=sph, pd=pd):
                pharg = sph.kpg @ pos.T  # (npw, na)
                ph = torch.exp(torch.complex(torch.zeros_like(pharg), -pharg))
                phik = pf * ph[:, acol].T  # (nprojU, npw), φ phased
                p = projectors(pd, pos)  # (nprojβ, npw) KB projectors
                bphi = torch.einsum("jg,mg->mj", p.conj(), phik)  # ⟨β_j|φ_m⟩
                return phik + torch.einsum("mj,ij,ig->mg", bphi, q, p)

            jvp_out = torch.func.jvp(build, (system.positions,), (tang,))
            out.append(jvp_out[1])
        return out

    def dproj(self, hk, sph):
        """∂p rows for atom a's columns: −i(k+G)_α ⊙ p (zero elsewhere)."""
        kpga = sph.kpg[:, self.alpha].to(CDTYPE)
        dp = torch.zeros_like(hk.p)
        dp[self.s0:self.s1] = -1j * kpga[None, :] * hk.p[self.s0:self.s1]
        return dp

    def dh_ds_psi(self, isp: int, ik: int, w_extra=None, d_extra=None,
                  hub_extra=None):
        """(δH|ψ⟩, δS|ψ⟩) over the window at k. Bare perturbation by
        default (local δv_loc, D re-screening, moving projectors); pass
        w_extra (grid) and d_extra (full D matrix) to add the converged
        self-consistent potential change for total-δψ reconstruction.
        hub_extra (nprojU, nprojU, apply convention) adds the converged
        Dudarev δD_U for the +U total response; δS carries no +U."""
        cs = self.cs
        hk = cs.hks[isp][ik]
        c = cs.c_win[isp][ik]
        sph = cs.system.spheres[ik]
        b = cs.b_win[isp][ik]
        dp = self.dproj(hk, sph)
        db = becp(dp, c)
        dscr = hk.dscr
        qf = hk.q
        dv = self.dv_r[isp] if w_extra is None else self.dv_r[isp] + w_extra
        dd = self.d_dscr[isp] if d_extra is None else self.d_dscr[isp] + d_extra
        psi_r = g_to_r(c, sph.flat_idx, cs.shape)
        dh = box_to_sphere(r_to_g(psi_r * dv), sph.flat_idx)
        dh = dh + (b @ dd.to(CDTYPE)) @ hk.p
        dh = dh + (b @ dscr) @ dp + (db @ dscr) @ hk.p
        ds = (b @ qf) @ dp + (db @ qf) @ hk.p
        if cs.hub is not None:
            # V_U = |Sφ⟩D_U⟨Sφ|; atom a's motion moves Sφ (bare), and the
            # self-consistent δn feeds hub_extra = δD_U (total response).
            # self.dsphi is built in __init__ under the same `cs.hub is not
            # None` condition on the same cs, so it's real here too.
            assert self.dsphi is not None
            sphi = hk.hub_sphi
            dsphi = self.dsphi[ik]
            hub_d = hk.hub_d
            bh = becp(sphi, c)
            dh = dh + (becp(dsphi, c) @ hub_d) @ sphi + (bh @ hub_d) @ dsphi
            if hub_extra is not None:
                dh = dh + (bh @ hub_extra) @ sphi
        return dh, ds, dp, db

    def window_response(self, isp, ik, warm, cg_tol, cg_max_iter,
                        w_extra=None, d_extra=None, hub_extra=None):
        """Window S-metric perturbation theory + complement Sternheimer at
        one k. Returns (dpsi, hmat, smat, db): the occupied-band orbital
        response (window part + Sternheimer complement), the window matrices
        ⟨m|δH|n⟩ and ⟨m|δS|n⟩, and the moving-projector becp change.
        Updates warm[ik] with the complement solution (the warm start).

        ``w_extra``/``d_extra`` add the converged self-consistent potential
        change for total-δψ reconstruction (see ``dh_ds_psi``)."""
        cs = self.cs
        c = cs.c_win[isp][ik]
        ns = cs.n_solve[isp][ik]
        eps = cs.eps_win[isp][ik]
        dh, ds, dp, db = self.dh_ds_psi(isp, ik, w_extra=w_extra,
                                        d_extra=d_extra, hub_extra=hub_extra)

        # window coefficients c_mn (m any window state, n occupied)
        hmat = torch.einsum("mg,ng->mn", c.conj(), dh)
        smat = torch.einsum("mg,ng->mn", c.conj(), ds)
        de = (eps[None, :] - eps[:, None]).to(CDTYPE)  # ε_n − ε_m
        num = hmat - smat * eps[None, :].to(CDTYPE)
        safe = de.abs() > 1e-8
        de_safe = torch.where(safe, de, torch.ones_like(de))
        # Non-degenerate m ≠ n: c_mn = ⟨m|δH − ε_n δS|n⟩/(ε_n − ε_m). The
        # ε_n ≈ ε_m entries (diagonal m = n and degenerate off-diagonal
        # m ≠ n) divide by ~0, so they take the S-orthonormality gauge. The
        # m = n normalization is exactly −½⟨n|δS|n⟩.
        #
        # For an OFF-diagonal degenerate pair the correct value depends on
        # which invariant the response feeds — the two limits of c_mn/c_nm
        # differ because the anti-Hermitian half of num/de diverges as
        # ε_n → ε_m while its Hermitian half stays −½⟨m|δS|n⟩:
        #   * density / becsum / normalization see only the Hermitian half,
        #     so −½⟨m|δS|n⟩ is the continuous limit (self.deg_full=False);
        #   * the SECOND-derivative assembly in ``hessian_column`` pairs δψ
        #     against the (non-stationary) ∂E/∂c gradient, which picks up the
        #     diverging anti-Hermitian half through its ∂/∂τ′; its continuous
        #     limit is the FULL metric coupling −⟨m|δS|n⟩ (self.deg_full=True).
        # The −½ substitution used for both is right for the density but
        # introduces a spurious discontinuity in the Hessian at exact
        # degeneracy (a ~0.5 % high optical Γ-phonon frequency on diamond Si);
        # −⟨m|δS|n⟩ restores continuity with the ε_n≠ε_m column (issue #141).
        cmn = torch.where(safe, num / de_safe, -0.5 * smat)
        if self.deg_full:
            band = torch.arange(cmn.shape[0], device=cmn.device)
            offdiag = band[:, None] != band[None, :]
            cmn = torch.where((~safe) & offdiag, -smat, cmn)
        dpsi_win = cmn.mT @ c  # δψ_n(win) = Σ_m c_mn ψ_m, rows n

        # complement: (H − ε_n S) δψ⊥ = −P_c†(δH − ε_n δS)|ψ_n⟩, occ n
        rhs = dh[:ns] - eps[:ns, None].to(CDTYPE) * ds[:ns]
        rhs = -(rhs - (rhs @ c.conj().T) @ cs.s_win[isp][ik])
        dperp = cs._sternheimer_k(isp, ik, rhs, warm[ik], cg_tol, cg_max_iter)
        warm[ik] = dperp
        return dpsi_win[:ns] + dperp, hmat, smat, db

    def bare_map_derivative(self, dpsi_warm, cg_tol: float=1e-10, cg_max_iter: int=400):
        """∂F/∂τ_{aα} at fixed input x: per-spin (δρ_out list, δbec_out list of
        per-atom mats, δn_hub).

        Window part from S-metric perturbation theory, complement part from the
        projected Sternheimer solve, plus the explicit motion of the becsum
        projectors and the augmentation phases. Each spin channel responds
        through its own frozen operators (V^σ, D^σ, bands), exactly as
        _ConvergedUSPP.apply_chi0 threads them; the 2.0 prefactor on δρ is the
        ψ*δψ+δψ*ψ conjugate pair, NOT the spin degeneracy (that lives in f_win,
        =2 for nspin=1 and =1 per channel for nspin=2). ``dpsi_warm`` is per
        spin (``dpsi_warm[isp][ik]``). δn_hub is the bare occupation-matrix
        response (per hub channel; None without +U)."""
        cs = self.cs
        system, grid = cs.system, cs.grid
        dev = system.positions.device
        kw = system.kweights
        nsp = cs.nspin
        drho_sm = [torch.zeros(cs.shape, dtype=RDTYPE, device=dev)
                   for _ in range(nsp)]
        dbec = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                 for (s0, s1) in system.atom_slices] for _ in range(nsp)]
        # DFT+U: the bare position derivative of the occupation matrix n^I_pq
        # (one hub channel for nspin=1 at the SCF's half-occupancy weight; two
        # channels for nspin=2, one per spin, same as apply_chi0)
        dnh = None
        if cs.hub is not None:
            dnh = [[torch.zeros(st["dim"], st["dim"], dtype=CDTYPE, device=dev)
                    for st in cs.hub.sites] for _ in range(cs.nsh)]
        for isp in range(nsp):
            for ik, sph in enumerate(system.spheres):
                hk, c = cs.hks[isp][ik], cs.c_win[isp][ik]
                ns = cs.n_solve[isp][ik]
                f = cs.f_win[isp][ik]
                wk = float(kw[ik])
                dpsi, _hmat, _smat, db = self.window_response(
                    isp, ik, dpsi_warm[isp], cg_tol, cg_max_iter)

                fw = f[:ns]
                psi_r = g_to_r(c[:ns], sph.flat_idx, cs.shape)
                dpsi_r = g_to_r(dpsi, sph.flat_idx, cs.shape)
                drho_sm[isp] += 2.0 * wk * torch.einsum(
                    "b,bxyz->xyz", fw, (psi_r.conj() * dpsi_r).real)

                # becsum: orbital response + explicit projector motion
                b = cs.b_win[isp][ik]
                b_d = becp(hk.p, dpsi) + db[:ns]
                for at, (s0, s1) in enumerate(system.atom_slices):
                    bo, bd = b[:ns, s0:s1], b_d[:, s0:s1]
                    m1 = torch.einsum("b,bi,bj->ij", fw.to(CDTYPE),
                                      bd.conj(), bo)
                    dbec[isp][at] += wk * (m1 + m1.conj().T)

                # +U occupations: n_pq = Σ wf ⟨Sφ_p|ψ⟩⟨ψ|Sφ_q⟩; the derivative
                # gets the orbital response ⟨Sφ|δψ⟩ and the explicit Sφ motion
                # ⟨∂Sφ|ψ⟩. Sφ is spin-independent, so self.dsphi is shared.
                if cs.hub is not None:
                    # dnh is built above under the same `cs.hub is not None`
                    # (same cs), and self.dsphi likewise from __init__.
                    assert dnh is not None and self.dsphi is not None
                    ich = min(isp, cs.nsh - 1)
                    bh = cs.bh_win[isp][ik]  # ⟨Sφ|ψ⟩ over the window
                    dbh = becp(hk.hub_sphi, dpsi) + becp(self.dsphi[ik], c)[:ns]
                    for si, st in enumerate(cs.hub.sites):
                        s0, d = st["start"], st["dim"]
                        m1 = torch.einsum(
                            "b,bp,bq->pq", fw.to(CDTYPE),
                            dbh[:, s0:s0 + d], bh[:ns, s0:s0 + d].conj())
                        dnh[ich][si] += cs.hub_w * wk * (m1 + m1.conj().T)
        dbec = [[0.5 * (m + m.conj().T) for m in ch] for ch in dbec]
        if cs.hub is not None:
            assert dnh is not None
            dnh = [[0.5 * (m + m.conj().T) for m in ch] for ch in dnh]

        # augmentation density (per spin): response becsum at fixed phases (the
        # shared becsum→ρ_aug builder) plus the explicit phase motion of atom a
        # with this channel's map-output becsum
        drho_out = []
        g_a = system.g_sphere[:, self.alpha].to(CDTYPE)
        sp_a = system.species_of_atom[self.a]
        for isp in range(nsp):
            drho_aug = aug_density_from_becsum(system, dbec[isp], cs.phase_pos)
            aug_sph = (-1j * g_a) * cs.phase_pos[:, self.a].conj() \
                * torch.einsum("ij,ijg->g",
                               cs.rho_ij_sp[isp][self.a].to(CDTYPE),
                               system.aug[sp_a].q_g)
            aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
            aug_box[system.sphere_idx] = aug_sph / cs.vol
            drho_aug = drho_aug + g_to_r_box(aug_box.reshape(cs.shape),
                                             real=True)
            drho_out.append(drho_sm[isp] / cs.vol + drho_aug)
        return drho_out, dbec, dnh


def bare_position_derivative(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int,
                             cg_tol: float = 1e-10,
                             cg_max_iter: int = 400):
    """∂F_map/∂τ_{aα} at the converged state (fixed input x*). Returns
    (δρ_out, δbec_out per atom) for nspin=1; per-spin lists (δρ_out^σ,
    δbec_out^σ) for nspin=2. Insulators, nspin 1 or 2, ±U."""
    _check_supported(res)
    if res.get("nspin", 1) not in (1, 2):
        raise NotImplementedError("position response: nspin must be 1 or 2")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                 zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                for isp in range(cs.nspin)]
        drho, dbec, _dnh = pert.bare_map_derivative(warm, cg_tol, cg_max_iter)
    if cs.nspin == 1:
        return drho[0], dbec[0]
    return drho, dbec


def _self_consistent_response(cs, bare_rho, bare_bec, bare_nh=None, *,
                              beta: float=0.3, history: int=12, inner_tol: float=1e-9,
                              max_inner: int=80, cg_tol: float=1e-10, cg_max_iter: int=300,
                              verbose: bool=False):
    """δx = (1 − χ̃K)⁻¹ δx_bare — the forward fixed point of the Newton
    finisher with the bare position derivative as the source. ``bare_rho`` and
    ``bare_bec`` are per-spin (grid list, per-atom list); ``bare_nh`` is the
    per-channel hub list or None. Returns per-spin (δρ*, δbec*, w_total) where
    w_total = (K δx) = (w_sp, d_ddd, d_hub) is the self-consistent potential
    change (needed to rebuild δψ later).

    The composite vector and its packing follow the same per-spin
    _pack/_unpack layout newton_polish / uspp_implicit use: per-spin grid
    fields, per-spin per-atom becsum (real; Hermitian response), then the
    per-channel Re/Im hub tail. χ̃ is block-diagonal over spin except the
    shared-Fermi δμ (dropped here — insulators only); K keeps its cross-spin
    Hartree/f_xc blocks. With DFT+U K maps δn to the Dudarev
    δD_U = −(U−J)·herm(δn) through cs.k_hub, and w_total gains that δD_U."""
    system = cs.system
    shape, n_pts = tuple(cs.shape), cs.grid.n_points
    nsp = cs.nspin
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    has_hub = cs.hub is not None and bare_nh is not None
    nsh = cs.nsh if has_hub else 0
    hub_dims = [st["dim"] for st in cs.hub.sites] if has_hub else None

    def _pack_all(rho_sp, bec_sp, nh):
        return _pack([r.to(RDTYPE) for r in rho_sp],
                     [[m.real.to(RDTYPE) for m in ch] for ch in bec_sp],
                     nh if has_hub else None)

    def _unpack_all(vec):
        return _unpack(vec, shape, n_pts, nbec, nsp, hub_dims, nsh)

    r_vec = _pack_all(bare_rho, bare_bec, bare_nh)
    dpsi_warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                  zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                 for isp in range(nsp)]
    d = r_vec.clone()
    mixer = AndersonMixer(history, beta)
    # Bound before the loop purely so ty can see `gn` as always assigned in
    # the `else` clause below (max_inner is always >= 1 in practice, same
    # "runs >=1 iteration" pattern as scf/loop.py/uspp_loop.py) -- overwritten
    # every real iteration.
    gn = float("inf")
    for it in range(1, max_inner + 1):
        d_rho, d_bec, d_nh = _unpack_all(d)
        w_sp = cs.k_hxc_grid(d_rho)
        d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in ch]
                                  for ch in d_bec])
        d_hub = cs.k_hub(d_nh) if has_hub else None
        chi_rho, chi_bec, chi_nh = cs.apply_chi0(
            w_sp, d_ddd, dpsi_warm, cg_tol, cg_max_iter, d_hub_sp=d_hub)
        g_vec = r_vec + _pack_all(chi_rho, chi_bec, chi_nh)
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
    d_rho, d_bec, d_nh = _unpack_all(d)
    w_sp = cs.k_hxc_grid(d_rho)
    d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in ch]
                              for ch in d_bec])
    d_hub = cs.k_hub(d_nh) if has_hub else None
    return d_rho, d_bec, (w_sp, d_ddd, d_hub)


def position_density_response(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int, *,
                              beta: float = 0.3, history: int = 12,
                              inner_tol: float = 1e-9, max_inner: int = 80,
                              cg_tol: float = 1e-10, cg_max_iter: int = 300,
                              verbose: bool = False):
    """Self-consistent dρ*/dτ_{aα} and dbecsum*/dτ_{aα} at the converged
    SCF point (analytic — no SCF re-runs). Returns (δρ*, δbec* per atom,
    w_total) for nspin=1; per-spin lists (δρ*^σ, δbec*^σ, w_total) for
    nspin=2. Insulators, nspin 1 or 2, ±U."""
    _check_supported(res)
    if res.get("nspin", 1) not in (1, 2):
        raise NotImplementedError("position response: nspin must be 1 or 2")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                 zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                for isp in range(cs.nspin)]
        bare_rho, bare_bec, bare_nh = pert.bare_map_derivative(
            warm, cg_tol, cg_max_iter)
        d_rho, d_bec, w_tot = _self_consistent_response(
            cs, bare_rho, bare_bec, bare_nh, beta=beta, history=history,
            inner_tol=inner_tol, max_inner=max_inner, cg_tol=cg_tol,
            cg_max_iter=cg_max_iter, verbose=verbose)
    if cs.nspin == 1:
        return d_rho[0], d_bec[0], w_tot
    return d_rho, d_bec, w_tot


def _total_orbital_response(cs, pert, w_grid, d_ddd, d_hub=None, cg_tol: float=1e-10,
                            cg_max_iter: int=400):
    """Per-spin δψ_n, δε_n, and the per-spin SMOOTH density derivative at the
    TOTAL perturbation (bare position motion + converged self-consistent
    potential change). One window-PT + Sternheimer pass per (spin, k).
    ``w_grid`` and ``d_ddd`` are per-spin; ``d_hub`` (per hub channel, per site
    δD_U from k_hub) adds the converged Dudarev V_U change to the total
    perturbation."""
    system = cs.system
    nsp = cs.nspin
    d_extra_sp = []
    for isp in range(nsp):
        d_extra = cs._aug_dmat(w_grid[isp])
        for a, (s0, s1) in enumerate(system.atom_slices):
            d_extra[s0:s1, s0:s1] += d_ddd[isp][a].real
        d_extra_sp.append(d_extra)
    hub_extra_sp: list[torch.Tensor | None] = [None] * nsp
    if d_hub is not None:
        # assemble the block-diagonal δD_U in the apply convention (D^T), the
        # same one apply_chi0 uses for the hub V_U perturbation; one channel
        # per spin (min(isp, nsh-1) folds nspin=1's shared channel)
        for isp in range(nsp):
            ich = min(isp, cs.nsh - 1)
            hub_extra = torch.zeros(cs.hub.nproj, cs.hub.nproj, dtype=CDTYPE,
                                    device=system.positions.device)
            for m, st in zip(d_hub[ich], cs.hub.sites, strict=True):
                s0, d = st["start"], st["dim"]
                hub_extra[s0:s0 + d, s0:s0 + d] = 0.5 * (m + m.conj().T)
            hub_extra_sp[isp] = hub_extra.conj()
    dpsi_all: list[list[torch.Tensor]] = [[] for _ in range(nsp)]
    deps_all: list[list[torch.Tensor]] = [[] for _ in range(nsp)]
    drho_sm = [torch.zeros(cs.shape, dtype=RDTYPE, device=system.positions.device)
               for _ in range(nsp)]
    for isp in range(nsp):
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
        for ik, sph in enumerate(system.spheres):
            c = cs.c_win[isp][ik]
            ns = cs.n_solve[isp][ik]
            f, eps = cs.f_win[isp][ik], cs.eps_win[isp][ik]
            wk = float(system.kweights[ik])
            dpsi, hmat, smat, _db = pert.window_response(
                isp, ik, warm, cg_tol, cg_max_iter, w_extra=w_grid[isp],
                d_extra=d_extra_sp[isp], hub_extra=hub_extra_sp[isp])
            deps = (hmat.diagonal().real - eps * smat.diagonal().real)[:ns]
            dpsi_all[isp].append(dpsi)
            deps_all[isp].append(deps)
            fw = f[:ns]
            psi_r = g_to_r(c[:ns], sph.flat_idx, cs.shape)
            dpsi_r = g_to_r(dpsi, sph.flat_idx, cs.shape)
            drho_sm[isp] += 2.0 * wk * torch.einsum(
                "b,bxyz->xyz", fw, (psi_r.conj() * dpsi_r).real)
    return dpsi_all, deps_all, [d / cs.vol for d in drho_sm]


def hessian_column(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int, *,
                   response_kw=None, verbose: bool = False):
    """d²E/dτ dτ_{aα} — one analytic Hessian column (na, 3), no SCF
    re-runs. The τ-graph of the force expression is differentiated once
    more along the direction (e_{aα}, δstate/δτ_{aα}): the scalar
    s = dE'/dλ (one create_graph backward plus real pairings with the
    state response) has ∂s/∂τ equal to the full mixed column, explicit
    ∂²E/∂τ∂τ' included. Insulators, nspin 1 or 2, ±U.

    nspin=2: the force-energy graph is spin-resolved exactly as
    paw_forces.forces_uspp (per-spin coeffs/occ/eps/becsum/ddd leaves, SpinXC
    on the total density), and the δstate pairing that closes the mixed second
    derivative runs over both channels; the per-spin state response comes from
    the same _self_consistent_response / _total_orbital_response above. The
    Hessian column itself is a single (na, 3) — d²E_total is spin-summed.

    With DFT+U the force-energy graph gains the Dudarev E_U(c, τ) term (the
    same in-graph n(τ) expression forces_uspp differentiates for the +U
    force), so g_c and g_pos pick up ∂E_U/∂c and ∂E_U/∂τ; the +U-total
    orbital response (V_U feedback threaded through _self_consistent_response
    / _total_orbital_response) then makes ∂s/∂τ' the full +U mixed column."""
    from gradwave.core.density import sigma_from_rho
    from gradwave.core.energies.ewald import ewald_energy
    from gradwave.core.energies.hartree import hartree_energy
    from gradwave.core.energies.local_pp import local_energy
    from gradwave.core.energies.nl_pp import nonlocal_energy
    from gradwave.core.hamiltonian import projectors
    from gradwave.postscf._response import spin_sigma_triple
    from gradwave.postscf.paw_forces import (
        _aug_at_fixed,
        _aug_from_becsum,
        _normalize_spin,
        rho_core_on_graph,
    )

    _check_supported(res)
    if res.get("nspin", 1) not in (1, 2):
        raise NotImplementedError("hessian_column: nspin must be 1 or 2")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("hessian_column: insulators only")
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    kw = system.kweights
    rkw = dict(response_kw or {})
    rkw.setdefault("verbose", verbose)

    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        nsp = cs.nspin
        pert = PositionPerturbation(cs, a, alpha)
        # the mixed second derivative needs the FULL metric coupling in the
        # degenerate window off-diagonal (−⟨m|δS|n⟩), the continuous limit of
        # the non-degenerate coefficient; the −½ density limit would leave a
        # spurious discontinuity in the Hessian at a band degeneracy (#141).
        pert.deg_full = True
        warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                 zip(cs.c_win[isp], cs.n_solve[isp], strict=True)]
                for isp in range(nsp)]
        bare_rho, bare_bec, bare_nh = pert.bare_map_derivative(warm)
        d_rho, d_bec, (w_grid, d_ddd, d_hub) = _self_consistent_response(
            cs, bare_rho, bare_bec, bare_nh, **rkw)
        dpsi_all, deps_all, drho_sm = _total_orbital_response(
            cs, pert, w_grid, d_ddd, d_hub=d_hub)
        dbec_tot = d_bec  # per-spin self-consistent becsum response (Hermitian)
        # ddd response at the converged becsum response (per spin)
        dddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in ch]
                                 for ch in dbec_tot])

    # rebuild the force-energy graph with STATE LEAVES (per spin, mirroring
    # paw_forces.forces_uspp's spin loop plus the extra δstate pairing)
    nspin, coeffs_s, occ_s, eigs_s, becsum_s, rho_sp = _normalize_spin(res)
    pos = system.positions.detach().clone().requires_grad_(True)
    is_paw = any(p.is_paw for p in system.paws)

    ddd_leaves_s: list[list[torch.Tensor]] = [[] for _ in range(nsp)]
    if is_paw:
        from gradwave.scf.paw_onsite import onecenters

        onec = onecenters(system, xc,
                          device=system.positions.device)  # cached
        for at, sp in enumerate(system.species_of_atom):
            if nsp == 1:
                _, ddd = onec[sp].energy_and_ddd(becsum_s[0][at].detach())
                ddd_leaves_s[0].append(
                    ddd.detach().clone().requires_grad_(True))
            else:
                _, ddd = onec[sp].energy_and_ddd(
                    [becsum_s[0][at].detach(), becsum_s[1][at].detach()])
                for isp in range(2):
                    ddd_leaves_s[isp].append(
                        ddd[isp].detach().clone().requires_grad_(True))

    ns_k_s = [cs.n_solve[isp] for isp in range(nsp)]
    c_leaves_s = [[coeffs_s[isp][ik][:ns_k_s[isp][ik]].detach().clone()
                   .requires_grad_(True)
                   for ik in range(len(coeffs_s[isp]))]
                  for isp in range(nsp)]
    eps_leaves = [eigs_s[isp].detach().clone().requires_grad_(True)
                  for isp in range(nsp)]
    rho_s_leaves = []
    for isp in range(nsp):
        rho_s_fixed = (rho_sp[isp].detach()
                       - _aug_at_fixed(res, system, isp)).detach()
        rho_s_leaves.append(rho_s_fixed.clone().requires_grad_(True))

    projs = [projectors(pd, pos) for pd in system.proj_data]
    phase_arg = system.g_sphere @ pos.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg),
                                     phase_arg))
    q = system.q_full.to(CDTYPE)

    # DFT+U: E_U(c, τ) as the in-graph n(τ) expression (the same term
    # forces_uspp adds for the +U force). g_c/g_pos then carry ∂E_U/∂c and
    # ∂E_U/∂τ, so the mixed column picks up the full +U second derivative.
    hub_sites = res.get("hub_sites")
    hub_phi_free = None
    if hub_sites is not None:
        from gradwave.scf.uspp_hubbard import phi_free_per_k

        hub_phi_free = phi_free_per_k(system, hub_sites)

    e = ewald_energy(pos, system.charges, grid.cell)
    becps_full_s: list[list[torch.Tensor]] = [[] for _ in range(nsp)]
    rho_chans = []
    for isp in range(nsp):
        occ = occ_s[isp].detach()
        eps_leaf = eps_leaves[isp]
        ns_k = ns_k_s[isp]
        ns0 = ns_k[0]
        assert all(n == ns0 for n in ns_k), "insulator: uniform occupied count"
        rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=pos.device)
                  for (s0, s1) in system.atom_slices]
        for ik in range(len(c_leaves_s[isp])):
            nsk = ns_k[ik]
            b = becp(projs[ik], c_leaves_s[isp][ik])
            becps_full_s[isp].append(b)
            w = (kw[ik] * occ[ik][:nsk]).to(CDTYPE)
            for at, (s0, s1) in enumerate(system.atom_slices):
                ba = b[:, s0:s1]
                rho_ij[at] = rho_ij[at] + torch.einsum("b,bi,bj->ij", w,
                                                       ba.conj(), ba)
        rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]
        rho_aug = _aug_from_becsum(system, rho_ij, phases)
        rho_chans.append(rho_s_leaves[isp] + rho_aug)

        e = e + nonlocal_energy(becps_full_s[isp],
                                system.proj_data[0].dij_full,
                                occ[:, :ns0], kw)
        for ik, b in enumerate(becps_full_s[isp]):
            nsk = ns_k[ik]
            quad = torch.einsum("bi,ij,bj->b", b.conj(), q, b).real
            e = e - (kw[ik] * occ[ik][:nsk] * eps_leaf[ik][:nsk] * quad).sum()
        if is_paw:
            for at in range(len(system.atom_slices)):
                e = e + (ddd_leaves_s[isp][at].to(CDTYPE)
                         * rho_ij[at]).sum().real
        if hub_sites is not None:
            from gradwave.scf.uspp_hubbard import hubbard_e_channel

            mult = 2.0 if nsp == 1 else 1.0
            e = e + mult * hubbard_e_channel(
                hub_sites, hub_phi_free, system.q_full, pos, system.spheres,
                projs, c_leaves_s[isp], becps_full_s[isp], occ[:, :ns0], kw,
                occ_scale=(0.5 if nsp == 1 else 1.0))

    # Not `sum(rho_chans)`: the int `0` default start makes the inferred type
    # `Tensor | int` even though rho_chans always has nsp (>= 1) entries.
    rho_tot = rho_chans[0]
    for rc in rho_chans[1:]:
        rho_tot = rho_tot + rc
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    rho_core = rho_core_on_graph(system, phases)
    # grads below take create_graph=True (a second backward follows), so keep
    # this E_xc eager, compiled aot_autograd cannot double-backward.
    with xc_eager():
        if nsp == 1:
            # nspin=1 is always paired with a collinear XCFunctional (same
            # convention as scf/uspp_loop.py's own xc dispatch).
            assert isinstance(xc, XCFunctional)
            rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
            sigma = (sigma_from_rho(rho_xc, grid.g_cart)
                     if xc.needs_gradient else None)
            e = e + xc.energy(rho_xc, vol, sigma)
        else:
            assert isinstance(xc, SpinXC)
            c2 = 0.0 if rho_core is None else 0.5 * rho_core
            r_u, r_d = rho_chans[0] + c2, rho_chans[1] + c2
            s_uu, s_dd, s_tt = spin_sigma_triple(xc, r_u, r_d, grid.g_cart)
            e = e + xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)
    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64,
                                 device=pos.device)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)
    e = e + hartree_energy(rho_g, grid.g2, vol) + local_energy(rho_g,
                                                               vloc_g, vol)

    # flat leaf list: pos, per-spin rho_s, per-spin eps, per-spin-per-k c,
    # per-spin-per-atom ddd — re-nested per spin after the backward
    c_flat = [c for ch in c_leaves_s for c in ch]
    ddd_flat = [d for ch in ddd_leaves_s for d in ch]
    leaves = [pos, *rho_s_leaves, *eps_leaves, *c_flat, *ddd_flat]
    grads = torch.autograd.grad(e, leaves, create_graph=True)
    g_pos = grads[0]
    off = 1
    g_rho = grads[off:off + nsp]
    off += nsp
    g_eps = grads[off:off + nsp]
    off += nsp
    g_c_flat = grads[off:off + len(c_flat)]
    off += len(c_flat)
    g_ddd_flat = grads[off:off + len(ddd_flat)]
    g_c_s, i = [], 0
    for isp in range(nsp):
        n = len(c_leaves_s[isp])
        g_c_s.append(g_c_flat[i:i + n])
        i += n
    g_ddd_s, i = [], 0
    for isp in range(nsp):
        n = len(ddd_leaves_s[isp])
        g_ddd_s.append(g_ddd_flat[i:i + n])
        i += n

    # directional derivative de'/dλ along (e_{aα}, δstate); torch's
    # complex grads are conjugate-Wirtinger, so the real pairing for a
    # complex leaf is Re⟨g, δz⟩ summed over both Wirtinger halves —
    # (g.conj()*δz).real reproduces d/dt e(z + t δz)
    s = g_pos[a, alpha]
    for isp in range(nsp):
        s = s + (g_rho[isp] * drho_sm[isp]).sum()
        for ik in range(len(c_leaves_s[isp])):
            s = s + (g_c_s[isp][ik].conj() * dpsi_all[isp][ik]).real.sum()
        for ik in range(len(c_leaves_s[isp])):
            nsk = ns_k_s[isp][ik]
            s = s + (g_eps[isp][ik][:nsk] * deps_all[isp][ik]).sum()
        for at in range(len(ddd_leaves_s[isp])):
            s = s + (g_ddd_s[isp][at] * dddd[isp][at].real).sum()
    (col,) = torch.autograd.grad(s, pos)
    return col.detach()
