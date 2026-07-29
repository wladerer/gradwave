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
apply_chi0/k_hub machinery the density-loss adjoint already carries. Coverage:
nspin=1, ±U, insulators, Γ-phonon scope (q = 0).
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
from gradwave.postscf._response import fxc_hvp
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


def _fxc_apply(cs, w: torch.Tensor) -> torch.Tensor:
    """f_xc·w at the converged density — the XC half of k_hxc_grid without
    the Hartree kernel (the NLCC core carries no Hartree). Shared HVP from
    postscf._response."""
    return fxc_hvp(cs.xc, cs.rho_xc, cs.grid, w)


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
        dv = self.dv_r if w_extra is None else self.dv_r + w_extra
        dd = self.d_dscr if d_extra is None else self.d_dscr + d_extra
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
        """∂F/∂τ_{aα} at fixed input x: (δρ_out, δbec_out per atom, δn_hub).

        Window part from S-metric perturbation theory, complement part
        from the projected Sternheimer solve, plus the explicit motion of
        the becsum projectors and the augmentation phases. δn_hub is the
        bare occupation-matrix response (per hub channel; None without +U)."""
        cs = self.cs
        system, grid = cs.system, cs.grid
        dev = system.positions.device
        kw = system.kweights
        isp = 0  # nspin=1 scope
        drho_sm = torch.zeros(cs.shape, dtype=RDTYPE, device=dev)
        dbec = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                for (s0, s1) in system.atom_slices]
        # DFT+U: the bare position derivative of the occupation matrix n^I_pq
        # (one hub channel for nspin=1, at the SCF's half-occupancy weight)
        dnh = None
        if cs.hub is not None:
            dnh = [[torch.zeros(st["dim"], st["dim"], dtype=CDTYPE, device=dev)
                    for st in cs.hub.sites]]
        for ik, sph in enumerate(system.spheres):
            hk, c = cs.hks[isp][ik], cs.c_win[isp][ik]
            ns = cs.n_solve[isp][ik]
            f = cs.f_win[isp][ik]
            wk = float(kw[ik])
            dpsi, _hmat, _smat, db = self.window_response(
                isp, ik, dpsi_warm, cg_tol, cg_max_iter)

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

            # +U occupations: n_pq = Σ wf ⟨Sφ_p|ψ⟩⟨ψ|Sφ_q⟩; the derivative gets
            # the orbital response ⟨Sφ|δψ⟩ and the explicit Sφ motion ⟨∂Sφ|ψ⟩
            if cs.hub is not None:
                # dnh is built above under the same `cs.hub is not None`
                # (same cs), and self.dsphi likewise from __init__.
                assert dnh is not None and self.dsphi is not None
                bh = cs.bh_win[isp][ik]  # ⟨Sφ|ψ⟩ over the window
                dbh = becp(hk.hub_sphi, dpsi) + becp(self.dsphi[ik], c)[:ns]
                for si, st in enumerate(cs.hub.sites):
                    s0, d = st["start"], st["dim"]
                    m1 = torch.einsum("b,bp,bq->pq", fw.to(CDTYPE),
                                      dbh[:, s0:s0 + d], bh[:ns, s0:s0 + d].conj())
                    dnh[0][si] += cs.hub_w * wk * (m1 + m1.conj().T)
        dbec = [0.5 * (m + m.conj().T) for m in dbec]
        if cs.hub is not None:
            assert dnh is not None
            dnh = [[0.5 * (m + m.conj().T) for m in dnh[0]]]

        # augmentation density: response becsum at fixed phases (the shared
        # becsum→ρ_aug builder) plus the explicit phase motion of atom a with
        # the map-output becsum, which stays local to this perturbation
        drho_aug = aug_density_from_becsum(system, dbec, cs.phase_pos)
        g_a = system.g_sphere[:, self.alpha].to(CDTYPE)
        sp_a = system.species_of_atom[self.a]
        aug_sph = (-1j * g_a) * cs.phase_pos[:, self.a].conj() \
            * torch.einsum("ij,ijg->g",
                           cs.rho_ij_sp[isp][self.a].to(CDTYPE),
                           system.aug[sp_a].q_g)
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
        aug_box[system.sphere_idx] = aug_sph / cs.vol
        drho_aug = drho_aug + g_to_r_box(aug_box.reshape(cs.shape), real=True)
        return drho_sm / cs.vol + drho_aug, dbec, dnh


def bare_position_derivative(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int,
                             cg_tol: float = 1e-10,
                             cg_max_iter: int = 400):
    """∂F_map/∂τ_{aα} at the converged state (fixed input x*). Returns
    (δρ_out, δbec_out per atom). Insulators, nspin=1, ±U."""
    _check_supported(res)
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("position response: nspin=1 only")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        drho, dbec, _dnh = pert.bare_map_derivative(warm, cg_tol, cg_max_iter)
        return drho, dbec


def _self_consistent_response(cs, bare_rho, bare_bec, bare_nh=None, *,
                              beta: float=0.3, history: int=12, inner_tol: float=1e-9,
                              max_inner: int=80, cg_tol: float=1e-10, cg_max_iter: int=300,
                              verbose: bool=False):
    """δx = (1 − χ̃K)⁻¹ δx_bare — the forward fixed point of the Newton
    finisher with the bare position derivative as the source. Returns the
    self-consistent (δρ*, δbec*, w_total) where w_total = (K δx) is the
    self-consistent potential change (needed to rebuild δψ later).

    With DFT+U the composite state x gains the occupation-matrix channel δn
    (packed as Re/Im tail, one channel for nspin=1); K maps it to the Dudarev
    δD_U = −(U−J)·herm(δn) through cs.k_hub, and w_total gains that δD_U."""
    system = cs.system
    shape, n_pts = tuple(cs.shape), cs.grid.n_points
    nbec = [s1 - s0 for (s0, s1) in system.atom_slices]
    has_hub = cs.hub is not None and bare_nh is not None
    hub_dims = [st["dim"] for st in cs.hub.sites] if has_hub else []
    base_len = n_pts + sum(n * n for n in nbec)

    def _pack_all(rho, bec, nh):
        # newton.py's _pack/_unpack always operate on per-spin lists (even
        # nspin=1 wraps its single channel in a length-1 list -- see
        # newton_polish's `[res.rho.detach().clone()]`); this module is
        # nspin=1-only scope, so wrap/unwrap the single channel explicitly
        # rather than passing the bare Tensor/flat list `_pack` expects a
        # list-of/list-of-lists for (previously "worked" only by the
        # accidental equivalence of concatenated row-major reshapes to a
        # single whole-tensor reshape -- correct by luck, not by contract).
        v = _pack([rho.to(RDTYPE)], [[m.real.to(RDTYPE) for m in bec]])
        if not has_hub:
            return v
        tail = []
        for m in nh[0]:
            tail.append(m.real.reshape(-1).to(RDTYPE))
            tail.append(m.imag.reshape(-1).to(RDTYPE))
        return torch.cat([v] + tail)

    def _unpack_all(vec):
        d_rho_s, d_bec_s = _unpack(vec[:base_len], shape, n_pts, nbec, nspin=1)
        d_rho, d_bec = d_rho_s[0], d_bec_s[0]
        if not has_hub:
            return d_rho, d_bec, None
        nh, off = [], base_len
        for d in hub_dims:
            re = vec[off:off + d * d].reshape(d, d)
            off += d * d
            im = vec[off:off + d * d].reshape(d, d)
            off += d * d
            nh.append(torch.complex(re, im))
        return d_rho, d_bec, [nh]

    r_vec = _pack_all(bare_rho, bare_bec, bare_nh)
    dpsi_warm = [[torch.zeros_like(c[:n_sv]) for c, n_sv in
                  zip(cs.c_win[0], cs.n_solve[0], strict=True)]]
    d = r_vec.clone()
    mixer = AndersonMixer(history, beta)
    # Bound before the loop purely so ty can see `gn` as always assigned in
    # the `else` clause below (max_inner is always >= 1 in practice, same
    # "runs >=1 iteration" pattern as scf/loop.py/uspp_loop.py) -- overwritten
    # every real iteration.
    gn = float("inf")
    for it in range(1, max_inner + 1):
        d_rho, d_bec, d_nh = _unpack_all(d)
        w_sp = cs.k_hxc_grid([d_rho])
        d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in d_bec]])
        d_hub = cs.k_hub(d_nh) if has_hub else None
        chi_rho, chi_bec, chi_nh = cs.apply_chi0(
            w_sp, d_ddd, dpsi_warm, cg_tol, cg_max_iter, d_hub_sp=d_hub)
        g_vec = r_vec + _pack_all(chi_rho[0], chi_bec[0], chi_nh)
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
    w_sp = cs.k_hxc_grid([d_rho])
    d_ddd = cs.hvp_onecenter([[m.to(torch.complex128) for m in d_bec]])
    d_hub = cs.k_hub(d_nh) if has_hub else None
    return d_rho, d_bec, (w_sp[0], d_ddd[0], d_hub)


def position_density_response(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int, *,
                              beta: float = 0.3, history: int = 12,
                              inner_tol: float = 1e-9, max_inner: int = 80,
                              cg_tol: float = 1e-10, cg_max_iter: int = 300,
                              verbose: bool = False):
    """Self-consistent dρ*/dτ_{aα} and dbecsum*/dτ_{aα} at the converged
    SCF point (analytic — no SCF re-runs). Insulators, nspin=1, ±U."""
    _check_supported(res)
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("position response: nspin=1 only")
    if res.get("smearing", "none") != "none":
        raise NotImplementedError("position response: fixed occupations "
                                  "only (insulators)")
    with torch.no_grad():
        cs = _ConvergedUSPP(res, xc)
        pert = PositionPerturbation(cs, a, alpha)
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        bare_rho, bare_bec, bare_nh = pert.bare_map_derivative(
            warm, cg_tol, cg_max_iter)
        d_rho, d_bec, w_tot = _self_consistent_response(
            cs, bare_rho, bare_bec, bare_nh, beta=beta, history=history,
            inner_tol=inner_tol, max_inner=max_inner, cg_tol=cg_tol,
            cg_max_iter=cg_max_iter, verbose=verbose)
    return d_rho, d_bec, w_tot


def _total_orbital_response(cs, pert, w_grid, d_ddd, d_hub=None, cg_tol: float=1e-10,
                            cg_max_iter: int=400):
    """δψ_n, δε_n, and the SMOOTH density derivative at the TOTAL
    perturbation (bare position motion + converged self-consistent
    potential change). One window-PT + Sternheimer pass per k.

    d_hub (per hub channel, per site δD_U from k_hub) adds the converged
    Dudarev V_U change to the total perturbation."""
    system = cs.system
    isp = 0
    d_extra = cs._aug_dmat(w_grid)
    for a, (s0, s1) in enumerate(system.atom_slices):
        d_extra[s0:s1, s0:s1] += d_ddd[a].real
    hub_extra = None
    if d_hub is not None:
        # assemble the block-diagonal δD_U in the apply convention (D^T), the
        # same one apply_chi0 uses for the hub V_U perturbation
        hub_extra = torch.zeros(cs.hub.nproj, cs.hub.nproj, dtype=CDTYPE,
                                device=system.positions.device)
        for m, st in zip(d_hub[0], cs.hub.sites, strict=True):
            s0, d = st["start"], st["dim"]
            hub_extra[s0:s0 + d, s0:s0 + d] = 0.5 * (m + m.conj().T)
        hub_extra = hub_extra.conj()
    dpsi_all, deps_all = [], []
    drho_sm = torch.zeros(cs.shape, dtype=RDTYPE)
    warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
            zip(cs.c_win[0], cs.n_solve[0], strict=True)]
    for ik, sph in enumerate(system.spheres):
        c = cs.c_win[isp][ik]
        ns = cs.n_solve[isp][ik]
        f, eps = cs.f_win[isp][ik], cs.eps_win[isp][ik]
        wk = float(system.kweights[ik])
        dpsi, hmat, smat, _db = pert.window_response(
            isp, ik, warm, cg_tol, cg_max_iter, w_extra=w_grid,
            d_extra=d_extra, hub_extra=hub_extra)
        deps = (hmat.diagonal().real - eps * smat.diagonal().real)[:ns]
        dpsi_all.append(dpsi)
        deps_all.append(deps)
        fw = f[:ns]
        psi_r = g_to_r(c[:ns], sph.flat_idx, cs.shape)
        dpsi_r = g_to_r(dpsi, sph.flat_idx, cs.shape)
        drho_sm += 2.0 * wk * torch.einsum(
            "b,bxyz->xyz", fw, (psi_r.conj() * dpsi_r).real)
    return dpsi_all, deps_all, drho_sm / cs.vol


def hessian_column(res: USPPResult, xc: XCFunctional | SpinXC, a: int, alpha: int, *,
                   response_kw=None, verbose: bool = False):
    """d²E/dτ dτ_{aα} — one analytic Hessian column (na, 3), no SCF
    re-runs. The τ-graph of the force expression is differentiated once
    more along the direction (e_{aα}, δstate/δτ_{aα}): the scalar
    s = dE'/dλ (one create_graph backward plus real pairings with the
    state response) has ∂s/∂τ equal to the full mixed column, explicit
    ∂²E/∂τ∂τ' included. Insulators, nspin=1, ±U.

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
    from gradwave.postscf.paw_forces import (
        _aug_at_fixed,
        _aug_from_becsum,
        rho_core_on_graph,
    )

    _check_supported(res)
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("hessian_column: nspin=1 only")
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
        pert = PositionPerturbation(cs, a, alpha)
        # the mixed second derivative needs the FULL metric coupling in the
        # degenerate window off-diagonal (−⟨m|δS|n⟩), the continuous limit of
        # the non-degenerate coefficient; the −½ density limit would leave a
        # spurious discontinuity in the Hessian at a band degeneracy (#141).
        pert.deg_full = True
        warm = [torch.zeros_like(c[:n_sv]) for c, n_sv in
                zip(cs.c_win[0], cs.n_solve[0], strict=True)]
        bare_rho, bare_bec, bare_nh = pert.bare_map_derivative(warm)
        d_rho, d_bec, (w_grid, d_ddd, d_hub) = _self_consistent_response(
            cs, bare_rho, bare_bec, bare_nh, **rkw)
        dpsi_all, deps_all, drho_sm = _total_orbital_response(
            cs, pert, w_grid, d_ddd, d_hub=d_hub)
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
        from gradwave.scf.paw_onsite import onecenters

        onec = onecenters(system, xc,
                          device=system.positions.device)  # cached
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
    # DFT+U: E_U(c, τ) as the in-graph n(τ) expression (the same term
    # forces_uspp adds for the +U force). g_c/g_pos then carry ∂E_U/∂c and
    # ∂E_U/∂τ, so the mixed column picks up the full +U second derivative.
    hub_sites = res.get("hub_sites")
    if hub_sites is not None:
        from gradwave.scf.uspp_hubbard import hubbard_e_channel, phi_free_per_k

        hub_phi_free = phi_free_per_k(system, hub_sites)
        e = e + 2.0 * hubbard_e_channel(
            hub_sites, hub_phi_free, system.q_full, pos, system.spheres,
            projs, c_leaves, becps_full, occ[:, :ns0], kw, occ_scale=0.5)
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    rho_core = rho_core_on_graph(system, phases)
    rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    # hessian_column asserts nspin=1 above, so xc is always the collinear
    # functional here (the SpinXC/nspin=2 pairing is enforced by this
    # module's own callers, same convention as scf/uspp_loop.py's dispatch).
    assert isinstance(xc, XCFunctional)
    # grads below take create_graph=True (a second backward follows), so keep
    # this E_xc eager, compiled aot_autograd cannot double-backward.
    with xc_eager():
        e = e + xc.energy(rho_xc, vol, sigma)
    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64,
                                 device=pos.device)
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
