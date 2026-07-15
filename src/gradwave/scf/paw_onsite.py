"""PAW one-center corrections (port of QE's PW/src/paw_onecenter.f90).

Per PAW atom with occupation matrix ρ_ij (m-expanded becsum):

    E_1c = Σ_{AE,PS} sgn · ( E_H[ρ_LM] + E_xc[ρ_LM, ρ_core] ),  sgn = +1 AE, −1 PS
    ρ_LM(r) = Σ_ij ρ_ij c^{LM}_{lm_i,lm_j} · pfunc_ij(r)        (r² included)
        AE: pfunc = u_i u_j;  PS: ũ_i ũ_j + q^L_ij(r)           (both zero past r_aug)
    ddd_ij = ∂E_1c/∂ρ_ij  (EXACT for the quadrature actually summed)

ddd feeds back into the SCF Hamiltonian (added to the screened D in
scf/uspp.py); the energy is added to the plane-wave total. Radial Hartree per
LM channel is the multipole integral solution (its ddd is the exact
derivative — self-adjoint). XC is evaluated pointwise on a QE-convention
angular grid (Gauss–Legendre × uniform φ, lmax = 3·l_max_rho plus 2 extra
l's, matching QE's grid) with gradients in spherical components (∂_r via
QE's 3-point nonuniform stencil, angular via ∂θ/∂φ tables of the real
harmonics); the XC ddd comes from autograd through that quadrature
(_xc_exact), NOT from QE's divergence-form v_xc — integration by parts on
the lm-truncated expansion makes ddd ≠ ∂E/∂ρ_ij at the 0.05–1% level, which
broke force↔energy consistency at 1e-2 eV/Å on spin O₂ (QE carries the same
inconsistency; energies are unaffected). All float64 per atom type —
setup-layer speed is irrelevant (meshes ~1100 × ~120 directions).

Conventions verified by unit tests: radial Poisson vs an analytic Gaussian
shell, QE's printed one-center energy, and ddd == FD(E_1c) in becsum space.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.gaunt import real_gaunt_table, ylm_np
from gradwave.pseudo.radial_torch import simpson_weights
from gradwave.pseudo.upf_paw import PAWData

# QE paw_variables constants (PAW_rad_init arguments)
LM_FACT = 3  # max l for the non-GGA angular grid: LM_FACT · lmax_rho
LM_FACT_X = 3  # same, GGA
XLM = 2  # extra projection l's for the GGA divergence


def _cumint(g: np.ndarray) -> np.ndarray:
    """Cumulative ∫ g di in the index variable, O(h⁴) (Simpson pairs +
    5/8/−1 rule for odd endpoints). g (..., n) → same shape."""
    n = g.shape[-1]
    out = np.zeros_like(g)
    pair = (g[..., 0:-2:2] + 4.0 * g[..., 1:-1:2] + g[..., 2::2]) / 3.0
    even = np.cumsum(pair, axis=-1)
    out[..., 2::2] = even
    # odd points: I[2m+1] = I[2m] + (5 g_2m + 8 g_{2m+1} − g_{2m+2})/12
    npairs = pair.shape[-1]
    odd = out[..., 0:2 * npairs:2] + (
        5.0 * g[..., 0:-2:2] + 8.0 * g[..., 1:-1:2] - g[..., 2::2]
    ) / 12.0
    out[..., 1:-1:2] = odd
    if n % 2 == 0:  # final even-count point: trapezoid closure
        out[..., -1] = out[..., -2] + 0.5 * (g[..., -2] + g[..., -1])
    return out


def _cumint_t(g: torch.Tensor) -> torch.Tensor:
    """Torch (differentiable) twin of _cumint — identical stencil, built
    functionally (stack/cat, no in-place) so autograd can traverse it."""
    n = g.shape[-1]
    pair = (g[..., 0:-2:2] + 4.0 * g[..., 1:-1:2] + g[..., 2::2]) / 3.0
    even = torch.cumsum(pair, dim=-1)  # I at indices 2, 4, …, 2·npairs
    even_prev = torch.cat([torch.zeros_like(even[..., :1]), even[..., :-1]], dim=-1)
    odd = even_prev + (
        5.0 * g[..., 0:-2:2] + 8.0 * g[..., 1:-1:2] - g[..., 2::2]
    ) / 12.0  # I at indices 1, 3, …, 2·npairs−1
    core = torch.stack([odd, even], dim=-1).reshape(*even.shape[:-1], -1)
    out = torch.cat([torch.zeros_like(g[..., :1]), core], dim=-1)
    if n % 2 == 0:  # final even-count point: trapezoid closure
        tail = out[..., -1:] + 0.5 * (g[..., -2:-1] + g[..., -1:])
        out = torch.cat([out, tail], dim=-1)
    return out


class OneCenter:
    """Per-species one-center integrator. Call energy_and_ddd per atom."""

    def __init__(self, paw: PAWData, xc):
        self.paw = paw
        self.xc = xc
        self.lmax_beta = max(b.l for b in paw.betas)
        self.lmax_rho = 2 * self.lmax_beta
        self.l2 = (self.lmax_rho + 1) ** 2
        r, rab = paw.r, paw.rab
        self.r = r
        self.rab = rab
        self.mesh = len(r)
        self.w_simp = simpson_weights(rab)  # full-mesh quadrature weights
        self.kkbeta = max(b.cutoff_idx for b in paw.betas)

        # angular grid, QE PAW_rad_init conventions
        gga = getattr(xc, "needs_gradient", False)
        if self.lmax_rho == 0:
            lmax, ladd = 0, 0
        elif gga:
            lmax, ladd = LM_FACT_X * self.lmax_rho, XLM
        else:
            lmax, ladd = LM_FACT * self.lmax_rho, 0
        rad_lmax = lmax + ladd
        self.ladd = ladd
        nth = (rad_lmax + 2) // 2
        nphi = rad_lmax + 1 + (rad_lmax % 2)
        from scipy.special import roots_legendre

        z, wz = roots_legendre(nth)
        phi = np.arange(nphi) * (2.0 * math.pi / nphi)
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        st = np.sqrt(1.0 - zz**2)
        self.dirs = np.stack(
            [st * np.cos(pp), st * np.sin(pp), zz], axis=-1
        ).reshape(-1, 3)
        self.ww = (wz[:, None] * np.full(nphi, 2.0 * math.pi / nphi)).reshape(-1)
        self.nx = len(self.ww)
        self.cos_th = self.dirs[:, 2].copy()
        self.sin_th = np.sqrt(1.0 - self.cos_th**2)

        lm_max = (rad_lmax + 1) ** 2
        self.lm_max = lm_max
        self.ylm = ylm_np(rad_lmax, self.dirs)  # (nx, lm_max)
        # ∂Y/∂θ and (1/sinθ)∂Y/∂φ by central FD in the angles (1e-5 → ~1e-10)
        h = 1e-5
        th = np.arccos(self.cos_th)
        ph = np.arctan2(self.dirs[:, 1], self.dirs[:, 0])

        def dirs_of(th_, ph_):
            return np.stack(
                [np.sin(th_) * np.cos(ph_), np.sin(th_) * np.sin(ph_), np.cos(th_)],
                axis=-1,
            )

        self.dylmt = (
            ylm_np(rad_lmax, dirs_of(th + h, ph)) - ylm_np(rad_lmax, dirs_of(th - h, ph))
        ) / (2 * h)
        self.dylmp = (
            ylm_np(rad_lmax, dirs_of(th, ph + h)) - ylm_np(rad_lmax, dirs_of(th, ph - h))
        ) / (2 * h) / self.sin_th[:, None]

        # Gaunt table restricted to the density expansion L ≤ lmax_rho
        self.gaunt = real_gaunt_table(self.lmax_beta)  # (l2, nlm_b, nlm_b)

        # m-expanded index map + per-pair radial functions (r² included),
        # zeroed beyond the augmentation sphere like QE's pfunc/ptfunc
        self.idx = []
        for i, b in enumerate(paw.betas):
            for m in range(2 * b.l + 1):
                self.idx.append((i, b.l, b.l * b.l + m))
        self.nm = len(self.idx)
        ircut = paw.aug_cutoff_idx
        self.ircut = ircut
        nb = paw.n_proj

        def cut(f):
            out = np.zeros(self.mesh)
            out[:ircut] = f[:ircut]
            return out

        self.pfunc_ae = {}
        self.pfunc_ps = {}  # (i, j) channel pairs; PS is per-L (aug added)
        for i in range(nb):
            for j in range(nb):
                key = (min(i, j), max(i, j))
                if key in self.pfunc_ae:
                    continue
                self.pfunc_ae[key] = cut(paw.aewfc[key[0]].rphi * paw.aewfc[key[1]].rphi)
                base = cut(paw.pswfc[key[0]].rphi * paw.pswfc[key[1]].rphi)
                per_l = {}
                li, lj = paw.betas[key[0]].l, paw.betas[key[1]].l
                for ll in range(abs(li - lj), li + lj + 1):
                    q = paw.qijl.get((key[0], key[1], ll))
                    f = base.copy()
                    if q is not None:
                        f[: len(q)] += q
                    per_l[ll] = f
                self.pfunc_ps[key] = (base, per_l)

        self.core_ae = paw.ae_core_rho if paw.ae_core_rho is not None else np.zeros(self.mesh)
        self.core_ps = paw.core_rho if paw.core_rho is not None else np.zeros(self.mesh)

    # ---------- building blocks ----------

    def rho_lm(self, rho_ij: np.ndarray, what: str) -> np.ndarray:
        """(mesh, l2) lm-expanded one-center density (r² included)."""
        out = np.zeros((self.mesh, self.l2))
        for a, (i, _li, lmi) in enumerate(self.idx):
            for b, (j, _lj, lmj) in enumerate(self.idx):
                w = rho_ij[a, b]
                if abs(w) < 1e-14:
                    continue
                key = (min(i, j), max(i, j))
                cy = self.gaunt[:, lmi, lmj]  # (l2,)
                nz = np.nonzero(np.abs(cy) > 1e-12)[0]
                for lm in nz:
                    ll = int(math.isqrt(lm))
                    if what == "ae":
                        f = self.pfunc_ae[key]
                    else:
                        base, per_l = self.pfunc_ps[key]
                        f = per_l.get(ll, base)
                    out[:, lm] += w * cy[lm] * f
        return out

    def hartree(self, rho_lm: np.ndarray):
        """(v_lm (mesh, l2) [eV], E_H [eV]) — multipole radial Poisson."""
        v = np.zeros_like(rho_lm)
        r = self.r
        for lm in range(self.l2):
            ll = int(math.isqrt(lm))
            f = rho_lm[:, lm]
            if not np.any(f):
                continue
            a_in = _cumint(f * r**ll * self.rab)
            b_all = _cumint(f * np.where(r > 0, r ** -(ll + 1), 0.0) * self.rab)
            b_out = b_all[-1] - b_all
            pref = 4.0 * math.pi * E2 / (2 * ll + 1)
            rl = np.where(r > 0, r ** -(ll + 1), 0.0)
            v[:, lm] = pref * (rl * a_in + r**ll * b_out)
        e = 0.5 * float((self.w_simp[:, None] * v * rho_lm).sum())
        return v, e

    def _lm2rad(self, f_lm: np.ndarray) -> np.ndarray:
        """(mesh, l2) → (mesh, nx) values along directions."""
        return f_lm @ self.ylm[:, : f_lm.shape[1]].T

    def _rad2lm(self, f_rad: np.ndarray, nlm: int) -> np.ndarray:
        """(mesh, nx) → (mesh, nlm) projection with the angular weights."""
        return f_rad @ (self.ww[:, None] * self.ylm[:, :nlm])

    # ---------- exact XC (autograd through the quadrature) ----------

    def _torch_tables(self):
        if not hasattr(self, "_tt"):
            def t(a):
                return torch.as_tensor(np.ascontiguousarray(a),
                                       dtype=torch.float64)

            r = self.r
            lls = np.arange(self.lmax_rho + 1)[:, None]  # (nl, 1)
            rl_neg = np.where(r > 0, r[None, :] ** -(lls + 1), 0.0)
            self._tt = dict(
                ylm=t(self.ylm[:, : self.l2]),
                dylmt=t(self.dylmt[:, : self.l2]),
                dylmp=t(self.dylmp[:, : self.l2]),
                rm2=t(np.where(r > 0, r**-2.0, 0.0)),
                rm3=t(np.where(r > 0, r**-3.0, 0.0)),
                wq=t(self.w_simp * r**2),  # radial quadrature incl. r²
                ww=t(self.ww),
                rp=t(r[2:] - r[1:-1]),
                rm_=t(r[:-2] - r[1:-1]),
                c0=(r[0] - r[1]) / (r[2] - r[1]),
                core_ae=t(self.core_ae),
                core_ps=t(self.core_ps),
                # radial-Poisson tables per l: integrands and prefactor powers
                h_a=t(r[None, :] ** lls * self.rab[None, :]),
                h_b=t(rl_neg * self.rab[None, :]),
                h_vn=t(rl_neg),
                h_vp=t(r[None, :] ** lls),
                wsimp=t(self.w_simp),
            )
        return self._tt

    # ---------- dense linear maps ρ_ij → ρ_lm (torch chain) ----------

    def _rho_lm_maps(self):
        """T_what (nf·l2, nm²) with rho_lm_t(ρ)[:nf] = (T @ ρ.flat).reshape —
        the EXACT linearization of rho_lm (same Gaunt/pfunc/per-L aug
        selection, same 1e-12 Gaunt threshold)."""
        if hasattr(self, "_T"):
            return self._T
        # actual radial support of every pair function used (aug q^L tails
        # may outrun aug_cutoff_idx on some files — scan, don't assume)
        nf = self.ircut
        for key in self.pfunc_ae:
            base, per_l = self.pfunc_ps[key]
            for f in [self.pfunc_ae[key], base, *per_l.values()]:
                nz = np.nonzero(f)[0]
                if len(nz):
                    nf = max(nf, int(nz[-1]) + 1)
        self._nf = nf
        T = {}
        for what in ("ae", "ps"):
            mat = np.zeros((nf, self.l2, self.nm, self.nm))
            for a, (i, _li, lmi) in enumerate(self.idx):
                for b, (j, _lj, lmj) in enumerate(self.idx):
                    key = (min(i, j), max(i, j))
                    cy = self.gaunt[:, lmi, lmj]
                    nz = np.nonzero(np.abs(cy) > 1e-12)[0]
                    for lm in nz:
                        ll = int(math.isqrt(lm))
                        if what == "ae":
                            f = self.pfunc_ae[key]
                        else:
                            base, per_l = self.pfunc_ps[key]
                            f = per_l.get(ll, base)
                        mat[:, lm, a, b] += cy[lm] * f[:nf]
            T[what] = torch.as_tensor(
                mat.reshape(nf * self.l2, self.nm * self.nm))
        self._T = T
        return T

    def rho_lm_t(self, rho_ij: torch.Tensor, what: str) -> torch.Tensor:
        """(mesh, l2) torch ρ_lm, differentiable in rho_ij (real (nm, nm))."""
        T = self._rho_lm_maps()[what]
        cut = (T @ rho_ij.reshape(-1)).reshape(self._nf, self.l2)
        pad = torch.zeros(self.mesh - self._nf, self.l2, dtype=cut.dtype)
        return torch.cat([cut, pad], dim=0)

    def hartree_t(self, rho_lm: torch.Tensor):
        """Torch twin of hartree(): (v_lm (mesh, l2) [eV], E_H [eV])."""
        tt = self._torch_tables()
        cols = []
        for lm in range(self.l2):
            ll = int(math.isqrt(lm))
            f = rho_lm[:, lm]
            a_in = _cumint_t(f * tt["h_a"][ll])
            b_all = _cumint_t(f * tt["h_b"][ll])
            b_out = b_all[-1] - b_all
            pref = 4.0 * math.pi * E2 / (2 * ll + 1)
            cols.append(pref * (tt["h_vn"][ll] * a_in + tt["h_vp"][ll] * b_out))
        v = torch.stack(cols, dim=1)
        e = 0.5 * (tt["wsimp"][:, None] * v * rho_lm).sum()
        return v, e

    def _rgrad_t(self, f: torch.Tensor, tt) -> torch.Tensor:
        """QE radial_gradient iflag=0 stencil, torch, f (mesh, nx)."""
        rp = tt["rp"][:, None]
        rm = tt["rm_"][:, None]
        mid = (rp**2 * (f[:-2] - f[1:-1]) - rm**2 * (f[2:] - f[1:-1])) / (
            rp * rm * (rp - rm))
        g0 = mid[0] + (mid[1] - mid[0]) * tt["c0"]
        return torch.cat([g0[None], mid, torch.zeros_like(f[:1])], dim=0)

    def _xc_exact(self, rho_lms: list, what: str):
        """(E_xc [eV], [dE_xc/dρ_lm (mesh, l2) numpy] per spin) with the
        derivative EXACT for the quadrature actually summed — autograd through
        density and gradient chains, no integration by parts. (The divergence
        form of v_xc is only δE/δρ up to lm-truncation error; using it as ddd
        broke force↔energy consistency at 1e-2 eV/Å on spin O₂.)"""
        with torch.enable_grad():  # callable under a no_grad SCF driver
            leaves = [torch.as_tensor(rl_np, dtype=torch.float64).requires_grad_(True)
                      for rl_np in rho_lms]
            e_xc = self._exc_t(leaves, what)
            gs = torch.autograd.grad(e_xc, leaves)
        return float(e_xc.detach()), [g.numpy() for g in gs]

    def _exc_t(self, rls: list, what: str) -> torch.Tensor:
        """E_xc [eV] as a torch scalar from (mesh, l2) torch ρ_lm tensors —
        the quadrature body shared by _xc_exact, e1c_t and energy_theta."""
        tt = self._torch_tables()
        spin = len(rls) == 2
        core = tt["core_ae"] if what == "ae" else tt["core_ps"]
        cfrac = 0.5 if spin else 1.0
        gga = getattr(self.xc, "needs_gradient", False)
        dens, grads = [], []
        for rl in rls:
            rho_rad = rl @ tt["ylm"].T  # (mesh, nx), r² included
            dens.append(rho_rad * tt["rm2"][:, None] + cfrac * core[:, None])
            if gga:
                dr = self._rgrad_t(dens[-1], tt)
                gth = (rl @ tt["dylmt"].T) * tt["rm3"][:, None]
                gph = (rl @ tt["dylmp"].T) * tt["rm3"][:, None]
                grads.append(torch.stack([dr, gth, gph]))
        if spin:
            if gga:
                g_tot = grads[0] + grads[1]
                e = self.xc.energy_density(
                    dens[0].reshape(-1), dens[1].reshape(-1),
                    (grads[0] ** 2).sum(0).reshape(-1),
                    (grads[1] ** 2).sum(0).reshape(-1),
                    (g_tot**2).sum(0).reshape(-1))
            else:
                e = self.xc.energy_density(dens[0].reshape(-1),
                                           dens[1].reshape(-1))
        else:
            sig = (grads[0] ** 2).sum(0).reshape(-1) if gga else None
            e = self.xc.energy_density(dens[0].reshape(-1), sig)
        return (e.reshape(self.mesh, self.nx) * tt["wq"][:, None]
                * tt["ww"][None, :]).sum()

    # ---------- fully in-graph one-center chain (torch) ----------

    def e1c_t(self, rho_ijs: list) -> torch.Tensor:
        """E_1c [eV] as a torch scalar, fully differentiable in the REAL
        (nm, nm) rho_ij tensors AND the XC-functional parameters: dense-T
        ρ_lm, torch radial Poisson, the exact angular XC quadrature."""
        e_tot = torch.zeros((), dtype=torch.float64)
        for what, sgn in (("ae", 1.0), ("ps", -1.0)):
            rls = [self.rho_lm_t(r, what) for r in rho_ijs]
            _, e_h = self.hartree_t(sum(rls))
            e_tot = e_tot + sgn * (e_h + self._exc_t(rls, what))
        return e_tot

    @staticmethod
    def _to_real_t(rho_ij) -> torch.Tensor:
        m = rho_ij.detach()
        if m.is_complex():
            m = m.real
        return m.cpu().to(torch.float64)

    def hvp_becsum(self, rho_ij, vec):
        """One-center Hessian-vector product ∂²E_1c/∂ρ_ij² · vec.

        rho_ij, vec: (nm, nm) tensors (nspin=1) or 2-lists (spin); returns
        the matching structure of real float64 (nm, nm) tensors. For many
        HVPs at one fixed rho_ij (the adjoint outer loop), hvp_factory
        amortizes the first-order graph."""
        return self.hvp_factory(rho_ij)(vec)

    def hvp_factory(self, rho_ij):
        """Reusable HVP at fixed rho_ij. Builds E_1c's first-order graph
        (forward + create_graph backward) once; every call is then a
        single retained second backward. The adjoint solver evaluates the
        HVP at the SAME converged becsum every outer iteration, and the
        factory result is bit-identical to the one-shot path (measured:
        870 → 524 ms per call on Ni kjpaw spin, 1.66×)."""
        spin = isinstance(rho_ij, (list, tuple))
        rhos = ([self._to_real_t(m) for m in rho_ij] if spin
                else [self._to_real_t(rho_ij)])
        with torch.enable_grad():
            leaves = [m.clone().requires_grad_(True) for m in rhos]
            e = self.e1c_t(leaves)
            gs = torch.autograd.grad(e, leaves, create_graph=True)

        def hvp(vec):
            vecs = ([self._to_real_t(m) for m in vec] if spin
                    else [self._to_real_t(vec)])
            with torch.enable_grad():
                inner = sum((g * v).sum()
                            for g, v in zip(gs, vecs, strict=True))
                hv = torch.autograd.grad(inner, leaves, retain_graph=True)
            hv = [h.detach() for h in hv]
            return hv if spin else hv[0]

        return hvp

    def energy_theta(self, rho_ij) -> torch.Tensor:
        """E_1c as a torch scalar with the XC-functional parameters ON THE
        GRAPH (densities fixed and detached) — the one-center piece of
        dE_total/dθ by stationarity. Hartree is θ-independent."""
        spin = isinstance(rho_ij, (list, tuple))
        rhos = [self._to_real_t(m) for m in (rho_ij if spin else [rho_ij])]
        with torch.enable_grad():
            return self.e1c_t(rhos)

    # ---------- per-atom driver ----------

    def energy_and_ddd(self, rho_ij):
        """One-center energy [eV] and ddd [eV] for one atom.

        rho_ij: (nm, nm) tensor (nspin=1) → returns (E, ddd);
        or [ρ↑_ij, ρ↓_ij] → returns (E, [ddd↑, ddd↓]).

        ddd = autograd through the full torch chain (dense-T ρ_lm → torch
        radial Poisson → angular XC quadrature), so it is the derivative of
        the energy actually returned, bit-for-bit consistent with e1c_t. (The
        earlier hand-assembled Hartree term truncated its quadrature at
        kkbeta while the pair functions run to aug_cutoff_idx > kkbeta on psl
        meshes — a 3e-7 relative inconsistency the energy-space FD exposes.)
        """
        spin = isinstance(rho_ij, (list, tuple))
        rhos = [self._to_real_t(m) for m in (rho_ij if spin else [rho_ij])]
        with torch.enable_grad():
            leaves = [m.clone().requires_grad_(True) for m in rhos]
            e = self.e1c_t(leaves)
            gs = torch.autograd.grad(e, leaves)
        e_tot = float(e.detach())
        ddds = [g.detach() for g in gs]
        return (e_tot, ddds) if spin else (e_tot, ddds[0])
