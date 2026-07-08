"""PAW one-center corrections (port of QE's PW/src/paw_onecenter.f90).

Per PAW atom with occupation matrix ρ_ij (m-expanded becsum):

    E_1c = Σ_{AE,PS} sgn · ( E_H[ρ_LM] + E_xc[ρ_LM, ρ_core] ),  sgn = +1 AE, −1 PS
    ρ_LM(r) = Σ_ij ρ_ij c^{LM}_{lm_i,lm_j} · pfunc_ij(r)        (r² included)
        AE: pfunc = u_i u_j;  PS: ũ_i ũ_j + q^L_ij(r)           (both zero past r_aug)
    ddd_ij = ∂E_1c/∂ρ_ij = Σ sgn Σ_LM c^{LM}_ij ∫ pfunc^L_ij (v_H + v_xc)_LM dr

ddd feeds back into the SCF Hamiltonian (added to the screened D in
scf/uspp.py); the energy is added to the plane-wave total. Radial Hartree per
LM channel is the multipole integral solution; XC is evaluated pointwise on a
QE-convention angular grid (Gauss–Legendre × uniform φ, lmax = 3·l_max_rho
plus 2 extra projection l's for GGA) with gradients assembled in spherical
components (∂_r via QE's 3-point nonuniform stencil, angular via ∂θ/∂φ tables
of the real harmonics). All in numpy/torch float64 per atom type — setup-layer
speed is irrelevant (meshes ~1100 × ~120 directions).

Conventions verified by unit tests: radial Poisson vs an analytic Gaussian
shell, ∇ρ and the divergence identity ∫div F = surface term on synthetic
band-limited fields, and (the real gate) QE's printed one-center energy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.gaunt import real_gaunt_table, ylm_np
from gradwave.pseudo.radial import simpson
from gradwave.pseudo.radial_torch import simpson_weights
from gradwave.pseudo.upf_paw import PAWData

# QE paw_variables constants (PAW_rad_init arguments)
LM_FACT = 3  # max l for the non-GGA angular grid: LM_FACT · lmax_rho
LM_FACT_X = 3  # same, GGA
XLM = 2  # extra projection l's for the GGA divergence


def _radial_gradient(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """QE radial_gradient iflag=0: 3-point nonuniform stencil; gf[-1]=0,
    gf[0] linearly extrapolated. f (..., n)."""
    gf = np.zeros_like(f)
    rp = r[2:] - r[1:-1]
    rm = r[:-2] - r[1:-1]
    gf[..., 1:-1] = (
        rp**2 * (f[..., :-2] - f[..., 1:-1]) - rm**2 * (f[..., 2:] - f[..., 1:-1])
    ) / (rp * rm * (rp - rm))
    gf[..., 0] = gf[..., 1] + (gf[..., 2] - gf[..., 1]) * (r[0] - r[1]) / (r[2] - r[1])
    return gf


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

    def xc_terms(self, rho_lm: np.ndarray, core: np.ndarray):
        """(v_lm (mesh, l2) [eV], E_xc [eV]) on the angular grid (LDA + GGA)."""
        r2 = self.r**2
        rm2 = np.where(r2 > 0, 1.0 / r2, 0.0)
        rho_rad = self._lm2rad(rho_lm)  # (mesh, nx), r² included
        rho_full = rho_rad * rm2[:, None] + core[:, None]  # true density

        rho_t = torch.tensor(rho_full.reshape(-1), requires_grad=True)
        gga = getattr(self.xc, "needs_gradient", False)
        if gga:
            # spherical gradient components
            dr = _radial_gradient(rho_full.T, self.r).T  # (mesh, nx)
            gth = (rho_lm @ self.dylmt[:, : self.l2].T) * (rm2 * np.where(
                self.r > 0, 1.0 / self.r, 0.0))[:, None]
            gph = (rho_lm @ self.dylmp[:, : self.l2].T) * (rm2 * np.where(
                self.r > 0, 1.0 / self.r, 0.0))[:, None]
            grad = np.stack([dr, gph, gth], axis=0)  # (3, mesh, nx) QE order r,φ,θ
            sigma_np = (grad**2).sum(axis=0)
            sigma_t = torch.tensor(sigma_np.reshape(-1), requires_grad=True)
            e_density = self.xc.energy_density(rho_t, sigma_t)
        else:
            sigma_t = None
            e_density = self.xc.energy_density(rho_t)
        e_sum = e_density.sum()
        grads = torch.autograd.grad(e_sum, [rho_t] + ([sigma_t] if gga else []))
        v1 = grads[0].detach().numpy().reshape(self.mesh, self.nx)
        e_pt = e_density.detach().numpy().reshape(self.mesh, self.nx)

        # energy: Σ_ix ww ∫ e_xc(r,ix) r² dr
        e_xc = float(self.ww @ simpson((e_pt * r2[:, None]).T, self.rab))
        v_lm = self._rad2lm(v1, self.l2)

        if gga:
            v2 = 2.0 * grads[1].detach().numpy().reshape(self.mesh, self.nx)
            # h = 2 ∂e/∂σ ∇ρ, r² included (QE convention); project to the FULL
            # lm_max (lmax + ladd) — that is what the extra XLM l's are for
            h = v2[None] * grad * r2[None, :, None]  # (3, mesh, nx)
            h_lm = np.stack([
                hc @ (self.ww[:, None] * self.ylm) for hc in h
            ])  # (3, mesh, lm_max)
            # divergence (QE PAW_divergence): angular part on directions
            aux = np.zeros((self.mesh, self.nx))
            for lm in range(self.lm_max):
                aux += np.outer(h_lm[1, :, lm], self.dylmp[:, lm])
                aux += np.outer(
                    h_lm[2, :, lm],
                    self.dylmt[:, lm] * self.sin_th + 2.0 * self.ylm[:, lm] * self.cos_th,
                )
            div_lm = aux @ (self.ww[:, None] * self.ylm[:, : self.l2])
            rm3 = np.where(self.r > 0, self.r ** -3.0, 0.0)
            div_lm = div_lm * rm3[:, None]
            for lm in range(self.l2):
                div_lm[:, lm] += _radial_gradient(h_lm[0, :, lm], self.r) * rm2
            v_lm = v_lm - div_lm
        return v_lm, e_xc

    def xc_terms_spin(self, rho_lm_s: list, core: np.ndarray):
        """Spin-polarized XC: (v_lm per spin [2×(mesh,l2)], E_xc). The core is
        split half/half between the channels (QE convention). GGA vector
        fields: h_σ = (2 e_{σσ} ∇ρ_σ + e_{tt} ∇ρ_tot)·r²."""
        r2 = self.r**2
        rm2 = np.where(r2 > 0, 1.0 / r2, 0.0)
        rm1 = np.where(self.r > 0, 1.0 / self.r, 0.0)
        rho_full, grads_np = [], []
        for rl in rho_lm_s:
            rho_rad = self._lm2rad(rl)
            rho_full.append(rho_rad * rm2[:, None] + 0.5 * core[:, None])
            dr = _radial_gradient(rho_full[-1].T, self.r).T
            gth = (rl @ self.dylmt[:, : self.l2].T) * (rm2 * rm1)[:, None]
            gph = (rl @ self.dylmp[:, : self.l2].T) * (rm2 * rm1)[:, None]
            grads_np.append(np.stack([dr, gph, gth], axis=0))
        ru = torch.tensor(rho_full[0].reshape(-1), requires_grad=True)
        rd = torch.tensor(rho_full[1].reshape(-1), requires_grad=True)
        gga = getattr(self.xc, "needs_gradient", False)
        if gga:
            g_tot = grads_np[0] + grads_np[1]
            s_uu = torch.tensor((grads_np[0] ** 2).sum(0).reshape(-1), requires_grad=True)
            s_dd = torch.tensor((grads_np[1] ** 2).sum(0).reshape(-1), requires_grad=True)
            s_tt = torch.tensor((g_tot**2).sum(0).reshape(-1), requires_grad=True)
            e_density = self.xc.energy_density(ru, rd, s_uu, s_dd, s_tt)
            leaves = [ru, rd, s_uu, s_dd, s_tt]
        else:
            e_density = self.xc.energy_density(ru, rd)
            leaves = [ru, rd]
        g = torch.autograd.grad(e_density.sum(), leaves)
        e_pt = e_density.detach().numpy().reshape(self.mesh, self.nx)
        e_xc = float(self.ww @ simpson((e_pt * r2[:, None]).T, self.rab))

        v_lms = []
        for isp in range(2):
            v1 = g[isp].detach().numpy().reshape(self.mesh, self.nx)
            v_lm = v1 @ (self.ww[:, None] * self.ylm[:, : self.l2])
            if gga:
                e_ss = g[2 + isp].detach().numpy().reshape(self.mesh, self.nx)
                e_tt = g[4].detach().numpy().reshape(self.mesh, self.nx)
                h = (2.0 * e_ss[None] * grads_np[isp] + e_tt[None] * g_tot) * r2[None, :, None]
                h_lm = np.stack([hc @ (self.ww[:, None] * self.ylm) for hc in h])
                aux = np.zeros((self.mesh, self.nx))
                for lm in range(self.lm_max):
                    aux += np.outer(h_lm[1, :, lm], self.dylmp[:, lm])
                    aux += np.outer(
                        h_lm[2, :, lm],
                        self.dylmt[:, lm] * self.sin_th
                        + 2.0 * self.ylm[:, lm] * self.cos_th,
                    )
                div_lm = aux @ (self.ww[:, None] * self.ylm[:, : self.l2])
                rm3 = np.where(self.r > 0, self.r ** -3.0, 0.0)
                div_lm = div_lm * rm3[:, None]
                for lm in range(self.l2):
                    div_lm[:, lm] += _radial_gradient(h_lm[0, :, lm], self.r) * rm2
                v_lm = v_lm - div_lm
            v_lms.append(v_lm)
        return v_lms, e_xc

    # ---------- per-atom driver ----------

    def energy_and_ddd(self, rho_ij):
        """One-center energy [eV] and ddd [eV] for one atom.

        rho_ij: (nm, nm) tensor (nspin=1) → returns (E, ddd);
        or [ρ↑_ij, ρ↓_ij] → returns (E, [ddd↑, ddd↓]).
        """
        spin = isinstance(rho_ij, (list, tuple))
        rhos = ([m.detach().cpu().numpy().real for m in rho_ij]
                if spin else [rho_ij.detach().cpu().numpy().real])
        e_tot = 0.0
        v_saved = {}  # what → list of per-spin (v_H + v_xc)_lm
        for what, core, sgn in (("ae", self.core_ae, 1.0), ("ps", self.core_ps, -1.0)):
            rls = [self.rho_lm(r, what) for r in rhos]
            v_h, e_h = self.hartree(sum(rls))
            if spin:
                v_xs, e_x = self.xc_terms_spin(rls, core)
                v_saved[what] = [v_h + v_x for v_x in v_xs]
            else:
                v_x, e_x = self.xc_terms(rls[0], core)
                v_saved[what] = [v_h + v_x]
            e_tot += sgn * (e_h + e_x)

        wk = simpson_weights(self.rab[: self.kkbeta])
        ddds = []
        for isp in range(len(rhos)):
            ddd = np.zeros((self.nm, self.nm))
            for a, (i, _li, lmi) in enumerate(self.idx):
                for b in range(a, self.nm):
                    j, _lj, lmj = self.idx[b]
                    key = (min(i, j), max(i, j))
                    cy = self.gaunt[:, lmi, lmj]
                    nz = np.nonzero(np.abs(cy) > 1e-12)[0]
                    val = 0.0
                    for lm in nz:
                        ll = int(math.isqrt(lm))
                        f_ae = self.pfunc_ae[key][: self.kkbeta]
                        base, per_l = self.pfunc_ps[key]
                        f_ps = per_l.get(ll, base)[: self.kkbeta]
                        val += cy[lm] * float(
                            (wk * f_ae * v_saved["ae"][isp][: self.kkbeta, lm]).sum()
                        )
                        val -= cy[lm] * float(
                            (wk * f_ps * v_saved["ps"][isp][: self.kkbeta, lm]).sum()
                        )
                    ddd[a, b] = ddd[b, a] = val
            ddds.append(torch.as_tensor(ddd, dtype=torch.float64))
        return (e_tot, ddds) if spin else (e_tot, ddds[0])
