"""r2SCAN meta-GGA (Furness, Kaplan, Ning, Perdew, Sun, JPCL 11, 8208 (2020)).

The regularized-restored SCAN functional: a τ-dependent meta-GGA that depends on
the kinetic-energy density through the regularized iso-orbital indicator

    ᾱ = (τ − τ_W)/(τ_unif + η τ_W),   τ_W = |∇ρ|²/(8ρ),
    τ_unif = (3/10)(3π²)^{2/3} ρ^{5/3},   η = 0.001.

Transcribed from libxc's own Maple source (mgga_x_r2scan / mgga_c_r2scan and
their scan/rscan/PW92 includes) so it matches libxc — hence QE's `input_dft`
='r2scan' — pointwise; validated against pyscf/libxc in tests. Written as a
differentiable PyTorch expression so v_xc, the meta-GGA v_τ = ∂e/∂τ, forces, and
the learnable-parameter graph all fall out of autograd exactly as for LDA/PBE.

Units mirror the rest of core.xc: ρ [e/Å³], σ = |∇ρ|² [e²/Å⁸], τ [e/Å⁵] arrive
in Å units and are converted to Hartree a.u. internally; e_xc is returned in
eV/Å³. Exchange spin-scales (E_x[ρ↑,ρ↓] = ½Σ E_x[2ρ_s]); correlation is a
total-density functional of (ρ, σ_tot, τ_tot, ζ).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.base import RHO_FLOOR_AU, XCFunctional, to_au
from gradwave.core.xc.spin import SpinXC

# ---------------------------------------------------------------------------
# shared constants (a.u.)
# ---------------------------------------------------------------------------
MU_GE = 10.0 / 81.0
K_FACTOR_C = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0)  # τ_unif reduced prefactor (per-spin)
XT2S = 1.0 / (2.0 * (3.0 * math.pi**2) ** (1.0 / 3.0))  # total-density |∇ρ|/ρ^{4/3} → s
X_FACTOR_C = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)

# exchange enhancement
_K1 = 0.065
_H0X = 1.174
_A1X = 4.9479
_C1X, _C2X, _DX, _ETA, _DP2 = 0.667, 0.8, 1.24, 0.001, 0.361
_B2 = math.sqrt(5913.0 / 405000.0)
_B1 = (511.0 / 13500.0) / (2.0 * _B2)
_B3 = 0.5
_B4 = MU_GE**2 / _K1 - 1606.0 / 18225.0 - _B1**2
# rSCAN switching polynomial coefficients, libxc order [c7, c6, ..., c0]
_RSCAN_FX = [-0.023185843322, 0.234528941479, -0.887998041597, 1.451297044490,
             -0.663086601049, -0.4445555, -0.667, 1.0]
# Σ_{i=1..8} i·ff[9-i] with ff 1-based over [c7..c0] ⇒ Σ i·fx[8-i] (0-based)
_C2SUM = sum(i * _RSCAN_FX[8 - i] for i in range(1, 9))
_CN = 20.0 / 27.0 + _ETA * 5.0 / 3.0
_C2X_COEF = -_C2SUM * (1.0 - _H0X)

# correlation
_C1C, _C2C, _DC = 0.64, 1.5, 0.7
_RSCAN_FC = [-0.051848879792, 0.516884468372, -1.915710236206, 3.061560252175,
             -1.535685604549, -0.4352, -0.64, 1.0]
_DFC2 = sum(i * _RSCAN_FC[7 - i] for i in range(1, 8))  # Σ_{i=1..7} i·fc[8-i] (0-based)
_MGAMMA = (1.0 - math.log(2.0)) / math.pi**2
_BETA_A, _BETA_B, _BETA_C = 0.066724550603149220, 0.1, 0.1778
_B1C, _B2C, _B3C = 0.0285764, 0.0889, 0.125541
_CHI_INFTY = 0.12802585262625815
_G_CNST = 2.363
_DP2C = 0.361
# PW92 (modified) correlation parameters, per row k = 0,1,2
_PW_A = [0.0310907, 0.01554535, 0.0168869]
_PW_ALPHA1 = [0.21370, 0.20548, 0.11125]
_PW_BETA1 = [7.5957, 14.1189, 10.357]
_PW_BETA2 = [3.5876, 6.1977, 3.6231]
_PW_BETA3 = [1.6382, 3.3662, 0.88026]
_PW_BETA4 = [0.49294, 0.62517, 0.49671]
_FZ20 = 1.709920934161365617563962776245
_FZ_DEN = 2.0 ** (4.0 / 3.0) - 2.0


def _f_zeta(z):
    return ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0) - 2.0) / _FZ_DEN


def _mphi(z):
    return 0.5 * ((1.0 + z) ** (2.0 / 3.0) + (1.0 - z) ** (2.0 / 3.0))


# ---------------------------------------------------------------------------
# exchange
# ---------------------------------------------------------------------------
def _ex_enhancement(p, alpha):
    """r2SCAN exchange enhancement F_x(p, ᾱ)."""
    # SCAN y-analogue with the r2SCAN gradient regularization
    r2x = (_CN * _C2X_COEF * torch.exp(-(p**2) / _DP2**4) + MU_GE) * p
    h1x = 1.0 + _K1 * r2x / (_K1 + r2x)
    # f(ᾱ): exp branch (ᾱ≤0), 7th-order polynomial (0<ᾱ≤2.5), decay (ᾱ>2.5)
    a_neg = torch.clamp(alpha, max=0.0)
    f_neg = torch.exp(-_C1X * a_neg / (1.0 - a_neg))
    a_sm = torch.clamp(alpha, min=0.0, max=2.5)
    f_sm = sum(_RSCAN_FX[7 - i] * a_sm**i for i in range(8))  # c0 + c1 a + ... + c7 a^7
    a_lg = torch.clamp(alpha, min=2.5)
    f_lg = -_DX * torch.exp(_C2X / (1.0 - a_lg))
    f_a = torch.where(alpha <= 0.0, f_neg,
                      torch.where(alpha <= 2.5, f_sm, f_lg))
    # gx(p): s = √p, √s = p^{1/4}
    s = torch.sqrt(torch.clamp(p, min=1e-30))
    gx = 1.0 - torch.exp(-_A1X / torch.sqrt(torch.clamp(s, min=1e-30)))
    return (h1x + f_a * (_H0X - h1x)) * gx


def _ex_unpol(n, sig, tau):
    """Exchange energy density [Ha/bohr³] for a spin-unpolarized density
    (n, sig, tau) in a.u. — the argument passed the spin-scaled 2ρ_s per channel."""
    n = torch.clamp(n, min=RHO_FLOOR_AU)
    kf = (3.0 * math.pi**2 * n) ** (1.0 / 3.0)
    eps_x_unif = -3.0 / (4.0 * math.pi) * kf  # Ha/particle
    p = XT2S**2 * sig / n ** (8.0 / 3.0)
    tau_w = sig / (8.0 * n)
    tau_unif = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0) * n ** (5.0 / 3.0)
    alpha = (tau - tau_w) / (tau_unif + _ETA * tau_w)
    return n * eps_x_unif * _ex_enhancement(p, alpha)


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------
def _pw92_g(k, rs):
    """PW92(-modified) g(k, rs) and its rs-derivative (analytic, a.u.)."""
    a, a1, b1, b2, b3, b4 = (_PW_A[k], _PW_ALPHA1[k], _PW_BETA1[k], _PW_BETA2[k],
                             _PW_BETA3[k], _PW_BETA4[k])
    rsh = torch.sqrt(rs)
    q1 = b1 * rsh + b2 * rs + b3 * rs * rsh + b4 * rs**2
    g = -2.0 * a * (1.0 + a1 * rs) * torch.log1p(1.0 / (2.0 * a * q1))
    q1p = b1 / (2.0 * rsh) + b2 + 1.5 * b3 * rsh + 2.0 * b4 * rs
    lp = -q1p / (q1 * (2.0 * a * q1 + 1.0))  # d/drs log1p(1/(2a q1))
    gp = -2.0 * a * (a1 * torch.log1p(1.0 / (2.0 * a * q1)) + (1.0 + a1 * rs) * lp)
    return g, gp


def _f_pw(rs, z):
    """PW92(-modified) correlation ε_c(rs, ζ) and dε_c/drs [Ha/particle]."""
    fz = _f_zeta(z)
    g1, g1p = _pw92_g(0, rs)
    g2, g2p = _pw92_g(1, rs)
    g3, g3p = _pw92_g(2, rs)
    e = g1 + z**4 * fz * (g2 - g1 + g3 / _FZ20) - fz * g3 / _FZ20
    ep = g1p + z**4 * fz * (g2p - g1p + g3p / _FZ20) - fz * g3p / _FZ20
    return e, ep


def _scan_eclda0(rs):
    """SCAN LSDA0 base ε and dε/drs (analytic)."""
    rsh = torch.sqrt(rs)
    den = 1.0 + _B2C * rsh + _B3C * rs
    e = -_B1C / den
    denp = _B2C / (2.0 * rsh) + _B3C
    ep = _B1C * denp / den**2
    return e, ep


def _scan_Gc(z):
    # one_minus_z_pow_n(z, 12) = 1 − z¹² (even-n telescoping); →1 at z=0, →0 at z=±1
    return (1.0 - _G_CNST * (2.0 ** (1.0 / 3.0) - 1.0) * _f_zeta(z)) * (1.0 - z**12)


def _scan_e0(rs, z, s):
    eclda0, _ = _scan_eclda0(rs)
    one_minus_ginf = -torch.expm1(-0.25 * torch.log1p(4.0 * _CHI_INFTY * s**2))
    h0 = _B1C * torch.log1p(torch.expm1(-eclda0 / _B1C) * one_minus_ginf)
    return (eclda0 + h0) * _scan_Gc(z)


def _mbeta(rs):
    return _BETA_A * (1.0 + _BETA_B * rs) / (1.0 + _BETA_C * rs)


def _ec(n, sig_tot, tau_tot, zeta):
    """r2SCAN correlation energy density [Ha/bohr³], total-density functional."""
    n = torch.clamp(n, min=RHO_FLOOR_AU)
    rs = (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0)
    z = torch.clamp(zeta, -1.0 + 1e-12, 1.0 - 1e-12)
    phi = _mphi(z)
    xt = torch.sqrt(torch.clamp(sig_tot, min=0.0)) / n ** (4.0 / 3.0)
    s = XT2S * xt
    tt = xt / (4.0 * 2.0 ** (1.0 / 3.0) * phi * torch.sqrt(rs))

    # regularized iso-orbital indicator (total). τ_unif carries the spin-scaling
    # factor ds(z) = ½((1+z)^{5/3}+(1-z)^{5/3}) = 1 at z=0 (from t_total(z,1,1)).
    ds = 0.5 * ((1.0 + z) ** (5.0 / 3.0) + (1.0 - z) ** (5.0 / 3.0))
    tau_w = sig_tot / (8.0 * n)
    tau_unif = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0) * n ** (5.0 / 3.0) * ds
    alpha = (tau_tot - tau_w) / (tau_unif + _ETA * tau_w)

    f_pw, dfpw = _f_pw(rs, z)
    w1 = torch.expm1(-f_pw / (_MGAMMA * phi**3))

    # ec1 = f_pw + fH  (r2SCAN gradient term, eqns S29–S34); r2scan_d(z) = ds
    r2d = ds
    ecl0, decl0 = _scan_eclda0(rs)
    gc = _scan_Gc(z)
    elsda0 = ecl0 * gc          # LSDA0
    elsda1 = f_pw               # LSDA1 = PW92
    delsda0 = decl0 * gc        # d/drs (z fixed)
    delsda1 = dfpw
    dy = _DFC2 / (27.0 * _MGAMMA * r2d * phi**3 * w1) * (
        20.0 * rs * (delsda0 - delsda1) - 45.0 * _ETA * (elsda0 - elsda1)
    ) * s**2 * torch.exp(-(s**4) / _DP2C**4)
    yy = _mbeta(rs) * tt**2 / (_MGAMMA * w1)
    one_minus_g = -torch.expm1(-0.25 * torch.log1p(4.0 * (yy - dy)))
    fH = _MGAMMA * phi**3 * torch.log1p(w1 * one_minus_g)
    ec1 = f_pw + fH

    # ec0 = SCAN e0
    ec0 = _scan_e0(rs, z, s)

    # f_c(ᾱ): exp branch, polynomial, decay (correlation params)
    a_neg = torch.clamp(alpha, max=0.0)
    fc_neg = torch.exp(-_C1C * a_neg / (1.0 - a_neg))
    a_sm = torch.clamp(alpha, min=0.0, max=2.5)
    fc_sm = sum(_RSCAN_FC[7 - i] * a_sm**i for i in range(8))
    a_lg = torch.clamp(alpha, min=2.5)
    fc_lg = -_DC * torch.exp(_C2C / (1.0 - a_lg))
    fc_a = torch.where(alpha <= 0.0, fc_neg,
                       torch.where(alpha <= 2.5, fc_sm, fc_lg))

    eps_c = ec1 + fc_a * (ec0 - ec1)
    return n * eps_c


# ---------------------------------------------------------------------------
# public functionals
# ---------------------------------------------------------------------------
class R2SCAN(XCFunctional):
    """Spin-unpolarized r2SCAN meta-GGA."""

    needs_gradient = True
    needs_tau = True

    def energy_density(self, rho, sigma=None, tau=None):
        if sigma is None or tau is None:
            raise ValueError("r2SCAN requires sigma = |∇ρ|² and tau")
        n = to_au(rho)
        sig = sigma * BOHR_ANG**8
        t = tau * BOHR_ANG**5
        e_x = _ex_unpol(n, sig, t)  # unpolarized: 2ρ_s = ρ
        e_c = _ec(n, sig, t, torch.zeros_like(n))
        # e_x, e_c are Ha/bohr³ = ρ_au·ε; return eV/Å³ via ρ[e/Å³]·ε·HARTREE_EV
        eps = (e_x + e_c) / torch.clamp(n, min=RHO_FLOOR_AU)  # Ha/particle
        return rho * eps * HARTREE_EV


class SpinR2SCAN(SpinXC):
    """Collinear spin-polarized r2SCAN meta-GGA."""

    needs_gradient = True
    needs_tau = True

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None,
                       sigma_tot=None, tau_up=None, tau_dn=None):
        if sigma_uu is None or tau_up is None:
            raise ValueError("spin r2SCAN requires per-spin sigma and tau")
        nu, nd = to_au(rho_up), to_au(rho_dn)
        s_uu = sigma_uu * BOHR_ANG**8
        s_dd = sigma_dd * BOHR_ANG**8
        s_tt = sigma_tot * BOHR_ANG**8
        tu, td = tau_up * BOHR_ANG**5, tau_dn * BOHR_ANG**5
        n = nu + nd
        # exchange: spin-scaled E_x = ½Σ E_x[2ρ_s, 4σ_ss, 2τ_s]
        e_x = 0.5 * (_ex_unpol(2.0 * nu, 4.0 * s_uu, 2.0 * tu)
                     + _ex_unpol(2.0 * nd, 4.0 * s_dd, 2.0 * td))
        zeta = (nu - nd) / torch.clamp(n, min=RHO_FLOOR_AU)
        e_c = _ec(n, s_tt, tu + td, zeta)
        eps = (e_x + e_c) / torch.clamp(n, min=RHO_FLOOR_AU)
        return (rho_up + rho_dn) * eps * HARTREE_EV
