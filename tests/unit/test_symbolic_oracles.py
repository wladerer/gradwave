"""Symbolic ground-truth oracles: pin numeric code against exact closed forms.

The real Gaunt coefficients and the LDA-PW92 XC derivatives both have exact
closed forms (real Gaunt from Wigner-3j; LDA e/v/f from the algebraic functional).
These tests freeze those exact values — derived once, offline, in SymPy — so the
numeric implementations are checked against MATHEMATICAL truth, not against each
other:

- core/gaunt.py builds the real Gaunt table by Gauss-Legendre quadrature; here it
  is pinned to exact rational×√ values, including two selection-rule zeros that the
  quadrature only reaches as ~1e-16 noise.
- xc/lda_pw92.py obtains v_xc/f_xc by autograd; here e_xc, v_xc, f_xc at a fixed
  density are pinned to the frozen closed-form values.

No SymPy dependency at runtime — the exact values are baked in. Fast tier.
"""

import math

import torch

from gradwave.core.gaunt import real_gaunt_table
from gradwave.core.xc.lda_pw92 import LDA_PW92

# --- exact real-Gaunt values c[L_idx, i, j] = ∫ Ȳ_L Ȳ_i Ȳ_j dΩ (gradwave's real
# spherical-harmonic convention), from sympy.physics.wigner.real_gaunt -----------
_SQRT_PI = math.sqrt(math.pi)
GAUNT_ORACLE = {
    (0, 0, 0): 1.0 / (2.0 * _SQRT_PI),                 # 1/(2√π)
    (1, 0, 1): 1.0 / (2.0 * _SQRT_PI),                 # 1/(2√π)
    (1, 1, 4): math.sqrt(5.0) / (5.0 * _SQRT_PI),      # √5/(5√π)
    (2, 1, 5): math.sqrt(15.0) / (10.0 * _SQRT_PI),    # √15/(10√π)
    (4, 1, 3): 0.0,   # selection rule (parity l₁+l₂+L odd) → EXACT zero
    (6, 1, 1): 0.0,   # selection rule → EXACT zero
}


def test_gaunt_matches_closed_form():
    """Quadrature real_gaunt_table reproduces the exact closed forms + zeros."""
    c = real_gaunt_table(2)  # d-channel: indices 0..8, L up to 4 (L_idx 0..24)
    for (li, i, j), exact in GAUNT_ORACLE.items():
        assert abs(c[li, i, j] - exact) < 1e-12, (
            f"c[{li},{i},{j}] = {c[li, i, j]} != {exact}"
        )


# --- exact LDA-PW92 e_xc/v_xc/f_xc at ρ = 0.3 e/Å³ (Slater x + PW92 c), frozen
# from the symbolic functional and its first two derivatives ---------------------
RHO_TEST = 0.3
LDA_ORACLE = {
    "e_xc": -2.523437885359381,      # eV/Å³
    "v_xc": -10.973065391009701,     # eV·Å³/e   (δE/δρ)
    "f_xc": -11.218811914527805,     # eV·Å⁶/e²  (δ²E/δρ²)
}


def test_lda_pw92_matches_closed_form():
    """LDA-PW92 energy density and its autograd v_xc/f_xc match the closed form."""
    func = LDA_PW92()
    rho = torch.tensor([RHO_TEST], dtype=torch.float64, requires_grad=True)
    e = func.energy_density(rho)
    (v,) = torch.autograd.grad(e.sum(), rho, create_graph=True)
    (f,) = torch.autograd.grad(v.sum(), rho)
    assert abs(e.item() - LDA_ORACLE["e_xc"]) < 1e-10
    assert abs(v.item() - LDA_ORACLE["v_xc"]) < 1e-10
    assert abs(f.item() - LDA_ORACLE["f_xc"]) < 1e-9
