"""Track 4 — Symbolic ground-truth oracles for the test suite.

The point of Tracks 1–2 is not only faster kernels but EXACT reference values.
Here the closed forms (derived once, offline, in SymPy) are FROZEN as constants,
so this test needs no SymPy at runtime — it pins gradwave's numerical code against
mathematical truth, turning "matches quadrature/autograd" (numeric-vs-numeric)
into "matches closed form" (numeric-vs-exact).

Run:  uv run pytest experiments/symbolic/track4_oracles.py -q
  or:  uv run python experiments/symbolic/track4_oracles.py
"""

from __future__ import annotations

import math

import torch

torch.set_default_dtype(torch.float64)

# --- Exact real-Gaunt values (Track 1, from sympy.physics.wigner.real_gaunt) ---
# c[L_idx, i, j] = ∫ Ȳ_L Ȳ_i Ȳ_j dΩ, gradwave real-harmonic convention.
GAUNT_ORACLE = {
    (0, 0, 0): 1.0 / (2.0 * math.sqrt(math.pi)),          # 1/(2√π)
    (1, 0, 1): 1.0 / (2.0 * math.sqrt(math.pi)),          # 1/(2√π)
    (1, 1, 4): math.sqrt(5.0) / (5.0 * math.sqrt(math.pi)),   # √5/(5√π)
    (2, 1, 5): math.sqrt(15.0) / (10.0 * math.sqrt(math.pi)),  # √15/(10√π)
    (4, 1, 3): 0.0,   # selection rule → EXACT zero (parity: l1+l2+L odd)
    (6, 1, 1): 0.0,   # selection rule → EXACT zero
}


def test_gaunt_matches_closed_form():
    from gradwave.core.gaunt import real_gaunt_table

    C = real_gaunt_table(2)
    for (L, i, j), exact in GAUNT_ORACLE.items():
        assert abs(C[L, i, j] - exact) < 1e-12, f"c[{L},{i},{j}]={C[L, i, j]} != {exact}"


# --- Exact LDA-PW92 e/v/f at a fixed density (Track 2, from the SymPy kernel) ---
# ρ = 0.3 e/Å³. Values frozen from the symbolic e_xc(ρ) and its first two
# derivatives; units eV/Å³, eV·Å³/e, eV·Å⁶/e² respectively.
RHO_TEST = 0.3
LDA_ORACLE = {
    "e_xc": -2.523437885359381,
    "v_xc": -10.973065391009701,
    "f_xc": -11.218811914527805,
}


def test_lda_pw92_matches_closed_form():
    from gradwave.core.xc.lda_pw92 import LDA_PW92

    func = LDA_PW92()
    rho = torch.tensor([RHO_TEST], requires_grad=True)
    e = func.energy_density(rho)
    (v,) = torch.autograd.grad(e.sum(), rho, create_graph=True)
    (f,) = torch.autograd.grad(v.sum(), rho)
    assert abs(e.item() - LDA_ORACLE["e_xc"]) < 1e-10
    assert abs(v.item() - LDA_ORACLE["v_xc"]) < 1e-10
    assert abs(f.item() - LDA_ORACLE["f_xc"]) < 1e-9


if __name__ == "__main__":
    test_gaunt_matches_closed_form()
    print("gaunt oracle: PASS (6 exact values incl. 2 selection-rule zeros)")
    test_lda_pw92_matches_closed_form()
    print("LDA-PW92 oracle: PASS (e, v_xc, f_xc vs frozen closed form)")
