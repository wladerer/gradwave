"""Track 1 — Real Gaunt coefficients as EXACT sparse tables (SymPy).

gradwave's core/gaunt.py builds c[LM,i,j] = ∫ Ȳ_LM Ȳ_i Ȳ_j dΩ by Gauss–Legendre
quadrature into a DENSE float array. This script shows the symbolic alternative:

  1. exact closed-form values (rational × √) from sympy.physics.wigner.real_gaunt,
     which — verified below — matches gradwave's real-harmonic convention exactly
     (cos-slot ↔ +m, sin-slot ↔ −m);
  2. the selection rules make the table ~92% structural zeros → a sparse (index,
     value) representation replaces the dense cube;
  3. the symbolic values are an exact oracle for the quadrature table (Track 4).

Run:  uv run --with sympy python experiments/symbolic/track1_gaunt.py
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp
from sympy.physics.wigner import real_gaunt

from gradwave.core.gaunt import real_gaunt_table


def idx_to_lm(idx: int) -> tuple[int, int, str]:
    """Dense slot → (l, m, kind). Ordering (l,0),(l,1c),(l,1s),(l,2c),(l,2s),…"""
    l = int(np.floor(np.sqrt(idx)))
    off = idx - l * l
    if off == 0:
        return l, 0, "0"
    m = (off + 1) // 2
    return l, m, "c" if off % 2 == 1 else "s"


def slot_to_signed_m(m: int, kind: str) -> int:
    """gradwave real-harmonic slot → SymPy real_gaunt signed m (cos→+m, sin→−m)."""
    return -m if kind == "s" else m


def build_symbolic_table(lmax_beta: int):
    """Exact real Gaunt table as a dict {(LM,i,j): sympy_value} for nonzeros only."""
    nb = (lmax_beta + 1) ** 2
    nL = (2 * lmax_beta + 1) ** 2
    exact: dict[tuple[int, int, int], sp.Expr] = {}
    for LM in range(nL):
        lL, mL, kL = idx_to_lm(LM)
        sL = slot_to_signed_m(mL, kL)
        for i in range(nb):
            li, mi, ki = idx_to_lm(i)
            si = slot_to_signed_m(mi, ki)
            for j in range(i, nb):  # symmetric in (i,j); fill j<i by mirror
                lj, mj, kj = idx_to_lm(j)
                sj = slot_to_signed_m(mj, kj)
                val = real_gaunt(lL, li, lj, sL, si, sj)
                if val != 0:
                    exact[(LM, i, j)] = sp.nsimplify(val)
    return exact, nb, nL


def main() -> None:
    lmax_beta = 2  # d-channel projectors: i,j ≤ 8, L ≤ 4
    print(f"=== Track 1: exact sparse real Gaunt table (lmax_beta={lmax_beta}) ===\n")

    t0 = time.perf_counter()
    exact, nb, nL = build_symbolic_table(lmax_beta)
    t_sym = time.perf_counter() - t0

    # ---- sparsity ----------------------------------------------------------
    # exact holds j>=i; the physical table is symmetric in (i,j).
    full_nonzero = sum(1 if i == j else 2 for (LM, i, j) in exact)
    dense_size = nL * nb * nb
    print(f"symbolic generation: {t_sym:.1f}s, {len(exact)} unique nonzeros "
          f"({full_nonzero} with (i,j) symmetry)")
    print(f"dense table size:    {nL}×{nb}×{nb} = {dense_size}")
    print(f"structural zeros:    {dense_size - full_nonzero}/{dense_size} "
          f"= {100 * (dense_size - full_nonzero) / dense_size:.1f}%")
    print(f"→ sparse storage keeps {full_nonzero} floats + indices vs {dense_size} "
          f"dense floats ({dense_size / max(full_nonzero,1):.1f}× smaller)\n")

    # ---- a few exact closed forms -----------------------------------------
    print("sample EXACT values (rational × √), vs quadrature float:")
    Cq = real_gaunt_table(lmax_beta)
    shown = 0
    for (LM, i, j), v in exact.items():
        if i != j and shown < 6:
            print(f"  c[{LM:2d},{i},{j}] = {str(v):<28} = {float(v):+.10f}  "
                  f"(quad {Cq[LM, i, j]:+.10f})")
            shown += 1

    # ---- verify against gradwave's quadrature table (Track 4 oracle) -------
    Csym = np.zeros_like(Cq)
    for (LM, i, j), v in exact.items():
        f = float(v)
        Csym[LM, i, j] = f
        Csym[LM, j, i] = f
    max_err = float(np.max(np.abs(Csym - Cq)))
    # every quadrature entry the symbolic table calls zero must actually be ~0:
    false_zero = float(np.max(np.abs(Cq[Csym == 0.0])))
    print(f"\nverification vs quadrature real_gaunt_table({lmax_beta}):")
    print(f"  max |symbolic − quadrature| over nonzeros = {max_err:.2e}")
    print(f"  max |quadrature| where symbolic == 0       = {false_zero:.2e}")
    ok = max_err < 1e-9 and false_zero < 1e-9
    print(f"\n{'PASS' if ok else 'FAIL'}: symbolic table reproduces quadrature to "
          f"{max(max_err, false_zero):.1e}, exact zeros are exact.")

    # ---- what the generated sparse kernel data would look like -------------
    print("\ngenerated sparse-table stub (first 3 rows of the COO the kernel ships):")
    coo = sorted(exact.items())[:3]
    for (LM, i, j), v in coo:
        print(f"  (L_idx={LM}, i={i}, j={j}, value={float(v):.12g})")


if __name__ == "__main__":
    main()
