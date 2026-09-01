"""Track 2 — Symbolic XC kernel codegen: CSE'd, tape-free, analytic f_xc.

gradwave's XC functionals get every derivative from autograd (base.py: "v_xc =
δE_xc/δρ is obtained by autograd"). For f_xc (needed by DFPT/phonons/response)
that means a DOUBLE backward through the functional's expression graph.

This proves the symbolic alternative on LDA-PW92 (the pipeline is functional-
agnostic; r2SCAN is the same recipe with a bigger expression):

  ρ ──SymPy──▶ e_xc(ρ)  ──diff──▶ v_xc, f_xc  ──cse──▶ generated torch kernel

Then it checks the generated kernel against gradwave's actual autograd v_xc and
double-backward f_xc, and times both. Everything mirrors xc/base.py exactly:
ρ[e/Å³] → ρ_au = ρ·a₀³, e_xc[eV/Å³] = ρ·(ε_x+ε_c)·Ha.

Run:  uv run --with sympy python experiments/symbolic/track2_xc_codegen.py
"""

from __future__ import annotations

import math
import time

import sympy as sp
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.lda_pw92 import _CX, _EC0, LDA_PW92

torch.set_default_dtype(torch.float64)


def symbolic_energy_density():
    """e_xc(ρ) [eV/Å³] as an exact SymPy expression of ρ [e/Å³] — same math as
    lda_pw92.py (Slater x + PW92 c), same unit boundary as xc/base.py."""
    rho = sp.symbols("rho", positive=True)
    a0_3 = sp.Float(BOHR_ANG, 20) ** 3
    rho_au = rho * a0_3

    eps_x = sp.Float(_CX, 20) * rho_au ** sp.Rational(1, 3)

    A, a1, b1, b2, b3, b4 = (sp.Float(c, 20) for c in _EC0)
    rs = (3 / (4 * sp.pi * rho_au)) ** sp.Rational(1, 3)
    srs = sp.sqrt(rs)
    q0 = -2 * A * (1 + a1 * rs)
    q1 = 2 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    eps_c = q0 * sp.log(1 + 1 / q1)  # = log1p(1/q1)

    e_xc = rho * (eps_x + eps_c) * sp.Float(HARTREE_EV, 20)
    return rho, e_xc


def main() -> None:
    print("=== Track 2: symbolic XC kernel codegen (LDA-PW92) ===\n")

    rho, e = symbolic_energy_density()
    v = sp.diff(e, rho)          # v_xc = ∂e_xc/∂ρ   (the LDA potential)
    f = sp.diff(e, rho, 2)       # f_xc = ∂²e_xc/∂ρ² (the response kernel)

    # ---- CSE: what makes "one big expr differentiated 3 ways" compact -------
    repl, reduced = sp.cse([e, v, f], optimizations="basic")
    print(f"common subexpressions found by CSE: {len(repl)}")
    print("  (shared rs, √rs, log-term feed e, v AND f — computed once)\n")

    # generated tape-free torch kernel: returns (e, v, f) in one forward pass.
    # sqrt/log dispatch on type so CSE-hoisted CONSTANTS (e.g. √π) evaluate on
    # Python floats while the ρ-dependent terms evaluate on tensors.
    def _sqrt(x):
        return torch.sqrt(x) if isinstance(x, torch.Tensor) else math.sqrt(x)

    def _log(x):
        return torch.log(x) if isinstance(x, torch.Tensor) else math.log(x)

    tmod = [{"log": _log, "sqrt": _sqrt, "pi": math.pi}]
    kernel = sp.lambdify(rho, [e, v, f], modules=tmod, cse=True)

    # ---- reference: gradwave's actual autograd path ------------------------
    func = LDA_PW92()
    rng = torch.Generator().manual_seed(0)
    rho_t = (torch.rand(4096, generator=rng) * 0.5 + 0.02).requires_grad_(True)

    e_ref = func.energy_density(rho_t)
    (v_ref,) = torch.autograd.grad(e_ref.sum(), rho_t, create_graph=True)  # autograd v_xc
    (f_ref,) = torch.autograd.grad(v_ref.sum(), rho_t)                     # DOUBLE backward f_xc

    with torch.no_grad():
        e_k, v_k, f_k = kernel(rho_t.detach())

    e_err = (e_k - e_ref.detach()).abs().max().item()
    v_err = (v_k - v_ref.detach()).abs().max().item()
    f_err = (f_k - f_ref.detach()).abs().max().item()
    print("generated kernel vs gradwave autograd (max abs err over 4096 points):")
    print(f"  e_xc : {e_err:.2e}")
    print(f"  v_xc : {v_err:.2e}   (vs single backward)")
    print(f"  f_xc : {f_err:.2e}   (vs DOUBLE backward)")
    ok = max(e_err, v_err, f_err) < 1e-10
    print(f"\n{'PASS' if ok else 'FAIL'}: symbolic kernel reproduces autograd "
          f"e/v/f to {max(e_err, v_err, f_err):.1e}\n")

    # ---- timing: kernel forward vs autograd (fwd + 2 backward) -------------
    big = (torch.rand(200_000, generator=rng) * 0.5 + 0.02)
    N = 50

    t0 = time.perf_counter()
    for _ in range(N):
        x = big.clone().requires_grad_(True)
        ee = func.energy_density(x)
        (vv,) = torch.autograd.grad(ee.sum(), x, create_graph=True)
        (ff,) = torch.autograd.grad(vv.sum(), x)
    t_autograd = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            kernel(big)
    t_kernel = (time.perf_counter() - t0) / N

    # fused kernel via torch.compile (the fair comparison: LDA is a tiny
    # expression, so unfused eager codegen can't win — fusion is the point).
    t_comp = float("nan")
    try:
        ckernel = torch.compile(kernel, fullgraph=False)
        ckernel(big)  # warm up / trigger compile
        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                ckernel(big)
        t_comp = (time.perf_counter() - t0) / N
    except Exception as exc:  # noqa: BLE001 — POC: report, don't crash
        print(f"(torch.compile unavailable: {type(exc).__name__})")

    print(f"e+v+f over 200k points, mean of {N} runs (CPU, float64):")
    print(f"  autograd (fwd + 2 backward): {t_autograd*1e3:7.2f} ms")
    print(f"  generated kernel, eager    : {t_kernel*1e3:7.2f} ms  "
          f"({t_autograd / t_kernel:.2f}× vs autograd)")
    if t_comp == t_comp:  # not NaN
        print(f"  generated kernel, compiled : {t_comp*1e3:7.2f} ms  "
              f"({t_autograd / t_comp:.2f}× vs autograd)")
    print("  (LDA is ~1 log + 2 powers; the real speed/memory win is r2SCAN's "
          "deep graph and analytic f_xc with NO double-backward.)")


if __name__ == "__main__":
    main()
