"""Symbolic f_xc codegen for PBE — analytic v_xc + f_xc, no autograd double-backward.

gradwave gets XC derivatives from autograd, and base.py forces f_xc onto the EAGER
path (its own comment: torch.compile/aot_autograd can't double-backward, and f_xc IS
a double backward). The DFPT/phonon/response f_xc (postscf._response.fxc_hvp) therefore
runs uncompiled through a deep tape.

This derives PBE's e_xc(ρ,σ) once in SymPy — matching gradwave's pbe.py / _pbe_kernels
exactly — CSE's it, and emits a flat, tape-free kernel returning
    e,  v_ρ=∂e/∂ρ,  v_σ=∂e/∂σ,  f_ρρ, f_ρσ, f_σσ
analytically. Validated against gradwave's autograd (single + DOUBLE backward), then
benchmarked: the forced-eager autograd f_xc vs the symbolic kernel (eager + compiled).

Run (asus):  uv run python experiments/symbolic/fxc_codegen.py
"""

from __future__ import annotations

import math
import time

import sympy as sp
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc._pbe_kernels import BETA, GAMMA, KAPPA, MU
from gradwave.core.xc.lda_pw92 import _CX, _EC0
from gradwave.core.xc.pbe import PBE

torch.set_default_dtype(torch.float64)


def pbe_energy_density_symbolic():
    """e_xc(ρ,σ) [eV/Å³] as an exact SymPy expression, mirroring pbe.py exactly."""
    rho, sigma = sp.symbols("rho sigma", positive=True)
    a0 = sp.Float(BOHR_ANG, 20)
    rho_au = rho * a0**3
    sigma_au = sigma * a0**8
    grad_au = sp.sqrt(sigma_au)

    kf = (3 * sp.pi**2 * rho_au) ** sp.Rational(1, 3)
    s2 = sigma_au / (2 * kf * rho_au) ** 2
    Fx = 1 + sp.Float(KAPPA, 20) - sp.Float(KAPPA, 20) / (
        1 + sp.Float(MU, 20) * s2 / sp.Float(KAPPA, 20))
    eps_x = sp.Float(_CX, 20) * rho_au ** sp.Rational(1, 3) * Fx

    # PW92 correlation (unpolarized), = lda_pw92._g_pw92(rs, _EC0)
    A, a1, b1, b2, b3, b4 = (sp.Float(c, 20) for c in _EC0)
    rs = (3 / (4 * sp.pi * rho_au)) ** sp.Rational(1, 3)
    srs = sp.sqrt(rs)
    q0 = -2 * A * (1 + a1 * rs)
    q1 = 2 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    eps_c_lda = q0 * sp.log(1 + 1 / q1)

    ks = sp.sqrt(4 * kf / sp.pi)
    t2 = sigma_au / (2 * ks * rho_au) ** 2
    g = sp.Float(GAMMA, 20)
    beta = sp.Float(BETA, 20)
    Ac = (beta / g) / (sp.exp(-eps_c_lda / g) - 1)
    num = 1 + Ac * t2
    den = 1 + Ac * t2 + (Ac * t2) ** 2
    H = g * sp.log(1 + (beta / g) * t2 * num / den)
    eps_c = eps_c_lda + H

    e_xc = rho * (eps_x + eps_c) * sp.Float(HARTREE_EV, 20)
    return rho, sigma, e_xc


def main():
    print("=== symbolic f_xc codegen for PBE ===\n")
    rho, sigma, e = pbe_energy_density_symbolic()
    firsts = [sp.diff(e, rho), sp.diff(e, sigma)]
    seconds = [sp.diff(e, rho, 2), sp.diff(e, rho, sigma), sp.diff(e, sigma, 2)]
    repl, _ = sp.cse([e, *firsts, *seconds], optimizations="basic")
    print(f"CSE common subexpressions across e, v_ρ, v_σ, f_ρρ, f_ρσ, f_σσ: {len(repl)}")

    # fold every pure-number subexpression (√π, exp(const), …) to a float literal
    # so the generated kernel is torch.compile-safe (torch.sqrt/log/exp only ever
    # see tensors — the isinstance-dispatch version broke Dynamo tracing).
    def fold(expr):
        return expr.replace(lambda a: a.is_number, lambda a: sp.Float(a.evalf(20)))

    exprs = [fold(x) for x in [e, *firsts, *seconds]]
    tmod = [{"sqrt": torch.sqrt, "log": torch.log, "exp": torch.exp}]
    kernel = sp.lambdify((rho, sigma), exprs, modules=tmod, cse=True)

    # ---- validate vs gradwave's PBE autograd (single + double backward) ----
    func = PBE()
    g = torch.Generator().manual_seed(0)
    r = (torch.rand(4096, generator=g) * 0.4 + 0.03).requires_grad_(True)
    sg = (torch.rand(4096, generator=g) * 0.05 + 1e-3).requires_grad_(True)
    e_ref = func.energy_density(r, sg)
    vr, vs = torch.autograd.grad(e_ref.sum(), (r, sg), create_graph=True)
    frr = torch.autograd.grad(vr.sum(), r, retain_graph=True)[0]
    frs = torch.autograd.grad(vr.sum(), sg, retain_graph=True)[0]
    fss = torch.autograd.grad(vs.sum(), sg)[0]

    with torch.no_grad():
        ek, vrk, vsk, frrk, frsk, fssk = kernel(r.detach(), sg.detach())
    errs = {"e": (ek - e_ref.detach()).abs().max().item(),
            "v_ρ": (vrk - vr.detach()).abs().max().item(),
            "v_σ": (vsk - vs.detach()).abs().max().item(),
            "f_ρρ": (frrk - frr).abs().max().item(),
            "f_ρσ": (frsk - frs).abs().max().item(),
            "f_σσ": (fssk - fss).abs().max().item()}
    print("\nmax abs error vs autograd (4096 pts):")
    for k, v in errs.items():
        print(f"  {k:5s}: {v:.2e}")
    ok = max(errs.values()) < 1e-9
    print(f"{'PASS' if ok else 'FAIL'}: symbolic PBE f_xc matches autograd to "
          f"{max(errs.values()):.1e}\n")

    # ---- benchmark: forced-eager autograd f_xc vs symbolic kernel ----
    # CUDA (Triton) avoids the CPU-Inductor /sbin/ldconfig call NixOS lacks.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = 30
    br = (torch.rand(200_000, generator=g) * 0.4 + 0.03).to(dev)
    bs = (torch.rand(200_000, generator=g) * 0.05 + 1e-3).to(dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def bench(fn, reps=N):
        fn(); sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        return (time.perf_counter() - t0) / reps

    def autograd_fxc():
        x = br.clone().requires_grad_(True)
        y = bs.clone().requires_grad_(True)
        ee = func.energy_density(x, y)
        a, b = torch.autograd.grad(ee.sum(), (x, y), create_graph=True)
        torch.autograd.grad(a.sum(), x, retain_graph=True)
        torch.autograd.grad(a.sum(), y, retain_graph=True)
        torch.autograd.grad(b.sum(), y)

    def eager_fxc():
        with torch.no_grad():
            kernel(br, bs)

    t_ag = bench(autograd_fxc)
    t_eager = bench(eager_fxc)

    t_comp = float("nan")
    try:
        ck = torch.compile(kernel)
        t_comp = bench(lambda: ck(br, bs))
    except Exception as exc:  # noqa: BLE001
        print(f"(torch.compile failed: {type(exc).__name__}: {str(exc)[:200]})")

    print(f"e + v_ρ,v_σ + f_ρρ,f_ρσ,f_σσ over 200k pts on {dev}, mean of {N}:")
    print(f"  autograd (fwd + double backward, forced eager): {t_ag*1e3:8.2f} ms")
    print(f"  symbolic kernel, eager                        : {t_eager*1e3:8.2f} ms  "
          f"({t_ag/t_eager:.1f}×)")
    if t_comp == t_comp:
        print(f"  symbolic kernel, torch.compile (Triton)       : {t_comp*1e3:8.2f} ms  "
              f"({t_ag/t_comp:.1f}×)")
    print("\n→ analytic f_xc is tape-free and compilable — the path base.py forces onto "
          "eager (autograd double-backward can't be compiled at all).")


if __name__ == "__main__":
    main()
