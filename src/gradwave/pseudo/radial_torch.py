"""Differentiable spherical Bessel transforms (torch mirror of radial.py).

Stress needs form factors v_loc(|G|), F_i(|k+G|), f_core(|G|) differentiable
in |G| (the G-vectors move under strain). Tracing autograd through the
quadrature would retain O(nq · nmesh · nterms) intermediates, so sbt_t is a
custom autograd.Function: the forward evaluates under no_grad and the
backward applies the analytic derivative

    dF/dq = ∫ g(r) · r · j_l'(qr) dr,   j_l' = [l·j_{l−1} − (l+1)·j_{l+1}]/(2l+1)

(the 1/x-free form — stable at q → 0; the l = 0 case reduces to −j₁).
Memory stays flat and values are machine-precision, matching radial.sph_jl:
the ascending series below x = 4 (converged to ~1e-17 in ≤ 40 terms), closed
trigonometric forms above (no recurrence, so no n > x instability at l = 4,
which the derivative of l = 3 needs).

Quadrature weights are radial.simpson's, frozen to torch once per mesh.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.pseudo._bessel_data import DOUBLE_FACTORIAL, SERIES_TERMS, SERIES_X
from gradwave.pseudo.radial import _simpson_index_weights

_CHUNK = 4_000_000  # max elements of one (nq_chunk, nmesh) block


def simpson_weights(rab: np.ndarray) -> np.ndarray:
    """Combined Simpson × mesh weights so that ∫f dr = Σ f·w (numpy, setup)."""
    return _simpson_index_weights(len(rab)) * rab


def jl_t(l: int, x: torch.Tensor) -> torch.Tensor:
    """j_l(x) for l ≤ 5, elementwise, x ≥ 0. Plain tensor math (call under
    no_grad — the analytic-derivative path in sbt_t exists precisely so this
    never needs to be traced)."""
    if not 0 <= l <= 5:
        raise ValueError(f"l={l} out of supported range 0..5")
    small = x < SERIES_X
    xs = torch.where(small, x, torch.full_like(x, 1.0))
    x2 = xs * xs
    term = xs.pow(l) / DOUBLE_FACTORIAL[l]
    acc = term
    for k in range(1, SERIES_TERMS):
        term = term * (-0.5 * x2) / (k * (2 * l + 2 * k + 1))
        acc = acc + term

    xb = torch.where(small, torch.full_like(x, SERIES_X), x)
    s, c = torch.sin(xb), torch.cos(xb)
    u = 1.0 / xb
    u2 = u * u
    if l == 0:
        big = s * u
    elif l == 1:
        big = s * u2 - c * u
    elif l == 2:
        big = (3.0 * u2 * u - u) * s - 3.0 * u2 * c
    elif l == 3:
        big = (15.0 * u2 * u2 - 6.0 * u2) * s - (15.0 * u2 * u - u) * c
    elif l == 4:
        big = (105.0 * u2 * u2 * u - 45.0 * u2 * u + u) * s + (
            -105.0 * u2 * u2 + 10.0 * u2
        ) * c
    else:  # l == 5, via upward recurrence coefficients (verified vs j4/j3)
        big = (945.0 * u2 * u2 * u2 - 420.0 * u2 * u2 + 15.0 * u2) * s + (
            -945.0 * u2 * u2 * u + 105.0 * u2 * u - u
        ) * c
    return torch.where(small, acc, big)


def _djl_t(l: int, x: torch.Tensor) -> torch.Tensor:
    """j_l'(x) via the 1/x-free three-term identity (l ≤ 4; needs j_{l+1})."""
    if l == 0:
        return -jl_t(1, x)
    return (l * jl_t(l - 1, x) - (l + 1) * jl_t(l + 1, x)) / (2 * l + 1)


def _contract(l: int, q: torch.Tensor, r: torch.Tensor, gw: torch.Tensor,
              deriv: bool) -> torch.Tensor:
    """Σ_r gw_r · j_l(q r)  (or gw_r · r · j_l'(q r)), chunked over q."""
    out = torch.empty_like(q)
    step = max(1, _CHUNK // max(1, r.numel()))
    weights = gw * r if deriv else gw
    for i0 in range(0, q.numel(), step):
        x = q[i0 : i0 + step, None] * r[None, :]
        kern = _djl_t(l, x) if deriv else jl_t(l, x)
        out[i0 : i0 + step] = kern @ weights
    return out


class _SBT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, gw, r, l):
        ctx.l = l
        ctx.save_for_backward(q, gw, r)
        with torch.no_grad():
            return _contract(l, q, r, gw, deriv=False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        q, gw, r = ctx.saved_tensors
        with torch.no_grad():
            d = _contract(ctx.l, q, r, gw, deriv=True)
        return grad_out * d, None, None, None


def sbt_t(l: int, gvals: torch.Tensor, r: torch.Tensor, w: torch.Tensor,
          q: torch.Tensor) -> torch.Tensor:
    """F(q) = ∫ g(r)·j_l(qr) dr, differentiable in q (first order only).

    gvals, r: (n,) mesh values (all r-powers already in g, as in radial.sbt);
    w: (n,) simpson_weights(rab); q: (nq,) ≥ 0.
    """
    return _SBT.apply(q, gvals * w, r, l)


class RadialTables:
    """Per-species mesh data frozen to torch once, for strain-differentiable
    form-factor evaluation (used by postscf/stress.py)."""

    def __init__(self, upf, device=None):
        from gradwave.constants import E2
        from gradwave.pseudo.local import RC_DEFAULT, _msh, _v_short_range, alpha_z

        dt = torch.float64
        n = _msh(upf)  # QE truncates local-channel integrals at 10 bohr
        self.r = torch.as_tensor(upf.r[:n], dtype=dt, device=device)
        self.w = torch.as_tensor(simpson_weights(upf.rab[:n]), dtype=dt, device=device)
        self.zval = upf.z_valence
        self.rc = RC_DEFAULT
        self.alpha = alpha_z(upf)
        vsr = _v_short_range(upf, RC_DEFAULT)
        self.vsr_r2 = torch.as_tensor(vsr * upf.r[:n] ** 2, dtype=dt, device=device)
        self.e2 = E2
        self.beta_l = [b.l for b in upf.betas]
        self.beta_g = [
            torch.as_tensor(b.rbeta * upf.r[: b.cutoff_idx], dtype=dt, device=device)
            for b in upf.betas
        ]
        self.beta_r = [self.r[: b.cutoff_idx] for b in upf.betas]
        # Simpson closure weights depend on the point count — each truncated
        # projector mesh needs its own rule, not a slice of the full-mesh one
        self.beta_w = [
            torch.as_tensor(simpson_weights(upf.rab[: b.cutoff_idx]), dtype=dt, device=device)
            for b in upf.betas
        ]
        if upf.core_rho is not None:
            self.core_g = torch.as_tensor(
                4.0 * np.pi * upf.r[:n] ** 2 * upf.core_rho[:n], dtype=dt, device=device
            )
        else:
            self.core_g = None

    def vloc_of_g(self, q: torch.Tensor) -> torch.Tensor:
        """v(|G|) [eV·Å³] for q > 0, differentiable in q (pseudo/local.py split)."""
        short = 4.0 * np.pi * sbt_t(0, self.vsr_r2, self.r, self.w, q)
        tail = -4.0 * np.pi * self.zval * self.e2 * torch.exp(
            -0.25 * (q * self.rc) ** 2
        ) / q**2
        return short + tail

    def beta_of_g(self, i: int, q: torch.Tensor) -> torch.Tensor:
        """F_i(q), differentiable in q (pseudo/kb.py convention)."""
        return sbt_t(self.beta_l[i], self.beta_g[i], self.beta_r[i], self.beta_w[i], q)

    def core_of_g(self, q: torch.Tensor) -> torch.Tensor:
        # No NLCC for this species: match the numpy contract and return zeros
        # rather than raising, so callers need not special-case core_rho=None.
        if self.core_g is None:
            return torch.zeros_like(q)
        return sbt_t(0, self.core_g, self.r, self.w, q)
