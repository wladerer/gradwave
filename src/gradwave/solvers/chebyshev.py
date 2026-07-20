"""Chebyshev-filtered subspace iteration (CheFSI) for the complex Hermitian
plane-wave Hamiltonian (Layer B).

This is the GPU-oriented alternative to block Davidson. Instead of a growing
subspace with a Rayleigh-Ritz per expansion round, each outer round applies a
degree-m Chebyshev polynomial of H to a FIXED-size band block through the
three-term recurrence, then does one QR and one small Rayleigh-Ritz. The bulk
of the work is the m Hamiltonian applies inside the filter, which are pure
batched FFT streams with no interleaved reduction, so the filter tolerates
low precision and the whole solve carries one host readback per round instead
of the dozen a growing-subspace Davidson needs.

The scaled filter recurrence is the standard one from Zhou, Saad, Tiago, and
Chelikowsky, "Self-consistent-field calculations using Chebyshev-filtered
subspace iteration", J. Comput. Phys. 219 (2006) 172. With the damp interval
[a, b] mapped to [-1, 1] by t(λ) = (λ − c)/e, c = (a+b)/2, e = (b−a)/2, and a0
a lower bound of the spectrum controlling the amplification scale:

    σ1 = e/(a0 − c)                          (constant)
    Y  = (H·X − c·X)·(σ1/e)
    for i = 2..m:
        σ2   = 1/(2/σ1 − σ)                  (σ starts at σ1)
        Ynew = (H·Y − c·Y)·(2σ2/e) − (σ·σ2)·X
        X, Y, σ = Y, Ynew, σ2

Everything above a is damped; everything below a (the wanted occupied and buffer
states) is amplified, most strongly near a0. Norm-conserving only: the standard
eigenproblem, no overlap. The generalized USPP/PAW metric is a separate entry
point (see the audit notes) and is not implemented here.

Runs entirely under torch.no_grad() — autograd must never see this.
"""

from __future__ import annotations

import torch

from gradwave.solvers.davidson import BatchedDavidsonResult, _orthonormalize_b


@torch.no_grad()
def _lanczos_bounds(
    h_apply,
    mask: torch.Tensor,  # (nk, npw_max) bool
    steps: int = 6,
    seed: int = 2024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-k spectrum bounds of H from a short k-step Lanczos on a random start.

    Returns (lo, hi) each (nk,), with hi a rigorous upper bound
    max(θ(T)) + ‖f‖ and lo a lower bound min(θ(T)) − ‖f‖ (Kaniel–Paige/
    residual bracket). A handful of steps suffices because only the extremes
    are needed, not accuracy. The upper bound is essentially set by the highest
    kinetic energy on the sphere, which Lanczos finds in a few steps.
    """
    nk, m = mask.shape
    device = mask.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(nk, 1, m, 2, generator=gen, dtype=torch.float64)
    v = torch.view_as_complex(noise).to(device).to(torch.complex128)
    v = v * mask[:, None, :]
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-30)

    alpha = torch.zeros(nk, steps, dtype=torch.float64, device=device)
    beta = torch.zeros(nk, steps, dtype=torch.float64, device=device)
    v_prev = torch.zeros_like(v)
    b_prev = torch.zeros(nk, 1, dtype=torch.float64, device=device)
    fnorm = torch.zeros(nk, dtype=torch.float64, device=device)
    for j in range(steps):
        f = h_apply(v.to(torch.complex128))  # (nk, 1, m)
        a = torch.einsum("kbg,kbg->kb", v.conj(), f).real  # (nk, 1)
        alpha[:, j] = a[:, 0]
        f = f - a[..., None] * v - b_prev[..., None] * v_prev
        bn = torch.linalg.norm(f, dim=-1).real  # (nk, 1)
        fnorm = bn[:, 0]
        if j + 1 < steps:
            beta[:, j + 1] = bn[:, 0]
            v_prev = v
            b_prev = bn
            v = f / bn[..., None].clamp_min(1e-30)

    # per-k tridiagonal T (symmetric) → extreme Ritz values
    t_mat = torch.diag_embed(alpha)
    if steps > 1:
        off = beta[:, 1:]
        t_mat = t_mat + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    theta = torch.linalg.eigvalsh(t_mat)  # (nk, steps) ascending
    hi = theta[:, -1] + fnorm
    lo = theta[:, 0] - fnorm
    return lo, hi


@torch.no_grad()
def _cheby_filter(
    h_apply,
    x: torch.Tensor,  # (nk, nb, npw_max)
    degree: int,
    a: torch.Tensor,  # (nk,) lower edge of the damp interval (largest wanted Ritz value)
    b: torch.Tensor,  # (nk,) upper spectrum bound
    a0: torch.Tensor,  # (nk,) lower spectrum bound (amplification scale point)
) -> torch.Tensor:
    """Scaled degree-`degree` Chebyshev filter applied to the block `x`.

    Amplifies the subspace with eigenvalues below `a`, damps [a, b]. Per-k
    scalars broadcast over the (nb, npw) block.
    """
    e = ((b - a) / 2).clamp_min(1e-30)[:, None, None]
    c = ((b + a) / 2)[:, None, None]
    # a0 must sit strictly below c; guard against a degenerate/near-flat window
    a0c = (a0[:, None, None] - c).clamp_max(-1e-30)
    sigma1 = e / a0c
    sigma = sigma1

    y = (h_apply(x) - c * x) * (sigma1 / e)
    x_prev = x
    for _ in range(2, degree + 1):
        sigma2 = 1.0 / (2.0 / sigma1 - sigma)
        y_new = (h_apply(y) - c * y) * (2.0 * sigma2 / e) - (sigma * sigma2) * x_prev
        x_prev = y
        y = y_new
        sigma = sigma2
    return y


@torch.no_grad()
def chebyshev_filtered_batched(
    h_apply,
    x0: torch.Tensor,  # (nk, nb, npw_max), padded slots zero
    t: torch.Tensor,  # (nk, npw_max) kinetic diagonal, 0 in padding (unused; signature parity)
    mask: torch.Tensor,  # (nk, npw_max) bool
    tol: float = 1e-9,
    max_iter: int = 40,
    degree: int = 10,
    n_lanczos: int = 6,
    n_buffer: int | None = None,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> BatchedDavidsonResult:
    """Converge the nb lowest eigenpairs of a FIXED H by Chebyshev-filtered
    subspace iteration. Drop-in for `davidson_batched`: same signature and
    result type (returns the nb requested bands), so an SCF loop can select it
    per iteration.

    Each outer round: filter the current Ritz block (degree H-applies),
    QR-orthonormalize, one H-apply for the Rayleigh-Ritz, one small eigh,
    residual check. The subspace never grows and there is no restart logic —
    the filter does the job the growing Davidson subspace did.

    The block carries `n_buffer` extra bands above the nb requested. The damp
    interval starts at the TOP Ritz value, so the highest band in the block
    sits on the filter's amplify/damp boundary and converges slowly; the buffer
    keeps that slow band out of the nb that are returned and gated. Without a
    buffer the nb-th band stalls at the edge (a real CheFSI failure mode, not a
    bug). Two buffer bands suffice for the usual case; the real SCF already
    keeps empty bands, so a caller can pass n_buffer=0 and widen x0 instead.

    `bounds` lets the caller pass a precomputed (lo, hi) spectrum bracket
    (reused across SCF iterations, since the spectrum barely moves) to skip the
    Lanczos estimate. `t` is accepted for signature parity with
    `davidson_batched` and is not otherwise used.
    """
    nk, nb, m = x0.shape
    rdtype = x0.real.dtype
    if n_buffer is None:
        n_buffer = min(max(2, (nb + 7) // 8), m - nb)
    nw = nb + n_buffer  # working block width

    if bounds is None:
        lo, hi = _lanczos_bounds(h_apply, mask, steps=n_lanczos)
    else:
        lo, hi = bounds
    lo = lo.to(x0.device)
    hi = hi.to(x0.device)

    x0w = x0
    if n_buffer:
        gen = torch.Generator(device="cpu").manual_seed(nb + 104729)
        pad = torch.view_as_complex(
            torch.randn(nk, n_buffer, m, 2, generator=gen, dtype=torch.float64)
        ).to(x0.device).to(x0.dtype)
        x0w = torch.cat([x0, pad], dim=1)

    v = _orthonormalize_b(x0w, mask)
    hv = h_apply(v)
    eig = torch.zeros(nk, nw, dtype=rdtype, device=x0.device)
    x = v[:, :nw]
    rn = torch.full((nk, nw), float("inf"), dtype=rdtype, device=x0.device)

    for it in range(1, max_iter + 1):
        s = torch.einsum("kig,kjg->kij", v.conj(), hv)
        s = 0.5 * (s + s.conj().transpose(-1, -2))
        w, u = torch.linalg.eigh(s)
        eig = w[:, :nw].real
        x = torch.einsum("kja,kjg->kag", u[:, :, :nw], v)
        hx = torch.einsum("kja,kjg->kag", u[:, :, :nw], hv)

        r = hx - eig[..., None] * x
        rn = torch.linalg.norm(r, dim=-1).real
        # gate only the nb bands that are returned; the buffer bands ride the
        # damp edge and are allowed to stay loose
        if float(rn[:, :nb].max()) < tol:
            return BatchedDavidsonResult(eig[:, :nb], x[:, :nb], it, rn[:, :nb])

        # damp everything above the highest block band; amplify below it. a0 is
        # a lower scale point: the sharper of the Lanczos bracket and the
        # running smallest Ritz value.
        a = eig[:, -1]
        a0 = torch.minimum(lo, eig[:, 0])
        y = _cheby_filter(h_apply, x, degree, a, hi, a0)
        v = _orthonormalize_b(y, mask)
        hv = h_apply(v)

    return BatchedDavidsonResult(eig[:, :nb], x[:, :nb], max_iter, rn[:, :nb])


@torch.no_grad()
def chebyshev_filtered_batched_ms(
    h_apply,
    x0: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    tol: float = 1e-9,
    max_iter: int = 40,
    degree: int = 10,
    n_lanczos: int = 6,
    crossover: float = 1e-5,
    mixed_precision: bool = True,
) -> BatchedDavidsonResult:
    """fp32-deep CheFSI: draft the whole subspace iteration in complex64 down
    to `crossover`, then polish in the input precision from the drafted vectors.

    Unlike the Davidson mixed-precision path, whose per-round fp64
    Rayleigh-Ritz caps the low-precision window, here the filter H-applies
    dominate the work and run in fp32 throughout the draft. The polish re-runs
    the same iteration in full precision from the warm start, so the returned
    eigenpairs are full precision regardless of the draft.

    Skipped (single full-precision solve) when mixed_precision is off, x0 is
    already low precision, or crossover ≥ tol.
    """
    from gradwave.dtypes import real_of

    low = torch.complex64
    if (not mixed_precision) or x0.dtype == low or crossover >= tol:
        return chebyshev_filtered_batched(
            h_apply, x0, t, mask, tol=tol, max_iter=max_iter,
            degree=degree, n_lanczos=n_lanczos,
        )
    hi_dtype = x0.dtype
    # bounds are cheap and precision-insensitive; compute once and reuse for
    # both the draft and the polish (_lanczos_bounds runs in fp64 internally)
    lo, hi = _lanczos_bounds(h_apply, mask, steps=n_lanczos)
    draft = chebyshev_filtered_batched(
        h_apply, x0.to(low), t.to(real_of(low)), mask,
        tol=crossover, max_iter=max_iter, degree=degree,
        bounds=(lo.to(real_of(low)), hi.to(real_of(low))),
    )
    x1 = draft.eigenvectors.to(hi_dtype)
    # renormalize per band: fp32 leaves ‖ψ‖ good only to ~1e-6, and the polish
    # orthonormalizes anyway, but a clean warm start converges faster
    x1 = x1 / torch.linalg.norm(x1, dim=-1, keepdim=True).clamp_min(1e-30)
    return chebyshev_filtered_batched(
        h_apply, x1, t, mask, tol=tol, max_iter=max_iter,
        degree=degree, n_lanczos=n_lanczos, bounds=(lo, hi),
    )
