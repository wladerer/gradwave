"""SoftModeDeflate P2: the deflated solve on planted near-critical operators.

Rigorous, fast validation of the deflation algorithm and the pre/post fork,
decoupled from the DFT cost: a real non-normal operator whose spectrum has one
soft mode driven to (and past) +1 plus a strong negative "Hartree-like" mode, so
the exact solution is known (dense solve) and both the near-critical and the
"gain > 1" regimes are exercised. The physical version (the real Si response
operator, coupling-scaled) lives in the integration test.
"""

import pytest
import torch

from gradwave.scf.soft_mode import (
    anderson_solve,
    deflated_solve,
    soft_subspace_from_operator,
)


def _near_critical_operator(lam_soft, n=40, hard=-2.20, noise=0.05, seed=11):
    """Non-normal operator with a planted spectrum: max-real mode at ``lam_soft``
    (the soft mode → +1), a strong negative charge-like mode at ``hard`` (so the
    complement spans a wide spectrum that a damped-Richardson smoother could not
    handle — Anderson smoothing is required), and the rest well inside (-0.5, 0.5).
    Upper-triangular so the eigenvalues are exactly the diagonal."""
    g = torch.Generator().manual_seed(seed)
    mid = torch.linspace(-0.5, 0.5, n - 2, dtype=torch.float64)
    diag = torch.cat([torch.tensor([hard, lam_soft], dtype=torch.float64), mid])
    upper = torch.triu(noise * torch.randn(n, n, generator=g, dtype=torch.float64), 1)
    return upper + torch.diag(diag)


@pytest.mark.parametrize("lam_soft,regime", [(0.999, "near-critical"),
                                             (1.05, "gain>1")])
def test_deflation_solves_where_baseline_struggles(lam_soft, regime):
    a = _near_critical_operator(lam_soft)
    n = a.shape[0]

    def apply(x):
        return a @ x

    g = torch.Generator().manual_seed(2)
    vbar = torch.randn(n, generator=g, dtype=torch.float64)
    vbar = vbar - vbar.mean()

    # ground truth: (1 − A) u = v̄
    u_star = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - a, vbar)

    q = soft_subspace_from_operator(apply, torch.zeros(n, dtype=torch.float64),
                                    krylov=24, n_modes=1, seed=4).vectors
    assert len(q) == 1

    base = anderson_solve(apply, vbar, tol=1e-8, max_iter=300, history=8)
    post = deflated_solve(apply, vbar, q, method="post", tol=1e-8, max_iter=300)
    pre = deflated_solve(apply, vbar, q, method="pre", tol=1e-8, max_iter=300)

    # both variants converge to the EXACT solution (correctness is the gate)
    for r in (post, pre):
        assert r.converged, (regime, r)
        err = float(torch.linalg.vector_norm(r.u - u_star)
                    / torch.linalg.vector_norm(u_star))
        assert err < 1e-5, (regime, err)
    # the two variants land on the same solution (relative — the near-critical
    # solution has a large amplified soft component, ‖u‖ ~ 1/(1−λ_soft))
    assert float(torch.linalg.vector_norm(post.u - pre.u)
                 / torch.linalg.vector_norm(pre.u)) < 1e-5

    # deflation is a decisive win: far fewer iterations than the baseline, or the
    # baseline fails outright (gain > 1). Record the counts in the message.
    assert (not base.converged) or (post.n_iter < base.n_iter), (
        regime, "base", base.n_iter, base.converged, "post", post.n_iter)


def _soft_cluster_operator(n=60, k=6, lo=0.985, hi=0.999, noise=0.03, seed=11):
    """A soft *cluster* of ``k`` modes near +1 plus a BENIGN complement in
    (-0.5, 0.5). The complement is trivially handled by the Anderson smoother, so
    this isolates the deflation effect: convergence is gated purely on removing the
    cluster, which is more modes than the smoother's history can resolve at once."""
    g = torch.Generator().manual_seed(seed)
    cluster = torch.linspace(lo, hi, k, dtype=torch.float64)
    rest = torch.linspace(-0.5, 0.5, n - k, dtype=torch.float64)
    diag = torch.cat([cluster, rest])
    upper = torch.triu(noise * torch.randn(n, n, generator=g, dtype=torch.float64), 1)
    return upper + torch.diag(diag)


def test_deflation_wins_decisively_on_a_soft_cluster():
    """The regime deflation is *for*: a soft cluster that stalls plain Anderson.

    Also settles the pre/post fork: for an INCOMPLETE deflation subspace, post
    (smooth-then-correct) is decisively better; for a complete one they tie — so
    post is never worse and is the variant to keep.
    """
    a = _soft_cluster_operator(k=6)
    n = a.shape[0]

    def apply(x):
        return a @ x

    g = torch.Generator().manual_seed(2)
    vbar = torch.randn(n, generator=g, dtype=torch.float64)
    vbar = vbar - vbar.mean()
    u_star = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - a, vbar)

    tol, cap = 1e-7, 400
    base = anderson_solve(apply, vbar, history=8, tol=tol, max_iter=cap)
    assert not base.converged  # the 6-mode soft cluster stalls plain Anderson

    q6 = soft_subspace_from_operator(apply, torch.zeros(n, dtype=torch.float64),
                                     krylov=30, n_modes=6, seed=4).vectors
    q1 = soft_subspace_from_operator(apply, torch.zeros(n, dtype=torch.float64),
                                     krylov=30, n_modes=1, seed=4).vectors

    d6 = deflated_solve(apply, vbar, q6, method="post", history=8, tol=tol, max_iter=cap)
    d1 = deflated_solve(apply, vbar, q1, method="post", history=8, tol=tol, max_iter=cap)

    # deflating the whole cluster rescues a non-converging baseline, exactly
    assert d6.converged, d6
    err = float(torch.linalg.vector_norm(d6.u - u_star)
                / torch.linalg.vector_norm(u_star))
    assert err < 1e-5, err
    assert d6.n_iter < cap // 3            # decisive (calibrated ~40 vs 400 cap)
    # capturing the whole cluster matters: 6 modes >> 1 mode
    assert d6.n_iter < d1.n_iter

    # pre/post fork: post is never worse; on the incomplete (1-mode) subspace it
    # is decisively better, so post is the variant to keep.
    p1 = deflated_solve(apply, vbar, q1, method="pre", history=8, tol=tol, max_iter=cap)
    assert d1.n_iter <= p1.n_iter
