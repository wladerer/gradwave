"""Per-block adaptive damping in PulayMixer.

A linear fixed-point map with one expansive block diverges under plain
damping at fixed alpha; the adaptive multiplier must detect the growing
block from its residual history and cut its step until the damped gain
falls below one, without touching the stable block. history=1 forces the
pure damped path, isolating the adaptation from DIIS."""

import torch

from gradwave.scf.mixing import PulayMixer


def _run(adapt):
    n = 8
    g2 = torch.linspace(0.0, 4.0, 2 * n, dtype=torch.float64)
    lam = torch.cat([
        torch.full((n,), 0.5, dtype=torch.float64),   # stable block
        torch.full((n,), -2.5, dtype=torch.float64),  # expansive under 0.7
    ]).to(torch.complex128)
    target = torch.ones(2 * n, dtype=torch.complex128)
    ids = torch.cat([torch.zeros(n, dtype=torch.int64),
                     torch.ones(n, dtype=torch.int64)])
    mixer = PulayMixer(g2, alpha=0.7, history=1, check_g0=False,
                       adapt_blocks=ids if adapt else None)
    x = torch.zeros(2 * n, dtype=torch.complex128)
    for _ in range(200):
        out = lam * (x - target) + target  # fixed point at `target`
        x = mixer.step(x, out)
        if not torch.isfinite(x).all() or float(x.abs().max()) > 1e12:
            return None
    return float((x - target).abs().max())


def test_adaptive_converges_where_plain_damping_diverges():
    plain = _run(adapt=False)
    assert plain is None or plain > 1.0, "premise: plain damping must fail"
    # the multiplier recovery rule hovers around unit gain in this pure
    # damped setting, so the tail is slow; DIIS owns the tail in real runs
    adapted = _run(adapt=True)
    assert adapted is not None and adapted < 1e-4, f"residual {adapted}"


def test_stable_run_is_untouched():
    """With no expansive block the multipliers never move and the
    trajectory is identical to the non-adaptive mixer."""
    n = 8
    g2 = torch.linspace(0.0, 4.0, n, dtype=torch.float64)
    lam = torch.full((n,), 0.4, dtype=torch.complex128)
    target = torch.ones(n, dtype=torch.complex128)
    xs = []
    for adapt in (False, True):
        mixer = PulayMixer(g2, alpha=0.7, history=4, check_g0=False,
                           adapt_blocks=(torch.zeros(n, dtype=torch.int64)
                                         if adapt else None))
        x = torch.zeros(n, dtype=torch.complex128)
        for _ in range(30):
            x = mixer.step(x, lam * (x - target) + target)
        xs.append(x)
    assert torch.equal(xs[0], xs[1])


def test_broyden_converges_expansive_map_near_newton():
    """Broyden-II on the same expansive linear map: the secant updates
    capture the per-direction gains, so it converges where plain damping
    diverges, in a handful of iterations (near-Newton for a linear map)."""
    from gradwave.scf.mixing import BroydenMixer

    n = 8
    g2 = torch.linspace(0.0, 4.0, 2 * n, dtype=torch.float64)
    lam = torch.cat([
        torch.full((n,), 0.5, dtype=torch.float64),
        torch.full((n,), -2.5, dtype=torch.float64),
    ]).to(torch.complex128)
    target = torch.ones(2 * n, dtype=torch.complex128)
    mixer = BroydenMixer(g2, alpha=0.7, history=8, check_g0=False)
    x = torch.zeros(2 * n, dtype=torch.complex128)
    for it in range(30):  # noqa: B007 — `it` is checked after the loop
        out = lam * (x - target) + target
        x = mixer.step(x, out)
        assert torch.isfinite(x).all()
        if float((x - target).abs().max()) < 1e-10:
            break
    assert float((x - target).abs().max()) < 1e-10, "did not converge"
    assert it < 15, f"took {it + 1} iterations (expected near-Newton)"


def test_johnson_converges_expansive_map():
    """Johnson's weighted modified Broyden (the QE scheme) on the same
    expansive map: pair normalization + w0 regularization must converge it
    from the plain-damping-divergent regime."""
    from gradwave.scf.mixing import JohnsonMixer

    n = 8
    g2 = torch.linspace(0.0, 4.0, 2 * n, dtype=torch.float64)
    lam = torch.cat([
        torch.full((n,), 0.5, dtype=torch.float64),
        torch.full((n,), -2.5, dtype=torch.float64),
    ]).to(torch.complex128)
    target = torch.ones(2 * n, dtype=torch.complex128)
    mixer = JohnsonMixer(g2, alpha=0.7, history=8, check_g0=False)
    x = torch.zeros(2 * n, dtype=torch.complex128)
    for _ in range(40):
        x = mixer.step(x, lam * (x - target) + target)
        assert torch.isfinite(x).all()
    assert float((x - target).abs().max()) < 1e-8
