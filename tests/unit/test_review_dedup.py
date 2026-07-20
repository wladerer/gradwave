"""Guards for the dedup wave: shared constants and cross-implementation parity
so the pieces that were unified cannot silently drift apart again.

- pseudo/_bessel_data.py holds the one set of small-argument series parameters
  and double-factorials that both spherical-Bessel evaluators (radial numpy,
  radial_torch) now import; numpy-vs-torch parity is pinned across l and a grid,
  and vloc_of_g numpy-vs-torch on real pseudopotentials.
- postscf/hessian.py reuses phonons.py's single derived cm⁻¹ constant, so the FD
  and analytic Γ-frequency routes share one value to the last digit.
- solvers/_ms.py is the one draft→polish skeleton behind davidson_batched_ms and
  chebyshev_filtered_batched_ms (skip predicate + full-precision parity).
- davidson single-k vs batched use compensating conjugation conventions; they
  must land on the same eigenvalues.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.pseudo import _bessel_data, radial, radial_torch
from gradwave.pseudo.local import vloc_of_g
from gradwave.pseudo.radial_torch import RadialTables, jl_t
from gradwave.pseudo.upf import parse_upf

PSE = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


# --------------------------------------------------------------------------- #
# Finding 1: shared Bessel constants + numpy/torch parity
# --------------------------------------------------------------------------- #
def test_bessel_constants_are_shared_objects():
    # both evaluators bind the SAME constants — no per-module copy to drift
    assert radial.SERIES_X is _bessel_data.SERIES_X
    assert radial.SERIES_TERMS is _bessel_data.SERIES_TERMS
    assert radial.DOUBLE_FACTORIAL is _bessel_data.DOUBLE_FACTORIAL
    assert radial_torch.SERIES_X is _bessel_data.SERIES_X
    assert radial_torch.SERIES_TERMS is _bessel_data.SERIES_TERMS
    assert radial_torch.DOUBLE_FACTORIAL is _bessel_data.DOUBLE_FACTORIAL
    # the unified (safer) pair
    assert _bessel_data.SERIES_X == 4.0
    assert _bessel_data.SERIES_TERMS == 40


def test_jl_numpy_torch_parity_grid():
    # dense grid straddling the series/trig crossover and out into the tail
    x = np.concatenate([
        np.linspace(0.0, 60.0, 60001),
        [1e-10, 3.999999, 4.0, 4.000001],
    ])
    xt = torch.from_numpy(x)
    for l in range(5):
        ref = radial.sph_jl(l, x)
        got = jl_t(l, xt).numpy()
        assert np.abs(got - ref).max() < 1e-14, l


def test_jl_series_matches_trig_at_crossover():
    # just below SERIES_X sph_jl takes the series branch; it must agree with the
    # closed trig form evaluated at the SAME point (continuity of the seam)
    x = 4.0 - 1e-6
    series = radial.sph_jl(2, np.array([x]))[0]
    trig = (3.0 / x**3 - 1.0 / x) * np.sin(x) - 3.0 / x**2 * np.cos(x)
    assert abs(float(series) - trig) < 1e-13


def test_numpy_l_range_guard():
    with pytest.raises(ValueError):
        radial.sph_jl(5, np.array([1.0]))  # numpy path only supports l ≤ 4


def test_vloc_of_g_numpy_torch_parity():
    for name in ("Si_ONCV_PBE-1.2.upf", "PD_Ni_PBE.upf"):
        upf = parse_upf(PSE / name)
        q = np.linspace(0.05, 15.0, 53)
        ref = vloc_of_g(upf, q)
        got = RadialTables(upf).vloc_of_g(torch.tensor(q)).numpy()
        assert np.abs(got - ref).max() < 1e-11, name


# --------------------------------------------------------------------------- #
# Finding 2: hessian.py reuses phonons.py's derived cm⁻¹ constant
# --------------------------------------------------------------------------- #
def test_hessian_reuses_phonons_constant():
    from gradwave.postscf import hessian, phonons

    # the FD route's constant IS the analytic route's derived value (identity,
    # not merely close) — the 8th-digit literal is gone
    assert hessian.SQRT_EV_AMU_ANG2_TO_CM1 == phonons._SQRT_EV_AMU_ANG2_TO_CM1
    assert hessian.SQRT_EV_AMU_ANG2_TO_CM1 == pytest.approx(
        521.4708983727718, rel=1e-12)


def test_gamma_phonons_matches_gamma_frequencies():
    from gradwave.postscf.hessian import gamma_phonons
    from gradwave.postscf.phonons import gamma_frequencies

    rng = np.random.default_rng(0)
    na = 3
    a = rng.standard_normal((3 * na, 3 * na))
    phi = a + a.T  # symmetric force constants
    masses = np.array([12.0, 15.999, 1.008])

    f_fd = gamma_phonons(phi, masses)
    f_an = gamma_frequencies(phi.reshape(na, 3, na, 3), masses)
    assert np.allclose(np.sort(f_fd), np.sort(f_an), rtol=1e-12, atol=1e-10)


# --------------------------------------------------------------------------- #
# Findings 4 & 5: shared draft→polish scaffolding; conjugation conventions
# --------------------------------------------------------------------------- #
def _hermitian(m, seed=7):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, m, generator=g, dtype=torch.float64)
         + 1j * torch.randn(m, m, generator=g, dtype=torch.float64))
    a = a + a.conj().T
    a = a + m * torch.eye(m, dtype=torch.complex128)  # push spectrum positive
    return a


def _batched_apply(a):
    at = a.transpose(-1, -2)  # (Hψ)_g = Σ_g' A_gg' ψ_g'  →  v @ Aᵀ
    return lambda v: v @ at.to(v.dtype)


def test_ms_skip_predicate_bitexact():
    from gradwave.solvers.chebyshev import (
        chebyshev_filtered_batched,
        chebyshev_filtered_batched_ms,
    )
    from gradwave.solvers.davidson import davidson_batched, davidson_batched_ms

    m, nb = 14, 3
    a = _hermitian(m)
    h = _batched_apply(a)
    x0 = torch.view_as_complex(torch.randn(1, nb, m, 2, dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(3)))
    t = torch.ones(1, m, dtype=torch.float64)
    mask = torch.ones(1, m, dtype=torch.bool)

    # mixed_precision off → identical to the plain batched solver
    base = davidson_batched(h, x0, t, mask, tol=1e-10)
    off = davidson_batched_ms(h, x0, t, mask, tol=1e-10, mixed_precision=False)
    assert torch.equal(base.eigenvalues, off.eigenvalues)

    cbase = chebyshev_filtered_batched(h, x0, t, mask, tol=1e-10)
    # crossover ≥ tol also skips the draft
    coff = chebyshev_filtered_batched_ms(h, x0, t, mask, tol=1e-10,
                                         crossover=1e-3)
    assert torch.equal(cbase.eigenvalues, coff.eigenvalues)


def test_ms_full_precision_result():
    from gradwave.solvers.chebyshev import chebyshev_filtered_batched_ms
    from gradwave.solvers.davidson import davidson_batched_ms

    m, nb = 16, 4
    a = _hermitian(m, seed=11)
    ref = torch.linalg.eigvalsh(a)[:nb]
    h = _batched_apply(a)
    x0 = torch.view_as_complex(torch.randn(1, nb, m, 2, dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(5)))
    t = torch.ones(1, m, dtype=torch.float64)
    mask = torch.ones(1, m, dtype=torch.bool)

    d = davidson_batched_ms(h, x0, t, mask, tol=1e-9, crossover=1e-4)
    assert torch.allclose(d.eigenvalues[0], ref, atol=1e-7)

    c = chebyshev_filtered_batched_ms(h, x0, t, mask, tol=1e-9, crossover=1e-4,
                                      degree=12)
    assert torch.allclose(c.eigenvalues[0], ref, atol=1e-7)


def test_davidson_singlek_batched_conjugation_agree():
    # single-k davidson() and davidson_batched() build the Rayleigh-Ritz matrix
    # with OPPOSITE conjugation conventions (both correct); they must converge
    # to the same eigenvalues on a k=1 problem.
    from gradwave.solvers.davidson import davidson, davidson_batched

    m, nb = 18, 4
    a = _hermitian(m, seed=17)
    ref = torch.linalg.eigvalsh(a)[:nb]
    at = a.transpose(-1, -2)
    x0 = torch.view_as_complex(torch.randn(nb, m, 2, dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(9)))
    t_g = torch.ones(m, dtype=torch.float64)

    sk = davidson(lambda v: v @ at, x0, t_g, tol=1e-10)
    bk = davidson_batched(lambda v: v @ at, x0[None], t_g[None],
                          torch.ones(1, m, dtype=torch.bool), tol=1e-10)
    assert torch.allclose(sk.eigenvalues, ref, atol=1e-8)
    assert torch.allclose(sk.eigenvalues, bk.eigenvalues[0], atol=1e-8)
