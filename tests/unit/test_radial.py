import numpy as np
import pytest
from scipy.integrate import simpson as scipy_simpson

from gradwave.pseudo.radial import sbt, simpson, sph_jl


def linear_mesh(n=601, rmax=6.0):
    r = np.linspace(0.0, rmax, n)
    rab = np.full(n, r[1] - r[0])
    return r, rab


def log_mesh(n=600, rmin=1e-5, rmax=8.0):
    # QE-style: r_i = rmin/Z * exp(i*dx) form; here plain geometric mesh
    i = np.arange(n)
    dx = np.log(rmax / rmin) / (n - 1)
    r = rmin * np.exp(i * dx)
    rab = r * dx  # dr/di
    return r, rab


@pytest.mark.parametrize(("mesh", "tol"), [(linear_mesh, 1e-9), (log_mesh, 1e-5)])
def test_simpson_against_analytic(mesh, tol):
    # ∫₀^R e^{-r} r² dr = 2 − e^{-R}(R² + 2R + 2); tolerance set by each
    # mesh's own Simpson truncation error, cross-checked against scipy.
    r, rab = mesh()
    f = np.exp(-r) * r**2
    ours = simpson(f, rab)
    R = r[-1]
    exact = 2.0 - np.exp(-R) * (R**2 + 2 * R + 2)
    assert abs(ours - exact) < tol
    assert abs(ours - scipy_simpson(f, x=r)) < 10 * tol


def test_simpson_even_point_count():
    r, rab = linear_mesh(n=602)
    f = r**3
    assert abs(simpson(f, rab) - r[-1] ** 4 / 4) < 1e-8


def test_sph_jl_small_argument():
    # j1 via sin/cos suffers total cancellation at small x; the series must not.
    x = np.array([1e-10, 1e-8, 1e-6, 1e-4])
    j1 = sph_jl(1, x)
    assert np.allclose(j1, x / 3.0, rtol=1e-13)
    j2 = sph_jl(2, x)
    assert np.allclose(j2, x**2 / 15.0, rtol=1e-12)
    # continuity across the series/trig switch at x = 2 (gap small enough
    # that the function's own variation j'·Δx ~ 1e-13 is below tolerance)
    for l in range(4):
        below, above = sph_jl(l, np.array([2.0 - 1e-13])), sph_jl(l, np.array([2.0 + 1e-13]))
        assert abs(below - above) < 1e-12
    # spot values against mpmath-grade references (sin/cos closed forms at safe x)
    xr = np.array([3.7])
    assert np.isclose(sph_jl(0, xr)[0], np.sin(3.7) / 3.7, rtol=1e-15)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_sbt_gaussian_identity(l):
    # ∫₀^∞ e^{-a r²} j_l(qr) r^{l+2} dr = q^l √π / (2^{l+2} a^{l+3/2}) · e^{-q²/4a}
    a = 1.7
    r, rab = linear_mesh(n=2001, rmax=10.0)
    q = np.array([1e-8, 0.1, 0.5, 1.0, 2.0, 5.0])
    g = np.exp(-a * r**2) * r ** (l + 2)
    ours = sbt(l, g, r, rab, q)
    ref = q**l * np.sqrt(np.pi) / (2 ** (l + 2) * a ** (l + 1.5)) * np.exp(-(q**2) / (4 * a))
    assert np.allclose(ours, ref, rtol=1e-8, atol=1e-14)
