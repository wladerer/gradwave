"""Harmonic thermodynamics against closed-form Einstein and Sommerfeld limits.

The phonon quantities are exercised on a synthetic Einstein DOS, a narrow
Gaussian spike of total weight 3N at a single frequency ω0, whose Cv, ZPE, and
free energy all have textbook forms. The electronic term is checked against its
own flat-DOS closed form. No SCF runs here, so the whole file is a fast unit
test.
"""

import numpy as np
import pytest

from gradwave.postscf import thermo
from gradwave.postscf.thermo import CM1_TO_EV, K_B


def _einstein_dos(omega0_cm=200.0, n_modes=6, width=1.0, npoints=20001):
    """Narrow Gaussian at ω0 with ∫g dω = n_modes, standing in for 3N δ(ω−ω0).

    The width is small against ω0 so the spike behaves like a delta for the
    thermodynamic integrals, and the fine grid keeps the discretized moment close
    to the analytic one.
    """
    grid = np.linspace(0.0, 5.0 * omega0_cm, npoints)
    g = np.exp(-0.5 * ((grid - omega0_cm) / width) ** 2)
    g *= n_modes / np.trapezoid(g, grid)
    return grid, g


def test_mode_count_recovers_3n():
    grid, g = _einstein_dos(n_modes=6)
    assert thermo.mode_count(grid, g) == pytest.approx(6.0, rel=1e-4)


def test_dulong_petit_high_T():
    # k_B T ≫ ħω0: every Einstein kernel saturates to 1, Cv → 3N·k_B.
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    cv_over_kb = thermo.heat_capacity_in_kB(grid, g, T=20000.0)
    assert cv_over_kb == pytest.approx(6.0, rel=0.02)


def test_heat_capacity_vanishes_at_low_T():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    assert thermo.heat_capacity(grid, g, T=1.0) == pytest.approx(0.0, abs=1e-30)
    assert thermo.heat_capacity(grid, g, T=0.0) == 0.0


def test_zero_point_energy_einstein():
    # ZPE = ½·(3N)·ħω0 for a spike at ω0.
    omega0_cm, n_modes = 200.0, 6
    grid, g = _einstein_dos(omega0_cm=omega0_cm, n_modes=n_modes)
    expected = 0.5 * n_modes * omega0_cm * CM1_TO_EV
    assert thermo.zero_point_energy(grid, g) == pytest.approx(expected, rel=1e-4)
    assert thermo.zero_point_energy(grid, g) > 0.0


def test_entropy_nonnegative_and_monotonic():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    temps = [50.0, 100.0, 200.0, 400.0, 800.0]
    s = [thermo.entropy(grid, g, T) for T in temps]
    assert all(v >= 0.0 for v in s)
    assert all(b > a for a, b in zip(s, s[1:], strict=False))
    assert thermo.entropy(grid, g, T=0.0) == 0.0


def test_free_energy_matches_u_minus_ts():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    for T in (25.0, 150.0, 300.0, 900.0):
        u = thermo.internal_energy_vib(grid, g, T)
        s = thermo.entropy(grid, g, T)
        f = thermo.free_energy_vib(grid, g, T)
        assert f == pytest.approx(u - T * s, abs=1e-8)


def test_debye_temperature_positive():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    theta = thermo.debye_temperature(grid, g)
    # A spike near ω0 gives θ_D ≈ √(5/3)·ħω0/k_B, so it must be finite and > 0.
    assert theta > 0.0
    expected = np.sqrt(5.0 / 3.0) * 200.0 * CM1_TO_EV / K_B
    assert theta == pytest.approx(expected, rel=0.02)


def test_electronic_heat_capacity_linear_flat_dos():
    # Flat DOS g(E_F) = g0: Cel = (π²/3)k_B²·g0·T, exactly linear in T.
    energy = np.linspace(-10.0, 10.0, 501)
    g0 = 3.5
    states = np.full_like(energy, g0)
    e_fermi = 0.0
    for T in (10.0, 100.0, 300.0):
        expected = (np.pi ** 2 / 3.0) * K_B ** 2 * g0 * T
        got = thermo.electronic_heat_capacity(energy, states, e_fermi, T)
        assert got == pytest.approx(expected, rel=1e-10)
    # Linearity: doubling T doubles Cel.
    c1 = thermo.electronic_heat_capacity(energy, states, e_fermi, 100.0)
    c2 = thermo.electronic_heat_capacity(energy, states, e_fermi, 200.0)
    assert c2 == pytest.approx(2.0 * c1, rel=1e-12)


def test_electronic_heat_capacity_spin_resolved_sums_channels():
    energy = np.linspace(-10.0, 10.0, 501)
    states_1d = np.full_like(energy, 4.0)
    states_2d = np.stack([np.full_like(energy, 2.0), np.full_like(energy, 2.0)])
    T = 100.0
    assert thermo.electronic_heat_capacity(energy, states_2d, 0.0, T) == pytest.approx(
        thermo.electronic_heat_capacity(energy, states_1d, 0.0, T), rel=1e-12
    )


def test_thermo_table_shape_and_scalars():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    temps = (0.0, 100.0, 300.0, 1000.0)
    t = thermo.thermo_table(grid, g, temps)
    # scalars
    assert t["mode_count"] == pytest.approx(6.0, rel=1e-4)
    assert t["zero_point_energy_eV"] > 0.0
    assert t["debye_temperature_K"] > 0.0
    # each T-dependent series matches the requested temperatures elementwise
    assert t["temperatures_K"] == [0.0, 100.0, 300.0, 1000.0]
    for key in ("free_energy_eV", "internal_energy_eV", "heat_capacity_eV_K", "entropy_eV_K"):
        assert len(t[key]) == len(temps)


def test_thermo_table_matches_single_quantity_functions():
    grid, g = _einstein_dos(omega0_cm=250.0, n_modes=6)
    temps = (0.0, 150.0, 600.0)
    t = thermo.thermo_table(grid, g, temps)
    assert t["zero_point_energy_eV"] == pytest.approx(thermo.zero_point_energy(grid, g))
    for i, T in enumerate(temps):
        assert t["free_energy_eV"][i] == pytest.approx(thermo.free_energy_vib(grid, g, T))
        assert t["internal_energy_eV"][i] == pytest.approx(thermo.internal_energy_vib(grid, g, T))
        assert t["heat_capacity_eV_K"][i] == pytest.approx(thermo.heat_capacity(grid, g, T))
        assert t["entropy_eV_K"][i] == pytest.approx(thermo.entropy(grid, g, T))


def test_thermo_table_thermodynamic_identities():
    grid, g = _einstein_dos(omega0_cm=200.0, n_modes=6)
    temps = (0.0, 100.0, 300.0, 800.0)
    t = thermo.thermo_table(grid, g, temps)
    for i, T in enumerate(temps):
        # F = U − T·S holds to rounding (shared discretization)
        f, u, s = t["free_energy_eV"][i], t["internal_energy_eV"][i], t["entropy_eV_K"][i]
        assert f == pytest.approx(u - T * s, abs=1e-9)
    # T=0: U = ZPE, S = 0, Cv = 0
    assert t["internal_energy_eV"][0] == pytest.approx(t["zero_point_energy_eV"])
    assert t["entropy_eV_K"][0] == 0.0
    assert t["heat_capacity_eV_K"][0] == 0.0
    # Dulong–Petit: Cv/k_B → 3N at high T
    assert t["heat_capacity_eV_K"][-1] / K_B == pytest.approx(6.0, rel=0.05)
