"""Phase 1 of the phase-diagram builder: quasi-harmonic single-phase
thermodynamics. Validated on a synthetic Einstein model with a known mode
Grueneisen, so no SCF runs here. The static energy is a spring in volume and the
phonon frequency scales as omega(V) = omega0 (V0/V)^gamma, for which the
thermodynamic Grueneisen alpha B V / Cv equals gamma at every temperature."""

import numpy as np

from gradwave.postscf.qha import qha


def _einstein(omega_cm, n_modes):
    grid = np.linspace(1.0, 3 * omega_cm, 6000)
    sigma = omega_cm * 0.004
    d = np.exp(-0.5 * ((grid - omega_cm) / sigma) ** 2)
    d *= n_modes / np.trapezoid(d, grid)
    return grid, d


def _model(gamma, v0=20.0, k=2.0, omega0=400.0, n_modes=6.0):
    vols = np.linspace(0.90 * v0, 1.12 * v0, 11)
    energies = 0.5 * k * (vols - v0) ** 2  # static well, minimum at v0
    dos = [_einstein(omega0 * (v0 / v) ** gamma, n_modes) for v in vols]
    return vols, energies, dos


def test_qha_recovers_gruneisen():
    gamma_in = 2.0
    vols, energies, dos = _model(gamma_in)
    temps = np.linspace(50, 900, 40)
    r = qha(vols, energies, dos, temps)

    # volume expands with temperature and alpha is positive
    assert r.volume[-1] > r.volume[0]
    assert np.all(r.thermal_expansion()[3:] > 0)
    # the thermodynamic Grueneisen recovers the input mode Grueneisen
    mid = (temps > 350) & (temps < 750)
    assert np.allclose(r.gruneisen()[mid], gamma_in, rtol=0.02), r.gruneisen()[mid]
    # Cp exceeds Cv once there is thermal expansion
    assert np.all(r.heat_capacity_p()[5:] >= r.cv[5:])


def test_qha_zero_gruneisen_has_no_expansion():
    # volume-independent frequencies mean no thermal expansion
    vols, energies, dos = _model(0.0)
    temps = np.linspace(50, 900, 30)
    r = qha(vols, energies, dos, temps)
    assert np.all(np.abs(r.thermal_expansion()) < 1e-7)
    assert abs(r.volume[-1] - r.volume[0]) < 1e-3
