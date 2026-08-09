"""Work function / electrode potential extraction (postscf/work_function.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gradwave.postscf.work_function import (
    U_SHE_ABS,
    electrode_potential,
    vacuum_level,
    work_function,
)


def _slab(v_lo, v_hi, e_f, nz=120):
    """A synthetic slab: density localized in the middle; the effective potential
    dips inside the slab and plateaus at v_lo / v_hi in the two vacuum regions."""
    z = torch.arange(nz, dtype=torch.float64)
    rho1 = torch.exp(-((z - nz / 2) ** 2) / (2 * 8.0**2))
    # left plateau v_lo, right plateau v_hi, a dip through the slab
    ramp = 0.5 * (1 + torch.tanh((z - nz / 2) / 6.0))  # 0 → 1
    v1 = v_lo + (v_hi - v_lo) * ramp - 3.0 * rho1  # dip where the density is

    def to3(a):
        return a.reshape(1, 1, nz).expand(6, 6, nz).clone()

    return SimpleNamespace(v_eff=to3(v1), rho=to3(rho1), fermi=e_f)


def test_symmetric_work_function_and_potentials():
    res = _slab(2.5, 2.5, e_f=0.3)
    assert vacuum_level(res) == pytest.approx(2.5, abs=1e-3)
    assert work_function(res) == pytest.approx(2.2, abs=1e-3)

    ep = electrode_potential(res, phi_pzc=2.0)
    assert ep.work_function == pytest.approx(2.2, abs=1e-3)
    assert ep.potential_vs_vacuum == pytest.approx(2.2, abs=1e-3)
    assert ep.potential_vs_she == pytest.approx(2.2 - U_SHE_ABS, abs=1e-3)
    assert ep.potential_vs_pzc == pytest.approx(0.2, abs=1e-3)  # Φ − Φ_PZC


def test_asymmetric_slab_two_faces():
    """A dipolar slab has two different face potentials."""
    res = _slab(2.0, 3.2, e_f=0.5)
    e_lo, e_hi = vacuum_level(res, both_faces=True)
    assert e_lo == pytest.approx(2.0, abs=2e-2)
    assert e_hi == pytest.approx(3.2, abs=2e-2)
    phi_lo, phi_hi = work_function(res, both_faces=True)
    assert phi_lo == pytest.approx(1.5, abs=2e-2)
    assert phi_hi == pytest.approx(2.7, abs=2e-2)
