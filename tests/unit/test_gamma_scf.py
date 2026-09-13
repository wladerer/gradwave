"""The Γ-point real-wavefunction path wired into the SCF loop.

`core/gamma.py` (GammaBasis / GammaHamiltonian / davidson_gamma) is validated
against the complex machinery at the operator level in `test_gamma.py`. This
module gates the *SCF integration*: a single-Γ calculation routed through the
real half-sphere solve must reproduce the complex path to machine precision
(total energy, eigenvalues, density), and the eligibility gate
(`_resolve_gamma_real`) must fall back to the complex path for every case that
is not provably safe.

Selection is via `GRADWAVE_GAMMA_REAL` (auto|1|0), read per call by
`gradwave.scf.loop._gamma_real_mode` — the tests set the env var directly
(the former import-frozen module global is gone).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import gradwave.scf.loop as loop
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(scope="module")
def o2_system():
    """O2 molecule in a box — a single Γ k-point, NC ONCV pseudo (eligible)."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    a = 8.0
    d = 1.21
    cell = np.diag([a, a, a])
    pos = np.array([[a / 2, a / 2, a / 2 - d / 2], [a / 2, a / 2, a / 2 + d / 2]])
    return setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY, kmesh=(1, 1, 1), nbands=8)


def _run(system, monkeypatch, mode):
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", mode)
    return scf(system, PBE(), smearing="gaussian", width=0.2, etol=1e-9,
               rhotol=1e-8, verbose=False, max_iter=80)


def test_scf_energy_matches_complex(o2_system, monkeypatch):
    """A full Γ-only SCF through the real path reproduces the complex path to
    machine precision — the correctness bar for the whole wiring."""
    res_c = _run(o2_system, monkeypatch, "0")  # complex path
    res_g = _run(o2_system, monkeypatch, "1")  # real Γ path (forced)

    assert res_c.converged and res_g.converged
    assert res_c.gamma_real is False
    assert res_g.gamma_real is True

    e_c = float(res_c.energies.total)
    e_g = float(res_g.energies.total)
    # both fixed points are the SAME density, so the totals must agree far
    # tighter than the SCF tolerance — machine precision on a ~-30 eV total.
    assert abs(e_g - e_c) / abs(e_c) < 1e-10, (e_g, e_c)

    # eigenvalues and Fermi level match to eigensolver round-off
    assert float((res_g.eigenvalues - res_c.eigenvalues).abs().max()) < 1e-7
    assert abs(res_g.fermi - res_c.fermi) < 1e-7
    # the converged densities are pointwise identical (gauge-invariant |ψ|²)
    assert float((res_g.rho - res_c.rho).abs().max()) < 1e-8


@pytest.fixture(scope="module")
def ni_fm_system():
    """FM fcc Ni at Γ, run as collinear nspin=2. Time-reversal maps ↑→↓, but
    WITHIN each spin channel V_σ(r) is real (the antiunitary is plain conjugation
    K, K²=+1), so each channel is solved on its own real half sphere. This pins
    that per-spin realification. (The Kramers K²=−1 obstruction is spinor-only
    — noncollinear/SOC — and never reaches this gate; see _resolve_gamma_real.)"""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "PD_Ni_PBE.upf")
    a = 3.52
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    return setup_system(cell, np.zeros((1, 3)), [0], [upf], ecut=45 * RY,
                        kmesh=(1, 1, 1), nbands=14, time_reversal=False)


def test_magnetic_nspin2_matches_complex(ni_fm_system, monkeypatch):
    """Collinear nspin=2 is SAFE: the real path solves each spin channel on its
    own real half sphere and reproduces the complex path to machine precision
    ([D-020]). Measured on FM Ni: rel ΔE = 0, moments identical."""
    kw = dict(nspin=2, start_mag=[0.5], smearing="gaussian", width=0.1,
              kerker=True, etol=1e-9, rhotol=1e-8, verbose=False, max_iter=200)
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "0")
    res_c = scf(ni_fm_system, LSDA_PW92(), **kw)
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "1")
    res_g = scf(ni_fm_system, LSDA_PW92(), **kw)

    assert res_c.converged and res_g.converged
    assert res_c.gamma_real is False
    assert res_g.gamma_real is True  # nspin=2 genuinely engages the real path
    e_c, e_g = float(res_c.energies.total), float(res_g.energies.total)
    assert abs(e_g - e_c) / abs(e_c) < 1e-9, (e_g, e_c)
    assert abs(float(res_g.mag_total) - float(res_c.mag_total)) < 1e-6
    assert float((res_g.eigenvalues - res_c.eigenvalues).abs().max()) < 1e-6
    assert float((res_g.rho - res_c.rho).abs().max()) < 1e-8


def test_auto_engages_on_gamma(o2_system, monkeypatch):
    """`auto` (the default) picks the real path for an eligible Γ calculation."""
    res = _run(o2_system, monkeypatch, "auto")
    assert res.converged
    assert res.gamma_real is True


def test_disabled_env_uses_complex(o2_system, monkeypatch):
    """`0` keeps the complex path even on an eligible Γ system."""
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "0")
    gb = loop._resolve_gamma_real(o2_system, PBE(), None, None, False, None)
    assert gb is None


def test_default_is_auto(o2_system, monkeypatch):
    """The path is ON BY DEFAULT: an UNSET GRADWAVE_GAMMA_REAL resolves to
    "auto" ([D-020], superseding [D-009]), so an eligible Γ-only run takes the
    real path. This guards the default against silently regressing to off."""
    monkeypatch.delenv("GRADWAVE_GAMMA_REAL", raising=False)
    assert loop._gamma_real_mode() == "auto"
    # and it actually engages on an eligible Γ system with the env var unset
    gb = loop._resolve_gamma_real(o2_system, PBE(), None, None, False, None)
    assert gb is not None


def test_gate_falls_back_for_multi_k(monkeypatch):
    """A multi-k calculation is ineligible: `auto` falls back, `1` raises."""
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    a = 6.0
    cell = np.diag([a, a, a])
    pos = np.array([[0.0, 0.0, 0.0]])
    multik = setup_system(cell, pos, [0], [upf], ecut=16 * RY, kmesh=(2, 1, 1),
                          nbands=6, use_symmetry=False)
    assert len(multik.spheres) > 1  # genuinely multi-k

    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "auto")
    assert loop._resolve_gamma_real(multik, PBE(), None, None, False, None) is None

    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "1")
    with pytest.raises(ValueError, match="k-points"):
        loop._resolve_gamma_real(multik, PBE(), None, None, False, None)


def test_gate_falls_back_for_blockers(o2_system, monkeypatch):
    """Operators the real GammaHamiltonian does not implement disqualify it:
    `auto` falls back, `1` raises with the blocker named."""
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "auto")
    # fp32 mixed-precision draft
    assert loop._resolve_gamma_real(o2_system, PBE(), None, None, True, None) is None

    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "1")
    with pytest.raises(ValueError, match="mixed-precision"):
        loop._resolve_gamma_real(o2_system, PBE(), None, None, True, None)


def test_gate_builds_gamma_basis(o2_system, monkeypatch):
    """On an eligible Γ system the gate returns a half-sphere basis roughly
    half the size of the full sphere (the memory lever)."""
    monkeypatch.setenv("GRADWAVE_GAMMA_REAL", "auto")
    gb = loop._resolve_gamma_real(o2_system, PBE(), None, None, False, None)
    assert gb is not None
    assert gb.nhalf == (gb.npw + 1) // 2
    assert gb.npw == o2_system.spheres[0].npw
