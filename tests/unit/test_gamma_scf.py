"""The Γ-point real-wavefunction path wired into the SCF loop.

`core/gamma.py` (GammaBasis / GammaHamiltonian / davidson_gamma) is validated
against the complex machinery at the operator level in `test_gamma.py`. This
module gates the *SCF integration*: a single-Γ calculation routed through the
real half-sphere solve must reproduce the complex path to machine precision
(total energy, eigenvalues, density), and the eligibility gate
(`_resolve_gamma_real`) must fall back to the complex path for every case that
is not provably safe.

Selection is via `GRADWAVE_GAMMA_REAL` (auto|1|0), read into the module global
`gradwave.scf.loop._GAMMA_REAL_ENV` at import — the tests monkeypatch that
global (env vars are captured once at import, so patching os.environ would not
take effect).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import gradwave.scf.loop as loop
from gradwave.core.xc.pbe import PBE
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
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", mode)
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


def test_auto_engages_on_gamma(o2_system, monkeypatch):
    """`auto` (the default) picks the real path for an eligible Γ calculation."""
    res = _run(o2_system, monkeypatch, "auto")
    assert res.converged
    assert res.gamma_real is True


def test_disabled_env_uses_complex(o2_system, monkeypatch):
    """`0` keeps the complex path even on an eligible Γ system."""
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "0")
    gb = loop._resolve_gamma_real(o2_system, PBE(), None, None, False, None)
    assert gb is None


def test_gate_falls_back_for_multi_k(monkeypatch):
    """A multi-k calculation is ineligible: `auto` falls back, `1` raises."""
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    a = 6.0
    cell = np.diag([a, a, a])
    pos = np.array([[0.0, 0.0, 0.0]])
    multik = setup_system(cell, pos, [0], [upf], ecut=16 * RY, kmesh=(2, 1, 1),
                          nbands=6, use_symmetry=False)
    assert len(multik.spheres) > 1  # genuinely multi-k

    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "auto")
    assert loop._resolve_gamma_real(multik, PBE(), None, None, False, None) is None

    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "1")
    with pytest.raises(ValueError, match="k-points"):
        loop._resolve_gamma_real(multik, PBE(), None, None, False, None)


def test_gate_falls_back_for_blockers(o2_system, monkeypatch):
    """Operators the real GammaHamiltonian does not implement disqualify it:
    `auto` falls back, `1` raises with the blocker named."""
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "auto")
    # fp32 mixed-precision draft
    assert loop._resolve_gamma_real(o2_system, PBE(), None, None, True, None) is None

    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "1")
    with pytest.raises(ValueError, match="mixed-precision"):
        loop._resolve_gamma_real(o2_system, PBE(), None, None, True, None)


def test_gate_builds_gamma_basis(o2_system, monkeypatch):
    """On an eligible Γ system the gate returns a half-sphere basis roughly
    half the size of the full sphere (the memory lever)."""
    monkeypatch.setattr(loop, "_GAMMA_REAL_ENV", "auto")
    gb = loop._resolve_gamma_real(o2_system, PBE(), None, None, False, None)
    assert gb is not None
    assert gb.nhalf == (gb.npw + 1) // 2
    assert gb.npw == o2_system.spheres[0].npw
