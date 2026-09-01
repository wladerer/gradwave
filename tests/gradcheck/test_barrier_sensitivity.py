"""Barrier parameter sensitivity dE_a/dλ — the envelope-theorem correctness proof.

The headline claim: at a *converged* (stationary) geometry the single autograd
backward ∂E/∂λ (density fixed, geometry fixed) equals the total derivative
dE*/dλ of the RE-RELAXED-geometry energy — because the geometry-response term
∂E/∂R · dR/dλ vanishes at ∇_R E = 0. We prove it three ways:

* ``test_h2_envelope_theorem`` — a single stationary state (H₂, one bond DOF):
  ∂E/∂λ (fixed geometry) == central FD of E*(λ) over the *re-relaxed* bond. The
  bond length b*(λ) genuinely moves with λ (dR/dλ ≠ 0), so the match is a real
  test that the geometry term cancels, not a triviality.
* ``test_h2o_barrier_fd_matches_autograd`` — a genuine barrier (H₂O linear
  bending saddle vs bent minimum): dE_a/dλ = ∂E_TS/∂λ − ∂E_IS/∂λ in one backward
  pass per endpoint == central FD of the FULL barrier E_a(λ) with both endpoints
  re-relaxed at λ ± h. This is the flagship correctness proof.
* ``test_barrier_potential_sensitivity_arithmetic`` /
  ``test_grand_canonical_envelope`` — the grand-canonical dΩ_a/dµ = −ΔN path.

λ here is the LearnableX exchange coefficient ``raw_mu``; the same machinery
serves a Hubbard U via ``manifolds=``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import LearnableX, energy_param_grads
from gradwave.postscf.barrier import (
    barrier_parameter_sensitivity,
    barrier_potential_sensitivity,
    grand_potential,
)
from gradwave.pseudo.upf import parse_upf

RY = 13.605693122994
PSEUDOS = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
BOX = 8.0
ECUT = 30 * RY
CELL = np.diag([BOX, BOX, BOX])
# 3-atom molecules in vacuum: the density residual floors ~a few e-10 from
# vacuum-region noise, so hold rhotol a notch looser than the 2-atom H2 case
# (still far tighter than the geometry stationarity that gates the match) and
# allow more iterations for the harder open-shell-ish linear/stretched probes.
_MOL_KW = dict(rhotol=2e-9, max_iter=250)


def _h():
    return parse_upf(str(PSEUDOS / "H_ONCV_PBE-1.2.upf"))


def _o():
    return parse_upf(str(PSEUDOS / "O_ONCV_PBE-1.2.upf"))


def _make_xc(raw_mu_shift: float = 0.0) -> LearnableX:
    xc = LearnableX(kappa=0.65, mu=0.18)  # off-PBE, generic
    if raw_mu_shift:
        with torch.no_grad():
            dict(xc.named_parameters())["raw_mu"].add_(raw_mu_shift)
    return xc


def _run(pos, species, upfs, xc, *, rhotol: float = 1e-10, max_iter: int = 120):
    from gradwave.scf.loop import scf, setup_system

    sysx = setup_system(CELL, pos, species, upfs, ecut=ECUT, kmesh=(1, 1, 1))
    res = scf(sysx, xc, smearing="none", etol=1e-11, rhotol=rhotol,
              max_iter=max_iter, verbose=False)
    assert res.converged, f"|drho| floor above rhotol={rhotol:.0e}"
    return res


# ---------------------------------------------------------------- H2 (1 DOF)
def _h2_pos(b: float) -> np.ndarray:
    c = BOX / 2
    return np.array([[c - b / 2, c, c], [c + b / 2, c, c]])


def _relax_h2(xc, b0: float = 0.76) -> float:
    from scipy.optimize import minimize_scalar

    upfs = [_h(), _h()]

    def e(b):
        return float(_run(_h2_pos(b), [0, 0], upfs, xc).energies.total)

    r = minimize_scalar(e, bracket=(b0 * 0.92, b0, b0 * 1.08), method="brent",
                        options={"xtol": 1e-8})
    return float(r.x)


@pytest.mark.standard
def test_h2_envelope_theorem():
    """∂E/∂λ at fixed relaxed geometry == FD of the re-relaxed-geometry energy."""
    torch.set_num_threads(8)
    upfs = [_h(), _h()]
    xc0 = _make_xc()
    b_star = _relax_h2(xc0)
    res = _run(_h2_pos(b_star), [0, 0], upfs, xc0)
    ag = float(energy_param_grads(res, xc0)["raw_mu"])

    h = 2e-3
    vals = []
    for sign in (+1, -1):
        xc = _make_xc(sign * h)
        b = _relax_h2(xc)  # geometry RE-RELAXED at λ ± h
        vals.append(float(_run(_h2_pos(b), [0, 0], upfs, xc).energies.total))
    fd = (vals[0] - vals[1]) / (2 * h)
    # the geometry term ∂E/∂b · db/dλ must cancel to tight tol
    assert abs(fd - ag) < 5e-4 * max(1.0, abs(fd)), (ag, fd)


# ------------------------------------------------------ H2O bending barrier
def _h2o_bent(r: float, half_angle: float) -> np.ndarray:
    c = BOX / 2
    return np.array([
        [c, c, c],
        [c + r * np.cos(half_angle), c + r * np.sin(half_angle), c],
        [c + r * np.cos(half_angle), c - r * np.sin(half_angle), c],
    ])


def _relax_bent(xc):
    from scipy.optimize import minimize

    upfs = [_o(), _h()]

    def e(p):
        return float(_run(_h2o_bent(p[0], p[1]), [0, 1, 1], upfs, xc, **_MOL_KW)
                     .energies.total)

    r = minimize(e, np.array([0.97, np.radians(52.0)]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 300})
    return r.x, float(r.fun)


def _relax_linear(xc, r0: float = 0.97):
    from scipy.optimize import minimize_scalar

    upfs = [_o(), _h()]

    def e(r):
        return float(_run(_h2o_bent(r, np.pi / 2), [0, 1, 1], upfs, xc, **_MOL_KW)
                     .energies.total)

    r = minimize_scalar(e, bracket=(r0 * 0.92, r0, r0 * 1.08), method="brent",
                        options={"xtol": 1e-8})
    return float(r.x), float(r.fun)


def _h2o_barrier(xc):
    upfs = [_o(), _h()]
    p_min, _ = _relax_bent(xc)
    r_ts, _ = _relax_linear(xc)
    res_is = _run(_h2o_bent(p_min[0], p_min[1]), [0, 1, 1], upfs, xc, **_MOL_KW)
    res_ts = _run(_h2o_bent(r_ts, np.pi / 2), [0, 1, 1], upfs, xc, **_MOL_KW)
    return res_is, res_ts


@pytest.mark.slow
def test_h2o_barrier_fd_matches_autograd():
    """dE_a/dλ (one backward per endpoint) == central FD of the full barrier."""
    torch.set_num_threads(8)
    xc0 = _make_xc()
    res_is, res_ts = _h2o_barrier(xc0)
    sens = barrier_parameter_sensitivity(res_is, res_ts, xc=xc0)
    ag = sens.dEa_dtheta["raw_mu"]
    # a real barrier: linear water sits well above the bent minimum
    assert sens.barrier_eV > 0.5, sens.barrier_eV

    h = 3e-3
    eas = []
    for sign in (+1, -1):
        xc = _make_xc(sign * h)
        ri, rt = _h2o_barrier(xc)  # BOTH endpoints re-relaxed at λ ± h
        eas.append(float(rt.energies.total) - float(ri.energies.total))
    fd = (eas[0] - eas[1]) / (2 * h)
    assert abs(fd - ag) < 2e-3 * max(1.0, abs(fd)), (ag, fd, eas)


# ------------------------------------------------------ canonical assembly / guards
def test_requires_a_parameter():
    r = SimpleNamespace(system=SimpleNamespace(positions=np.zeros((2, 3))),
                        energies=SimpleNamespace(total=torch.tensor(0.0)))
    with pytest.raises(ValueError, match="xc="):
        barrier_parameter_sensitivity(r, r)


def test_rejects_mismatched_atom_count():
    ri = SimpleNamespace(system=SimpleNamespace(positions=np.zeros((2, 3))),
                         energies=SimpleNamespace(total=torch.tensor(0.0)))
    rt = SimpleNamespace(system=SimpleNamespace(positions=np.zeros((3, 3))),
                         energies=SimpleNamespace(total=torch.tensor(0.0)))
    with pytest.raises(ValueError, match="atom count"):
        barrier_parameter_sensitivity(ri, rt, xc=_make_xc())


# ------------------------------------------------------ grand-canonical (dE_a/dU)
def _fake_cmu(n_electrons: float, free_energy: float, mu: float, natoms: int = 1):
    """A minimal stand-in for a constant-µ SCFResult (only the fields the grand-
    canonical path reads)."""
    return SimpleNamespace(
        system=SimpleNamespace(positions=np.zeros((natoms, 3))),
        fermi=mu,
        n_electrons=n_electrons,
        energies=SimpleNamespace(free_energy=torch.tensor(free_energy)),
    )


def test_barrier_potential_sensitivity_arithmetic():
    """dΩ_a/dµ = −ΔN, dE_a/dU = +ΔN, Ω = F − µN — pure arithmetic, no SCF."""
    mu = -3.0
    ri = _fake_cmu(n_electrons=8.00, free_energy=-100.0, mu=mu)
    rt = _fake_cmu(n_electrons=8.25, free_energy=-99.0, mu=mu)
    s = barrier_potential_sensitivity(ri, rt)
    assert s.delta_N == pytest.approx(0.25)
    assert s.domega_a_dmu == pytest.approx(-0.25)
    assert s.dEa_dU == pytest.approx(0.25)
    assert s.omega_is_eV == pytest.approx(-100.0 - mu * 8.00)
    assert s.omega_ts_eV == pytest.approx(-99.0 - mu * 8.25)
    assert s.barrier_omega_eV == pytest.approx(s.omega_ts_eV - s.omega_is_eV)
    assert grand_potential(ri) == pytest.approx(s.omega_is_eV)


def test_potential_sensitivity_rejects_mu_mismatch():
    ri = _fake_cmu(8.0, -100.0, mu=-3.0)
    rt = _fake_cmu(8.2, -99.0, mu=-2.5)
    with pytest.raises(ValueError, match="SAME"):
        barrier_potential_sensitivity(ri, rt)


@pytest.mark.slow
def test_grand_canonical_envelope():
    """dΩ/dµ = −N at a constant-µ ESM slab (grand-canonical Hellmann–Feynman)."""
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.scf.loop import scf, setup_system

    torch.set_num_threads(8)
    na = parse_upf(str(PSEUDOS / "Na_ONCV_PBE_sr.upf"))
    cell = np.diag([6.0, 6.0, 16.0])
    pos = np.array([[3.0, 3.0, 8.0]])
    common = dict(smearing="fermi-dirac", width=0.3, etol=1e-8, rhotol=1e-7,
                  max_iter=200, verbose=False, boundary="open_z_metal")

    def run(target_mu):
        sysx = setup_system(cell, pos, [0], [na], ecut=26 * RY)
        r = scf(sysx, LDA_PW92(), target_mu=target_mu, **common)
        assert r.converged
        return r

    # neutral Fermi level as the working µ
    sysx = setup_system(cell, pos, [0], [na], ecut=26 * RY)
    rf = scf(sysx, LDA_PW92(), smearing="fermi-dirac", width=0.3, etol=1e-8,
             rhotol=1e-7, max_iter=200, verbose=False, boundary="open_z_metal")
    mu0 = rf.fermi
    r0 = run(mu0)
    dmu = 0.05
    r_up = run(mu0 + dmu)
    r_dn = run(mu0 - dmu)

    fd = (grand_potential(r_up) - grand_potential(r_dn)) / (2 * dmu)
    minus_N = -float(r0.n_electrons)
    # dΩ/dµ = −N to the FD accuracy (N itself floats ~O(0.1) e over ±dµ)
    assert abs(fd - minus_N) < 0.02 * abs(minus_N), (fd, minus_N)

    # r_up/r_dn are at different µ, so the same-electrode-potential guard trips
    with pytest.raises(ValueError, match="SAME"):
        barrier_potential_sensitivity(r_dn, r_up)
