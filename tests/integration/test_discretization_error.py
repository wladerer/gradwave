"""Plane-wave (Ecut) discretization-error estimator.

Covers the invariants that do not need a reference (charge conservation of the
complement correction, definite energy lowering), the exact nspin=2 nonmagnetic
limit, the symmetric-vs-full-BZ force error agreement, and a low->high cutoff
comparison for both formalisms. See postscf/discretization_error.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.discretization_error import (
    estimate_density_error,
    estimate_force_error,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.uspp import scf_uspp, setup_uspp

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_cell(a=5.43):
    return a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3])


def _downsample(rho, shape_dst):
    ftil = torch.fft.fftshift(torch.fft.fftn(rho) / rho.numel())
    sl = [slice((s - d) // 2, (s - d) // 2 + d)
          for s, d in zip(rho.shape, shape_dst, strict=True)]
    ftil_c = torch.fft.ifftshift(ftil[sl[0], sl[1], sl[2]])
    n = shape_dst[0] * shape_dst[1] * shape_dst[2]
    return (torch.fft.ifftn(ftil_c) * n).real


def _corr(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.dot(a, b)
                 / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-30))


def test_nc_complement_invariants():
    """int(drho) = 0 (correction orthogonal to occupied) and dE < 0."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    est = estimate_density_error(res, ecut_large=35 * RY)

    vol, n = system.grid.volume, system.grid.n_points
    nelec = float(res.rho.sum()) * vol / n
    dq_per_e = float(est.drho.sum()) * vol / n / nelec
    assert abs(dq_per_e) < 1e-3           # charge-conserving
    assert est.denergy < 0.0              # definite lowering
    assert -5.0 < est.denergy < 0.0       # sane magnitude (eV)


def test_nspin2_nonmagnetic_limit_matches_nspin1():
    """nspin=2 with zero moment reproduces the nspin=1 estimate exactly."""
    torch.set_num_threads(4)
    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")

    def make():
        return setup_system(4.05 / 2 * FCC, np.zeros((1, 3)), [0], [al],
                            ecut=18 * RY, kmesh=(2, 2, 2), nbands=10)

    kw = dict(smearing="gaussian", width=0.1, etol=1e-10, rhotol=1e-9,
              verbose=False)
    r1 = scf(make(), PBE(), **kw)
    r2 = scf(make(), SpinPBE(), nspin=2, start_mag=[0.0], **kw)
    e1 = estimate_density_error(r1, ecut_large=36 * RY)
    e2 = estimate_density_error(r2, ecut_large=36 * RY)

    assert float((e1.drho - e2.drho).abs().max()) < 1e-8
    assert abs(e1.denergy - e2.denergy) < 1e-8


def test_symmetric_force_error_matches_full_bz():
    """IBZ + symmetrize reproduces the full-BZ force error and is invariant."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    shift = 0.12 * np.array([1.0, 1, 1]) / np.sqrt(3)  # keeps 3-fold + mirrors
    pos = np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]]) + np.array([[0, 0, 0], shift])

    def run(use_symmetry):
        system = setup_system(a / 2 * FCC, pos, [0, 0], [upf], ecut=20 * RY,
                              kmesh=(4, 4, 4), use_symmetry=use_symmetry)
        res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
                  verbose=False)
        est = estimate_density_error(res, ecut_large=50 * RY)
        return res, est, estimate_force_error(res, est)

    res_s, _es, f_s = run(True)
    _res_n, _en, f_n = run(False)
    assert res_s.system.sym is not None and res_s.system.sym.n_ops > 1

    assert float((f_s - f_n).abs().max()) < 1e-4          # matches full BZ
    from gradwave.symmetry import symmetrize_forces
    resid = f_s - symmetrize_forces(f_s, res_s.system.sym, res_s.system.grid.cell)
    assert float(resid.abs().max()) < 1e-10               # exactly invariant


@pytest.mark.slow
def test_nc_low_to_high_reduces_density_error():
    """The estimate correlates with the true low->high change and reduces it."""
    torch.set_num_threads(8)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()

    def run(ecut):
        system = setup_system(cell, pos, [0, 0], [upf], ecut=ecut, kmesh=(2, 2, 2))
        return scf(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
                   verbose=False)

    lo, hi = run(12 * RY), run(28 * RY)
    est = estimate_density_error(lo, ecut_large=28 * RY)
    rho_hi_on_lo = _downsample(hi.rho, lo.system.grid.shape)
    true_err = rho_hi_on_lo - lo.rho

    assert _corr(est.drho, true_err) > 0.5
    before = float((lo.rho - rho_hi_on_lo).abs().sum())
    after = float((lo.rho + est.drho - rho_hi_on_lo).abs().sum())
    assert after < before                                  # reduces the error


@pytest.mark.slow
def test_uspp_density_and_energy_error():
    """USPP/PAW: positive density correlation and a sane energy ratio."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])

    def run(ecut):
        return scf_uspp(setup_uspp(cell, pos, [0, 0], [paw], ecut=ecut,
                                   kmesh=(2, 2, 2), ecutrho=4 * ecut),
                        PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=80)

    lo, hi = run(12 * RY), run(30 * RY)
    est = estimate_density_error(lo, ecut_large=30 * RY, xc=PBE())

    grid = lo["system"].grid
    vol, n = grid.volume, grid.n_points
    nelec = float(lo["rho"].sum()) * vol / n
    assert abs(float(est.drho.sum()) * vol / n / nelec) < 1e-3   # ~charge conserving

    rho_hi_on_lo = _downsample(hi["rho"], grid.shape)
    assert _corr(est.drho, rho_hi_on_lo - lo["rho"]) > 0.5

    true_de = float(hi["energies"].free_energy) - float(lo["energies"].free_energy)
    assert true_de < 0.0
    assert 0.5 < est.denergy / true_de < 1.5                     # right sign, right scale
