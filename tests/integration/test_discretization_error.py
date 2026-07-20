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

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.discretization_error import (
    estimate_density_error,
    estimate_eigenvalue_error,
    estimate_force_error,
    estimate_gap_error,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
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


def test_eigenvalue_error_reproduces_energy_error():
    """δε <= 0 for every band, and the occupation-weighted sum of the occupied
    shifts reproduces denergy (the two paths compute the same per-band term)."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    err = estimate_density_error(res, ecut_large=35 * RY)
    eige = estimate_eigenvalue_error(res, ecut_large=35 * RY)

    assert all(float(de.max()) <= 1e-12 for de in eige.deig)   # definite lowering
    de_tot = 0.0
    for ik in range(len(system.spheres)):
        occ, de = res.occupations[ik], eige.deig[ik]
        de_tot += float(system.kweights[ik]) * float((occ[:de.shape[0]] * de).sum())
    assert abs(de_tot - err.denergy) < 1e-6 * abs(err.denergy)  # == denergy


@pytest.mark.slow
def test_gap_error_moves_toward_high_cutoff():
    """The extrapolated gap eps+δε is closer to the high-cutoff gap than the
    raw low-cutoff gap (the correction reduces the basis-set gap error)."""
    torch.set_num_threads(8)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()

    def run(ecut):
        system = setup_system(cell, pos, [0, 0], [upf], ecut=ecut, kmesh=(4, 4, 4))
        return scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
                   verbose=False)

    lo, hi = run(14 * RY), run(30 * RY)
    ge_lo = estimate_gap_error(lo, estimate_eigenvalue_error(lo, ecut_large=30 * RY))
    gap_hi = estimate_gap_error(
        hi, estimate_eigenvalue_error(hi, ecut_large=75 * RY))["gap_eV"]

    assert ge_lo["gap_eV"] > 0.0 and not ge_lo["direct"]        # Si indirect gap
    err_raw = abs(ge_lo["gap_eV"] - gap_hi)
    err_corr = abs(ge_lo["gap_extrapolated_eV"] - gap_hi)
    assert err_corr < err_raw                                   # correction helps


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


def test_nspin2_force_error_matches_nspin1():
    """Force error for nspin=2 at zero moment reproduces the nspin=1 estimate.

    The nonlocal channel now sums over spin channels; a nonmagnetic run splits
    the nspin=1 orbitals into two identical half-occupied channels, so the two
    estimates must agree to round-off. Uses a displaced two-atom cell so there
    is a real force (and force error) to compare.
    """
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    shift = 0.10 * np.array([1.0, 1, 1]) / np.sqrt(3)
    pos = (np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]])
           + np.array([[0, 0, 0], shift]))

    def make():
        return setup_system(a / 2 * FCC, pos, [0, 0], [upf], ecut=18 * RY,
                            kmesh=(2, 2, 2), use_symmetry=False, nbands=8)

    kw = dict(smearing="gaussian", width=0.1, etol=1e-10, rhotol=1e-9,
              verbose=False)
    r1 = scf(make(), PBE(), **kw)
    r2 = scf(make(), SpinPBE(), nspin=2, start_mag=[0.0, 0.0], **kw)
    e1 = estimate_density_error(r1, ecut_large=40 * RY)
    e2 = estimate_density_error(r2, ecut_large=40 * RY)
    f1 = estimate_force_error(r1, e1)
    f2 = estimate_force_error(r2, e2)

    assert float(f1.abs().max()) > 1e-4                 # a real signal to match
    assert float((f1 - f2).abs().max()) < 1e-6          # spin summation exact
    # The nspin=2 nonlocal channel loops over spins with per-spin δφ and
    # occupations; the moment magnitude hits no separate branch, so the
    # zero-moment limit already exercises the full spin path. A converged
    # magnetic-metal force reference (Fe/Co/Ni ONCV) is too costly for CI.


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


@pytest.mark.slow
def test_uspp_eigenvalue_reproduces_energy_error():
    """USPP/PAW eigenvalue error: every δε <= 0, and the occupation-weighted sum
    of the occupied generalized shifts reproduces the density-path denergy."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    res = scf_uspp(setup_uspp(a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3]),
                              [0, 0], [paw], ecut=12 * RY, kmesh=(2, 2, 2),
                              ecutrho=48 * RY),
                   PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=80)
    err = estimate_density_error(res, ecut_large=30 * RY, xc=PBE())
    eige = estimate_eigenvalue_error(res, ecut_large=30 * RY, xc=PBE())

    assert all(float(de.max()) <= 1e-9 for de in eige.deig)     # definite lowering
    system = res["system"]
    de_tot = 0.0
    for ik in range(len(system.spheres)):
        occ, de = res["occupations"][ik], eige.deig[ik]
        de_tot += float(system.kweights[ik]) * float((occ[:de.shape[0]] * de).sum())
    assert abs(de_tot - err.denergy) < 1e-6 * abs(err.denergy)   # == denergy
    # Si is an insulator: the gap tool resolves and the correction is finite
    gap = estimate_gap_error(res, eige)
    assert gap["gap_eV"] > 0.0


def _disp_paw_cell(a=5.43, shift=np.array([0.11, -0.06, 0.04])):
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]]) + np.array([[0, 0, 0], shift])
    return cell, pos


@pytest.mark.slow
def test_uspp_force_error_vs_high_cutoff():
    """USPP/PAW force error on a displaced cell: δF correlates with the true
    low->high force change and reduces it, with a sane magnitude ratio.

    Exercises the augmentation, S-orthogonality, and one-center ddd channels of
    the P(eps) propagation against a finite-cutoff force reference."""
    torch.set_num_threads(8)
    from gradwave.postscf.paw_forces import forces_uspp

    paw = parse_upf_paw(FIX / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell, pos = _disp_paw_cell()

    def run(ecut):
        return scf_uspp(setup_uspp(cell, pos, [0, 0], [paw], ecut=ecut,
                                   kmesh=(2, 2, 2), ecutrho=4 * ecut),
                        PBE(), etol=1e-10, rhotol=1e-9, verbose=False, max_iter=80)

    lo, hi = run(12 * RY), run(30 * RY)
    err = estimate_density_error(lo, ecut_large=30 * RY, xc=PBE())
    dF = estimate_force_error(lo, err, xc=PBE())
    true_dF = forces_uspp(hi, PBE()) - forces_uspp(lo, PBE())

    assert float(dF.abs().max()) > 1e-3                 # a real signal
    assert _corr(dF, true_dF) > 0.9                     # right direction
    assert 0.4 < float(dF.norm() / true_dF.norm()) < 1.6   # right scale
    f_lo = forces_uspp(lo, PBE())
    before = float((f_lo - forces_uspp(hi, PBE())).abs().sum())
    after = float((f_lo + dF - forces_uspp(hi, PBE())).abs().sum())
    assert after < before                               # reduces the error


@pytest.mark.slow
def test_uspp_nspin2_force_error_matches_nspin1():
    """USPP/PAW force error at zero moment: nspin=2 reproduces nspin=1 to
    round-off (the per-spin loop, per-spin smooth-density and becsum responses,
    and the one-center HVP all reduce to the single-channel case)."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell, pos = _disp_paw_cell(shift=np.array([0.11, -0.06, 0.04]))
    kw = dict(smearing="gaussian", width=0.1, etol=1e-10, rhotol=1e-9,
              verbose=False, max_iter=80)

    def mk(ecut):
        return setup_uspp(cell, pos, [0, 0], [paw], ecut=ecut, kmesh=(2, 2, 2),
                          ecutrho=4 * ecut)

    r1 = scf_uspp(mk(12 * RY), PBE(), **kw)
    f1 = estimate_force_error(r1, estimate_density_error(r1, ecut_large=30 * RY,
                                                         xc=PBE()), xc=PBE())
    r2 = scf_uspp(mk(12 * RY), SpinPBE(), nspin=2, start_mag=[0.0, 0.0], **kw)
    f2 = estimate_force_error(r2, estimate_density_error(r2, ecut_large=30 * RY,
                                                         xc=SpinPBE()), xc=SpinPBE())
    assert float(f1.abs().max()) > 1e-3
    assert float((f1 - f2).abs().max()) < 1e-6


# --------------------------------------------------------------------------- #
#  Non-collinear (spinor) path                                                #
# --------------------------------------------------------------------------- #


def _nc_occ(res, scheme="gaussian", width=0.1):
    from gradwave.core.occupations import (
        SCHEMES,
        find_fermi,
        occupations_and_entropy,
    )
    system = res.system
    mu = find_fermi(res.eigenvalues, system.kweights, SCHEMES[scheme], width,
                    system.n_electrons, degeneracy=1.0)
    occ, _ = occupations_and_entropy(res.eigenvalues, mu, SCHEMES[scheme], width,
                                     degeneracy=1.0)
    return occ, system


@pytest.mark.slow
def test_noncollinear_nonmagnetic_matches_collinear():
    """A nonmagnetic spinor SCF reduces to the collinear run, so the spinor
    complement correction must reproduce the collinear density/energy error.

    This pins the spinor machinery (doubled up/down axis, rebuilt exchange field,
    degeneracy-1 occupations) against the tested collinear estimator."""
    torch.set_num_threads(8)
    si = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    a = 5.43
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])

    def mk():
        return setup_system(a / 2 * FCC, pos, [0, 0], [si], ecut=15 * RY,
                            kmesh=(2, 2, 2), use_symmetry=False, time_reversal=False)

    rc = scf(mk(), PBE(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    ec = estimate_density_error(rc, ecut_large=35 * RY)

    xc = NoncollinearXC(SpinPBE())
    rn = scf_noncollinear(mk(), xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]], width=0.1,
                          etol=1e-9, rhotol=1e-8, verbose=False, nonmagnetic=True)
    en = estimate_density_error(rn, ecut_large=35 * RY, xc=xc,
                                smearing="gaussian", width=0.1)

    assert abs(en.denergy - ec.denergy) < 1e-4 * abs(ec.denergy)   # matches collinear
    assert float((en.drho - ec.drho).abs().max()) < 1e-5
    vol, n = rn.system.grid.volume, rn.system.grid.n_points
    assert abs(float(en.drho.sum()) * vol / n) < 1e-6             # charge-conserving

    # eigenvalue error reproduces the spinor denergy (degeneracy-1 occupations)
    eige = estimate_eigenvalue_error(rn, ecut_large=35 * RY, xc=xc,
                                     smearing="gaussian", width=0.1)
    assert all(float(de.max()) <= 1e-9 for de in eige.deig)
    occ, system = _nc_occ(rn)
    de_tot = sum(float(system.kweights[ik]) * float((occ[ik] * eige.deig[ik]).sum())
                 for ik in range(len(system.spheres)))
    assert abs(de_tot - en.denergy) < 1e-6 * abs(en.denergy)


@pytest.mark.slow
def test_soc_discretization_error_invariants_and_direction():
    """Fully-relativistic (SOC) spinor error: charge-conserving, definite energy
    lowering, self-consistent eigenvalue sum, and a density correction that
    correlates with — and reduces — the true low->high change. Exercises the
    enlarged spin-orbit projector path."""
    torch.set_num_threads(8)
    ga = parse_upf(FIX / "Ga_ONCV_PBE_FR-1.0.upf")
    as_ = parse_upf(FIX / "As_ONCV_PBE_FR-1.1.upf")
    a = 5.653
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    xc = NoncollinearXC(SpinPBE())

    def run(ecut, fft=None):
        system = setup_system(cell, pos, [0, 1], [ga, as_], ecut=ecut * RY,
                              kmesh=(2, 2, 2), nbands=13, use_symmetry=False,
                              time_reversal=False, fft_shape=fft)
        assert system.is_fr
        return system, scf_noncollinear(
            system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]], smearing="gaussian",
            width=0.1, etol=1e-7, rhotol=1e-6, verbose=False)

    slo, lo = run(24)
    est = estimate_density_error(lo, ecut_large=55 * RY, xc=xc,
                                 smearing="gaussian", width=0.1)
    vol, n = slo.grid.volume, slo.grid.n_points
    assert abs(float(est.drho.sum()) * vol / n) < 1e-6            # charge-conserving
    assert est.denergy < 0.0                                     # definite lowering

    eige = estimate_eigenvalue_error(lo, ecut_large=55 * RY, xc=xc,
                                     smearing="gaussian", width=0.1)
    assert all(float(de.max()) <= 1e-9 for de in eige.deig)
    occ, system = _nc_occ(lo)
    de_tot = sum(float(system.kweights[ik]) * float((occ[ik] * eige.deig[ik]).sum())
                 for ik in range(len(system.spheres)))
    assert abs(de_tot - est.denergy) < 1e-6 * abs(est.denergy)

    _shi, hi = run(44, fft=tuple(slo.grid.shape))
    true_err = hi.rho - lo.rho
    assert _corr(est.drho, true_err) > 0.8                       # right direction
    before = float((lo.rho - hi.rho).abs().sum())
    after = float((lo.rho + est.drho - hi.rho).abs().sum())
    assert after < before                                       # reduces the error
