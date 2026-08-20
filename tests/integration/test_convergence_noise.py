"""Statistical noise floor, one-run ecut curve, and the convergence driver.

Fast tier (unmarked): tiny diamond-Si systems, Gamma-only, ecut <= 20 Ry.
Covers the Janssen et al. (npj Comput. Mater. 2024) statistical half:

  - the rigid-translation noise floor is positive, small, and roughly
    independent of the probe count (postscf.convergence_noise);
  - the one-run reversed-annulus err(ecut') curve agrees with an actual
    lower-ecut run's error against the high-ecut reference within a factor
    ~2-3 (postscf.discretization_error.estimate_ecut_error_curve);
  - the api.recommend_convergence driver assembles the evidence and applies
    the target-vs-noise-floor rule consistently.
"""

import json

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.convergence_noise import estimate_noise_floor
from gradwave.postscf.discretization_error import estimate_ecut_error_curve
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, SI_ONCV, si_fcc, si_upf

SCF_KW = dict(smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)


@pytest.fixture(scope="module")
def si_base():
    """One converged 12 Ry Gamma diamond-Si run, shared by the noise tests."""
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(1, 1, 1))
    res = scf(system, PBE(), **SCF_KW)
    assert res.converged
    return system, res


def test_noise_floor_translation_probes(si_base):
    """sigma_E from rigid translations: positive, tiny, and stable (within
    stats) against the number of probes; every probe converges."""
    torch.set_num_threads(2)
    system, res = si_base
    xc = PBE()
    nf3 = estimate_noise_floor(res, xc, kmesh=(1, 1, 1),
                               n_translations=3, seed=0)
    nf5 = estimate_noise_floor(res, xc, kmesh=(1, 1, 1),
                               n_translations=5, seed=1)
    for nf in (nf3, nf5):
        assert nf.all_converged
        assert nf.sigma_e > 1e-12                 # a real, resolved scatter
        assert nf.sigma_e_per_atom < 1e-3         # ... but a tiny one (eV/atom)
        assert nf.spread_e >= nf.sigma_e          # range bounds the std
    # the floor is a scale estimate: 3 vs 5 probes agree within broad stats
    ratio = nf5.sigma_e / nf3.sigma_e
    assert 1.0 / 8.0 < ratio < 8.0
    # shifts stay strictly inside one grid spacing per axis
    upper = 1.0 / np.asarray(system.grid.shape, dtype=float)
    fr = np.asarray(nf3.shifts_frac)
    assert np.all(fr > 0.0) and np.all(fr < upper[None, :])
    # probes are re-converged fresh systems, not copies of the base energy
    assert len(set(nf5.energies)) == 5


def test_noise_floor_guards(si_base):
    _system, res = si_base
    with pytest.raises(ValueError):
        estimate_noise_floor(res, PBE(), kmesh=(1, 1, 1), n_translations=1)
    with pytest.raises(ValueError):
        estimate_noise_floor(res, PBE(), kmesh=(1, 1, 1), n_volumes=3)


def test_ecut_curve_one_run_vs_direct():
    """The reversed-annulus curve from ONE 20 Ry run reproduces the measured
    E(12 Ry) - E(20 Ry) on the pinned grid within a factor 3."""
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    upf = si_upf()
    sys_hi = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY,
                          kmesh=(1, 1, 1))
    res_hi = scf(sys_hi, PBE(), **SCF_KW)
    assert res_hi.converged

    curve = estimate_ecut_error_curve(
        res_hi, [10 * RY, 12 * RY, 14 * RY], include_tail=False)
    # error decreases monotonically as the candidate cutoff grows
    assert curve.denergy[0] > curve.denergy[1] > curve.denergy[2] > 0.0
    assert curve.tail == 0.0 and curve.error == curve.denergy

    # direct check: an actual 12 Ry run on the SAME (pinned) FFT grid
    sys_lo = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY,
                          kmesh=(1, 1, 1), fft_shape=tuple(sys_hi.grid.shape))
    res_lo = scf(sys_lo, PBE(), **SCF_KW)
    assert res_lo.converged
    direct = (float(res_lo.energies.free_energy)
              - float(res_hi.energies.free_energy))
    assert direct > 0.0                          # lower cutoff = higher energy
    est = curve.denergy[1]
    assert 1.0 / 3.0 < est / direct < 3.0, (
        f"one-run estimate {est:.3e} eV vs direct {direct:.3e} eV")

    # the tail term is a positive add-on toward the infinite-basis error
    curve_t = estimate_ecut_error_curve(res_hi, [12 * RY], ecut_large=40 * RY)
    assert curve_t.tail > 0.0
    assert curve_t.error[0] == pytest.approx(curve_t.denergy[0] + curve_t.tail)

    # guards: candidates must sit strictly below the run cutoff
    with pytest.raises(ValueError):
        estimate_ecut_error_curve(res_hi, [25 * RY])
    with pytest.raises(ValueError):
        estimate_ecut_error_curve(res_hi, [])


def test_recommend_convergence_driver_si():
    """End-to-end driver on Si: evidence blocks assembled, rule applied
    consistently, output JSON-serializable."""
    from ase import Atoms

    from gradwave.api import recommend_convergence
    from gradwave.inputs import Input, KPointsParams, SmearingParams

    torch.set_num_threads(2)
    cell, pos = si_fcc()
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=PSEUDOS, pseudo_map={"Si": SI_ONCV},
        ecut=10 * RY, xc="pbe", kpoints=KPointsParams(mesh=(1, 1, 1)),
        smearing=SmearingParams(type="none"))
    target = 1e-3
    block = recommend_convergence(
        inp, target=target, ecut_run=20 * RY,
        ecut_candidates=[10 * RY, 14 * RY], kmeshes=[(1, 1, 1), (2, 2, 2)],
        kpoint_exponent=2.0, n_translations=3, seed=0, verbose=False)

    json.dumps(block)                            # summary-block contract

    errs = [d["error_eV_per_atom"] for d in block["ecut_curve"]]
    assert errs[0] > errs[1] > 0.0               # systematic curve decreases
    floor = block["noise_floor_eV_per_atom"]
    assert 0.0 < floor < 1e-3                    # measured Si floor is tiny
    assert block["achievable"] == (target >= floor)
    assert block["effective_target_eV_per_atom"] == max(target, floor)
    assert block["noise"]["all_converged"]

    # the rule's outputs are consistent with its own evidence
    eff = block["effective_target_eV_per_atom"]
    rec = block["recommended_ecut_eV"]
    if rec is not None and rec != block["ecut_run_eV"]:
        by_ecut = {d["ecut_eV"]: d["error_eV_per_atom"]
                   for d in block["ecut_curve"]}
        assert by_ecut[rec] <= eff
    assert block["recommended_kmesh"] is not None
    assert len(block["kpoint"]["meshes"]) == 2


def test_recommend_convergence_rejects_bad_inputs():
    from ase import Atoms

    from gradwave.api import recommend_convergence
    from gradwave.inputs import Input, KPointsParams

    cell, pos = si_fcc()
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(atoms=atoms, pseudo_dir=PSEUDOS, pseudo_map={"Si": SI_ONCV},
                ecut=10 * RY, xc="pbe", kpoints=KPointsParams(mesh=(1, 1, 1)))
    with pytest.raises(ValueError):
        recommend_convergence(inp, target=0.0)
    with pytest.raises(ValueError):
        recommend_convergence(inp, target=1e-3, ecut_run=5 * RY)
