"""Energy-metric convergence gate, end to end through the SCF loop.

Two contracts: (a) selecting the energy gate converges a well-behaved system to
the SAME fixed point as the density gate (same F, iteration count within a
couple), and (b) omitting the option changes nothing — the default density gate
is bit-for-bit unchanged. See scf.loop.scf(energy_metric=...) and
postscf._response.kernel_energy_error.
"""

from pathlib import Path

import numpy as np

from gradwave.core.xc.pbe import PBE
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_system(a=5.43, ecut_ry=12.0, kmesh=(2, 2, 2)):
    from gradwave.pseudo.upf import parse_upf

    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return setup_system(cell, pos, [0, 0], [upf], ecut=ecut_ry * RY, kmesh=kmesh)


def test_energy_gate_matches_density_gate():
    """On a well-behaved insulator the energy gate lands on the density gate's
    fixed point: same free energy to ~1e-8 eV, iteration count within a couple.
    The energy-metric value is recorded only on the energy-gate run."""
    import torch

    torch.set_num_threads(4)
    res_rho = scf(_si_system(), PBE(), smearing="none",
                  etol=1e-9, rhotol=1e-8, diago_tol=1e-9, verbose=False)
    res_en = scf(_si_system(), PBE(), smearing="none",
                 etol=1e-9, rhotol=1e-8, diago_tol=1e-9,
                 energy_metric=True, entol=1e-9, verbose=False)
    assert res_rho.converged and res_en.converged
    # same fixed point: the free energies agree to well under 1e-8 eV
    assert abs(res_en.energies.free_energy - res_rho.energies.free_energy) < 1e-8
    # the energy gate is at least as efficient (on an insulator the energy error
    # settles a few iterations before the density residual reaches rhotol)
    assert res_en.n_iter <= res_rho.n_iter

    # the recorder carries the per-iteration estimate only for the energy run
    assert res_en.recorder.iters[-1]["e_metric"] is not None
    assert res_en.recorder.summarize()["energy_metric_eV"] is not None
    assert res_rho.recorder.iters[-1]["e_metric"] is None
    # final estimate sits at/under the threshold that stopped the run
    assert abs(res_en.recorder.iters[-1]["e_metric"]) < 1e-9

    # the Harris-Foulkes/KS gap rides along as the zero-machinery bracket:
    # recorded each iteration, second order in the residual, so it collapses to
    # ~the entol scale at the converged step and shrinks from its early value
    gaps = [i["e_hf_gap"] for i in res_en.recorder.iters]
    assert all(g is not None for g in gaps)
    assert abs(gaps[-1]) < 1e-7
    assert abs(gaps[-1]) < abs(gaps[1])
    assert "harris_foulkes_gap_eV" in res_en.recorder.summarize()


def test_omitting_the_option_is_bit_for_bit():
    """The default (density) gate is unchanged when the new knobs are left off:
    an explicit energy_metric=False run is byte-identical to a bare run, and no
    energy metric is recorded."""
    import torch

    torch.set_num_threads(4)
    common = dict(smearing="none", etol=1e-9, rhotol=1e-8, diago_tol=1e-9, verbose=False)
    res_a = scf(_si_system(), PBE(), **common)
    res_b = scf(_si_system(), PBE(), energy_metric=False, entol=1e-6, **common)
    assert res_a.n_iter == res_b.n_iter
    assert res_a.energies.free_energy == res_b.energies.free_energy  # bit-for-bit
    assert res_a.recorder.iters[-1]["e_metric"] is None
    assert "energy_metric_eV" not in res_a.recorder.summarize()
