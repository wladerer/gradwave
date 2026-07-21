"""Newton-Krylov SCF finisher: quadratic convergence with the exact
response Jacobian (χ̃ with the Fermi-surface channel + autograd kernels).

Observed on Si kjpaw: |F(x)−x| 5.5e-5 → 5.4e-6 → 1.8e-9 in two Newton
steps from a deliberately loose SCF, with the polished free energy 4e-7
meV from a deep-converged reference. The residual EVALUATION floors at
the eigensolver noise (~2e-9 here); the finisher detects the floor and
stops honestly."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.newton import newton_polish
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
@pytest.mark.slow
def test_newton_polish_si_insulator():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    xc = LearnableX()

    def make():
        return setup_uspp(cell, pos, [0, 0], [paw], ecut=15 * RY,
                          kmesh=(2, 2, 2), ecutrho=60 * RY)

    loose = scf_uspp(make(), xc, etol=1e-4, rhotol=1e-3, verbose=False,
                     max_iter=40)
    ref = scf_uspp(make(), xc, etol=1e-12, rhotol=1e-10, verbose=False,
                   max_iter=80)
    pol = newton_polish(loose, xc, tol=5e-9)
    assert pol["converged"]
    assert len(pol["newton"]) <= 3, pol["newton"]
    # quadratic contraction between the first two steps
    assert pol["newton"][1] < 1e-2 * pol["newton"][0]
    df = abs(float(pol["energies"].free_energy)
             - float(ref["energies"].free_energy))
    assert df < 1e-6, f"polished F off by {df:.2e} eV"


@pytest.mark.slow
def test_newton_polish_al_metal():
    """Smeared metal: the Jacobian carries the Fermi-surface occupation
    channel (divided differences, δμ) — the finisher must converge
    through it, not despite it."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    xc = PBE()

    def make():
        return setup_uspp(cell, pos, [0], [paw], ecut=20 * RY,
                          kmesh=(2, 2, 2), ecutrho=100 * RY, nbands=8)

    loose = scf_uspp(make(), xc, smearing="gaussian", width=0.5,
                     etol=1e-4, rhotol=1e-3, verbose=False, max_iter=60)
    ref = scf_uspp(make(), xc, smearing="gaussian", width=0.5,
                   etol=1e-12, rhotol=1e-10, verbose=False, max_iter=120)
    pol = newton_polish(loose, xc, tol=5e-9)
    assert pol["converged"]
    df = abs(float(pol["energies"].free_energy)
             - float(ref["energies"].free_energy))
    assert df < 1e-6, f"polished F off by {df:.2e} eV"
