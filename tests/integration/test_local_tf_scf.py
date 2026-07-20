"""The local Thomas–Fermi preconditioner must not move the SCF fixed point.

Like the Kerker filter it reduces to, `precond="local_tf"` only reshapes the
mixing step, so the converged free energy and eigenvalues must match the bare
`precond="kerker"` run to solver precision. These gates protect the wiring
(mixer `precond_op` hook, the NC and USPP drivers, and the module) against
regressions; the operator's three analytic limits are pinned separately in
`tests/unit/test_local_tf.py`, and the iteration-count win on inhomogeneous
slabs lives in `benchmarks/bench_precond.py`.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
AL_A = 4.05
AL_CELL = AL_A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.standard
def test_nc_local_tf_matches_kerker():
    """fcc Al (a metal, where the preconditioner is meaningful): the local-TF
    and bare-Kerker runs reach the same fixed point to solver precision."""
    torch.set_num_threads(4)
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")

    def run(precond):
        system = setup_system(AL_CELL, np.array([[0.0, 0, 0]]), [0], [al],
                              ecut=20 * RY, kmesh=(6, 6, 6), nbands=10,
                              use_symmetry=True)
        return scf(system, PBE(), smearing="gaussian", width=0.3, etol=1e-9,
                   rhotol=1e-8, max_iter=80, verbose=False, precond=precond)

    rk = run("kerker")
    rt = run("local_tf")
    assert rk.converged and rt.converged
    assert abs(float(rk.energies.free_energy)
               - float(rt.energies.free_energy)) < 1e-8
    assert torch.allclose(rk.eigenvalues, rt.eigenvalues, atol=1e-6)


@pytest.mark.standard
def test_uspp_local_tf_matches_kerker():
    """USPP/PAW twin of the NC gate — exercises the `scf_uspp` precond wiring
    on the composite (ρ, becsum) mixing vector (local-TF on the ρ-total block
    only)."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(4)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")

    def run(precond):
        s = setup_uspp(AL_CELL, [[0.0, 0, 0]], [0], [paw], ecut=20 * RY,
                       ecutrho=120 * RY, kmesh=(6, 6, 6), nbands=10,
                       use_symmetry=True)
        return scf_uspp(s, PBE(), smearing="gaussian", width=0.3, etol=1e-9,
                        rhotol=1e-8, max_iter=80, mixing_scheme="johnson",
                        verbose=False, precond=precond)

    rk = run("kerker")
    rt = run("local_tf")
    assert rk["converged"] and rt["converged"]
    assert abs(float(rk["energies"].free_energy)
               - float(rt["energies"].free_energy)) < 1e-8
    for ek, et in zip(rk["eigenvalues"], rt["eigenvalues"], strict=True):
        assert torch.allclose(ek, et, atol=1e-6)


def test_precond_rejects_unknown_name():
    """A typo'd preconditioner name fails fast rather than silently running
    bare damping."""
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")
    system = setup_system(AL_CELL, np.array([[0.0, 0, 0]]), [0], [al],
                          ecut=15 * RY, kmesh=(2, 2, 2), nbands=8,
                          use_symmetry=True)
    with pytest.raises(ValueError, match="precond"):
        scf(system, PBE(), smearing="gaussian", width=0.5, max_iter=1,
            verbose=False, precond="kerkr")
