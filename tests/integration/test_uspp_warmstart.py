"""Warm-starting scf_uspp from a previous result (scan chaining).

start_from seeds (ρ, becsum) from a converged result on the same FFT grid,
with the density rescaled by the volume ratio so the electron count is
conserved. The converged fixed point must be independent of the start; the
iteration count should not grow. Observed: dF 2e-13 across a 5.43 → 5.48 Å
volume step, and 20 → 11 iterations for the smeared spin case restarted
from its own solution."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import happly_tally, reset_happly_tally
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
def _si(paw, a, fft_shape=None):
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return setup_uspp(cell, pos, [0, 0], [paw], ecut=15 * RY,
                      kmesh=(2, 2, 2), ecutrho=60 * RY, fft_shape=fft_shape)


@pytest.mark.slow
def test_warmstart_same_fixed_point_fewer_iterations():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    s_a = _si(paw, 5.43)
    shape = tuple(s_a.grid.shape)
    r_a = scf_uspp(s_a, PBE(), etol=1e-10, rhotol=1e-9, verbose=False,
                   max_iter=80)
    assert r_a["converged"]

    r_cold = scf_uspp(_si(paw, 5.48, shape), PBE(), etol=1e-10, rhotol=1e-9,
                      verbose=False, max_iter=80)
    r_warm = scf_uspp(_si(paw, 5.48, shape), PBE(), etol=1e-10, rhotol=1e-9,
                      verbose=False, max_iter=80, start_from=r_a)
    fc = float(r_cold["energies"].free_energy)
    fw = float(r_warm["energies"].free_energy)
    assert abs(fc - fw) < 5e-8, f"warm fixed point moved: {abs(fc - fw):.2e}"
    assert r_warm["n_iter"] <= r_cold["n_iter"]

    # nspin=2 seeding shapes: restart a smeared spin run from itself
    r2 = scf_uspp(_si(paw, 5.43), SpinPBE(), nspin=2, start_mag=[0.3],
                  smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                  verbose=False, max_iter=80)
    r2w = scf_uspp(_si(paw, 5.43), SpinPBE(), nspin=2, start_mag=[0.3],
                   smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
                   verbose=False, max_iter=80, start_from=r2)
    df = abs(float(r2["energies"].free_energy)
             - float(r2w["energies"].free_energy))
    assert df < 5e-8
    assert r2w["n_iter"] < r2["n_iter"]


@pytest.mark.slow
def test_orbital_reuse_cuts_davidson_work():
    """On the generalized (S-metric) Davidson, seeding from the previous step's
    eigenvectors cuts H-applications at a fixed density warm start. A
    positions-only ionic move keeps the G-spheres, so the prior orbitals seed
    the solver directly; the density+becsum warm start is held identical (only
    the orbital seed differs), isolating the eigensolver saving from the density
    effect. BatchedHamiltonian.apply is the H|ψ⟩ chokepoint the counter taps."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def sysr(rattle):
        pos = np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]])
        pos[1] = pos[1] + rattle
        return setup_uspp(cell, pos, [0, 0], [paw], ecut=15 * RY,
                          kmesh=(2, 2, 2), ecutrho=60 * RY, use_symmetry=False)

    kw = dict(etol=1e-10, rhotol=1e-9, verbose=False, max_iter=80)
    r0 = scf_uspp(sysr(0.0), PBE(), **kw)
    # identical density + becsum warm start; the only difference is the orbitals
    r0_no_orb = dataclasses.replace(r0, coeffs=None)

    reset_happly_tally()
    r_den = scf_uspp(sysr(0.03), PBE(), start_from=r0_no_orb, **kw)
    h_den = happly_tally()
    reset_happly_tally()
    r_reuse = scf_uspp(sysr(0.03), PBE(), start_from=r0, **kw)
    h_reuse = happly_tally()

    f_den = float(r_den["energies"].free_energy)
    f_reuse = float(r_reuse["energies"].free_energy)
    assert abs(f_den - f_reuse) < 1e-9, "orbital reuse moved the fixed point"
    assert h_reuse < h_den, (
        f"orbital reuse did not cut H-applications: {h_reuse} !< {h_den}")


def test_warmstart_rejects_grid_mismatch():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    s_a = _si(paw, 5.43)
    fake = {"system": s_a, "nspin": 1}
    bigger = tuple(d + 5 for d in s_a.grid.shape)
    with pytest.raises(ValueError, match="same FFT grid"):
        scf_uspp(_si(paw, 5.43, bigger), PBE(), verbose=False,
                 max_iter=1, start_from=fake)
