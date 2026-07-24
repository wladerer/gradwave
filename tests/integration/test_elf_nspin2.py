"""Spin-resolved ELF (nspin=2 unblock).

elf() now returns a per-spin field (2, n1, n2, n3) for collinear nspin=2, each
channel built from (τ_σ, ρ_σ) with the spin-scaled TF reference. Two checks:

- ferromagnetic fcc Ni: the two channels genuinely differ (ELF↑ ≠ ELF↓) and the
  field stays in (0, 1];
- nonmagnetic Si limit: with ρ↑=ρ↓ each channel reproduces the spin-restricted
  ELF exactly (D_σ/D_h,σ = D/D_h), which pins down the 2^{2/3} spin factor.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf import volumetric as V
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc, si_upf

pytestmark = pytest.mark.standard


def test_elf_nspin2_ferromagnetic_ni_is_spin_resolved():
    torch.set_num_threads(8)
    a = 3.52
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    ni = parse_upf(PSEUDOS / "PD_Ni_PBE.upf")
    system = setup_system(cell, np.zeros((1, 3)), [0], [ni], ecut=45 * RY,
                          kmesh=(2, 2, 2), nbands=14, time_reversal=False)
    res = scf(system, LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.5], etol=1e-7, rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged and res.mag_total > 0.5

    e = V.elf(res)
    assert e.shape == (2, *system.grid.shape)
    assert np.all(np.isfinite(e))
    assert e.min() > 0.0 and e.max() <= 1.0 + 1e-9
    # genuinely spin-polarized → the channels differ
    assert np.abs(e[0] - e[1]).max() > 1e-3


def test_elf_nspin2_nonmagnetic_limit_matches_restricted():
    torch.set_num_threads(4)
    cell, pos = si_fcc()
    upf = si_upf()

    def make():
        return setup_system(cell, pos, [0, 0], [upf], ecut=15 * RY,
                            kmesh=(2, 2, 2), nbands=12)

    r1 = scf(make(), LDA_PW92(), smearing="gaussian", width=0.05,
             etol=1e-9, rhotol=1e-8, verbose=False)
    r2 = scf(make(), LSDA_PW92(), smearing="gaussian", width=0.05, nspin=2,
             start_mag=[0.0, 0.0], etol=1e-9, rhotol=1e-8, verbose=False)
    assert r1.converged and r2.converged and abs(r2.mag_total) < 1e-6

    e1 = V.elf(r1)
    e2 = V.elf(r2)
    assert e1.ndim == 3  # nspin=1 stays 3-D
    assert e2.shape == (2, *r2.system.grid.shape)
    # both channels coincide and equal the spin-restricted ELF
    assert np.abs(e2[0] - e2[1]).max() < 1e-6
    assert np.abs(e2[0] - e1).max() < 2e-3
