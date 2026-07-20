"""Spin-orbit coupling via fully-relativistic UPF j-projectors.

GaAs valence SO splitting Δ₀ is the textbook validation: 4-fold Γ8 above
the 2-fold Γ7 split-off. QE (lspinorb, identical SG15-FR pseudos) gives
Δ₀ = 0.336 eV (experiment 0.34)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc(5.653)


def make_system():
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")
    as_ = parse_upf(FIX / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")
    ref = json.loads((FIX / "gaas_so_ci" / "reference.json").read_text())
    return setup_system(CELL, POS, [0, 1], [ga, as_], ecut=40 * RY, kmesh=(2, 2, 2),
                        nbands=13, fft_shape=ref["fft_dims"], time_reversal=False), ref


def test_collinear_scf_rejects_fr_pseudos():
    system, _ = make_system()
    from gradwave.core.xc.pbe import PBE

    with pytest.raises(ValueError, match="spinor"):
        scf(system, PBE(), smearing="gaussian", verbose=False)


@pytest.mark.slow
def test_gaas_so_splitting_vs_qe():
    torch.set_num_threads(8)
    system, ref = make_system()
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.1,
                           etol=1e-7, rhotol=1e-6, verbose=False)
    f = float(res.energies.free_energy)
    assert abs(f - ref["etot_eV"]) / 2 * 1000 < 0.1  # meV/atom

    ig = [i for i, sp in enumerate(system.spheres)
          if np.abs(sp.k_frac).max() < 1e-9][0]
    eg = np.sort(res.eigenvalues[ig].cpu().numpy())
    gamma8 = eg[14:18]  # 4-fold valence top
    gamma7 = eg[12:14]  # 2-fold split-off
    assert np.ptp(gamma8) < 2e-3 and np.ptp(gamma7) < 2e-3
    delta0 = gamma8.mean() - gamma7.mean()
    qe_e = np.array(ref["eigenvalues_eV"])
    kq = np.array(ref["k_points_tpiba"])
    igq = int(np.argmin(np.linalg.norm(kq, axis=1)))
    eq = np.sort(qe_e[igq])
    delta0_qe = eq[14:18].mean() - eq[12:14].mean()
    assert abs(delta0 - delta0_qe) < 2e-3, (delta0, delta0_qe)
    assert 0.25 < delta0 < 0.45  # physical window around exp. 0.34
