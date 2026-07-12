"""+U on PAW, physical gate: FM fcc Ni, U = 3 eV on 3d, vs QE.

Observed: E_U 0.004 meV, F 1.3 meV, Tr[n] per channel to 4 decimals
(channel-swapped when the SCF lands on the -m branch — degenerate without
SOC). Metallic residual floors at ~1e-1 like the plain Ni spin gate, so the
gate is energies/moment/occupations, not the converged flag."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


@pytest.mark.slow
def test_paw_hubbard_ni_vs_qe():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "ni_paw_hubbard_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell = np.array([[0.0, 1.76, 1.76], [1.76, 0.0, 1.76], [1.76, 1.76, 0.0]])
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=50 * RY,
                        kmesh=(4, 4, 4), ecutrho=400 * RY, nbands=18,
                        fft_shape=ref["fft_dims"])
    r = scf_uspp(system, SpinPBE(), nspin=2, start_mag=[0.5],
                 smearing="gaussian", width=0.1, etol=1e-5, rhotol=5e-4,
                 mixing_alpha=0.3, verbose=False, max_iter=80,
                 hubbard=[HubbardManifold(species=0, l=2, u=3.0)])

    dF = abs(float(r["energies"].free_energy) - ref["etot_eV"]) * 1000
    assert dF < 5.0, f"F off by {dF:.2f} meV"
    deu = abs(float(r["energies"].hubbard) - ref["hubbard_eV"]) * 1000
    assert deu < 1.0, f"E_U off by {deu:.3f} meV"
    assert abs(abs(r["mag_total"]) - abs(ref["mag_muB"])) < 0.02
    tr = sorted(float(torch.trace(r["hub_occ"][isp][0]).real)
                for isp in range(2))
    qe_tr = sorted(ref["tr_ns_updown"])
    assert max(abs(a - b) for a, b in zip(tr, qe_tr, strict=True)) < 0.01
