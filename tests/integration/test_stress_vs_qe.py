"""Stress tensor vs Quantum ESPRESSO at identical parameters.

The fixed-basis (Nielsen–Martin) stress convention is shared: at the same
ecut/k/UPF the two codes carry the same basis-set incompleteness, so the
analytic tensors agree to the SCF convergence level (observed ≤ 0.006 kbar
across 90–14000 kbar magnitudes). QE prints −(1/Ω)∂E/∂ε; gradwave returns
+(1/Ω)∂E/∂ε, hence the sign flip in the assertions.

Cases probe distinct terms: Si (sheared low-symmetry cell + displaced atom →
full anisotropic tensor), Al (smeared metal, Kerker path), MgO (ionic, two
species), Ni (NLCC core stress — slow).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stress import _energy_strained, stress, stress_kbar
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994

SI_CELL = np.array([
    [0.0108600000, 2.7041400000, 2.7285750000],
    [2.7421500000, 0.0162900000, 2.7231450000],
    [2.7530100000, 2.7095700000, 0.0054300000],
])
SI_POS = np.array([[0.0, 0.0, 0.0], [1.4265050000, 1.3275000000, 1.3842875000]])

CASES = {
    "si_stress_ci": dict(
        cell=SI_CELL, pos=SI_POS,
        pseudos=("Si_ONCV_PBE-1.2.upf",), species=[0, 0],
        ecut=30 * RY, kmesh=(2, 2, 2), smearing="none", nbands=None, nat=2, slow=False,
    ),
    "al_stress_ci": dict(
        cell=np.array([
            [0.0121500000, 2.0331000000, 2.0209500000],
            [2.0088000000, 0.0121500000, 2.0209500000],
            [2.0209500000, 2.0452500000, 0.0000000000],
        ]),
        pos=np.zeros((1, 3)),
        pseudos=("Al_ONCV_PBE-1.2.upf",), species=[0],
        ecut=20 * RY, kmesh=(4, 4, 4), smearing="gaussian", nbands=10, nat=1, slow=False,
    ),
    "mgo_stress_ci": dict(
        cell=np.array([
            [0.0, 2.07441, 2.07441], [2.07441, 0.0, 2.07441], [2.07441, 2.07441, 0.0],
        ]),
        pos=np.array([[0.0, 0.0, 0.0], [2.07441, 2.07441, 2.07441]]),
        pseudos=("Mg_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf"), species=[0, 1],
        ecut=40 * RY, kmesh=(2, 2, 2), smearing="none", nbands=None, nat=2, slow=False,
    ),
    # NLCC: PseudoDojo Ni has a sharp model core; also exercises QE's 10-bohr
    # msh truncation of the local channel (the PD_Ni mesh reaches 13.7 bohr)
    "ni_stress_ci": dict(
        cell=np.array([
            [0.0, 1.7776, 1.7776], [1.7776, 0.0, 1.7776], [1.7776, 1.7776, 0.0],
        ]),
        pos=np.zeros((1, 3)),
        pseudos=("PD_Ni_PBE.upf",), species=[0],
        ecut=60 * RY, kmesh=(4, 4, 4), smearing="gaussian", nbands=14, nat=1, slow=True,
    ),
}


def run_case(name):
    cfg = CASES[name]
    ref = json.loads((FIX / name / "reference.json").read_text())
    upfs = [parse_upf(FIX / "pseudos" / p) for p in cfg["pseudos"]]
    system = setup_system(
        cfg["cell"], cfg["pos"], cfg["species"], upfs,
        ecut=cfg["ecut"], kmesh=cfg["kmesh"], nbands=cfg["nbands"],
        fft_shape=ref["fft_dims"],
    )
    res = scf(system, PBE(), smearing=cfg["smearing"], width=0.1,
              etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged, name

    de_mev = abs(float(res.energies.free_energy) - ref["etot_eV"]) / cfg["nat"] * 1000
    assert de_mev < 1.0, f"{name}: energy off by {de_mev:.4f} meV/atom"

    sig = stress(res, PBE())
    sig_qe = np.array(ref["stress_ev_a3"])  # QE sign: −(1/Ω)∂E/∂ε
    dmax_kbar = float(stress_kbar(torch.as_tensor(np.abs(-sig.cpu().numpy() - sig_qe))).max())
    assert dmax_kbar < 0.05, f"{name}: stress off by {dmax_kbar:.4f} kbar"
    return res


@pytest.mark.standard
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if not c["slow"]])
def test_stress_vs_qe_ci(name):
    torch.set_num_threads(4)
    run_case(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c["slow"]])
def test_stress_vs_qe_slow(name):
    torch.set_num_threads(8)
    run_case(name)


@pytest.mark.standard
def test_stress_with_symmetry():
    """IBZ k-set + stress symmetrization must reproduce the full-mesh QE
    tensor: the strain derivative of the IBZ-restricted expression is only
    correct after averaging over the point group (QE does the same)."""
    torch.set_num_threads(4)
    cfg = CASES["mgo_stress_ci"]
    ref = json.loads((FIX / "mgo_stress_ci" / "reference.json").read_text())
    upfs = [parse_upf(FIX / "pseudos" / p) for p in cfg["pseudos"]]
    system = setup_system(cfg["cell"], cfg["pos"], cfg["species"], upfs,
                          ecut=cfg["ecut"], kmesh=cfg["kmesh"],
                          fft_shape=ref["fft_dims"], use_symmetry=True)
    assert system.sym is not None and system.sym.n_ops > 1
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    sig = stress(res, PBE())
    sig_qe = np.array(ref["stress_ev_a3"])
    dmax = float(stress_kbar(torch.as_tensor(np.abs(-sig.cpu().numpy() - sig_qe))).max())
    assert dmax < 0.05, f"symmetrized stress off by {dmax:.4f} kbar"
    off = sig.cpu().numpy() - np.diag(np.diag(sig.cpu().numpy()))
    assert np.abs(off).max() < 1e-10  # cubic: exactly diagonal after averaging


def test_stress_autograd_vs_fd():
    """Autograd of the strained expression vs central finite differences —
    validates the ε-parameterization wiring independent of QE."""
    torch.set_num_threads(4)
    cfg = CASES["si_stress_ci"]
    upfs = [parse_upf(FIX / "pseudos" / p) for p in cfg["pseudos"]]
    system = setup_system(cfg["cell"], cfg["pos"], cfg["species"], upfs,
                          ecut=12 * RY, kmesh=(1, 1, 1))
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged

    # the ε=0 expression must reproduce the SCF energy
    e0 = _energy_strained(res, PBE(), torch.zeros(3, 3, dtype=torch.float64))
    assert abs(float(e0) - float(res.energies.total)) < 1e-7

    sig = stress(res, PBE(), symmetrize=False).cpu().numpy()
    d = 1e-6

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (float(_energy_strained(res, PBE(), ep))
                - float(_energy_strained(res, PBE(), -ep))) / (2 * d)

    for i, j in [(0, 0), (0, 1), (2, 1)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / system.grid.volume
        assert abs(sig[i, j] - fd_sym) < 1e-7, (i, j, sig[i, j], fd_sym)
