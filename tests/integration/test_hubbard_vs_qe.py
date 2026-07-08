"""DFT+U validation: NiO AFM-II with Hubbard U on the Ni 3d manifold vs QE 7.5
(HUBBARD `atomic` projectors, same PseudoDojo pseudos, cell, ecut, k-mesh,
smearing). The physical +U observables — Hubbard energy, occupation matrix,
and magnetization — must match QE; the absolute total energy carries a
constant PseudoDojo NLCC/semicore reference offset that cancels in differences,
so it is NOT asserted here (see reference.json note).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


@pytest.mark.slow
def test_nio_afm_hubbard_vs_qe():
    torch.set_num_threads(8)
    ref = json.load(open(FIX / "nio_afm_u_ci" / "reference.json"))
    cell = np.array(ref["cell_angstrom"])
    frac = np.array(ref["positions_crystal"])
    ni = parse_upf(FIX / "pseudos" / ref["pseudos"]["Ni"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(cell, frac @ cell, [0, 0, 1, 1], [ni, o],
                          ecut=ref["ecutwfc_ry"] * RY, kmesh=tuple(ref["kmesh"]),
                          nbands=40)
    res = scf(system, SpinPBE(), smearing=ref["smearing"],
              width=ref["degauss_ry"] * RY, etol=1e-6, rhotol=1e-5, verbose=False,
              nspin=2, start_mag=[+0.5, -0.5, 0, 0],
              hubbard=[HubbardManifold(species=0, l=2, u=ref["U_eV"], j=0.0)],
              max_iter=100)
    assert res.converged

    e_u = float(res.energies.hubbard)
    assert abs(e_u - ref["hubbard_energy_eV"]) < 0.02, e_u  # E_U within 20 meV

    # occupation matrix: the two Ni are opposite spin (AFM), traces vs QE
    up1 = float(torch.trace(res.hub_occ[0][0]).real)
    dn1 = float(torch.trace(res.hub_occ[1][0]).real)
    up2 = float(torch.trace(res.hub_occ[0][1]).real)
    dn2 = float(torch.trace(res.hub_occ[1][1]).real)
    assert abs(up1 - ref["ni_d_occ_up"]) < 0.02
    assert abs(dn1 - ref["ni_d_occ_dn"]) < 0.02
    assert abs((up2 - dn2) + (up1 - dn1)) < 1e-3  # exactly antiferromagnetic
    assert abs((up1 + dn1) - ref["ni_d_total"]) < 0.02

    # magnetization (spin-density integral)
    assert abs(res.mag_abs - ref["abs_magnetization_uB"]) < 0.05
    assert abs(res.mag_total) < 1e-3  # AFM: net zero


@pytest.mark.slow
def test_nio_linear_response_u_vs_hp():
    """The code computes its own U (Cococcioni linear response) vs QE hp.x DFPT.

    hp.x with nq=1,1,1 perturbs the same single cell as gradwave's rigid-probe
    finite difference, so the two are directly comparable. Reference generated
    with the two-step insulator procedure (see reference.json note).
    """
    from gradwave.postscf.hubbard_u import linear_response_u

    torch.set_num_threads(8)
    ref = json.load(open(FIX / "nio_hp" / "reference.json"))
    cell = np.array(ref["cell_angstrom"])
    frac = np.array(ref["positions_crystal"])
    ni = parse_upf(FIX / "pseudos" / ref["pseudos"]["Ni"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(cell, frac @ cell, [0, 0, 1, 1], [ni, o],
                          ecut=ref["ecutwfc_ry"] * RY, kmesh=tuple(ref["kmesh"]),
                          nbands=40)
    out = linear_response_u(system, SpinPBE(), l=2, species=0, site=0, alpha=0.1,
                            smearing="gaussian", width=0.05,
                            scf_kwargs=dict(etol=1e-7, rhotol=1e-6, verbose=False,
                                            nspin=2, start_mag=[+0.5, -0.5, 0, 0],
                                            max_iter=150))
    # localizing perturbation: |chi0| > |chi|, both negative
    assert out["chi0"] < out["chi"] < 0.0
    assert abs(out["U_eV"] - ref["hubbard_U_eV"]) < 0.15, out["U_eV"]


@pytest.mark.slow
def test_nio_energy_derivative_u_hellmann_feynman():
    """dE_total/dU from the Hellmann-Feynman identity Σ ½Tr[n(1−n)] vs
    finite-difference SCF re-runs — U as a first-class differentiable parameter.

    ecut 40 Ry is required: at 30 Ry the semicore PseudoDojo Ni doesn't converge
    and lands on a different occupation branch, breaking the FD comparison."""
    from gradwave.postscf.hubbard_u import energy_derivative_u

    torch.set_num_threads(8)
    ref = json.load(open(FIX / "nio_hp" / "reference.json"))
    cell = np.array(ref["cell_angstrom"])
    frac = np.array(ref["positions_crystal"])
    ni = parse_upf(FIX / "pseudos" / ref["pseudos"]["Ni"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(cell, frac @ cell, [0, 0, 1, 1], [ni, o],
                          ecut=40 * RY, kmesh=(2, 2, 2), nbands=40)
    kw = dict(width=0.05, etol=1e-7, rhotol=1e-6, verbose=False, nspin=2,
              start_mag=[+0.5, -0.5, 0, 0], max_iter=150, smearing="gaussian")
    u0, du = 5.0, 0.1
    man = lambda u: [HubbardManifold(species=0, l=2, u=u, j=0.0)]  # noqa: E731
    res0 = scf(system, SpinPBE(), hubbard=man(u0), **kw)
    assert res0.converged
    de_hf = energy_derivative_u(res0, man(u0))
    resp = scf(system, SpinPBE(), hubbard=man(u0 + du), **kw)
    resm = scf(system, SpinPBE(), hubbard=man(u0 - du), **kw)
    assert resp.converged and resm.converged
    de_fd = (float(resp.energies.total) - float(resm.energies.total)) / (2 * du)
    assert abs(de_hf - de_fd) < 1e-4, (de_hf, de_fd)


@pytest.mark.slow
def test_nio_linear_response_u_autodiff():
    """Analytic (Sternheimer) linear-response U: no finite differences, no
    probe SCF re-runs — one ground state, Hxc kernel via autograd HVP. Must
    match hp.x DFPT and the FD implementation (which it replaces)."""
    from gradwave.postscf.hubbard_u import linear_response_u_autodiff

    torch.set_num_threads(8)
    ref = json.load(open(FIX / "nio_hp" / "reference.json"))
    cell = np.array(ref["cell_angstrom"])
    frac = np.array(ref["positions_crystal"])
    ni = parse_upf(FIX / "pseudos" / ref["pseudos"]["Ni"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(cell, frac @ cell, [0, 0, 1, 1], [ni, o],
                          ecut=ref["ecutwfc_ry"] * RY, kmesh=tuple(ref["kmesh"]),
                          nbands=40)
    out = linear_response_u_autodiff(
        system, SpinPBE(), l=2, species=0, site=0, smearing="gaussian", width=0.05,
        scf_kwargs=dict(etol=1e-7, rhotol=1e-6, verbose=False, nspin=2,
                        start_mag=[+0.5, -0.5, 0, 0], max_iter=150))
    # infinitesimal response vs the FD (alpha=0.1) columns of this session
    assert abs(out["chi0"] - (-0.21360)) < 2e-3
    assert abs(out["chi"] - (-0.08733)) < 2e-3
    assert abs(out["U_eV"] - ref["hubbard_U_eV"]) < 0.15, out["U_eV"]  # vs hp.x
    assert out["n_outer"] < 50  # Anderson-accelerated fixed point
