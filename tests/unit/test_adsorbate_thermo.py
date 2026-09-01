"""Adsorbate / gas-phase free-energy thermochemistry.

Cross-checks :mod:`gradwave.postscf.adsorbate_thermo` (and its api wrappers)
against ASE's ``IdealGasThermo`` / ``HarmonicThermo`` on known molecules (H2, CO,
O2, H2O) with literature frequencies, plus the CHE ΔG(U, pH) shift, imaginary-
mode handling, and the differentiable dΔG/dU angle. Pure arithmetic — no SCF.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

ase_build = pytest.importorskip("ase.build")
ase_units = pytest.importorskip("ase.units")
ase_thermo = pytest.importorskip("ase.thermochemistry")

from ase.build import molecule  # noqa: E402
from ase.thermochemistry import HarmonicThermo, IdealGasThermo  # noqa: E402
from ase.units import invcm  # noqa: E402

from gradwave.constants import KB_EV  # noqa: E402
from gradwave.postscf import adsorbate_thermo as at  # noqa: E402

T = 298.15
P = 1.0e5

# Literature harmonic frequencies (cm^-1) for the gas references.
_GAS = {
    "H2": ("linear", 2, 0.0, [4401.0]),
    "CO": ("linear", 1, 0.0, [2143.0]),
    "O2": ("linear", 2, 1.0, [1580.0]),
    "H2O": ("nonlinear", 2, 0.0, [1595.0, 3657.0, 3756.0]),
}


@pytest.mark.parametrize("name", list(_GAS))
def test_ideal_gas_matches_ase(name: str) -> None:
    """ZPE / U / H / S / G of a gas molecule match ASE's IdealGasThermo.

    Feeds identical vib energies and geometry to both; the only difference is the
    CODATA vintage of the constants (ASE 3.29 uses CODATA-2014, gradwave 2018),
    so agreement is to ~1e-7 eV, well inside a tight tolerance.
    """
    geom, sigma, spin, freqs_cm = _GAS[name]
    atoms = molecule(name)
    vib_ev = [f * invcm for f in freqs_cm]
    ase_ig = IdealGasThermo(
        vib_energies=vib_ev, geometry=geom, atoms=atoms,
        symmetrynumber=sigma, spin=spin, potentialenergy=0.0,
    )
    g_ase = ase_ig.get_gibbs_energy(T, P, verbose=False)
    h_ase = ase_ig.get_enthalpy(T, verbose=False)
    s_ase = ase_ig.get_entropy(T, P, verbose=False)
    u_ase = ase_ig.get_internal_energy(T, verbose=False)
    zpe_ase = ase_ig.get_ZPE_correction()

    r = at.ideal_gas_thermo(
        vib_ev, geom, temperature=T, potentialenergy=0.0,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=sigma, spin=spin, pressure=P,
    )
    assert float(r["zero_point_energy"]) == pytest.approx(zpe_ase, abs=1e-9)
    assert float(r["internal_energy"]) == pytest.approx(u_ase, abs=1e-5)
    assert float(r["enthalpy"]) == pytest.approx(h_ase, abs=1e-5)
    assert float(r["entropy"]) == pytest.approx(s_ase, abs=1e-7)
    assert float(r["gibbs_energy"]) == pytest.approx(g_ase, abs=1e-5)
    assert r["n_imag"] == 0


def test_pressure_dependence_matches_ase() -> None:
    """Partial-pressure entropy term reproduces ASE at a non-standard pressure."""
    atoms = molecule("CO")
    vib_ev = [2143.0 * invcm]
    p = 100.0  # 1 mbar
    ase_ig = IdealGasThermo(
        vib_energies=vib_ev, geometry="linear", atoms=atoms,
        symmetrynumber=1, spin=0.0,
    )
    g_ase = ase_ig.get_gibbs_energy(T, p, verbose=False)
    r = at.ideal_gas_thermo(
        vib_ev, "linear", temperature=T,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=1, spin=0.0, pressure=p,
    )
    assert float(r["gibbs_energy"]) == pytest.approx(g_ase, abs=1e-5)


def test_harmonic_adsorbate_matches_ase() -> None:
    """U / S / F of a harmonic adsorbate match ASE's HarmonicThermo."""
    vib_cm = [2000.0, 1500.0, 1000.0, 500.0, 200.0, 100.0]
    vib_ev = [f * invcm for f in vib_cm]
    e_pot = -5.0
    ase_h = HarmonicThermo(vib_energies=vib_ev, potentialenergy=e_pot)
    u_ase = ase_h.get_internal_energy(T, verbose=False)
    s_ase = ase_h.get_entropy(T, verbose=False)
    f_ase = ase_h.get_helmholtz_energy(T, verbose=False)

    r = at.harmonic_thermo(vib_ev, temperature=T, potentialenergy=e_pot)
    assert float(r["internal_energy"]) == pytest.approx(u_ase, abs=1e-6)
    assert float(r["entropy"]) == pytest.approx(s_ase, abs=1e-8)
    assert float(r["helmholtz_energy"]) == pytest.approx(f_ase, abs=1e-6)
    assert r["n_imag"] == 0


def test_zpe_and_free_energy_ordering() -> None:
    """ZPE is exact ½Σħω and F ≤ U (entropy lowers the free energy at T>0)."""
    vib_ev = np.array([0.25, 0.10, 0.05])
    r = at.harmonic_thermo(vib_ev, temperature=T)
    assert float(r["zero_point_energy"]) == pytest.approx(0.5 * vib_ev.sum())
    assert float(r["helmholtz_energy"]) < float(r["internal_energy"])


def test_imaginary_mode_raises_and_drops() -> None:
    """A negative (imaginary) mode raises by default, drops with the flag."""
    vib_ev = [0.20, 0.10, -0.03]  # one imaginary mode
    with pytest.raises(ValueError, match="imaginary"):
        at.harmonic_thermo(vib_ev, temperature=T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = at.harmonic_thermo(vib_ev, temperature=T, ignore_imag_modes=True)
    assert r["n_imag"] == 1
    # ZPE now only over the two real modes.
    assert float(r["zero_point_energy"]) == pytest.approx(0.5 * (0.20 + 0.10))


def test_transition_state_one_imaginary_mode() -> None:
    """A first-order TS has exactly one imaginary mode; dropping it is clean."""
    vib_ev = [0.30, 0.20, 0.10, -0.05]
    e_clean, n_imag = at.clean_vib_energies(vib_ev, ignore_imag_modes=True)
    assert n_imag == 1
    assert e_clean.numel() == 3


def test_che_hydrogen_reference_is_half_h2() -> None:
    """The CHE H reference is exactly ½ G(H2)."""
    atoms = molecule("H2")
    vib_ev = [4401.0 * invcm]
    g_h2 = at.ideal_gas_thermo(
        vib_ev, "linear", temperature=T, potentialenergy=-6.7,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=2, spin=0.0,
    )["gibbs_energy"]
    ref = at.hydrogen_reference(g_h2)
    assert float(ref) == pytest.approx(0.5 * float(g_h2))


def test_electrode_potential_shift_formula() -> None:
    """ΔG(U, pH) = ΔG0 + n·(eU + kT·ln10·pH), exact and per-electron."""
    dg0 = -0.35
    for u, ph, n in [(0.0, 0.0, 1), (0.5, 3.0, 1), (-0.2, 7.0, 2)]:
        got = at.electrode_potential_shift(
            dg0, potential_v=u, ph=ph, temperature=T, n_electrons=n
        )
        expected = dg0 + n * (u + KB_EV * T * np.log(10.0) * ph)
        assert float(got) == pytest.approx(expected, abs=1e-12)
    # U = 0, pH = 0 is the CHE reference: no shift.
    assert float(at.electrode_potential_shift(dg0)) == pytest.approx(dg0)


def test_delta_g_ads_assembly_matches_hand_composition() -> None:
    """ΔG_ads = F(slab+ads) − E_slab − ½·G(H2) reproduces a by-hand assembly."""
    atoms = molecule("H2")
    e_slab_ads, e_slab, e_gas = -105.30, -100.00, -6.70
    ads_vib = [c * invcm for c in (1200.0, 800.0, 700.0, 400.0, 350.0, 300.0)]
    h2_vib = [4401.0 * invcm]

    r = at.adsorption_free_energy(
        energy_slab_ads=e_slab_ads, energy_slab=e_slab, energy_gas=e_gas,
        temperature=T, ads_vib_energies_ev=ads_vib, gas_vib_energies_ev=h2_vib,
        gas_geometry="linear", gas_masses_amu=atoms.get_masses(),
        gas_moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        gas_symmetrynumber=2, gas_spin=0.0, stoich_gas=0.5,
    )
    f_ads = at.harmonic_thermo(
        ads_vib, temperature=T, potentialenergy=e_slab_ads
    )["helmholtz_energy"]
    g_h2 = at.ideal_gas_thermo(
        h2_vib, "linear", temperature=T, potentialenergy=e_gas,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=2, spin=0.0,
    )["gibbs_energy"]
    expected = float(f_ads) - e_slab - 0.5 * float(g_h2)
    assert float(r["delta_g"]) == pytest.approx(expected, abs=1e-12)
    assert r["n_imag_ads"] == 0


def test_delta_g_ads_differentiable_in_energy_and_potential() -> None:
    """dΔG/d(E_slab+ads) = 1 and, after the CHE shift, dΔG/dU = n via autograd."""
    atoms = molecule("H2")
    e_slab_ads = torch.tensor(-105.30, dtype=torch.float64, requires_grad=True)
    u = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    ads_vib = torch.tensor(
        [c * invcm for c in (1200.0, 800.0, 700.0, 400.0, 350.0, 300.0)],
        dtype=torch.float64,
    )
    h2_vib = [4401.0 * invcm]

    r = at.adsorption_free_energy(
        energy_slab_ads=e_slab_ads, energy_slab=-100.0, energy_gas=-6.70,
        temperature=T, ads_vib_energies_ev=ads_vib, gas_vib_energies_ev=h2_vib,
        gas_geometry="linear", gas_masses_amu=atoms.get_masses(),
        gas_moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        gas_symmetrynumber=2, gas_spin=0.0, stoich_gas=0.5,
    )
    dg_u = at.electrode_potential_shift(
        r["delta_g"], potential_v=u, ph=0.0, temperature=T, n_electrons=1
    )
    dg_u.backward()
    assert e_slab_ads.grad is not None
    assert float(e_slab_ads.grad) == pytest.approx(1.0, abs=1e-10)
    assert float(u.grad) == pytest.approx(1.0, abs=1e-10)  # Nernstian slope n=1


def test_dG_differentiable_in_frequencies() -> None:
    """ΔG is differentiable in the mode energies (Sabatier-fit gradient path)."""
    ads_vib = torch.tensor([0.15, 0.10, 0.08, 0.05, 0.03, 0.02], requires_grad=True)
    r = at.harmonic_thermo(ads_vib, temperature=T, potentialenergy=-5.0)
    r["helmholtz_energy"].backward()
    assert ads_vib.grad is not None
    # ∂F/∂ħω = ∂ZPE/∂ħω + ... ; at least the ½ from ZPE, positive overall here.
    assert torch.all(ads_vib.grad > 0.0)


def test_api_wrapper_matches_postscf_and_detects_geometry() -> None:
    """The ASE-aware api wrapper reproduces the postscf result and geometry."""
    from gradwave.api import molecule_ideal_gas_thermo
    from gradwave.api.thermochem import _detect_geometry

    atoms = molecule("H2O")
    assert _detect_geometry(atoms) == "nonlinear"
    assert _detect_geometry(molecule("CO2")) == "linear"

    vib_ev = [f * invcm for f in (1595.0, 3657.0, 3756.0)]
    r_api = molecule_ideal_gas_thermo(
        atoms, vib_ev, temperature=T, symmetrynumber=2, spin=0.0,
    )
    r_ref = at.ideal_gas_thermo(
        vib_ev, "nonlinear", temperature=T,
        masses_amu=atoms.get_masses(),
        moments_of_inertia_amu_a2=atoms.get_moments_of_inertia(),
        symmetrynumber=2, spin=0.0,
    )
    assert float(r_api["gibbs_energy"]) == pytest.approx(
        float(r_ref["gibbs_energy"]), abs=1e-12
    )


def test_cm1_ev_roundtrip() -> None:
    """cm^-1 ↔ eV conversion round-trips and matches ASE's invcm."""
    freqs = np.array([100.0, 1500.0, 4401.0])
    ev = at.cm1_to_ev(freqs)
    assert torch.allclose(at.ev_to_cm1(ev), torch.as_tensor(freqs))
    # Same convention as ASE's invcm to ~1e-8 relative.
    assert float(ev[-1]) == pytest.approx(4401.0 * invcm, rel=1e-7)


def test_monatomic_gas_has_no_rotation_or_vibration() -> None:
    """A monatomic gas (Ar) has only translational + electronic entropy."""
    r = at.ideal_gas_thermo(
        [], "monatomic", temperature=T,
        masses_amu=[39.948], spin=0.0, pressure=P,
    )
    terms = r["entropy_terms"]
    assert float(terms["rotational"]) == 0.0
    assert float(terms["vibrational"]) == 0.0
    assert float(terms["translational"]) > 0.0
