"""Free-energy thermochemistry reachable from the YAML input, end to end.

`postscf.adsorbate_thermo` was importable (its wrappers re-exported from
`gradwave.api`) but had no `run()`/Input/CLI task glue. This gate covers the
wiring for the new `thermochem` task: the three modes (harmonic / ideal_gas /
adsorption) each build a summary block whose headline free energy matches a
direct call to the underlying postscf function, and the human report renders.

No SCF runs here, so it is a fast-tier test; the physics of the free-energy
models is validated in tests/unit/test_adsorbate_thermo.py (not touched here).
"""

from pathlib import Path

import pytest


def _write(tmp_path: Path, body: str):
    from gradwave.inputs import load_input

    p = tmp_path / "in.yaml"
    p.write_text(body + f"\noutput:\n  dir: {tmp_path}\n")
    return load_input(p)


_BOX = """
structure:
  cell: [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]]
  positions:
    cart: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.741]]
  species: [H, H]
"""


def test_harmonic_mode_matches_postscf(tmp_path):
    from gradwave.api import run
    from gradwave.postscf.adsorbate_thermo import cm1_to_ev, harmonic_thermo

    inp = _write(tmp_path, _BOX + """
task: thermochem
thermochem:
  mode: harmonic
  temperature: 300.0
  energy: -5.0
  freqs_cm: [1200.0, 800.0, 400.0]
""")
    assert inp.task == "thermochem" and inp.thermochem.mode == "harmonic"
    summary = run(inp, verbose=False)
    tc = summary["thermochem"]

    ref = harmonic_thermo(cm1_to_ev([1200.0, 800.0, 400.0]),
                          temperature=300.0, potentialenergy=-5.0)
    assert tc["free_energy_eV"] == pytest.approx(float(ref["helmholtz_energy"]), abs=1e-9)
    assert tc["zero_point_energy_eV"] == pytest.approx(
        float(ref["zero_point_energy"]), abs=1e-9)
    assert "thermochemistry" in (tmp_path / "thermochem.out").read_text()


def test_ideal_gas_mode_uses_structure(tmp_path):
    from gradwave.api import run

    inp = _write(tmp_path, _BOX + """
task: thermochem
thermochem:
  mode: ideal_gas
  temperature: 298.15
  energy: -6.8
  freqs_cm: [4400.0]
  symmetrynumber: 2
  spin: 0.0
""")
    summary = run(inp, verbose=False)
    tc = summary["thermochem"]
    # H2 is linear (auto-detected from the diatomic structure)
    assert tc["geometry"] == "linear"
    # a gas Gibbs energy has a substantial entropy term: G < H at 298 K
    assert tc["gibbs_energy_eV"] < tc["enthalpy_eV"]
    assert tc["entropy_eV_per_K"] > 0.0


def test_adsorption_mode_matches_and_electrode_shift(tmp_path):
    from gradwave.api import run
    from gradwave.api.thermochem import adsorption_free_energy_from_atoms
    from gradwave.postscf.adsorbate_thermo import (
        cm1_to_ev,
        electrode_potential_shift,
    )

    inp = _write(tmp_path, _BOX + """
task: thermochem
thermochem:
  mode: adsorption
  temperature: 298.15
  energy_slab_ads: -100.0
  energy_slab: -98.0
  energy_gas: -6.8
  stoich_gas: 0.5
  ads_freqs_cm: [1000.0, 800.0, 800.0]
  gas_freqs_cm: [4400.0]
  gas_symmetrynumber: 2
  electrode_potential_v: -0.2
  ph: 0.0
  n_electrons: 1
""")
    summary = run(inp, verbose=False)
    tc = summary["thermochem"]

    ref = adsorption_free_energy_from_atoms(
        energy_slab_ads=-100.0, energy_slab=-98.0, energy_gas=-6.8,
        temperature=298.15, ads_vib_energies_ev=cm1_to_ev([1000.0, 800.0, 800.0]),
        gas_vib_energies_ev=cm1_to_ev([4400.0]), gas_atoms=inp.atoms,
        gas_symmetrynumber=2, stoich_gas=0.5)
    assert tc["delta_g_eV"] == pytest.approx(float(ref["delta_g"]), abs=1e-9)

    # the CHE shift moves ΔG by n·(eU) at pH 0: U=-0.2 V, n=1 → −0.2 eV
    shifted = electrode_potential_shift(ref["delta_g"], potential_v=-0.2,
                                        ph=0.0, temperature=298.15, n_electrons=1)
    assert tc["electrode"]["free_energy_shifted_eV"] == pytest.approx(
        float(shifted), abs=1e-9)
    assert tc["electrode"]["free_energy_shifted_eV"] == pytest.approx(
        tc["delta_g_eV"] - 0.2, abs=1e-9)


def test_adsorption_requires_energies(tmp_path):
    from gradwave.inputs import InputError

    with pytest.raises(InputError, match="requires"):
        _write(tmp_path, _BOX + """
task: thermochem
thermochem:
  mode: adsorption
  ads_freqs_cm: [1000.0]
  gas_freqs_cm: [4400.0]
""")
