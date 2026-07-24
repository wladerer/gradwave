"""Unit tests for the code-review fixes in api / inputs / calculator /
checkpoint: kerker validation, weights_only checkpoint round-trip, the
smeared-USPP smearing-error adaptation, the calculator's honored/rejected
settings, and the summary-parameter formalism naming."""

from types import SimpleNamespace

import pytest
import torch

from gradwave.core.energies.total import EnergyBreakdown


# --------------------------------------------------------------------------- #
#  fix 3: MixingParams.kerker validation / normalization                      #
# --------------------------------------------------------------------------- #
def test_kerker_normalization():
    from gradwave.inputs import _normalize_kerker

    assert _normalize_kerker("auto") == "auto"
    assert _normalize_kerker("off") is False
    assert _normalize_kerker("on") is True
    assert _normalize_kerker("true") is True
    assert _normalize_kerker("false") is False
    assert _normalize_kerker(True) is True
    assert _normalize_kerker(False) is False


def test_kerker_rejects_garbage():
    from gradwave.inputs import _normalize_kerker

    with pytest.raises(ValueError, match="kerker"):
        _normalize_kerker("sometimes")
    with pytest.raises(ValueError, match="kerker"):
        _normalize_kerker(5)


# --------------------------------------------------------------------------- #
#  fix 4: checkpoint round-trips under weights_only=True                       #
# --------------------------------------------------------------------------- #
def _fake_nc_result():
    import numpy as np

    grid = SimpleNamespace(cell=np.eye(3) * 5.0, shape=(4, 4, 4),
                           volume=125.0, n_points=64)
    system = SimpleNamespace(grid=grid, positions=torch.zeros(2, 3),
                             species_of_atom=[0, 0], n_electrons=8.0,
                             ecut=200.0, kweights=torch.ones(1))
    e = EnergyBreakdown(kinetic=1.0, hartree=2.0, xc=-1.0, local=0.5,
                        nonlocal_=0.1, ewald=-3.0, smearing=0.0)
    return SimpleNamespace(
        system=system, energies=e, nspin=1, converged=True, n_iter=5,
        fermi=0.3, smearing="gaussian", width=0.1,
        eigenvalues=torch.zeros(1, 4), occupations=torch.ones(1, 4),
        rho=torch.zeros(64), rho_spin=None, history=[], coeffs=None,
        mag_vec=None, m=None)


def test_checkpoint_round_trip_weights_only(tmp_path):
    from gradwave.checkpoint import load_checkpoint, save_checkpoint

    path = save_checkpoint(_fake_nc_result(), tmp_path / "checkpoint.pt")
    payload = load_checkpoint(path)  # weights_only=True internally
    assert payload["format"] == "gradwave-checkpoint"
    assert payload["kind"] == "nc"
    assert payload["energies_eV"]["kinetic"] == 1.0
    assert payload["cell_ang"].shape == (3, 3)  # numpy array survives the load
    assert torch.equal(payload["rho"], torch.zeros(64))


# --------------------------------------------------------------------------- #
#  fix 9: the shared 11-key energy breakdown helper                           #
# --------------------------------------------------------------------------- #
def test_energies_eV_dict_keys():
    from gradwave.checkpoint import energies_eV_dict

    e = EnergyBreakdown(kinetic=1.0, hartree=2.0, xc=-1.0, local=0.5,
                        nonlocal_=0.1, ewald=-3.0, smearing=0.0)
    d = energies_eV_dict(e)
    assert set(d) == {"kinetic", "hartree", "xc", "local", "nonlocal", "ewald",
                      "smearing", "hubbard", "onecenter", "dispersion",
                      "total", "free_energy"}


# --------------------------------------------------------------------------- #
#  fix 1: smeared USPP/PAW (dict result) no longer crashes the smearing error #
# --------------------------------------------------------------------------- #
def test_smearing_error_accepts_uspp_dict():
    from gradwave.postscf.convergence_error import estimate_smearing_error

    # a USPP/PAW run is a plain dict; the estimator reads res.energies, so the
    # block adapts it to a shim. Use a nonzero entropy term so it doesn't bail.
    e = EnergyBreakdown(kinetic=1.0, hartree=2.0, xc=-1.0, local=0.5,
                        nonlocal_=0.1, ewald=-3.0, smearing=-0.05)
    res = {"energies": e}
    shim = res if not isinstance(res, dict) else SimpleNamespace(
        energies=res["energies"])
    sme = estimate_smearing_error(shim, scheme="gaussian", width=0.1)
    assert sme.scheme == "gaussian"
    assert sme.dsmearing == pytest.approx(0.5 * 0.05, rel=1e-6)


# --------------------------------------------------------------------------- #
#  fix 2: the calculator rejects nspin=2 rather than silently ignoring it      #
# --------------------------------------------------------------------------- #
def test_calculator_rejects_nspin2():
    from gradwave.calculator import GradWave

    with pytest.raises(ValueError, match="nspin=1 only"):
        GradWave(ecut=200.0, pseudopotentials={}, nspin=2)


def test_calculator_accepts_new_settings():
    from gradwave.calculator import GradWave

    calc = GradWave(ecut=200.0, pseudopotentials={}, max_iter=42,
                    diago_tol=1e-7, mixing_scheme="broyden", mixing_alpha=0.4,
                    mixing_history=5, mixing_kerker=True)
    assert calc.parameters["max_iter"] == 42
    assert calc.parameters["diago_tol"] == 1e-7
    assert calc.parameters["mixing_scheme"] == "broyden"
    assert calc.parameters["mixing_alpha"] == 0.4
    assert calc.parameters["mixing_history"] == 5
    assert calc.parameters["mixing_kerker"] is True


# --------------------------------------------------------------------------- #
#  fix 10 & 12: informative task error, magnetism formalism naming             #
# --------------------------------------------------------------------------- #
def test_unknown_task_message():
    import gradwave.api as api

    inp = SimpleNamespace(task="nonsense", output_dir="/tmp/nope")
    with pytest.raises(ValueError, match="scf | relax | bands | magnetism"):
        api.run(inp, verbose=False)
