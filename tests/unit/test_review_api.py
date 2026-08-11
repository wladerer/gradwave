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
#  the calculator now supports collinear spin (nspin=2); it still rejects      #
#  noncollinear/SOC (nspin=4), which has no calculator path                    #
# --------------------------------------------------------------------------- #
def test_calculator_accepts_nspin2_rejects_noncollinear():
    from gradwave.calculator import GradWave

    # collinear spin constructs (the SCF/forces/stress spin path is wired)
    GradWave(ecut=200.0, pseudopotentials={}, nspin=2)
    # noncollinear/spin-orbit remains gated
    with pytest.raises(ValueError, match="collinear"):
        GradWave(ecut=200.0, pseudopotentials={}, nspin=4)


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

    inp = SimpleNamespace(task="nonsense", output_dir="/tmp/nope", distributed=False)
    with pytest.raises(ValueError, match="scf | relax | bands | magnetism"):
        api.run(inp, verbose=False)


# --------------------------------------------------------------------------- #
#  scf.mixing.precond routes to the loop for both formalisms                   #
# --------------------------------------------------------------------------- #
def _mk_input(tmp_path, pseudo, precond_line):
    from gradwave.inputs import load_input
    from tests.helpers import PSEUDOS

    body = f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: {pseudo}}}
ecut: 400
{precond_line}"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


@pytest.mark.parametrize("pseudo, target_mod, fname", [
    ("C_ONCV_PBE-1.2.upf", "gradwave.scf.loop", "scf"),            # NC path
    ("C.pbe-n-kjpaw_psl.1.0.0.UPF", "gradwave.scf.uspp", "scf_uspp"),  # PAW path
])
def test_precond_reaches_the_scf_call(tmp_path, monkeypatch, pseudo, target_mod, fname):
    """scf.mixing.precond must land as the precond= kwarg on whichever loop the
    formalism dispatches to. build_system and the loop are both stubbed so the
    test asserts only on the plumbing, not on a real SCF."""
    import importlib

    import gradwave.api as api

    inp = _mk_input(tmp_path, pseudo, "scf: {mixing: {precond: local_tf}}\n")

    captured: dict = {}
    # patch the leaf module (api.scf holds run_scf's build_system global), not
    # the package re-export — the latter would leave the real build_system live
    monkeypatch.setattr("gradwave.api.scf.build_system", lambda _inp: object())
    mod = importlib.import_module(target_mod)

    def fake_loop(*_args, **kw):
        captured.update(kw)
        return SimpleNamespace()

    monkeypatch.setattr(mod, fname, fake_loop)
    api.run_scf(inp, verbose=False)
    assert captured["precond"] == "local_tf"


def test_precond_omitted_defaults_to_kerker_at_the_scf_call(tmp_path, monkeypatch):
    """Regression: an input with no precond key must reach the loop with the
    prior default (precond='kerker'), so existing inputs are unchanged."""
    import gradwave.api as api
    import gradwave.scf.loop as loop

    inp = _mk_input(tmp_path, "C_ONCV_PBE-1.2.upf", "")

    captured: dict = {}
    # patch the leaf module (api.scf holds run_scf's build_system global), not
    # the package re-export — the latter would leave the real build_system live
    monkeypatch.setattr("gradwave.api.scf.build_system", lambda _inp: object())

    def fake_scf(*_args, **kw):
        captured.update(kw)
        return SimpleNamespace()

    monkeypatch.setattr(loop, "scf", fake_scf)
    api.run_scf(inp, verbose=False)
    assert captured["precond"] == "kerker"


# --------------------------------------------------------------------------- #
#  shared dispersion compute core (api._compute_dispersion)                    #
#                                                                             #
#  api._apply_dispersion (YAML pipeline) and GradWave._apply_dispersion       #
#  (calculator) used to carry two byte-for-byte copies of the D3/D4 resolve+  #
#  evaluate block — a drift hazard. They now both delegate to                 #
#  api._compute_dispersion; these lock that seam to the direct config path so #
#  neither caller can silently diverge from postscf/dispersion*.              #
# --------------------------------------------------------------------------- #
# rattled CO in a box: C and O both D3/D4-covered, within dispersion range.
_DISP_POS = torch.tensor([[3.15, 3.20, 3.20], [3.15, 3.20, 4.33]], dtype=torch.float64)
_DISP_CELL = torch.eye(3, dtype=torch.float64).numpy() * 6.4
_DISP_Z = [6, 8]


def test_compute_dispersion_d3_matches_direct_config():
    from gradwave.api import _compute_dispersion
    from gradwave.postscf.dispersion import (
        D3Config,
        dispersion_energy,
        dispersion_forces,
        dispersion_stress,
    )

    cfg = D3Config.resolve("pbe", cutoff_ang=21.2, cn_cutoff_ang=10.6)
    cell_t = torch.as_tensor(_DISP_CELL, dtype=torch.float64)
    e_ref = float(dispersion_energy(_DISP_POS, cell_t, _DISP_Z, cfg))
    f_ref = dispersion_forces(_DISP_POS, _DISP_CELL, _DISP_Z, cfg)
    s_ref = dispersion_stress(_DISP_POS, _DISP_CELL, _DISP_Z, cfg)

    terms = _compute_dispersion(
        _DISP_POS, _DISP_CELL, _DISP_Z, method="d3", functional="pbe",
        cutoff_ang=21.2, cn_cutoff_ang=10.6,
    )
    assert float(terms.energy) == pytest.approx(e_ref, rel=1e-12)
    assert torch.allclose(terms.forces, f_ref)
    assert torch.allclose(terms.stress, s_ref)
    # the resolved config the summary block reports its damping from
    assert (terms.cfg.s6, terms.cfg.s8, terms.cfg.a1, terms.cfg.a2) == (
        cfg.s6, cfg.s8, cfg.a1, cfg.a2)


def test_compute_dispersion_d4_matches_direct_config():
    from gradwave.api import _compute_dispersion
    from gradwave.postscf.dispersion_d4 import (
        D4Config,
        dispersion_energy,
        dispersion_forces,
    )

    cfg = D4Config.resolve("pbe", charge=0.0, cutoff_ang=21.2, cn_cutoff_ang=10.6)
    cell_t = torch.as_tensor(_DISP_CELL, dtype=torch.float64)
    e_ref = float(dispersion_energy(_DISP_POS, cell_t, _DISP_Z, cfg))
    f_ref = dispersion_forces(_DISP_POS, _DISP_CELL, _DISP_Z, cfg)

    terms = _compute_dispersion(
        _DISP_POS, _DISP_CELL, _DISP_Z, method="d4", functional="pbe",
        charge=0.0, cutoff_ang=21.2, cn_cutoff_ang=10.6,
    )
    assert float(terms.energy) == pytest.approx(e_ref, rel=1e-12)
    assert torch.allclose(terms.forces, f_ref)


def test_compute_dispersion_skips_stress_when_not_needed():
    """The calculator passes need_stress=False when 'stress' isn't requested;
    the shared core must then skip the stress evaluation and return None."""
    from gradwave.api import _compute_dispersion

    terms = _compute_dispersion(
        _DISP_POS, _DISP_CELL, _DISP_Z, method="d3", functional="pbe",
        cutoff_ang=21.2, cn_cutoff_ang=10.6, need_stress=False,
    )
    assert terms.stress is None
    assert terms.forces is not None


def test_compute_dispersion_uncovered_element_raises():
    """An element with no vendored reference raises (callers catch and degrade
    to their own not-available shape)."""
    from gradwave.api import _compute_dispersion

    with pytest.raises((ValueError, NotImplementedError)):
        _compute_dispersion(
            _DISP_POS, _DISP_CELL, [6, 118], method="d3", functional="pbe",
            cutoff_ang=21.2, cn_cutoff_ang=10.6,
        )
