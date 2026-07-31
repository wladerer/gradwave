"""The four magnetic spinor knobs (scf.magnetic) and the spinor energy gate,
from the YAML surface through to scf_noncollinear.

Input validation (accept/reject), the johnson moment-collapse guard, and
monkeypatch plumbing that each knob reaches the driver with the campaign's
defaults left bit-for-bit unchanged when omitted. No SCF runs here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gradwave import api
from gradwave.inputs import InputError, load_input
from gradwave.scf.noncollinear import _resolve_mag_mixing_alpha
from tests.helpers import PSEUDOS


def _write(tmp_path, extra: str) -> Path:
    body = f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
noncollinear: true
smearing: {{type: gaussian, width: 0.1}}
{extra}"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


# --------------------------------------------------------------------------- #
#  Input validation: scf.magnetic accept / reject                             #
# --------------------------------------------------------------------------- #


def test_magnetic_defaults(tmp_path):
    inp = load_input(_write(tmp_path, ""))
    mg = inp.scf.magnetic
    assert mg.mixer == "pulay"
    assert mg.spin_precond is False
    assert mg.mixing_alpha is None
    assert mg.diago_schedule == "linear"


def test_magnetic_parses(tmp_path):
    inp = load_input(_write(
        tmp_path,
        "scf: {magnetic: {mixer: johnson, spin_precond: true, "
        "mixing_alpha: 0.3, diago_schedule: quadratic}}\n"))
    mg = inp.scf.magnetic
    assert mg.mixer == "johnson"
    assert mg.spin_precond is True
    assert mg.mixing_alpha == pytest.approx(0.3)
    assert mg.diago_schedule == "quadratic"


@pytest.mark.parametrize("scheme", ["pulay", "johnson", "broyden"])
def test_magnetic_mixer_accepts_every_scheme(tmp_path, scheme):
    inp = load_input(_write(tmp_path, f"scf: {{magnetic: {{mixer: {scheme}}}}}\n"))
    assert inp.scf.magnetic.mixer == scheme


def test_magnetic_mixer_rejects_unknown(tmp_path):
    with pytest.raises(InputError, match="scf.magnetic.mixer must be"):
        load_input(_write(tmp_path, "scf: {magnetic: {mixer: diis}}\n"))


def test_magnetic_diago_schedule_rejects_unknown(tmp_path):
    with pytest.raises(InputError, match="scf.magnetic.diago_schedule must be"):
        load_input(_write(tmp_path, "scf: {magnetic: {diago_schedule: cubic}}\n"))


def test_magnetic_unknown_key_rejected(tmp_path):
    with pytest.raises(InputError, match="did you mean"):
        load_input(_write(tmp_path, "scf: {magnetic: {mixr: johnson}}\n"))


# --------------------------------------------------------------------------- #
#  The johnson moment-collapse guard                                          #
# --------------------------------------------------------------------------- #


def test_resolve_mag_mixing_alpha_guard():
    """An unset m step keeps the pulay/broyden guard max(alpha, 0.6) but drops to
    0.3 for johnson (its normalized update inverts the boost into a collapse
    accelerant). An explicit value always wins."""
    # pulay/broyden default: the moment-collapse guard, bit-for-bit as before
    assert _resolve_mag_mixing_alpha("pulay", None, 0.5) == 0.6
    assert _resolve_mag_mixing_alpha("broyden", None, 0.5) == 0.6
    assert _resolve_mag_mixing_alpha("pulay", None, 0.7) == 0.7
    # johnson default: the safe scheme-dependent step
    assert _resolve_mag_mixing_alpha("johnson", None, 0.5) == 0.3
    assert _resolve_mag_mixing_alpha("johnson", None, 0.7) == 0.3
    # explicit value overrides for every scheme
    assert _resolve_mag_mixing_alpha("johnson", 0.5, 0.5) == 0.5
    assert _resolve_mag_mixing_alpha("pulay", 0.2, 0.5) == 0.2


# --------------------------------------------------------------------------- #
#  Plumbing: each knob reaches scf_noncollinear                               #
# --------------------------------------------------------------------------- #


def _capture(monkeypatch):
    import gradwave.scf.noncollinear as ncmod

    captured: dict = {}

    def fake(system, xc, **kw):
        captured.update(kw)
        return "SENTINEL"

    monkeypatch.setattr(ncmod, "scf_noncollinear", fake)
    return captured


def test_knobs_reach_driver(tmp_path, monkeypatch):
    captured = _capture(monkeypatch)
    inp = load_input(_write(
        tmp_path,
        "scf:\n"
        "  convergence: energy\n"
        "  entol: 5.0e-7\n"
        "  magnetic: {mixer: johnson, spin_precond: true, "
        "mixing_alpha: 0.3, diago_schedule: quadratic}\n"))
    out = api._run_scf_noncollinear(inp, object(), verbose=False)
    assert out == "SENTINEL"
    assert captured["mag_mixer"] == "johnson"
    assert captured["spin_precond"] is True
    assert captured["mag_mixing_alpha"] == pytest.approx(0.3)
    assert captured["mag_diago_schedule"] == "quadratic"
    assert captured["energy_metric"] is True
    assert captured["entol"] == pytest.approx(5e-7)


def test_omitted_knobs_pass_driver_defaults(tmp_path, monkeypatch):
    """Omitting scf.magnetic and scf.convergence reaches the driver with its own
    defaults, so the campaign's stock behaviour is unchanged."""
    captured = _capture(monkeypatch)
    inp = load_input(_write(tmp_path, ""))
    api._run_scf_noncollinear(inp, object(), verbose=False)
    assert captured["mag_mixer"] == "pulay"
    assert captured["spin_precond"] is False
    assert captured["mag_mixing_alpha"] is None
    assert captured["mag_diago_schedule"] == "linear"
    assert captured["energy_metric"] is False
    assert captured["entol"] == pytest.approx(1e-6)
