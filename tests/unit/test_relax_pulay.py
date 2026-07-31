"""Pulay (basis-incompleteness) stress correction on the vc-relax path (#217).

The fixed-basis (Nielsen-Martin) stress systematically under-pressures a
too-small basis, so a variable-cell relax silently drives soft cells toward
spuriously small volumes and still reports converged. The fix threads
``postscf.stress_error.estimate_pressure_error`` through the calculator
(``pulay_stress_correction``) and auto-enables it for supported vc-relax
inputs (``relax.pulay_correction: auto``).

Covered here (fast tier):
- the applied correction shifts exactly the stress diagonal by the estimated
  pressure error, leaving energy, forces, and shear components untouched
  (estimator monkeypatched — its own accuracy is validated in
  tests/integration/test_stress_error.py);
- the input-surface gating: auto resolves on only for supported vc inputs,
  explicit true errors when unsupported, fixed-cell relax never applies it
  (bit-for-bit: the calculator parameter is simply False);
- the calculator's own guards (shifted mesh, USPP) fail loud, not silent.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave.calculator import GradWave
from gradwave.inputs import InputError, load_input
from tests.helpers import RY, pseudo

SI = pseudo("Si_ONCV_PBE-1.2.upf")
SI_PAW = pseudo("Si.pbe-n-kjpaw_psl.1.0.0.UPF")


def _si_atoms():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _kw(**extra):
    return dict(
        ecut=8 * RY, pseudopotentials={"Si": SI}, xc="lda", kpts=(1, 1, 1),
        use_symmetry=False, **extra)


# ------------------------------------------------------- correction application

def test_correction_shifts_stress_diagonal_only(monkeypatch):
    """Corrected stress = uncorrected − P_err·I (Voigt diagonal); energy and
    forces bit-identical; ``last_pulay_pressure_gpa`` reports the correction."""
    p_err = 0.0125  # eV/Å³, the estimator's P_exact − P_coarse
    monkeypatch.setattr(
        "gradwave.postscf.stress_error.estimate_pressure_error",
        lambda res, xc, **kw: {"pressure_error_eV_A3": p_err,
                               "pressure_error_kbar": p_err * 1602.176634})

    plain = _si_atoms()
    plain.calc = GradWave(**_kw())
    corr = _si_atoms()
    corr.calc = GradWave(**_kw(pulay_stress_correction=True))

    s0, s1 = plain.get_stress(), corr.get_stress()
    np.testing.assert_allclose(s1[:3], s0[:3] - p_err, rtol=0, atol=1e-14)
    np.testing.assert_allclose(s1[3:], s0[3:], rtol=0, atol=1e-14)
    assert plain.get_potential_energy() == corr.get_potential_energy()
    np.testing.assert_array_equal(plain.get_forces(), corr.get_forces())
    assert plain.calc.last_pulay_pressure_gpa is None
    assert corr.calc.last_pulay_pressure_gpa == pytest.approx(
        p_err * 160.2176634)


def test_correction_off_is_bit_for_bit_default():
    """The parameter defaults off and is absent from the stress path: two
    default calculators agree exactly (no hidden estimator call)."""
    a1 = _si_atoms()
    a1.calc = GradWave(**_kw())
    a2 = _si_atoms()
    a2.calc = GradWave(**_kw(pulay_stress_correction=False))
    np.testing.assert_array_equal(a1.get_stress(), a2.get_stress())
    assert a2.calc.last_pulay_pressure_gpa is None


def test_shifted_mesh_guard(monkeypatch):
    """A shifted k-mesh must fail loud: the estimator's frozen strained
    rebuild would silently reconstruct the wrong k-point set."""
    kw = _kw(kshift=(1, 1, 1), pulay_stress_correction=True)
    kw["kpts"] = (2, 2, 2)
    atoms = _si_atoms()
    atoms.calc = GradWave(**kw)
    with pytest.raises(ValueError, match="Γ-centered"):
        atoms.get_stress()


def test_uspp_guard():
    """USPP/PAW has no stress-error estimator: requesting the correction on
    that path raises before the SCF rather than silently skipping."""
    atoms = _si_atoms()
    atoms.calc = GradWave(
        ecut=12 * RY, pseudopotentials={"Si": SI_PAW}, xc="pbe",
        kpts=(1, 1, 1), use_symmetry=False, pulay_stress_correction=True)
    with pytest.raises(NotImplementedError, match="norm-conserving only"):
        atoms.get_potential_energy()


# ----------------------------------------------------------- input-surface gate

def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def _si_yaml(extra: str = "", pseudo_name: str = "Si_ONCV_PBE-1.2.upf") -> str:
    pdir = Path(SI).parent
    return f"""
structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions: {{frac: [[0, 0, 0], [0.25, 0.25, 0.25]]}}
  species: [Si, Si]
pseudopotentials:
  dir: {pdir}
  map: {{Si: {pseudo_name}}}
ecut: 200.0
task: relax
{extra}"""


def _resolve(tmp_path, extra, **kw):
    from gradwave.api import _resolve_pulay_correction

    inp = load_input(_write(tmp_path, _si_yaml(extra, **kw)))
    return _resolve_pulay_correction(inp, verbose=False)


def test_auto_on_for_supported_vc_relax(tmp_path):
    assert _resolve(tmp_path, "symmetry: false\nrelax:\n  cell: true\n") is True


def test_auto_off_fixed_cell_and_explicit_false(tmp_path):
    # fixed-cell: never applied (the calculator parameter is plain False)
    assert _resolve(tmp_path, "relax:\n  cell: false\n") is False
    # explicit off wins even when supported
    assert _resolve(
        tmp_path,
        "symmetry: false\nrelax:\n  cell: true\n  pulay_correction: false\n",
    ) is False


def test_auto_off_with_symmetry_warns_not_errors(tmp_path):
    # symmetry: true (the default) is outside the estimator's contract: auto
    # degrades to uncorrected with a visible warning, not an error
    assert _resolve(tmp_path, "relax:\n  cell: true\n") is False


def test_explicit_true_unsupported_raises(tmp_path):
    with pytest.raises(InputError, match="pulay_correction"):
        _resolve(tmp_path,
                 "relax:\n  cell: true\n  pulay_correction: true\n")


def test_explicit_true_supported_and_uspp_rejected(tmp_path):
    assert _resolve(
        tmp_path,
        "symmetry: false\nrelax:\n  cell: true\n  pulay_correction: true\n",
    ) is True
    with pytest.raises(InputError, match="NC-only"):
        _resolve(
            tmp_path,
            "symmetry: false\nrelax:\n  cell: true\n  pulay_correction: true\n",
            pseudo_name="Si.pbe-n-kjpaw_psl.1.0.0.UPF")


def test_fixed_cell_calculator_param_is_false(tmp_path):
    """Bit-for-bit for fixed-cell relax: the built calculator carries
    pulay_stress_correction=False, so the stress path is untouched."""
    from gradwave.api import _build_relax_calc

    inp = load_input(_write(tmp_path, _si_yaml("relax:\n  cell: false\n")))
    calc = _build_relax_calc(inp, verbose=False)
    assert calc.parameters["pulay_stress_correction"] is False
