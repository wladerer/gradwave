"""Schema + output surface for the io catch-up: hybrid xc (pbe0/hse), the fixed
spin moment key (tot_magnetization), the COHP projections sub-block, and the
method-aware dispersion label in the text report. Parse-only / pure formatting,
so these run in the fast tier.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def _base(extra: str = "") -> str:
    return f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
{extra}"""


# ---------------------------------------------------------------- gap 1: hybrids

def test_pbe0_enables_full_hybrid(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("xc: pbe0\n")))
    assert inp.xc == "pbe"                      # semilocal base for the registries
    assert inp.hybrid.enabled
    assert inp.hybrid.name == "pbe0"
    assert inp.hybrid.mode == "full"
    assert inp.hybrid.alpha == pytest.approx(0.25)


def test_hse_is_screened_with_omega(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("xc: hse\n")))
    assert inp.hybrid.enabled
    assert inp.hybrid.mode == "short_range"
    assert inp.hybrid.omega == pytest.approx(0.2)


def test_hybrid_block_overrides_alpha_and_omega(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(
        tmp_path, _base("xc: hse\nhybrid: {alpha: 0.4, omega: 0.15}\n")))
    assert inp.hybrid.alpha == pytest.approx(0.4)
    assert inp.hybrid.omega == pytest.approx(0.15)


def test_plain_xc_leaves_hybrid_disabled(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("xc: pbe\n")))
    assert not inp.hybrid.enabled


@pytest.mark.parametrize("extra, needle", [
    ("hybrid: {alpha: 0.25}\n", "hybrid block needs a hybrid functional"),
    ("xc: pbe0\nnspin: 2\nsmearing: {type: gaussian}\n", "spin-unpolarized"),
    ("xc: pbe0\nhybrid: {alpha: 1.5}\n", "hybrid.alpha must be in"),
    ("xc: pbe0\nhybrid: {mode: bogus}\n", "hybrid.mode must be"),
])
def test_hybrid_validation_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


# ------------------------------------------------------ gap 3: fixed spin moment

def test_tot_magnetization_parses_for_nspin2(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(
        tmp_path, _base("nspin: 2\ntot_magnetization: 2.0\n")))
    assert inp.tot_magnetization == pytest.approx(2.0)


def test_tot_magnetization_defaults_none(tmp_path):
    from gradwave.inputs import load_input

    assert load_input(_write(tmp_path, _base())).tot_magnetization is None


def test_tot_magnetization_rejected_for_nspin1(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="nspin: 2"):
        load_input(_write(tmp_path, _base("tot_magnetization: 1.0\n")))


# ------------------------------------------------------------------ gap 5: COHP

def test_cohp_subblock_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "projections:\n"
        "  enabled: true\n"
        "  cohp: {enabled: true, pairs: [[0, 1]], rcut: 2.5, width: 0.2}\n")))
    c = inp.projections.cohp
    assert c.enabled
    assert c.pairs == ((0, 1),)
    assert c.rcut == pytest.approx(2.5)
    assert c.width == pytest.approx(0.2)


def test_cohp_default_disabled(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("projections: true\n")))
    assert not inp.projections.cohp.enabled


@pytest.mark.parametrize("extra, needle", [
    ("projections: {cohp: {rcut: -1.0}}\n", "rcut must be positive"),
    ("projections: {cohp: {pairs: [1]}}\n", "list of \\[i, j\\]"),
    ("projections: {cohp: {bogus: 1}}\n", "unknown key"),
])
def test_cohp_validation_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


# ---------------------------------------------- gap 4: method-aware D4 disp label

def _summary_with_dispersion(method: str) -> dict:
    """A minimal task summary carrying only a dispersion block — enough for
    format_output to render the structure/parameters/dispersion sections."""
    return {
        "code": {"name": "gradwave", "version": "test", "created": "2026-01-01T00:00:00"},
        "task": "scf",
        "structure": {
            "cell_ang": [[0.0, 1.78, 1.78], [1.78, 0.0, 1.78], [1.78, 1.78, 0.0]],
            "positions_ang": [[0.0, 0.0, 0.0], [0.89, 0.89, 0.89]],
            "species": ["C", "C"],
            "n_atoms": 2,
        },
        "parameters": {
            "formalism": "nc", "xc": "pbe", "ecut_eV": 680.28,
            "kmesh": [4, 4, 4], "nspin": 1, "smearing": "none", "width_eV": 0.1,
            "symmetry": True, "pseudos": {"C": "C_ONCV_PBE-1.2.upf"},
        },
        "dispersion": {
            "available": True,
            "method": method,
            "functional": "pbe",
            "damping": {"s6": 1.0, "s8": 1.2, "a1": 0.4, "a2_bohr": 5.0},
            "energy_eV": -0.1234,
            "energy_per_atom_eV": -0.0617,
            "forces_eV_ang": [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]],
            "stress_eV_ang3": [[0.0, 0, 0], [0, 0.0, 0], [0, 0, 0.0]],
        },
    }


def test_format_output_labels_d4_block():
    from gradwave.output import format_output

    text = format_output(_summary_with_dispersion("d4-bj"))
    assert "D4(BJ) dispersion" in text
    assert "D3(BJ)" not in text
    # per-atom dispersion forces surfaced in the text report (was JSON-only)
    assert "max |F_disp|" in text


def test_format_output_still_labels_d3_block():
    from gradwave.output import format_output

    text = format_output(_summary_with_dispersion("d3-bj"))
    assert "D3(BJ) dispersion" in text
    assert "D4(BJ)" not in text
