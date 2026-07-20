"""Layer-C input validation: unknown-key rejection, structure reading, and the
`gradwave validate` dry-run. Parse-only, no SCF, so these run in the fast tier.
"""

from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

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


def test_baseline_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base()))
    assert inp.atoms.get_chemical_formula() == "C2"
    assert inp.ecut == pytest.approx(680.28)


@pytest.mark.parametrize("extra, needle", [
    ("task: relax\nrelax: {optimzer: bfgs}\n", "optimizer"),   # sub-block typo
    ("kpoint: {mesh: [4, 4, 4]}\n", "kpoints"),                # top-level typo
    ("scf: {mixing: {alfa: 0.5}}\n", "alpha"),                 # nested typo
])
def test_unknown_key_suggests_the_right_one(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="did you mean"):
        load_input(_write(tmp_path, _base(extra)))
    # the suggestion names the intended key
    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


@pytest.mark.parametrize("extra, needle", [
    ("nspin: 3\n", "nspin must be 1 or 2"),
    ("task: bandz\n", "unknown task"),
    ("kpoints: {mesh: [4, 4]}\n", "3 entries"),
    ("xc: b3lyp\n", "unknown xc"),
])
def test_value_range_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


@pytest.mark.parametrize("mode", ["noncollinear: true\n", "task: magnetism\n"])
def test_symmetry_true_rejected_for_magnetic_modes(tmp_path, mode):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="symmetry: true is invalid"):
        load_input(_write(tmp_path, _base(mode + "symmetry: true\n")))


@pytest.mark.parametrize("mode", ["noncollinear: true\n", "task: magnetism\n"])
def test_symmetry_defaults_off_for_magnetic_modes(tmp_path, mode):
    from gradwave.inputs import load_input

    # no symmetry key: a magnetic spinor run defaults symmetry off so a minimal
    # input runs, instead of tripping the SCF driver's guard.
    inp = load_input(_write(tmp_path, _base(mode)))
    assert inp.symmetry is False


def test_symmetry_default_on_for_plain_scf(tmp_path):
    from gradwave.inputs import load_input

    assert load_input(_write(tmp_path, _base())).symmetry is True


def test_nonmagnetic_keeps_symmetry_and_pins_moment(tmp_path):
    from gradwave.inputs import load_input

    # spin-orbit only: m ≡ 0, so Kramers keeps the full symmetry
    inp = load_input(_write(tmp_path, _base("noncollinear: true\nnonmagnetic: true\n")))
    assert inp.nonmagnetic is True
    assert inp.symmetry is True


def test_nonmagnetic_allows_explicit_symmetry_true(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(
        tmp_path, _base("noncollinear: true\nnonmagnetic: true\nsymmetry: true\n")))
    assert inp.symmetry is True


def test_nonmagnetic_requires_noncollinear(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="nonmagnetic requires noncollinear"):
        load_input(_write(tmp_path, _base("nonmagnetic: true\n")))


def test_error_message_carries_the_filename(tmp_path):
    from gradwave.inputs import InputError, load_input

    p = _write(tmp_path, _base("nspin: 3\n"))
    with pytest.raises(InputError, match=str(p)):
        load_input(p)


def test_missing_required_key(tmp_path):
    from gradwave.inputs import InputError, load_input

    body = _base().replace("ecut: 680.28", "")
    with pytest.raises(InputError, match="missing required key 'ecut'"):
        load_input(_write(tmp_path, body))


# ---- structure reading through ASE ---------------------------------------

def _si() -> Atoms:
    return Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                 cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]], pbc=True)


def _struct_input(tmp_path, structblock: str) -> Path:
    return _write(tmp_path, f"""
structure: {structblock}
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: 400
""")


def test_reads_bare_filename(tmp_path):
    from gradwave.inputs import load_input

    ase_write(tmp_path / "si.xyz", _si(), format="extxyz")
    inp = load_input(_struct_input(tmp_path, "si.xyz"))
    assert inp.atoms.get_chemical_formula() == "Si2"
    assert inp.atoms.cell.rank == 3


def test_file_mapping_with_format_and_index(tmp_path):
    from gradwave.inputs import load_input

    ase_write(tmp_path / "traj.xyz", [_si(), _si(), _si()], format="extxyz")
    inp = load_input(_struct_input(tmp_path, "{file: traj.xyz, format: extxyz, index: 0}"))
    assert inp.atoms.get_chemical_formula() == "Si2"


def test_molecule_without_cell_is_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    ase_write(tmp_path / "mol.xyz", Atoms("Si2", positions=[[0, 0, 0], [0, 0, 2.3]]),
              format="xyz")
    with pytest.raises(InputError, match="no 3D cell"):
        load_input(_struct_input(tmp_path, "mol.xyz"))


def test_multiframe_slice_is_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    ase_write(tmp_path / "traj.xyz", [_si(), _si()], format="extxyz")
    with pytest.raises(InputError, match="single integer index"):
        load_input(_struct_input(tmp_path, "{file: traj.xyz, index: ':'}"))


# ---- the validate subcommand ---------------------------------------------

def test_validate_command_ok(tmp_path, capsys):
    from gradwave.cli import main

    rc = main(["validate", str(_write(tmp_path, _base("kpoints: {mesh: [4, 4, 4]}\n")))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ok:" in out and "C2" in out and "mesh [4, 4, 4]" in out


def test_validate_command_reports_error(tmp_path, capsys):
    from gradwave.cli import main

    rc = main(["validate", str(_write(tmp_path, _base("nspin: 7\n")))])
    err = capsys.readouterr().err
    assert rc == 1
    assert "nspin must be 1 or 2" in err
