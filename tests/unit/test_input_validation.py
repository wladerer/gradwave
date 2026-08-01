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
    # 'linear' parses but no mixer implements it — reject at parse time, not
    # deep in the SCF where the mixer builder raises a bare ValueError.
    ("scf: {mixing: {scheme: linear}}\n", "unknown mixing scheme"),
    # precond selects the density preconditioner; only kerker | local_tf exist.
    ("scf: {mixing: {precond: tf}}\n", "unknown mixing precond"),
    ("elastic: {mode: relax}\n", "clamped.*relaxed"),
    ("elastic: {fmax: 0}\n", "elastic.fmax"),
    ("elastic: {max_steps: 0}\n", "elastic.max_steps"),
])
def test_value_range_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


def test_elastic_mode_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base()))
    assert inp.elastic.mode == "clamped"  # default stays the historical tensor

    inp = load_input(_write(tmp_path, _base(
        "task: elastic\nelastic: {mode: relaxed, fmax: 0.005, max_steps: 40}\n")))
    assert inp.elastic.mode == "relaxed"
    assert inp.elastic.fmax == pytest.approx(0.005)
    assert inp.elastic.max_steps == 40


def test_precond_defaults_to_kerker(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base()))
    assert inp.scf.mixing.precond == "kerker"


def test_precond_local_tf_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("scf: {mixing: {precond: local_tf}}\n")))
    assert inp.scf.mixing.precond == "local_tf"


def test_mixing_scheme_defaults_to_auto(tmp_path):
    # the default defers to each formalism's resolver rather than pinning pulay
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base()))
    assert inp.scf.mixing.scheme == "auto"


def test_mixing_scheme_auto_and_explicit_parse(tmp_path):
    from gradwave.inputs import load_input

    auto = load_input(_write(tmp_path, _base("scf: {mixing: {scheme: auto}}\n")))
    assert auto.scf.mixing.scheme == "auto"
    exp = load_input(_write(tmp_path, _base("scf: {mixing: {scheme: johnson}}\n")))
    assert exp.scf.mixing.scheme == "johnson"


def test_auto_scheme_resolves_to_johnson_for_nspin2(tmp_path):
    """The auto default maps to None at the api boundary so BOTH the NC and the
    USPP/PAW resolver pick johnson for a collinear-spin (nspin=2) run — the
    scheme an input-file user gets without naming one."""
    from gradwave.api import _mixing_scheme
    from gradwave.inputs import load_input
    from gradwave.scf.loop import _resolve_mixing_scheme
    from gradwave.scf.uspp_loop import _resolve_uspp_mixing_scheme

    inp = load_input(_write(tmp_path, _base()))
    scheme = _mixing_scheme(inp)
    assert scheme is None
    assert _resolve_mixing_scheme(scheme, nspin=2) == "johnson"
    assert _resolve_uspp_mixing_scheme(scheme, nspin=2) == "johnson"


def test_explicit_scheme_bypasses_resolver(tmp_path):
    # an explicit scheme reaches the driver unchanged (resolver is a no-op on it)
    from gradwave.api import _mixing_scheme
    from gradwave.inputs import load_input
    from gradwave.scf.loop import _resolve_mixing_scheme

    inp = load_input(_write(tmp_path, _base("scf: {mixing: {scheme: pulay}}\n")))
    scheme = _mixing_scheme(inp)
    assert scheme == "pulay"
    assert _resolve_mixing_scheme(scheme, nspin=2) == "pulay"


@pytest.mark.parametrize("mode", ["noncollinear: true\n", "task: magnetism\n"])
def test_symmetry_true_rejected_for_magnetic_modes(tmp_path, mode):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="symmetry: true is invalid"):
        load_input(_write(tmp_path, _base(mode + "symmetry: true\n")))


def test_volumetric_shorthand_and_mapping(tmp_path):
    from gradwave.inputs import load_input

    short = load_input(_write(tmp_path, _base("output: {volumetric: true}\n")))
    assert short.output_volumetric.density is True
    assert short.output_volumetric.any()

    full = load_input(_write(tmp_path, _base(
        "output:\n"
        "  volumetric:\n"
        "    density: true\n"
        "    elf: true\n"
        "    bands: [[3, 0], [4, 1]]\n"
        "    format: xsf\n")))
    v = full.output_volumetric
    assert (v.density, v.elf, v.format) == (True, True, "xsf")
    assert v.bands == ((3, 0), (4, 1))

    chg = load_input(_write(tmp_path, _base(
        "output: {volumetric: {density: true, format: chgcar}}\n")))
    assert chg.output_volumetric.format == "chgcar"


@pytest.mark.parametrize("block, needle", [
    ("output: {volumetric: {densty: true}}\n", "did you mean"),   # field typo
    ("output: {volumetric: {format: vtk}}\n", "cube', 'xsf' or 'chgcar"),  # bad format
    ("output: {volumetric: {bands: [3, 0]}}\n", "band, kpoint"),   # not pairs
])
def test_volumetric_errors(tmp_path, block, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(block)))


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


def test_dispersion_shorthand_enables(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base("dispersion: true\n")))
    assert inp.dispersion.enabled is True
    assert inp.dispersion.functional is None  # defaults to the SCF xc at run time


def test_dispersion_block_overrides(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "dispersion:\n  functional: pbe0\n  cutoff: 18.0\n  a2: 4.9\n")))
    assert inp.dispersion.enabled is True
    assert inp.dispersion.functional == "pbe0"
    assert inp.dispersion.cutoff == pytest.approx(18.0)
    assert inp.dispersion.a2 == pytest.approx(4.9)


def test_dispersion_cutoff_ordering_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="cn_cutoff"):
        load_input(_write(tmp_path, _base(
            "dispersion:\n  cutoff: 5.0\n  cn_cutoff: 9.0\n")))


def test_dispersion_unknown_subkey_suggests(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="did you mean"):
        load_input(_write(tmp_path, _base("dispersion:\n  cutof: 20.0\n")))


# --- selective dynamics (structure.fixed) --------------------------------------

def _with_fixed(fixed_line: str, extra: str = "") -> str:
    """A two-atom C2 inline structure carrying a `fixed:` line in the structure
    block, plus optional top-level `extra` keys."""
    return f"""
structure:
  cell: [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
  positions: {{cart: [[0, 0, 0], [0.89175, 0.89175, 0.89175]]}}
  species: [C, C]
  {fixed_line}
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
{extra}"""


def test_fixed_default_is_none(tmp_path):
    from gradwave.inputs import load_input

    # no fixed key: the historical path, mask is None
    inp = load_input(_write(tmp_path, _base()))
    assert inp.fixed is None


def test_fixed_index_form_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _with_fixed("fixed: [0]")))
    assert inp.fixed is not None
    assert inp.fixed.shape == (2, 3)
    assert inp.fixed[0].all()          # atom 0 fully fixed
    assert not inp.fixed[1].any()      # atom 1 free


def test_fixed_per_axis_form_parses(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _with_fixed(
        "fixed: [[true, true, true], [true, false, false]]")))
    assert inp.fixed is not None
    assert inp.fixed.tolist() == [[True, True, True], [True, False, False]]


def test_fixed_file_form_parses(tmp_path):
    from gradwave.inputs import load_input

    a = Atoms("C2", positions=[[0, 0, 0], [0.9, 0.9, 0.9]],
              cell=[[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]],
              pbc=True)
    ase_write(str(tmp_path / "geo.xyz"), a, format="extxyz")
    body = f"""
structure: {{file: geo.xyz, fixed: [1]}}
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: 680.28
"""
    inp = load_input(_write(tmp_path, body))
    assert inp.fixed is not None
    assert inp.fixed[1].all() and not inp.fixed[0].any()


@pytest.mark.parametrize("fixed_line, needle", [
    ("fixed: [2]", "out of range"),                          # index >= natoms
    ("fixed: [0, 0]", "duplicate"),                          # repeated index
    ("fixed: [[true, true, true]]", "2 atoms"),              # per-atom length off
    ("fixed: [[true, true], [false, false, false]]", "3 booleans"),  # row len
    ("fixed: [[true, true, 1], [false, false, false]]", "boolean"),  # non-bool
    ("fixed: [[false, false, false], [false, false, false]]", "pins no axes"),
    ("fixed: []", "non-empty"),                              # empty
    ("fixed: [0, [true, true, true]]", "not a mix"),         # mixed spellings
])
def test_fixed_validation_rejects(tmp_path, fixed_line, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError) as exc:
        load_input(_write(tmp_path, _with_fixed(fixed_line)))
    if needle is not None:
        assert needle in str(exc.value)


def test_fixed_forces_symmetry_off_by_default(tmp_path):
    from gradwave.inputs import load_input

    # symmetry defaults True, but selective dynamics forces it off
    inp = load_input(_write(tmp_path, _with_fixed("fixed: [0]")))
    assert inp.symmetry is False


def test_fixed_with_explicit_symmetry_true_rejected(tmp_path):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match="selective dynamics"):
        load_input(_write(tmp_path, _with_fixed("fixed: [0]", "symmetry: true\n")))
