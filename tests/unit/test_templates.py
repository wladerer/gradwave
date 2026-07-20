"""Every `gradwave init` template must stay valid against the input schema.

Each template inlines an example structure and placeholder pseudopotential
paths. The test swaps the pseudo block for a real fixture file (mapping every
species to one existing UPF — `load_input` checks that the file exists, not that
it matches the element) and runs the whole template through `load_input`, so a
renamed or dropped schema key breaks the test rather than shipping a stale
example.
"""

from pathlib import Path

import pytest
import yaml

from gradwave import templates
from gradwave.inputs import _load_structure, load_input
from tests.helpers import PSEUDOS

_FIXTURE_UPF = "Si_ONCV_PBE-1.2.upf"   # any real UPF; existence is all that is checked


@pytest.mark.parametrize("name", templates.names())
def test_template_parses(name, tmp_path):
    raw = yaml.safe_load(templates.render(name))
    species = set(_load_structure(raw["structure"], Path(".")).get_chemical_symbols())
    raw["pseudopotentials"] = {"dir": str(PSEUDOS),
                               "map": {s: _FIXTURE_UPF for s in species}}
    p = tmp_path / "in.yaml"
    p.write_text(yaml.safe_dump(raw))
    inp = load_input(p)                 # raises InputError on any schema drift
    assert len(inp.atoms) >= 1
    # a magnetic spinor run cannot use IBZ symmetry reduction; the driver would
    # raise at runtime, so those templates must resolve symmetry off. A
    # spin-orbit-only run (nonmagnetic) pins m ≡ 0 and keeps symmetry.
    magnetic_spinor = (inp.noncollinear and not inp.nonmagnetic) or \
        inp.task == "magnetism"
    if magnetic_spinor:
        assert inp.symmetry is False, f"{name} must set symmetry: false"


def test_summaries_cover_every_template():
    assert set(templates.summaries()) == set(templates.names())
    assert all(desc for desc in templates.summaries().values())


def test_render_unknown_raises():
    with pytest.raises(KeyError):
        templates.render("does-not-exist")


def test_init_lists_templates(capsys):
    from gradwave.cli import main

    assert main(["init"]) == 0
    out = capsys.readouterr().out
    for name in templates.names():
        assert name in out


def test_init_writes_file_and_refuses_clobber(tmp_path, capsys):
    from gradwave.cli import main

    out = tmp_path / "relax.yaml"
    assert main(["init", "relax", "-o", str(out)]) == 0
    assert out.exists() and "task: relax" in out.read_text()

    # a second write without --force is refused
    assert main(["init", "relax", "-o", str(out)]) == 1
    assert "exists" in capsys.readouterr().err
    # --force overwrites
    assert main(["init", "relax", "-o", str(out), "--force"]) == 0


def test_init_unknown_template_errors(capsys):
    from gradwave.cli import main

    assert main(["init", "nope"]) == 1
    assert "unknown template" in capsys.readouterr().err


def test_init_stdout_roundtrips_through_validate(tmp_path, capsys):
    """`gradwave init scf` emits to stdout; the emitted text validates once the
    pseudo path is pointed at real files."""
    from gradwave.cli import main

    assert main(["init", "scf"]) == 0
    text = capsys.readouterr().out
    raw = yaml.safe_load(text)
    raw["pseudopotentials"] = {"dir": str(PSEUDOS), "map": {"Si": _FIXTURE_UPF}}
    p = tmp_path / "in.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert main(["validate", str(p)]) == 0
