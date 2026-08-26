"""Tests for the doc-truth-decay guard (scripts/check_doc_refs.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_doc_refs.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_doc_refs", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _repo(tmp_path: Path) -> Path:
    """A minimal repo tree with one real source file."""
    (tmp_path / "src" / "gradwave" / "core").mkdir(parents=True)
    (tmp_path / "src" / "gradwave" / "core" / "batch.py").write_text("x = 1\n")
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_resolves_real_reference(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "ok.md").write_text("The apply lives in `src/gradwave/core/batch.py`.\n")
    assert mod.find_broken_references(root) == []


def test_flags_missing_reference(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "bad.md").write_text(
        "line one\nSee `src/gradwave/core/moved.py` for details.\n"
    )
    broken = mod.find_broken_references(root)
    assert len(broken) == 1
    doc, lineno, token = broken[0]
    assert doc.name == "bad.md"
    assert lineno == 2
    assert token == "src/gradwave/core/moved.py"


def test_bare_and_backticked_both_matched(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "d.md").write_text(
        "backticked `tests/unit/gone.py` and bare tests/unit/also_gone.py here\n"
    )
    tokens = {t for _, _, t in mod.find_broken_references(root)}
    assert tokens == {"tests/unit/gone.py", "tests/unit/also_gone.py"}


@pytest.mark.parametrize(
    "line",
    [
        "a glob src/gradwave/*.py should be skipped",
        "a placeholder `src/gradwave/<module>.py` skipped",
        "brace `src/gradwave/{a,b}.py` skipped",
        "ellipsis src/gradwave/.../x.py skipped",
        "a var scripts/$NAME.py skipped",
    ],
)
def test_placeholder_patterns_skipped(tmp_path: Path, line: str):
    root = _repo(tmp_path)
    (root / "docs" / "p.md").write_text(line + "\n")
    assert mod.find_broken_references(root) == []


def test_urls_not_matched(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "u.md").write_text(
        "See https://github.com/x/gradwave/blob/main/src/gradwave/core/gone.py now\n"
    )
    # the path segment is inside a URL (preceded by '/') → not a repo reference
    assert mod.find_broken_references(root) == []


def test_generated_artifact_not_flagged(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "g.md").write_text("grep `docs/symbols.txt` for the API\n")
    assert mod.find_broken_references(root) == []


def test_ignore_list_suppresses(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "docs" / "i.md").write_text("provenance: benchmarks/results/local/out.json\n")
    assert len(mod.find_broken_references(root)) == 1
    assert mod.find_broken_references(root, ignore={"benchmarks/results/local/out.json"}) == []


def test_extra_dir_resolves_against_root(tmp_path: Path):
    """A memory-style dir outside the repo is scanned, but its path references
    resolve relative to --root (memory uses repo-relative paths)."""
    root = _repo(tmp_path)
    memory = tmp_path.parent / "memory"
    memory.mkdir()
    (memory / "note.md").write_text(
        "toeplitz wired into `src/gradwave/core/batch.py` (ok) and "
        "`src/gradwave/core/deleted.py` (broken)\n"
    )
    broken = mod.find_broken_references(root, extra_dirs=(memory,))
    tokens = {t for _, _, t in broken}
    assert tokens == {"src/gradwave/core/deleted.py"}


def test_skips_dot_dirs(tmp_path: Path):
    root = _repo(tmp_path)
    venv_doc = root / ".venv" / "pkg"
    venv_doc.mkdir(parents=True)
    (venv_doc / "readme.md").write_text("refers to src/gradwave/core/nope.py\n")
    assert mod.find_broken_references(root) == []


def test_main_exit_codes(tmp_path: Path, capsys):
    root = _repo(tmp_path)
    (root / "docs" / "clean.md").write_text("ok `src/gradwave/core/batch.py`\n")
    assert mod.main(["--root", str(root)]) == 0

    (root / "docs" / "dirty.md").write_text("bad src/gradwave/core/x.py\n")
    assert mod.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "src/gradwave/core/x.py" in out
