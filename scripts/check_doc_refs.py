#!/usr/bin/env python3
"""Doc-truth-decay guard: verify that file-path references in the docs resolve.

A doc that says code lives in ``core/batch.py`` after it has moved, been
renamed, or deleted is worse than no doc — it sends the next reader (human or
agent) to look in the wrong place, or to rebuild something that already exists.
This scans the markdown surface for repo-relative path references and fails on
any that do not point at a real file, so that class of rot is caught at CI time
instead of in a future session.

Scope (v1): **path references** — high precision, the most common rot class.
A reference is any backtick-quoted or bare token that looks like a repo file
under a known top-level directory with a known extension, e.g.
``src/gradwave/core/batch.py`` or ``tests/unit/test_flapw_efg.py``. Tokens with
glob/placeholder characters (``*``, ``<>``, ``{}``, ``...``, ``$``) are skipped —
they are patterns, not concrete paths. URLs are skipped naturally (a ``/`` before
the root dir fails the boundary check).

Out of scope (v1): ``module.symbol`` API references and *semantic* staleness
(a claim about an existing file that is no longer true). Those need symbol- and
usage-resolution; see docs/verification.md for the planned Phase 2.

Usage:
    uv run python scripts/check_doc_refs.py                # scan repo docs
    uv run python scripts/check_doc_refs.py --extra-dir DIR  # also scan DIR/*.md
                                                            # (e.g. an out-of-repo
                                                            #  memory directory)

Exit status is 1 if any broken reference is found, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Top-level directories whose contents are real, checkable repo artifacts.
_ROOTS = (
    "src",
    "tests",
    "scripts",
    "benchmarks",
    "examples",
    "experiments",
    "docs",
    ".github",
)

# Extensions that denote a concrete file (not a directory or a bare name).
_EXTS = (
    "py",
    "md",
    "toml",
    "sh",
    "yml",
    "yaml",
    "cfg",
    "ini",
    "rst",
    "txt",
    "lock",
    "json",
)

# A path token: <root>/<...>.<ext>, bounded so it is not a sub-string of a
# longer word or a URL path. The leading (?<![\w./-]) rejects a preceding
# slash (so ``github.com/.../src/x.py`` inside a URL does not match) or word
# char; the trailing (?![\w/-]) rejects ``batch.python`` / ``batch.py/more``.
_PATH_RE = re.compile(
    r"(?<![\w./-])"
    r"((?:" + "|".join(re.escape(r) for r in _ROOTS) + r")/[\w./-]+?"
    r"\.(?:" + "|".join(_EXTS) + r"))"
    r"(?![\w/-])"
)

# Placeholder / glob markers → the token is a pattern, not a concrete path.
_PLACEHOLDER = re.compile(r"[*<>{}$]|\.\.\.")

# Generated / gitignored artifacts that legitimately may not exist in a clean
# checkout. Referencing them in docs is fine; do not flag.
_GENERATED = frozenset(
    {
        "docs/symbols.txt",
    }
)

_SKIP_DIR_PARTS = frozenset({".git", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"})


def _iter_markdown(root: Path):
    """Yield every tracked-looking markdown file under *root*."""
    for path in sorted(root.rglob("*.md")):
        if _SKIP_DIR_PARTS & set(path.relative_to(root).parts):
            continue
        yield path


def _load_ignore(ignore_file: Path) -> set[str]:
    """Read a committed allow-list of intentionally-absent references (one repo-
    relative path per line; ``#`` comments and blank lines ignored)."""
    out: set[str] = set()
    if ignore_file.is_file():
        for raw in ignore_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def find_broken_references(
    root: Path,
    *,
    extra_dirs: tuple[Path, ...] = (),
    ignore: frozenset[str] | set[str] = frozenset(),
) -> list[tuple[Path, int, str]]:
    """Return ``(doc, lineno, token)`` for every path reference that does not
    resolve to a file under *root*. Docs are scanned under *root* and each of
    *extra_dirs*; every reference resolves relative to *root* (memory files use
    repo-relative paths)."""
    ignore = set(ignore) | _GENERATED
    scanned: list[Path] = list(_iter_markdown(root))
    for extra in extra_dirs:
        scanned.extend(_iter_markdown(extra))

    broken: list[tuple[Path, int, str]] = []
    for doc in scanned:
        try:
            text = doc.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _PATH_RE.finditer(line):
                token = match.group(1)
                if _PLACEHOLDER.search(token) or token in ignore:
                    continue
                if not (root / token).exists():
                    broken.append((doc, lineno, token))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: cwd)",
    )
    parser.add_argument(
        "--extra-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="additional directory of *.md to scan (repeatable); references "
        "still resolve relative to --root. Use for an out-of-repo memory dir.",
    )
    parser.add_argument(
        "--ignore",
        type=Path,
        default=None,
        help="allow-list file of intentionally-absent references "
        "(default: <root>/scripts/doc_refs_ignore.txt if present)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    ignore_file = args.ignore
    if ignore_file is None:
        ignore_file = root / "scripts" / "doc_refs_ignore.txt"
    ignore = _load_ignore(ignore_file)

    broken = find_broken_references(
        root,
        extra_dirs=tuple(d.resolve() for d in args.extra_dir),
        ignore=ignore,
    )

    if not broken:
        print("doc-refs: all path references resolve.")
        return 0

    for doc, lineno, token in broken:
        try:
            shown = doc.resolve().relative_to(root)
        except ValueError:
            shown = doc
        print(f"{shown}:{lineno}: broken path reference '{token}'")
    print(
        f"\ndoc-refs: {len(broken)} broken reference(s). "
        "Fix the doc, or if the reference is intentional add it to "
        f"{ignore_file.name}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
