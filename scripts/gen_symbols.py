"""Generate docs/symbols.txt — a flat, greppable index of gradwave's public API.

Run via the Makefile (griffe lives in the `docs` dependency-group):

    make symbols                       # -> docs/symbols.txt
    uv run --group docs python scripts/gen_symbols.py

One line per public symbol:

    module.symbol<TAB>signature<TAB>first docstring line

so an agent can `grep stress docs/symbols.txt` instead of scanning 124 files to
find the canonical helper. "Public" matches the mkdocs convention (mkdocs.yml:
`filters: ["!^_"]`) — every name whose final component does not start with an
underscore. The index is generated from source, so it never drifts once
regenerated; the curated "capability -> symbol" table in CLAUDE.md is the
companion that says which symbol to reach for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import griffe

PACKAGE = "gradwave"
OUT = Path(__file__).resolve().parent.parent / "docs" / "symbols.txt"

# Kinds we emit a line for. Modules/attributes are traversed but not themselves
# listed (attributes are mostly constants already covered by the docstring line
# of their module; list them too if that ever proves useful).
EMIT_KINDS = {"function", "class"}


def _public(name: str) -> bool:
    """mkdocs `!^_` filter: keep names whose final component isn't underscored."""
    return not name.rsplit(".", 1)[-1].startswith("_")


def _signature(obj: griffe.Object) -> str:
    """`name(param, param, ...)` for functions, `name` for classes."""
    params = getattr(obj, "parameters", None)
    if params is None:
        return obj.name
    names = [p.name for p in params if p.name not in ("self", "cls")]
    return f"{obj.name}({', '.join(names)})"


def _first_doc_line(obj: griffe.Object) -> str:
    doc = getattr(obj, "docstring", None)
    if doc is None or not doc.value:
        return ""
    return doc.value.strip().splitlines()[0].strip()


def _walk(obj: griffe.Object, rows: list[tuple[str, str, str]]) -> None:
    for member in obj.members.values():
        # `from __future__ import annotations` and other re-exports arrive as
        # aliases; resolving their kind/docstring can raise. Skip them.
        if member.is_alias:
            continue
        if not _public(member.path):
            continue
        try:
            kind = member.kind.value
            if kind in EMIT_KINDS:
                rows.append((member.path, _signature(member), _first_doc_line(member)))
            if member.members:
                _walk(member, rows)
        except griffe.AliasResolutionError:
            continue


def main() -> int:
    pkg = griffe.load(PACKAGE, search_paths=["src"])
    rows: list[tuple[str, str, str]] = []
    _walk(pkg, rows)
    rows.sort(key=lambda r: r[0])

    header = (
        f"# gradwave public API index — {len(rows)} symbols. "
        "Regenerate with `make symbols`.\n"
        "# Format: dotted.path<TAB>signature<TAB>summary. "
        "Grep this before reimplementing; see the capability table in CLAUDE.md.\n"
    )
    body = "".join(f"{path}\t{sig}\t{doc}\n" for path, sig, doc in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + body)
    print(f"wrote {OUT.relative_to(Path.cwd())} ({len(rows)} symbols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
