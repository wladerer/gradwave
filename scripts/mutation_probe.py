#!/usr/bin/env python3
"""Scoped mutation-testing probe: does the FAST test suite actually kill bugs?

mutmut 3.x's copy-to-`mutants/` sandbox is incompatible with our editable
src-layout install (imports resolve to the original src, so the copied mutation
never takes effect). This does the same job directly: parse a target module,
generate one AST mutant at a time (operator swaps, comparison flips, constant
perturbations, `and`↔`or`), patch the file IN PLACE — where the editable install
picks it up — run a fast covering test command, and record whether some test
FAILS (mutant *killed*) or all pass (mutant *survived* = a hole the fast suite
misses). Reverts after each.

    uv run python scripts/mutation_probe.py            # forces.py vs the fast force tests
    uv run python scripts/mutation_probe.py --target src/gradwave/postscf/stress.py

Survivors are printed as `file:line  <mutation>  — <original source line>` so
each is an actionable "add/extend a test that pins this".
"""

from __future__ import annotations

import argparse
import ast
import copy
import subprocess
import sys
import time
from pathlib import Path

# Fast tests that exercise the force/stress path without the slow FD/QE anchors:
# the metamorphic + property invariants (E, forces, stress identities) and the
# off-stationarity E↔H self-consistency gate.
DEFAULT_TESTS = [
    "tests/property/test_scf_invariants.py",
    "tests/integration/test_metamorphic_invariance.py",
    "tests/unit/test_energy_hamiltonian_consistency.py",
]

_CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_BIN_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def _describe(node) -> str:
    return type(node).__name__


def generate(tree: ast.AST):
    """Yield (lineno, description, mutated_tree). Each mutant flips exactly one
    node relative to a fresh deep copy of the original tree."""
    nodes = list(ast.walk(tree))
    for idx, node in enumerate(nodes):
        muts = []
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
            muts.append(("binop", _BIN_SWAP[type(node.op)]()))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and type(node.ops[0]) in _CMP_SWAP:
            muts.append(("compare", _CMP_SWAP[type(node.ops[0])]()))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
            muts.append(("boolop", _BOOL_SWAP[type(node.op)]()))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            muts.append(("drop-neg", None))  # -x -> x
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            muts.append(("const+1", node.value + 1))
        for kind, payload in muts:
            new = copy.deepcopy(tree)
            target = list(ast.walk(new))[idx]
            if kind == "binop":
                target.op = payload
            elif kind == "compare":
                target.ops = [payload]
            elif kind == "boolop":
                target.op = payload
            elif kind == "drop-neg":
                # replace the UnaryOp node by its operand via a parent rewrite
                target.op = ast.UAdd()  # -x -> +x (numerically = x)
            elif kind == "const+1":
                target.value = payload
            yield getattr(node, "lineno", 0), kind, new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="src/gradwave/postscf/forces.py")
    ap.add_argument("--tests", nargs="*", default=DEFAULT_TESTS)
    ap.add_argument("--limit", type=int, default=0, help="cap mutants (0 = all)")
    a = ap.parse_args()

    target = Path(a.target)
    original_src = target.read_text()
    src_lines = original_src.splitlines()
    tree = ast.parse(original_src)

    mutants = list(generate(tree))
    if a.limit:
        mutants = mutants[:: max(1, len(mutants) // a.limit)][: a.limit]
    print(f"# mutation probe: {target} — {len(mutants)} mutants vs {len(a.tests)} fast test files",
          flush=True)

    cmd = ["uv", "run", "python", "-m", "pytest", *a.tests,
           "-x", "-q", "-p", "no:cacheprovider", "-o", "addopts="]
    survivors, killed, errored = [], 0, 0
    t0 = time.time()
    try:
        for i, (lineno, kind, mtree) in enumerate(mutants):
            try:
                mutated_src = ast.unparse(ast.fix_missing_locations(mtree))
            except Exception:
                continue
            target.write_text(mutated_src)
            r = subprocess.run(cmd, capture_output=True, text=True)
            line = src_lines[lineno - 1].strip() if 0 < lineno <= len(src_lines) else ""
            if r.returncode == 0:
                survivors.append((lineno, kind, line))
                tag = "SURVIVED"
            elif r.returncode == 5:  # no tests collected — treat as errored
                errored += 1
                tag = "no-tests"
            else:
                killed += 1
                tag = "killed"
            print(f"  [{i+1}/{len(mutants)}] L{lineno} {kind:9} {tag}", flush=True)
    finally:
        target.write_text(original_src)  # ALWAYS restore

    dt = time.time() - t0
    n = len(mutants)
    score = 100.0 * killed / max(1, killed + len(survivors))
    print(f"\n# ==== result ({dt:.0f}s) ====", flush=True)
    print(f"# {n} mutants: {killed} killed, {len(survivors)} SURVIVED, {errored} no-tests",
          flush=True)
    print(f"# fast-suite mutation score on {target.name}: {score:.0f}% "
          f"(killed / (killed+survived))", flush=True)
    if survivors:
        print("# survivors (fast suite misses these — candidates for a new/extended test):",
              flush=True)
        for lineno, kind, line in survivors:
            print(f"  {target}:{lineno}  {kind:9} — {line}", flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    sys.exit(main())
