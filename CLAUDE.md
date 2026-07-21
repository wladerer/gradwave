# gradwave

Differentiable plane-wave DFT for periodic solids in PyTorch.

## Environment

Always run project commands through `uv run`. A bare `pytest` or `python` picks up
the ambient base venv (not the project environment) and fails collection on nearly
every test file. `uv run` resolves the correct environment with no manual activation.

```bash
uv sync            # create/update the managed venv with all dev deps
uv run pytest ...  # run tests in the project environment
uv run ruff check  # lint
```

This is NixOS. Do not suggest `pip install`, `pip install -e .`, or `python -m venv`.
Dependencies are declared in `pyproject.toml` and installed with `uv sync`.

## Tests

The suite is tiered by pytest marker. A test carries at most one tier marker; an
unmarked test is fast-tier by definition. Default to the fast gate for local work
and pre-commit checks. Reserve the heavier tiers for the situations named below.

| tier | select | wall time | when |
|---|---|---|---|
| fast | `-m "not standard and not slow and not torture and not gpu"` | ~80 s | every commit |
| standard | `-m "not slow and not torture and not gpu"` | ~10 min | CI |
| nightly | `-m "not torture and not gpu"` | hours | nightly / pre-release |
| torture | `-m torture` | >10 min each | manual, when the subsystem changes |

`pyproject.toml` sets `addopts = "-q -n auto"`, so runs parallelize across cores
via pytest-xdist. Pass `-n0` to disable parallelism when debugging a single test.

QE-comparison fixtures are committed under `tests/fixtures/qe`. CI never runs Quantum
ESPRESSO. Regenerate fixtures with `tests/fixtures/qe/regenerate.py` (QE via
`nix shell nixpkgs#quantum-espresso`).

## Layout

`src/gradwave/` holds the package. Notable modules and subpackages:

- `core/`, `grids.py`, `kpoints.py`, `symmetry.py` — plane-wave basis, k-points, symmetry
- `scf/`, `solvers/`, `postscf/` — the SCF loop, eigensolvers, post-SCF analysis
- `pseudo/` — pseudopotentials
- `inputs.py`, `templates.py`, `cli.py`, `api.py` — input parsing, init templates, CLI, public API
- `analysis.py` — frames and plotting, imports pandas/matplotlib lazily (the
  `analysis` optional-dependency group); core and CLI run without them

Tests live in `tests/{unit,integration,gradcheck}` with shared fixtures in
`tests/fixtures` and helpers in `tests/helpers.py`.

## Definition of done

Before opening a PR, from the worktree:

1. `uv run ruff check` is clean.
2. `uv run pytest -m "not standard and not slow and not torture and not gpu"` passes.
3. The branch is rebased on `main` so conflicts surface locally rather than at merge.
4. Regenerate `uv.lock` (`uv lock`) only if dependencies changed, and commit it last.

CI runs ruff and the standard tier on every PR, so let the green check stand in for
re-running the standard suite by hand.
