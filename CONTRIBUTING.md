# Contributing to gradwave

Thanks for working on gradwave. This page is the short version of the
contributor workflow. The project instructions in `CLAUDE.md` are the source of
truth where the two overlap, and `pyproject.toml` holds the tool configuration
referenced below.

## Environment

gradwave uses `uv` for a managed virtual environment. Create or update it with
one command.

```bash
uv sync            # managed venv with all dev dependencies
```

Run every project command through `uv run`. A bare `pytest` or `python` picks up
the ambient base venv rather than the project environment and fails collection on
nearly every test file, so `uv run` is not optional.

This is NixOS. Do not `pip install`, `pip install -e .`, or `python -m venv`.
Dependencies are declared in `pyproject.toml` and installed with `uv sync`.

`uv run` prints a `VIRTUAL_ENV=... does not match the project environment`
warning when another venv is already active in the shell. The warning is
harmless, `uv` uses the project environment regardless.

## Make shortcuts

The `Makefile` targets already go through `uv run` and carry the correct tier
markers, so prefer them over hand-written commands.

| target | what it runs |
|---|---|
| `make hooks` | install the pre-commit hooks (run once per clone) |
| `make check` | lint plus the fast test gate (the pre-commit check) |
| `make test-fast` | the fast tier, every commit |
| `make test-standard` | the standard tier, what CI runs |
| `make fmt` | format with ruff |
| `make lock` | regenerate `uv.lock` after a dependency change |

## Tests

The suite is tiered by pytest marker. A test carries at most one tier marker, and
an unmarked test is fast-tier by definition. Default to the fast gate for local
work.

| tier | select | wall time | when |
|---|---|---|---|
| fast | `-m "not standard and not slow and not torture and not gpu"` | ~80 s | every commit |
| standard | `-m "not slow and not torture and not gpu"` | ~10 min | CI |
| nightly | `-m "not torture and not gpu"` | hours | nightly / pre-release |
| torture | `-m torture` | >10 min each | manual, when the subsystem changes |

`pyproject.toml` sets `addopts = "-q -n auto"`, so runs parallelize across cores
via pytest-xdist. Pass `-n0` to disable parallelism when debugging a single test.

Invoke pytest as `uv run python -m pytest ...`. Bare `uv run pytest` picks up the
ambient venv and fails collection with a `ModuleNotFoundError`.

QE-comparison fixtures are committed under `tests/fixtures/qe` and CI never runs
Quantum ESPRESSO. Regenerate them with `tests/fixtures/qe/regenerate.py` (QE via
`nix shell nixpkgs#quantum-espresso`).

## Definition of done

Before opening a pull request, from your worktree.

1. `uv run ruff check` is clean.
2. `uv run ty check` is clean. `ty` runs at error severity over the growing
   typed-file list in `pyproject.toml`'s `[[tool.ty.overrides]]` and warn-only
   elsewhere. Never let a listed file regress, and add a file to that list only
   once it is fully and cleanly typed.
3. `uv run lint-imports` is clean (the import contracts in `pyproject.toml`'s
   `[tool.importlinter]`).
4. `uv run pytest -m "not standard and not slow and not torture and not gpu"`
   passes.
5. The branch is rebased on `main`, so conflicts surface locally rather than at
   merge.
6. Regenerate `uv.lock` (`uv lock`) only if dependencies changed, and commit it
   last.

CI runs ruff, ty, and the standard tier on every pull request, so let the green
check stand in for re-running the standard suite by hand.

## Typing conventions

Typing is a gradual rollout, configured in `pyproject.toml` under `[tool.ty]`.
Every rule is demoted to `warn` by default and enforced at `error` over the
explicit override list, so the untyped bulk stays visible without failing CI.
Put `@override` (`typing_extensions.override` on py<3.12) on any subclass method
that overrides a base method, which turns a typo'd signature into a type error
rather than a silent no-op. See the "Typing" section of `CLAUDE.md` for the full
policy on `jaxtyping` and where the runtime shape check is worth its cost.
