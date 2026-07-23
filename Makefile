# Canonical dev commands. Run `make <target>` instead of hand-writing the long
# pytest marker strings (shorter to type, impossible to get the markers wrong).
# Everything goes through `uv run` so the project env is used, not the base venv.

.PHONY: help test test-fast test-standard test-nightly lint imports fmt lock check hooks profile

BENCH ?= bench_scf
ARGS  ?= cpu 8 nosym

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

test-fast: ## fast gate (~80 s): run on every commit
	uv run pytest -m "not standard and not slow and not torture and not gpu"

test: test-fast ## alias for the fast gate

test-standard: ## standard tier (~10 min): what CI runs
	uv run pytest -m "not slow and not torture and not gpu"

test-nightly: ## nightly tier (hours): pre-release
	uv run pytest -m "not torture and not gpu"

lint: ## ruff, concise output
	uv run ruff check --output-format=concise

imports: ## enforce package-boundary contracts (import-linter)
	uv run lint-imports

profile: ## sample-profile a benchmark -> speedscope json (BENCH=bench_scf ARGS="cpu 8 nosym"); open at speedscope.app
	uv run --with py-spy py-spy record --rate 200 --format speedscope \
	  --output profile.speedscope.json -- \
	  $$(uv run python -c "import sys; print(sys.executable)") benchmarks/$(BENCH).py $(ARGS)

fmt: ## ruff autofix + format
	uv run ruff check --fix
	uv run ruff format

lock: ## refresh uv.lock after a dependency change
	uv lock

check: lint imports test-fast ## pre-push gate: lint + import contracts + fast tests

hooks: ## install git hooks (ruff on commit, fast gate on push)
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
