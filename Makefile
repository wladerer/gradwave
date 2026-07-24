# Canonical dev commands. Run `make <target>` instead of hand-writing the long
# pytest marker strings (shorter to type, impossible to get the markers wrong).
# Everything goes through `uv run` so the project env is used, not the base venv.

.PHONY: help test test-fast test-standard test-nightly lint imports fmt lock check hooks profile queue-init q-test q-status dashboard dashboard-push

BENCH ?= bench_scf
ARGS  ?= cpu 8 nosym

# Local fast-gate parallelism. `addopts = -n auto` in pyproject spawns one
# worker per core, which OOMs memory-tight laptops (8 concurrent fp64 SCFs).
# Cap the local fast gate to a safe worker count; override with `make test-fast
# FAST_JOBS=8`. CI and the standard/nightly tiers keep `-n auto`.
FAST_JOBS ?= 4

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

test-fast: ## fast gate (~80 s): run on every commit (local -n$(FAST_JOBS); override FAST_JOBS=)
	uv run pytest -n$(FAST_JOBS) -m "not standard and not slow and not torture and not gpu"

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

# --- job queue (pueue) — see docs/queue.md ---------------------------------
# Route heavy runs through the shared per-host queue so multiple agents don't
# thrash the laptop. `gwq` is a plain-python wrapper over pueue (no uv needed).

queue-init: ## create pueue groups with this host's slot budget (once per box)
	./scripts/gwq init

q-test: ## queue the fast gate on this host (throttled by the `test` group)
	./scripts/gwq test-fast

q-status: ## live queue view across the fleet (thinkpad + asus)
	./scripts/gwq status

DASH_HOST ?= homelab

dashboard: ## generate the fleet dashboard -> dashboard.html (open it in a browser)
	./scripts/dashboard.py --collect

dashboard-push: ## generate and push the dashboard to $(DASH_HOST) for tailscale-serve
	./scripts/dashboard.py --collect --out /tmp/gwdash.html
	ssh $(DASH_HOST) 'mkdir -p ~/gwdash'
	rsync -az /tmp/gwdash.html $(DASH_HOST):gwdash/index.html
	@echo "pushed to $(DASH_HOST):~/gwdash/index.html — serve once with:"
	@echo "  ssh $(DASH_HOST) 'sudo tailscale serve --bg --set-path / \$$HOME/gwdash'"
