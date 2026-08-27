# Canonical dev commands. Run `make <target>` instead of hand-writing the long
# pytest marker strings (shorter to type, impossible to get the markers wrong).
# Everything goes through `uv run` so the project env is used, not the base venv.

.PHONY: help test test-fast test-standard test-nightly lint imports typecheck fmt lock check hooks symbols profile queue-init q-test q-status dashboard dashboard-push worktrees worktrees-prune

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
	uv run pytest --dist loadscope -m "not slow and not torture and not gpu"

test-nightly: ## nightly tier (hours): pre-release
	uv run pytest -m "not torture and not gpu"

lint: ## ruff, concise output
	uv run ruff check --output-format=concise

imports: ## enforce package-boundary contracts (import-linter)
	uv run lint-imports

doc-refs: ## fail on stale file-path references in the docs (doc-truth-decay guard)
	uv run python scripts/check_doc_refs.py

typecheck: ## ty: error on the typed file list (pyproject [[tool.ty.overrides]]), warn-only elsewhere
	uv run ty check

profile: ## sample-profile a benchmark -> speedscope json (BENCH=bench_scf ARGS="cpu 8 nosym"); open at speedscope.app
	uv run --with py-spy py-spy record --rate 200 --format speedscope \
	  --output profile.speedscope.json -- \
	  $$(uv run python -c "import sys; print(sys.executable)") benchmarks/$(BENCH).py $(ARGS)

mutation: ## on-demand mutation probe of one module (run on asus, NOT in CI): TARGET=src/... TESTS="tests/..."
	uv run python scripts/mutation_probe.py --target $(TARGET) --tests $(TESTS)

fmt: ## ruff autofix + format
	uv run ruff check --fix
	uv run ruff format

lock: ## refresh uv.lock after a dependency change
	uv lock

check: lint imports typecheck doc-refs test-fast ## pre-push gate: lint + import contracts + typecheck + doc refs + fast tests

hooks: ## install git hooks (ruff on commit, fast gate on push)
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

symbols: ## regenerate docs/symbols.txt — greppable public-API index for agents
	uv run --group docs python scripts/gen_symbols.py

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
	uv run python scripts/dashboard.py --collect

worktrees: ## fleet worktree overview — drift, stale branches, cross-worktree file overlap
	uv run python scripts/worktrees.py

worktrees-prune: ## remove stale (merged) + clean + idle worktrees under .claude/worktrees/
	uv run python scripts/worktrees.py --prune

dashboard-push: ## generate and push the dashboard to $(DASH_HOST) for tailscale-serve
	uv run python scripts/dashboard.py --collect --out /tmp/gwdash.html
	ssh $(DASH_HOST) 'mkdir -p ~/gwdash'
	rsync -az /tmp/gwdash.html $(DASH_HOST):gwdash/index.html
	@echo "pushed to $(DASH_HOST):~/gwdash/index.html — serve once with (absolute path: \$$HOME is /root under sudo):"
	@echo "  ssh $(DASH_HOST) 'sudo tailscale serve --bg /home/wladerer/gwdash'"
