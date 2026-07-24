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

Prefer the `Makefile` shortcuts over hand-writing the long commands: `make test-fast`,
`make test-standard`, `make lint`, `make fmt`, `make check` (lint + fast gate), `make lock`.
They already go through `uv run` and carry the correct tier markers.

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

## Running commands efficiently

Long-lived commands (test runs, SCFs) should be launched in the background writing to a
log that ends with an `EXIT=$?` marker, then polled by grepping for that marker. Do not
`tail -f | pipe` a live run: the pipe buffers and hides results until the process exits.

Do not `pkill -f <pattern>` when the pattern also appears in the killing command's own line
(for example `pkill -f "pytest -m"` matches the shell running it and self-terminates). Kill
by PID via a self-excluding match instead, e.g. `kill $(pgrep -f '[.]venv/bin/pytest')`, or
stop the background task by its id.

Keep terminal output small: `git status -s`, `git log --oneline`, `git diff --stat`,
`ruff check --output-format=concise`, and `pytest --tb=short`. `GIT_PAGER=cat` avoids pager
stalls.

## Job queue (pueue) — route heavy runs through it

When multiple agents run at once, don't launch heavy test/benchmark runs as raw
background jobs — they thrash the laptop (three `make test-fast` = 12 xdist workers
on 8 cores). Submit through the shared per-host queue instead, so the `pueued`
daemon enforces a fixed slot budget per group no matter how many agents submit:

```bash
./scripts/gwq test-fast            # queued; the `test` group caps concurrency
./scripts/gwq bench bench_scf cpu 8 nosym   # captured -> benchmarks/results/<host>/
./scripts/gwq --host asus bench bench_scf cpu 8 nosym
./scripts/gwq status               # live queue across thinkpad + asus
./scripts/gwq log <id>             # tail a job's output
```

Benches from the thinkpad default to asus (keep perf off the laptop). Queued jobs
run against the canonical `~/github/gradwave`, not your worktree — pull it first if
you need a specific revision. Keep pueue coarse (one job = one whole run/sweep); a
future Dask sweep nests inside a single `gwq sweep` slot. Full reference and the
home-manager install snippet: **`docs/queue.md`**. If `gwq` reports pueue missing,
it isn't installed on that box yet — point the user at `docs/queue.md` (needs a
willnix rebuild), don't fall back to raw runs silently.

## Parallel agents & worktrees

Several agents run at once, each in its own worktree under `.claude/worktrees/`.
Worktrees isolate tracked files (you cannot clobber another agent's code), so the
real hazards are drift, stale clutter, and two agents editing the same module.
Rules that keep the fleet from tangling:

- **One worktree = one branch = one task**, and name the worktree after the branch.
  Never run two agents in the same worktree; never reuse a worktree for a new task
  (make a fresh one).
- **Branch from fresh `origin/main`.** Never stack a branch on another *unmerged*
  branch — main is squash-merged, so stacking guarantees conflicts on merge.
- **Keep branches short-lived and rebase on `origin/main` before opening/merging**
  the PR, so conflicts surface locally. Long-lived branches drift and rot.
- **Check for collisions before and during work:** `make worktrees` shows every
  worktree's drift, flags stale (merged) branches, and — the part that's otherwise
  invisible — lists files edited in more than one active worktree. If your file
  shows up there, coordinate before both sides diverge further.
- **Prune merged worktrees** with `make worktrees-prune` (removes only stale, clean,
  idle worktrees under `.claude/worktrees/`; never the primary checkout or a busy one).
- **Shared state is NOT worktree-isolated** — the git stash stack (never bare
  `git stash`; use a WIP commit), the primary `~/github/gradwave` checkout, and the
  `willnix` config repo (treat as single-writer; two agents editing it *will* clobber).

## Remote compute (asus)

A second NixOS box is reachable at `ssh asus` (Tailscale + LAN): 22 cores and an
RTX 3050 (6 GB). It is a synced peer — `uv` is installed and gradwave lives at the
same path (`~/github/gradwave`) — so offloading a job is just:

```bash
ssh asus 'cd ~/github/gradwave && git pull && uv sync && uv run <cmd>'
```

Use it for embarrassingly-parallel, self-contained work: benchmark sweeps
(`benchmarks/`), inverse-design and delta-gauge scans, fixture regeneration — each
worker runs a full SCF and returns numbers. Do not split a single
differentiable/autograd computation across machines: PyTorch autograd graphs are
process-local and do not serialize.

GPU caveat: the RTX 3050 has crippled fp64 (~1/64 of fp32), so for float64 SCF the
22 CPU cores usually beat the GPU; it only helps fp32-tolerant kernels. torch sees
CUDA on asus only because willnix puts the driver on the nix-ld search path
(`programs.nix-ld.libraries = [ config.hardware.nvidia.package ]` in hosts/asus);
if `torch.cuda.is_available()` ever returns False, check that line and rebuild. Verify:
`ssh asus 'cd ~/github/gradwave && uv run python -c "import torch; print(torch.cuda.is_available())"'`.

For occasional offload, plain SSH (optionally GNU `parallel -S :,asus`) is enough.
Reach for `dask.distributed` (scheduler local, `dask worker` on each host, GPU
workers tagged `--resources GPU=1`) only when sweeps get large enough to want a
dashboard, retries, and automatic placement.

## Definition of done

Run `make hooks` once per clone to install the pre-commit hooks (ruff on commit,
fast gate on push). Before opening a PR, from the worktree:

1. `uv run ruff check` is clean.
2. `uv run pytest -m "not standard and not slow and not torture and not gpu"` passes.
3. The branch is rebased on `main` so conflicts surface locally rather than at merge.
4. Regenerate `uv.lock` (`uv lock`) only if dependencies changed, and commit it last.

CI runs ruff and the standard tier on every PR, so let the green check stand in for
re-running the standard suite by hand.
