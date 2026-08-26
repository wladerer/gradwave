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
- `inputs/` — the input layer: `models.py` (the `Input`/`*Params` dataclass
  schema), `parse.py` (loading/validation); a leaf, imports no physics
- `api/` — the driver layer (Layer C), split by task: `system`, `scf`, `relax`,
  `eos`, `elastic`, `phonons`, `dispersion`, `summary`, `dispatch` (+ `_common`)
- `io/` — reporting and serialization: `output`, `checkpoint`, `runinfo`,
  `templates`, and `analysis` (frames/plotting; imports pandas/matplotlib
  lazily via the `analysis` optional-dependency group — core and CLI run
  without them)
- `cli.py`, `calculator.py`, `distributed.py` — entry points above the api layer

Tests live in `tests/{unit,integration,gradcheck}` with shared fixtures in
`tests/fixtures` and helpers in `tests/helpers.py`.

## Capability map — reach for these, don't reimplement

Before writing a helper, check whether one of these canonical symbols already
does it. This table is the judgement layer (which symbol is right, what not to
touch); for a full greppable list of every public symbol with its signature,
run `make symbols` to generate the gitignored **`docs/symbols.txt`**, then grep
it (`grep stress docs/symbols.txt`). Import from the leaf module — the physics
subpackage `__init__.py` files are empty; underscore-prefixed names are
internal. Exceptions: `api/` and `inputs/` re-export their public surface from
`__init__` (the historical flat-module paths, so `from gradwave.api import run`
and `from gradwave.inputs import Input` stay valid). When monkeypatching an
`api` internal in a test, patch the owning leaf module (e.g.
`gradwave.api.relax._build_relax_calc`), not the package re-export — the
drivers call their own module globals.

| If you need to… | Use | Not |
|---|---|---|
| Run a task from an `Input` (dispatches scf/relax/eos/…) | `api.run` (or `api.run_scf`/`run_relax`/`run_eos`) | drive `scf/` internals by hand |
| Build the right system (auto NC vs USPP/PAW) | `api.build_system` | call `scf.loop.setup_system` / `scf.uspp_setup.setup_uspp` directly unless you need the specific formalism |
| Use gradwave as an ASE calculator | `calculator.GradWave` | wrap `api.run` yourself |
| Parse / validate an input file | `inputs.load_input` → `inputs.Input` | hand-parse TOML/YAML |
| Unit conversions (Ha→eV, Ry→eV, Bohr→Å, ℏ²/2m, e²) | `constants.*` (`HARTREE_EV`, `RY_EV`, `BOHR_ANG`, `HBAR2_2M`, `E2`, `KB_EV`) | hardcode a conversion factor anywhere |
| Working precision / paired real dtype | `dtypes.RDTYPE`/`CDTYPE`, `dtypes.real_of(cdtype)` | write `torch.float64` inline |
| FFT real↔G transforms | `core.fftbox.{r_to_g, g_to_r, g_to_r_box}` (batched: `core.batch.g_to_r_b`) | roll your own `fftn`/`ifftn` (fftbox is normative for sign/normalization) |
| Build the FFT grid / plane-wave G-sphere | `grids.build_fft_grid`, `grids.build_gsphere` | re-derive good FFT sizes or the ecut sphere |
| Monkhorst-Pack k-mesh; reduce to the IBZ | `kpoints.monkhorst_pack`; `symmetry.reduce_mesh` | reimplement MP folding or symmetry reduction |
| Space group / symmetrize density / forces | `symmetry.find_spacegroup`, `symmetry.RhoSymmetrizer`, `symmetry.symmetrize_forces` | call spglib directly |
| The SCF loop (pick the formalism explicitly) | `scf.loop.scf` (NC), `scf.uspp_loop.scf_uspp` (USPP/PAW), `scf.noncollinear.scf_noncollinear` (spinor/SOC) | write a mixing/diagonalization loop |
| Eigensolver | `solvers.davidson_batched` (workhorse), `solvers.chebyshev_filtered_batched` | hand-code Davidson |
| Forces / stress | `postscf.forces.forces`, `postscf.stress.stress` (`stress_kbar` to convert) | recompute Hellmann-Feynman/Pulay terms |
| Bands / DOS / PDOS / phonons / EOS | `postscf.{bands.band_structure, dos.kpm_dos, pdos.projected_dos, phonons, eos.fit_bm3}` | |
| Load a pseudopotential (NC or PAW) | `pseudo.upf.parse_upf`, `pseudo.upf_paw.parse_upf_paw` (unified: `api._load_upf`, path-cached) | re-parse UPF XML; re-implement the radial FT (`pseudo.radial.sbt`) |
| Build the result summary / serialize / render | `api.build_summary`, `io.checkpoint.save_checkpoint`, `io.output.format_output` | hand-roll the summary-dict schema |
| Warm-start an SCF from a checkpoint | `io.checkpoint.load_checkpoint` → `io.checkpoint.as_start_from` (pass as `scf(..., start_from=)`) | |
| Load results into pandas frames / plot | `io.analysis.{load, scf_frame, bands_frame, plot_bands, …}` (needs the `analysis` extra) | parse the JSON by hand |

**Built but not task-wired (library-only).** These `postscf` modules are
importable and unit-tested but have **no Input / CLI / JSON surface** — `api.run`
does not reach them, so don't assume a task exists: `qha`, `convex_hull`,
`phase_diagram`, `composition_design`, `lattice_mc` (`lattice_mc` is a fixed-J
Ising model, not a fitted cluster expansion). Harmonic `thermo` **is** wired —
`run_phonons` emits an F/U/Cv/S(T) + ZPE + θ_D `thermo` block whenever it builds
a DOS. Natural next finishes: `qha` (reuses the phonon `thermo` bridge across
volumes) and `convex_hull` formation energies (blocked on reference-energy
provenance — needs caller-supplied elemental refs).

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

`gh pr checks <n>` intermittently fails with "no commit found on the pull request" even when
the PR clearly exists — query `gh api repos/<owner>/<repo>/commits/<sha>/check-runs` directly
instead, it doesn't have this problem. Re-check `gh pr view --json mergeable` immediately
before merging even if CI showed green earlier in the same session — another PR merging in
the interim can silently re-trigger a conflict that wasn't there a minute ago.

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
- **Verify a branch's real state before rebasing it, and verify the result after.** A
  worktree's local branch can silently drift from the actual remote branch (a stale
  checkout, an earlier operation that never pushed). Before `git rebase origin/main`,
  confirm `git rev-parse HEAD` matches the remote branch's real tip (`git fetch` +
  `git rev-parse origin/<branch>`) — a rebase with nothing real to replay still reports
  "Successfully rebased" while silently dropping every commit. After rebasing, diff the
  result against the commit's own pre-rebase parent (`git diff --stat <old-parent>
  <old-commit>` vs `git diff --stat origin/main HEAD`) rather than trusting a clean
  exit code as proof nothing was lost.
- **Split verification from destructive action into separate tool calls.** `cd <path>
  && pwd` (and `git status`) to confirm location and state, THEN `git reset --hard` /
  `git rebase` / etc. as its own following call. Chaining a verification step and a
  destructive command in one shot is easy to get wrong and hard to audit after the fact.
- **A background job needs a live `Monitor`, not a hope of being resumed.** If you're
  pausing your turn to wait on a background command, arm your own `Monitor` on its
  completion marker before you stop — otherwise there's no reliable signal that you're
  actually still working, and you'll generate repeated no-op "still waiting" notifications
  instead of one real one. If an investigation branches (e.g. you abandon one system or
  approach for another), explicitly stop the abandoned branch's `Monitor` rather than
  leaving it to fire on its own timeout well after the real work is already done.

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
2. `uv run ty check` is clean (error-level gate on the growing typed-file list in
   `pyproject.toml`'s `[[tool.ty.overrides]]`; warn-only (non-blocking) everywhere
   else — see "Typing" below).
3. `uv run lint-imports` is clean (the import contracts in `pyproject.toml`'s
   `[tool.importlinter]`, where `scf -> postscf._response` goes through the
   shared-response-kernel exception list).
4. `uv run pytest -m "not standard and not slow and not torture and not gpu"` passes.
5. The branch is rebased on `main` so conflicts surface locally rather than at merge.
6. Regenerate `uv.lock` (`uv lock`) only if dependencies changed, and commit it last.

CI runs ruff, ty, and the standard tier on every PR, so let the green check
stand in for re-running the standard suite by hand.

## Typing

Gradual rollout, not a flag day: `ty` (Astral's type checker) runs with every rule
demoted to `warn` by default (`[tool.ty.rules] all = "warn"`, plus
`[tool.ty.terminal] error-on-warning = false`), so the untyped bulk of
`src/gradwave` stays visible but never fails CI, and at `error` severity over an
explicit, growing file list (`[[tool.ty.overrides]]`). Only add a file to that list
once it is fully, cleanly typed — never let a listed file regress. `jaxtyping`
annotates tensor shape/dtype where it is meaningful (e.g.
`Complex[Tensor, "nk nb npw_max"]`), replacing shape-in-a-comment conventions;
static shape strings are not checked by `ty` (shapes are runtime data here), so
pair a `jaxtyped`-annotated function with `@jaxtyped(typechecker=beartype)` where
the runtime check is worth its (small, O(1)) cost — not blanket-applied, never on
a hot inner loop (Davidson, the SCF step) where dispatch overhead has proven to
matter, and not forced onto a signature where jaxtyping's shape-consistency model
doesn't actually fit (e.g. a `list[Tensor]` whose elements are legitimately
different shapes — see `calculator._remap_coeffs_to_spheres`).

Going forward, put `@override` (PEP 698, `typing_extensions.override` on py<3.12,
`typing.override` from 3.12) on any subclass method that overrides a base class
method (e.g. `Smearing`/`XCFunctional`/`SpinXC` subclasses) — it catches a typo'd
signature or a renamed base method as a type error instead of a silent no-op
override. This was not consistently applied before the gradual typing rollout;
add it whenever you touch a subclass method, don't do a repo-wide sweep for it.
