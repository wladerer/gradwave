# SCF eigensolver research — living knowledge base

A shared log for the eigensolver research program. Every worker (Davidson
baseline, CheFSI, and the planned LOBPCG / subset Rayleigh-Ritz / adaptive
scheduler) appends here: the benchmark battery's numbers, what works and what
does not on which systems, and the evolving adaptive strategy.

## How this connects to the code

- **Uniform interface + registry:** `gradwave/solvers/registry.py`. Every block
  eigensolver registers under a name and implements

  ```
  solve(apply_H, X0, precond, mask, *, tol, nbands, **kw) -> EigResult
  ```

  where `EigResult = (eigenvalues[nk,nb], eigenvectors[nk,nb,npw], n_iter,
  residual_norms[nk,nb], diagnostics: dict)`. `precond` is the kinetic diagonal
  `T` (what the Teter/plane-wave preconditioner is built from); solvers that need
  no preconditioner ignore it. The SCF loop selects a solver purely by name via
  `scf(..., eigensolver="<name>")` — `scf/loop.py::_solve_bands` dispatches
  through `registry.get`, and `_validate_scf_args` accepts any registered name.

- **To add a solver:** implement the adapter signature, `register("name", fn)`
  at import time, and it is immediately selectable in `scf()` and picked up by
  the battery (which runs every registered solver).

- **Battery runner:** `benchmarks/solver_battery/run.py`. Runs the full SCF with
  each registered solver across the systems below, writes one JSON per system to
  `results/`, an aggregate `results/summary.json`, and regenerates the results
  table in this file.

## Systems (the battery)

Small on purpose — a few minutes each; this is a harness, not a converged-physics
study. The correctness gate compares each solver to Davidson on the *same*
Hamiltonian, so cutoffs/k-meshes only need to give a well-conditioned H.

| Name | Class | System | Notes |
|---|---|---|---|
| `si_insulator` | insulator | Si diamond, 2 atoms | fixed occupations, no smearing |
| `mgo_insulator` | insulator | MgO rocksalt | wide gap; Mg semicore |
| `al_metal` | simple metal | Al fcc | free-electron-like, Gaussian smearing |
| `cu_metal` | noble metal | Cu fcc | narrow 3d bands near E_F |
| `fe_fm_metal` | ferromagnet | bcc Fe, collinear nspin=2 | Stoner metal; Johnson mixing |
| `cr_afm_metal` | antiferromagnet | 2-atom bcc Cr | opposite start moments |
| `bi_heavy` | heavy (scalar-rel) | Bi fcc | heavy element, dense spectrum |

**Known gap — true SOC:** spin-orbit coupling has no collinear representation;
it lives in the spinor SCF (`scf/noncollinear.py`), which is a *separate* solver
family the registry does not yet wrap. `bi_heavy` uses a scalar-relativistic
pseudo as a heavy-element stand-in. Wiring the spinor block solver into the same
registry (so a fully-relativistic Pt/Bi enters the battery) is a tracked
extension point.

## Results

<!-- RESULTS TABLE START -->
_Not yet run. Populate with `PYTHONPATH=src uv run python benchmarks/solver_battery/run.py`._
<!-- RESULTS TABLE END -->

Columns: **SCF iters** (outer self-consistency steps), **Wall** (full SCF wall
time), **eigh** (accumulated `torch.linalg.eigh` self-time = the Rayleigh-Ritz
cost) and its **share** of wall, **Converged** + **final |Δρ|** (the last
self-consistency residual), **E** (converged total energy), and **ΔE vs
Davidson** (the correctness gate; must be ≲ 1e-6 eV — anything larger is flagged
⚠ and is itself a finding).

## What works / what doesn't — and on which systems

_Filled in from the first battery. Structure to keep as the program grows:_

- **Insulators (Si, MgO):** _pending first run._
- **Simple / noble metals (Al, Cu):** _pending._
- **Magnetic (FM Fe, AFM Cr):** _pending._
- **Heavy / relativistic (Bi):** _pending._
- **Correctness flags:** _any solver whose converged energy disagreed with
  Davidson beyond the gate — pending._

## Adaptive strategy (evolving)

The end goal is a scheduler that picks (or switches) the eigensolver per system —
and possibly per SCF iteration — from cheap, observable features. Stub to grow:

- **Signals available today:** system class (insulator vs metal vs magnetic),
  band count, plane-wave count per k, k-count (batch width), the running SCF
  residual, and per-solver `eigh_share` (how RR-bound the solve is).
- **Hypotheses to test** (unverified until the battery + LOBPCG/subset-RR land):
  - Growing-subspace Davidson should dominate when the Rayleigh-Ritz `eigh` is a
    small share of wall (few bands, cheap H-apply); a filter (CheFSI) or a
    subset method should win when `eigh_share` is large (many bands / many k).
  - CheFSI needs a buffer above the wanted bands and good spectrum bounds; its
    edge is GPU / high-band-count regimes with one host sync per round.
  - Metals (dense DOS at E_F) vs insulators (clean gap) likely want different
    tolerances / restart policies.
- **Decision rule:** _none yet — to be fit once ≥3 solvers have battery data._

## Changelog

- Harness bootstrapped: uniform interface + registry, Davidson & CheFSI adapters,
  battery runner, this knowledge base.
