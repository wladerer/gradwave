# Thermochemistry / formation layer — completion plan

Status: **scoped, not scheduled** (2026-08-15). Deprioritized — thermodynamics is
not expected to be used rigorously for a while. This is a record so the work can
be picked up cleanly later.

## The finding

Phases 4–5 of the [thermochem/mechanical buildout] (vibrational thermodynamics;
formation/cohesive energies) are **already implemented as clean, unit-tested
library modules** — they are simply not wired into the task/input/CLI/reporting
surface. The gap is mostly *plumbing*, plus one genuine physics gap
(reference-energy provenance for formation energies).

| Module | Computes | Wired as a task? |
|---|---|---|
| `postscf/thermo.py` | harmonic F(T), U, Cv, S, ZPE, θ_D from a phonon DOS; Sommerfeld Cₑₗ | no |
| `postscf/qha.py` | quasi-harmonic V(T), G(T,P), α(T), Cp, Grüneisen | no |
| `postscf/convex_hull.py` | binary formation energy, lower hull, hull distance, ground states | no |
| `postscf/phase_diagram.py` | T–x diagram by common tangent, binodal, Tc | no |
| `postscf/composition_design.py` | differentiable composition surrogate + optimizer (torch) | no |
| `postscf/lattice_mc.py` | configurational Cv/S/F via fixed-J Ising Metropolis | no |
| `postscf/phonons_supercell.py` | frozen-phonon dispersion **+ phonon DOS** | **yes** (`phonons`) |

Only `eos`, `elastic`, `phonons` are tasks (whitelisted in `inputs/parse.py` and
`api/dispatch.py`'s `_POSTSCF_RUNNERS`). Critically, `run_phonons` builds the
phonon DOS but **never calls `thermo.py`** — even the one wired path stops at the
DOS.

## The task-wiring checklist (per capability)

Each of the below is the same mechanical sequence: `*Params` dataclass +
`Input` field + `_ALLOWED_TOP` entry + the two `task` whitelists
(`inputs/parse.py`, `api/dispatch.py`) + a `run_<task>` driver + summary block +
output formatter + `cli` print + plot `--kind` + tests + a `docs/examples` recipe.

## Phase 4 — Vibrational thermodynamics (pure wiring)

- **4a. Phonons → thermo bridge (highest value/effort; do first).** In
  `run_phonons`, call `thermo.{zero_point_energy, free_energy_vib,
  internal_energy_vib, heat_capacity, entropy, debye_temperature}` on the DOS it
  already builds, over a temperature grid, and emit a `thermo` sub-block. Add
  `temperatures` to `PhononParams`; add output/CLI/plot for F(T)/Cv/S. Both halves
  exist and share units — roughly a day, standalone PR. ZPE and finite-T free
  energies then fall out of every phonon run.
- **4b. QHA task (`task: qha`).** New driver composing an EOS-style volume scan
  with a per-volume phonon DOS → `qha()` for α(T), G(T,P), Grüneisen. More effort
  (N phonon runs), naturally parallel over volumes.

## Phase 5 — Formation & cohesive energies (wiring + the one real decision)

- **5a. Reference-energy provenance — a design decision (below).**
- **5b. Cohesive energy** (`E_bulk/atom − E_isolated_atom`): needs a
  spin-polarized isolated-atom-in-a-box energy; small helper.
- **5c. Formation energy + convex hull task (`task: formation`).** Wire
  `convex_hull` with the reference model. Real capability gap: it is **binary-only**
  today; N-ary (grand-potential) hull is the non-trivial physics addition.
- **5d. Finite-T formation (optional, combines 4+5):** fold `F_vib(T)` into
  formation energies / hull distances.

### The reference-energy decision

Formation energy = E − Σ μᵢ. Options for the reference chemical potentials:

- **(a) User-supplied** (`references: {Al: -3.74, …}`) — simplest, most flexible,
  no database. **Recommended first step.**
- **(b) Computed on demand** — gradwave runs the elemental reference phases as a
  sub-step (rigorous, self-consistent settings, but slow; needs a reference-structure
  catalog).
- **(c) Small committed reference set** at pinned settings (like the delta-gauge
  pseudos).

Recommendation: ship (a), add (b) as a helper, skip a large (c) database. Formation
energies are only as good as reference consistency, and (a) makes that the user's
explicit, visible choice.

## Phase 6 — Configurational (deferred, specialized)

Wire `phase_diagram`, `composition_design`, `lattice_mc` as tasks. Lower priority.
`lattice_mc` is a fixed-J Ising model, so real alloy use needs a cluster-expansion
fit (J from DFT) that does not yet exist; `composition_design`'s differentiable
surrogate (which rides the `scf.alchemical` engine) is the nearest existing thing.

## Suggested order

**4a → 4b → 5a(user-supplied) → 5c → 5b → 5d**, Phase 6 deferred. 4a alone is the
biggest value-per-effort and a clean standalone PR.

[thermochem/mechanical buildout]: the original 6-phase plan; Phases 1–3 (EOS,
elastic, supercell/q-mesh phonons) shipped in PRs #34, #41, #61.
