# LCAO / confined-NAO re-investigation (scoping)

Status: scoping memo, no build. Branch `research/lcao-nao-scoping`.
Date: 2026-09-10.

## What this is NOT re-litigating

There is a prior *measured* negative — [[lcao-startingwfc-low-ceiling]] (2026-08-08):
seeding the **first Davidson diagonalization** from PP_PSWFC pseudo-atomic
orbitals (QE `startingwfc='atomic'`) does not robustly help, because
`scf.loop.scf` warm-starts each SCF step's Davidson from the *previous* step's
eigenvectors, so the wavefunction seed only touches step 1 of ~8–11. That door
is closed and this memo does not reopen it.

This memo attacks the doors that verdict did **not** close. Its own summary
carries the key: *"SCF step count is set by density mixing, not the wfc seed."*
That sentence is the premise of Door A — the **density trajectory**, not the
wavefunction seed, is the lever, and the wfc-seed verdict says nothing about it.

## Door A — self-consistent LCAO draft SCF → PW polish

### The idea

gradwave starts every SCF from a superposition-of-atomic-densities (SAD) guess
(`scf.loop._seed_density` → `sad_density`) and Pulay mixing then relaxes that
density from scratch on every run. A tiny self-consistent LCAO/NAO SCF (basis of
~10–30 orbitals/atom, generalized eigenproblem of dimension ~50–500 — trivial
next to the PW solve) could hand PW a nearly-self-consistent density, so PW
starts partway converged and spends fewer iterations relaxing ρ.

The handoff machinery already exists and needs no new plumbing:

- `scf.loop.scf(start_from=…)` accepts a previous `SCFResult` *or* the checkpoint
  view from `io.checkpoint.as_start_from`.
- `scf.common.warm_start_densities` reads `rho` (nspin=1) / `rho_spin` (nspin=2)
  off `start_from`, validates the FFT grid matches, and rescales by the volume
  ratio so the electron count is exactly conserved. **A density-only dict
  `{"system": ns(grid=…), "nspin": 1, "rho": ρ}` is a complete, valid warm start**
  — no wavefunctions required. This is exactly the shape an LCAO draft would
  produce: a density on the PW FFT grid, no PW coefficients.
- When `start_from is not None` the first-step Davidson tolerance also tightens
  (`first_tol` 1e-3 → 1e-6), matching a "we already have a good density" restart.

So the only open question is empirical: **how many PW-SCF iterations are spent
"just relaxing the density," i.e. what is the gap between a cold SAD start and a
perfect-density start?** That gap is the *exact upper bound* on Door A — a real
LCAO draft hands over an LCAO-quality (approximate) density, which can only do
worse than the system's own converged density. If the gap is small, Door A is
dead regardless of how good the LCAO basis is. This ceiling needs **no LCAO code
at all** to measure.

### The ceiling measurement (asus, iteration-count only, no wall-time claim)

Method: run each system cold (SAD), save the converged density, re-run
`start_from` that density (density only, no PW wavefunctions = the honest LCAO
handoff), and also `start_from` the full `SCFResult` (density + PW coefficients =
absolute floor). PBE, `use_symmetry=False`, `etol=1e-8`, `rhotol=1e-7`. Run on
asus 2026-09-10 (gradwave f82c05f) at nice 19 alongside other load — iteration
counts only, no wall-time claim.

| system | regime | cold `n_iter` | density-only warm | full restart | **Door-A gap** |
|---|---|---|---|---|---|
| Al-4 (fcc conv, 4³ k, Fermi-Dirac 0.02 Ry, 40 Ry) | metal | 10 | 4 | 4 | **6 (60%)** |
| Si-8 (diamond conv, 2³ k, no smear, 30 Ry) | insulator | 11 | 4 | 2 | **7 (64%)** |

`dE(cold vs warm)` = 2.7e-12 eV (Al) / 1.1e-13 eV (Si) — both trajectories reach
the same fixed point; warm-start changes only the starting point, never the
converged answer.

Two structural observations beyond the headline gap:

- **The density carries essentially the whole benefit.** On the metal,
  full-restart (density + wavefunctions) is *identical* to density-only (4 = 4):
  the PW coefficients add nothing once the density is right — exactly consistent
  with the prior wfc-seed negative. On the insulator the wfc add another 2 iters
  (4 → 2), still minor next to the 7-iter density gap.
- The gap is uniform across metal/insulator here (~60%): at these small sizes
  the SAD→converged density relaxation dominates the iteration count in both
  regimes.

**Verdict (Door A): the ceiling is large — 6–7 of 10–11 iterations (~60%) are
density relaxation.** Well above the ≤2-iter kill threshold; the door is alive.
Caveats before celebrating: (a) this is the *perfect*-density limit — an
LCAO-quality draft density lands somewhere between SAD and converged (a minimal
basis has no interstitial variational freedom), so the realizable fraction of
the 6–7 iters is unknown and could plausibly be half or less; (b) the iterations
saved are the *early, cheap* ones (loose adaptive Davidson tolerances), so
wall-time savings < iteration savings; (c) the LCAO draft SCF is cheap on the
eigensolve side (dim ~50–500) but still pays grid-side Hartree/XC FFTs per draft
iteration, which must be counted. The honest next probe, if pursued: measure the
realizable fraction with a *cheap approximate* density that needs no new code —
e.g. a 3–4-iteration half-ecut PW draft SCF on the same FFT grid, density handed
over via the same `start_from` dict. If a sloppy draft density recovers most of
the 6-iter gap, an LCAO basis is unnecessary; if it recovers little, LCAO won't
do better either.

### What other codes report

- **QE** `startingpot='file'` vs `'atomic'`: reusing a converged charge density
  "substantially" cuts SCF iterations, but the docs give no number and this is
  the *converged-density* reuse case (relax/restart), not an LCAO draft. It is
  the same perfect-density-start limit the ceiling measurement bounds directly.
- **GPAW**: PW/FD modes *already* build their initial wavefunctions by
  diagonalizing an LCAO Hamiltonian in a minimal atomic-orbital basis on the SAD
  effective potential — i.e. GPAW does the LCAO step for the *wavefunctions* by
  default, but its initial *density* is still SAD. So even GPAW does not do a
  self-consistent LCAO **density** handoff; guidance is qualitative ("LCAO
  converges more easily; use it as a starting point"), no published iteration
  saving for a density handoff.
- No code in the sweep publishes a quantitative "self-consistent LCAO-density
  start saved N iterations in a PW run" figure. This is genuinely unmeasured in
  the literature, which is why the in-tree ceiling measurement is the deciding
  datum.

## Door B — confined-NAO projection layer

### The real limitation

`postscf/pdos.py`, `postscf/band_center.py`, and `postscf/cohp.py` project bands
onto pseudo-atomic orbitals read from **PP_PSWFC** (norm-conserving) / **PP_AECHI
/ chi** (PAW) tables in the pseudopotential file. When those tables are absent —
SG15 ONCV, many ONCV sets — `pdos._orbitals_for` raises
`"{element}: the pseudopotential carries no PP_PSWFC atomic orbital"` and PDOS /
d-band-center / COHP are simply unavailable. This is a real, user-facing
capability gap (memory: COHP/PDOS need PseudoDojo PD_*_std pseudos today).

### Why `sto_basis.py` does NOT already solve it

`pseudo/sto_basis.py` (`fit_contracted_sto`) fits a contracted STO **to an
existing PP_PSWFC/PP_AECHI orbital** as its curve-fit target. It presupposes the
very table Door B lacks — it reshapes an orbital, it does not generate one from
the pseudopotential. So it is not a fallback for PSWFC-less pseudos.

### What a minimal confined-NAO generator needs

SIESTA/ABACUS/FHI-aims all generate NAOs the same way: solve the isolated
pseudo-atom radial Kohn-Sham problem for the valence shells under an added
confinement potential (SIESTA "energy shift" ΔE_PAO, or FHI-aims' smooth
cutoff), giving strictly (or softly) localized R_nl(r). The pieces already in
`pseudo/`:

- **Reusable now**: `radial.simpson` (log-mesh integration), `radial.sph_jl` /
  `radial.sbt` (radial → q-space Bessel transform — produces exactly the form
  the projector code consumes), the UPF mesh (`r`, `rab`), the local potential
  (`pseudo/local.py`) and KB projectors (`pseudo/kb.py`) that define the
  pseudo-atom's Hamiltonian, and `AtomicOrbital.rchi = r·R_nl` as the output
  container. `pdos.py:196` already documents a hook to substitute a custom
  radial for the raw PSWFC table — a generated NAO drops straight in.
- **Missing (the build)**: a radial bound-state solver — a Numerov/shooting or
  a matrix (finite-difference) eigensolver — for the semilocal+KB pseudo-atom on
  the log mesh, plus a confinement potential (start with the SIESTA soft/hard
  confinement, one `rc` per shell from an energy-shift target). A frozen-potential
  single solve (no atomic self-consistency) is the minimal first cut; full atomic
  SCF (radial Hartree via `simpson` cumulative integral + LDA XC) is the accurate
  version. ~200–400 LOC for the frozen-potential version, ~600–800 for atomic SCF.

### Validation

Cross-check generated NAOs against elements where a PSWFC-bearing pseudo exists
(the fixture tree has both ONCV-with-PSWFC and PSWFC-less sets). Compare (a) the
radial shape R_nl(r) inside the pseudization radius, (b) the resulting PDOS /
d-band-center against the PSWFC-projected reference on the same SCF. Agreement of
the *projection-derived observables* (not absolute orbital normalization) is the
bar, mirroring the FLAPW "validate on splittings, not absolutes" discipline.

## Door C — NAO subspace as a CheFSI bootstrap (assess only)

`solvers.chebyshev_filtered_batched` needs a starting occupied subspace to filter;
today the auto-gate for CheFSI sits above the cell sizes currently reachable on
the available hardware (small-cell wall is latency, not arithmetic — see the SCF
perf memories). A confined-NAO block would be a natural nb-sized starting
subspace for the large-nb slab regime CheFSI targets (nb ~500+), replacing the
identity/random start. But: (a) the same first-step-only ceiling as the wfc-seed
negative applies once the SCF is warm-started step-to-step; (b) CheFSI's own
subspace is carried across SCF steps too, so the NAO seed again only touches
step 1; (c) the regime where this would matter is not currently reachable, so it
cannot be measured. **Assessment: do not build.** Revisit only if/when the CheFSI
large-nb slab regime becomes a live, benchmarkable workload, and even then expect
a first-step-only payoff. No effort estimate — gated behind hardware that does not
exist in the loop yet.

## Recommendations (ranked)

1. **Door A — measured ceiling is 6–7 of 10–11 iterations (~60%): PROBE NEXT,
   but with the zero-code draft first.** The plumbing is already 100% present
   (`start_from` + `warm_start_densities` accept a density-only dict). Before
   writing any LCAO code, run the half-ecut PW draft probe above to measure the
   *realizable* fraction of the ceiling with an imperfect density. Build the
   LCAO draft only if (i) the cheap draft leaves most of the gap on the table
   AND (ii) a wall-time (not just iteration-count) projection survives the
   early-iterations-are-cheap caveat. Effort: probe ~half a day on asus;
   LCAO draft SCF proper ~1–2 weeks (radial NAO generator + crystal S/H
   assembly + generalized eigensolve + density on the FFT grid).
2. **Door B — confined-NAO generator. BUILD (conditional).** A genuine
   capability win (PDOS/COHP/d-band-center for any pseudo), not perf, so it is
   not subject to any warm-start ceiling. Recommended scope: frozen-potential
   radial solver + SIESTA-style confinement first (~200–400 LOC), validated
   against PSWFC-bearing pseudos; escalate to atomic SCF only if the
   frozen-potential orbitals miss the projection bar. Condition: confirm real
   user demand for projections on PSWFC-less pseudos beyond the current
   PseudoDojo workaround. Effort: ~2–4 days frozen-potential, ~1–1.5 weeks with
   atomic SCF + validation.
3. **Door C — CheFSI NAO bootstrap. DO NOT BUILD.** First-step-only ceiling +
   unreachable regime. Reassess only when large-nb slab CheFSI is benchmarkable.

Note that Doors A and B **share the same missing component** — a radial
pseudo-atom solver with confinement. If Door B is built, Door A's incremental
cost drops to just the crystal-level LCAO assembly + generalized eigensolve, so
the two decisions are coupled: build B first, and B's radial solver de-risks and
cheapens any later A prototype.
