# AutoAPW campaign — code-quality / complexity / dedup inventory

Read-only inventory over the ~26-PR AutoAPW influx. Scope: `src/gradwave/flapw/*`,
`src/gradwave/postscf/{gipaw,kgeometry_nmr,kgeometry,_response}.py` and their tests.
Base: worktree `flapw-code-quality` @ `fce343b`. Line refs are approximate (concurrent
edits in flight). **No source was modified. No SCF / heavy test was run.**

Ranking key: **value ÷ risk**. RISK = LOW (pure move / static, bit-preserving) /
MED (behavior-affecting but a fast test guards it) / HIGH (behavior-affecting, no fast
guard — validate on asus before touching).

Baseline signals gathered:
- `ruff check` over the whole scope: **All checks passed** (no unused imports / dead-import
  hits — F401 is clean, so "unused imports" is a non-finding here).
- No `@override` gap: the scope has no subclass-method overrides (`functionals.py` is free
  functions; `gipaw.PAWOnSite`, the kgeometry_nmr helper classes have no base class). Do not
  force `@override` here.
- No stale "not built" docstrings remain: `gipaw.py` module docstring (lines ~78-84) already
  reads *"are now built (#349 Phase 1, #351 Phase 2)"* and line ~373 is honest ("an operator
  awaiting the assembled X_β … not yet an absolute"). The task's premise (that it claims the
  S-metric Sternheimer "is not built") is **already fixed** — verify, no action.

---

## SAFE TO DO NOW — static / bit-preserving, no asus

| # | Item | file:line | What / why | RISK | Guarding test |
|---|------|-----------|------------|------|---------------|
| S1 | **Split the 377-line `_multi_iterate` function** (CORRECTED — see note) | `flapw/scf.py:1265` (fn, 377 lines) | The single worst complexity hotspot in the campaign. `_multi_iterate` does 5+ distinct jobs: (1) build per-key `species`/`El` dicts (:1298-1310), (2) k-independent radial channels + `lodat` + nsph integrals + `us` (:1316-1327), (3) per-k solve dispatch (:1354-1381), (4) density accumulation, (5) Weinert potential + joint Anderson mixing + convergence gate. **CORRECTION (2026-08-22, verified against the source):** this row originally claimed a *297-line `_geom_for` closure* to hoist — that is WRONG. `_geom_for` (:1345-1352) is an 8-line k-geometry cache helper; the ~286 lines that follow it (:1354-1640) are the function body proper (per-k dispatch/density/Weinert/mixing), NOT inside `_geom_for`. So the fix is a genuine **function split** into ~3 module-level helpers (species/El setup; per-k solve dispatch; density+Weinert+mixing tail), not a trivial closure hoist. `_multi_iterate` is already refactored around `_MultiCtx`/`_MultiState`/`SolveKArgs`/`_solve_k_args`/`_lapw_multi_k`, so the split threads existing structs — behavior-preserving in intent but NOT byte-trivial. | MED (reclassified from LOW) | Only slow-tier FLAPW SCF integration tests (`tests/integration/test_flapw_scf.py`, `test_flapw_multi.py`) exercise `_multi_iterate` — no fast unit guard. **DEFER to asus** for the SCF validation. |
| S2 | **Dedup PW92 correlation block in `functionals.py`** | `flapw/functionals.py:16` (`vxc_lda`) and `:36` (`fxc_lda`) | ~8 identical lines (the `A,a1,b1..b4`, `denom`, `ec = -2A(1+a1 rs)log(1+1/denom)` PW92 setup) copy-pasted in both functions. Extract `_pw92_ec(rs)` returning the correlation-energy expression as a tensor op; both call it (one under `create_graph=True`, one detached — the autograd mode stays at the call site, so the helper is graph-agnostic and behavior-preserving). | LOW | `test_flapw_*` XC paths; `fxc_lda` is validated elementwise vs central-difference of `vxc_lda` (docstring, ~1e-10) — that FD test directly guards the split. |
| S3 | **Extract shared complex-Ylm-on-directions helper** (LOW VALUE) | `flapw/scf.py:88` (`_ylm_star`), `flapw/coulomb.py:134` (`gvec_ylm_tables`) | The angle-extraction + evaluation block repeats: `θ = arccos(clip(z/‖v‖))`, `φ = arctan2(y,x)`, small-norm mask, then `scipy.special.sph_harm_y(l,m,θ,φ)`. A shared `_complex_ylm_on_dirs(vecs, lmax)` is bit-for-bit identical. Caveat (per the Ylm audit): these are two *independent* scipy evaluators over *different domains* (plane-wave `ks` vs the n³ G-grid) with their own caches, so the shared surface is only the ~6-line direction→`sph_harm_y` core — modest value. Distinct from the **real**-harmonic `core.ylm.ylm_all` / `core.gaunt.ylm_np` (see D3). | LOW | `test_flapw_weinert.py` (covers `gvec_ylm_tables`), FLAPW SCF (`_ylm_star`). |
| S4 | **Doc audit pass (verify-only)** | `postscf/gipaw.py:78-84,373` | Confirm the already-updated Sternheimer status text stays accurate now that #349/#351 shipped. No edit needed unless a residual "Phase 2 / not built" phrasing is found elsewhere; grep of the scope for `not built|not yet|deprecated` returned only honest/accurate hits. | LOW | n/a (doc) |

**Highest-value TRULY-safe-now item: S2** (S1 was reclassified MED/defer-to-asus after the
297-line-closure premise was found wrong — see the S1 correction note). S2 (functionals PW92
dedup) is fully guarded by a fast unit test and is done in this pass.

**On S1 (deferred):** the 377-line `_multi_iterate` is the campaign's single biggest
readability/maintenance liability, but the split is a real function decomposition guarded only
by slow-tier SCF tests, so it waits for asus. Historical (now-corrected) rationale below:
the `SolveKArgs`/`_solve_k_args` scaffolding at scf.py:1931/1957
already exists to receive it), and it is bit-preserving.

---

## DEFER TO ASUS VALIDATION — numeric-dedup, behavior-affecting

| # | Item | file:line | What / why | RISK | Guarding test |
|---|------|-----------|------------|------|---------------|
| D1 | **Unify `flapw/mixing.anderson_next` onto `core/_anderson.AndersonMixer`** | `flapw/mixing.py:9` vs `core/_anderson.py:21` | **Same math** (Type-II Anderson; the "difference-against-last-iterate" window in `anderson_next` and the "consecutive-secant" window in `AndersonMixer` span the *same* subspace, so the projected step `u + β r − (ΔU+βΔR)γ` is algebraically identical). They differ in (a) state model (stateless history-list fn vs stateful class) and (b) the least-squares driver: `anderson_next` uses `lstsq(driver="gelsd")`, `AndersonMixer` uses Tikhonov-damped normal equations. Because (b) differs, unification is **not** bit-preserving — FLAPW SCF eigenvalues will move at the ~solver-tolerance level. A stateless thin wrapper over the class would consolidate but needs a full-SCF regression. `solvers.deflation.anderson_solve` is *not* a third mixer — it is a driver loop that already wraps `AndersonMixer`, so it is not part of this dedup. | MED→HIGH | **`anderson_next` has NO dedicated unit test** — only referenced by `test_review_errors.py` (an error-path test, not a numeric guard). Direct coverage is only the slow-tier FLAPW SCF. Treat as HIGH until an asus SCF regression confirms. |
| D2 | **Route FLAPW `spherical_jn` sites through the accurate `pseudo.radial.sph_jl`** | `flapw/lapw.py:27,137-156`; `flapw/coulomb.py:172,199` | The pseudo side is **already deduped** (`radial.sph_jl` ↔ `radial_torch.jl_t` share `_bessel_data` constants, pinned to 1e-14 by `test_review_dedup.py`), but the FLAPW side still calls raw `scipy.special.spherical_jn`, which `sph_jl`'s own docstring flags as only ~1e-9 accurate at small x. **Two sub-cases:** (a) `lapw.match_ab` uses `spherical_jn(l)` / `spherical_jn(l-1)` with l≤lmax(≤4) evaluated at *large* x=‖k+G‖R → scipy is fine there; routing through `sph_jl` is accuracy-neutral but still moves bits on a hot path → **MED**. (b) `coulomb.sphere_interstitial_moments` calls `spherical_jn(bl+1)` and `sphere_pseudocharge_ft` calls `spherical_jn(bl+npow+1)` with npow=`clip(round(R·gmax/6),2,8)` and bl up to fullpot_lmax — **the l argument routinely reaches 7 (interstitial) and 11–15 (pseudocharge, production fullpot_lmax=6), far exceeding `sph_jl`'s hard l≤4 cap (guarded by `test_numpy_l_range_guard`)**. So the coulomb consolidation is **infeasible** without first extending `sph_jl` to higher l, *and* these are the small-x-sensitive `j/x^(npow+1)` sites where scipy's 1e-9 could actually matter → **HIGH**. Recommendation: do (a) only if measured worthwhile; treat (b) as "known limitation, do not touch without extending the series impl + asus Weinert-accuracy check". | MED (lapw) / HIGH (coulomb) | lapw: FLAPW SCF integration (slow-tier). coulomb: `test_flapw_weinert.py` guards the moment/pseudocharge path but not against a higher-accuracy Bessel reference. |
| D3 | **Ylm/Gaunt across `core/*` vs `flapw/*` — mostly NOT mergeable** | `core/{ylm.ylm_all, gaunt.ylm_np/real_gaunt_table, spinor_proj.complex_ylm}`, `scf/paw_symmetry.ylm_rotation_matrices`, `flapw/efg.{ylm_rotations_complex:65, gaunt_matrix:97}` | Equivalence classes: **(real-harmonic eval)** `core.ylm.ylm_all`, `core.gaunt.ylm_np`; **(real-harmonic rotation matrices)** `scf.paw_symmetry.ylm_rotation_matrices`; **(complex-harmonic eval)** `core.spinor_proj.complex_ylm`, `flapw/*` scipy sites; **(complex rotation)** `efg.ylm_rotations_complex`; **(Gaunt triple-integral)** `core.gaunt.real_gaunt_table` (real basis, closed-form) vs `efg.gaunt_matrix` (complex basis, quadrature). These are **legitimately distinct** (real vs complex basis; evaluation vs rotation vs triple-integral) — the only real dedup is *within* the flapw complex-eval family (that is S3, safe). `efg.ylm_rotations_complex` and `scf.paw_symmetry.ylm_rotation_matrices` are the complex/real halves of the same construction but cannot share code across the basis divide without a convention refactor → do NOT force. | HIGH (if attempted) | `test_ylm_properties.py`, `test_review_core.py` (core real Ylm); `test_flapw_efg.py`, `test_flapw_dfpt.py` (efg gaunt/rotations). |
| D4 | **Radial-Poisson family — assess, likely distinct** | `flapw/efg.py:292 (lx_sphere_poisson), :307 (l2_sphere_poisson)`, `flapw/coulomb.py:214 (radial_poisson_to_R)`, `flapw/scf.py:371 (_weinert_potential), :894 (_weinert_multi)` | All solve a radial Poisson / multipole integral but for different L and boundary conditions (interior-only vs matched-at-R vs full Weinert). Probable shared inner radial-integral kernel, but the wrappers are legitimately different physics. Low value, defer. | MED | `test_flapw_weinert.py`, `test_flapw_efg.py`. |

---

## Coverage map (which dedup targets are guarded)

| Target | Fast unit test? | Verdict |
|--------|-----------------|---------|
| `pseudo.radial.sph_jl` ↔ `radial_torch.jl_t` | YES — `test_review_dedup.py` (1e-14 parity + l-range guard) | already deduped, safe |
| `flapw/*` complex-Ylm block (S3) | YES — `test_flapw_weinert.py`, `test_flapw_efg.py` | safe to consolidate |
| `functionals.py` PW92 (S2) | YES — `fxc_lda` FD-vs-`vxc_lda` self-check | safe to consolidate |
| `flapw/mixing.anderson_next` (D1) | **NO dedicated numeric test** (only `test_review_errors.py`) | UNCOVERED → asus SCF regression required |
| `flapw/lapw` / `coulomb` scipy Bessel (D2) | Indirect only (SCF integration; `test_flapw_weinert` for coulomb) | defer; coulomb case infeasible (l>4) |
| Ylm/Gaunt cross-package (D3) | YES per family (`test_ylm_properties`, `test_flapw_efg`) but merge is cross-basis | do not merge |

## Other complexity hotspots (locate-only, lower priority than S1)

- `flapw/scf.py:551 _lapw_multi_k` (~165 lines) — the single-iteration multi-k secular build; candidate to split the LO / non-spherical-augmentation branches.
- `postscf/kgeometry_nmr.py:648 induced_current_q` (~158 lines) and `:951 sigma_shielding` (~126 lines) — long but linear-response assembly with distinct physical stages; split by stage if touched.
- `flapw/scf.py:1112 _multi_setup` (~113) and `:1692 crystal_scf_multi` (~121) — orchestration; acceptable, lower priority.
