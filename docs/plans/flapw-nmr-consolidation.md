# FLAPW / NMR consolidation plan

Status: **planning** (written 2026-08-22, near the natural resting point of the AutoAPW campaign).

The campaign shipped an all-electron FLAPW code (`gradwave.flapw`), trustworthy EFG,
and a GIPAW shielding stack (`gradwave.postscf.gipaw`, `kgeometry_nmr`) across ~26 PRs.
The physics is validated per-material against Elk 11.0.2. What is **not** done is the
consolidation that turns a proven-per-material research capability into something a
non-author can drive. This document is the forward plan; it supersedes the stale
"remaining" section of `experiments/autoapw/STATUS.md` (which lists since-completed
work as open TODOs).

This is a plan, not a spec — each bucket becomes its own PR (or small stack) when picked up.

---

## Decisions locked in this session

- **Retire the finite-q shielding route.** `sigma_shielding` / `induced_current_q`
  (finite-q, ~217 s per eval on NC Si 2×2×2) is superseded by the analytic
  `sigma_shielding_dq` (∂/∂q, ~4.2× faster, the production path). Rather than teach the
  slow finite-q route about USPP (the PR-A/PR-B split from `abs_sigma_scoping.md`), we
  **remove finite-q and generalize the analytic route to USPP/PAW** instead. See Bucket 3.
  - Consequence: absolute-σ PR-A (finite-q + `uspp=` plumbing) is *not* the endpoint —
    it is at most a stepping-stone / reduction-gate reference. The real target is
    analytic-USPP. Revisit whether PR-A is even worth landing once analytic-USPP is scoped.

- **The universal ion-type basis construction is buildable now** (see Bucket 2). The
  per-material hand-tuning of `los=` / `el_override=` is a missing *policy*, not a missing
  *capability*: `flapw.atom.atomic_scf` already returns the per-element KS eigenvalues the
  policy needs.

---

## Bucket 1 — API / driver unification (do first: it makes everything else usable)

Today `gradwave.flapw` and the NMR postscf calls run entirely parallel to the PW stack,
with no `Input` schema, no `api.run` dispatch, and no calculator/CLI surface. You drive
`crystal_scf_multi` and the EFG/shielding functions by hand. This is the biggest *usability*
gap and the reason the accuracy gains are author-only.

1. **`Input` schema for FLAPW + NMR.** Extend `inputs/models.py` with an FLAPW system
   block (muffin-tin radii, lmax, LO/basis spec — see Bucket 2) and an NMR task block
   (EFG | shielding, nuclei selection, k-mesh). `inputs/` stays a leaf (no physics import).
2. **`api` drivers.** Add `api.run_flapw` (dispatches `crystal_scf_multi`) and `api.run_nmr`
   (EFG via `efg_tensor_full`; shielding via the analytic route from Bucket 3), following
   the split-by-task `api/` layout. Wire into `api.dispatch` so `api.run(Input)` routes an
   FLAPW/NMR input the same as scf/relax/eos.
3. **Summary / serialize / report.** EFG (V_zz, η, C_Q per site) and shielding (σ_iso,
   anisotropy, per-nucleus) get first-class `build_summary` entries + `io.output` rendering,
   instead of raw dict returns.
4. **Calculator / CLI surface.** Optional: expose EFG/shielding through `calculator.GradWave`
   and a `gradwave nmr` CLI verb once the api layer is stable.

Acceptance: a committed example `Input` file runs rutile-TiO₂ EFG end-to-end through
`api.run` and reproduces the campaign's hand-driven numbers.

---

## Bucket 2 — Universal ion-type basis construction (the "why is there no universal
construction?" answer)

**Why there isn't one yet.** The campaign built the LAPW basis *primitives* and validated
them per-material with hand-tuned specs:
- confined semicore LO (`_build_lo(confine=True)`, φ(R)=φ'(R)=0),
- unconfined HELO (`_build_lo(confine=False)`, high trial energy — the anion under-capture fix, #353),
- (pending) a contracted / lower-energy lever for the light-cation over-capture (task #48).

What is missing is the **selection policy** that maps an element to its basis. Every real
all-electron code (Elk, exciting, WIEN2k) derives this automatically from the atomic solve:
semicore states within a window below the valence get a confined LO; a HELO is added per l
at a high trial energy; the linearization energy E_l is set at the atomic eigenvalue / band
center. gradwave has all the ingredients — **`flapw.atom.atomic_scf(symbol, …)` already
returns the per-element converged KS eigenvalues** (`{"2s": −35.67, "2p": −13.20, …}`) — but
the glue mapping those eigenvalues to `{E_l, LO specs}` was never written, because proving
the physics one material at a time didn't need it.

**The construction.** A single policy function, e.g.

```
default_basis_spec(symbol, r_mt, target="standard") -> {los, el_override}
```

driven by `atomic_scf`:
- set E_l per l from the highest occupied atomic eigenvalue of that l (band-center refinement optional);
- for each semicore state within ~[E_valence − Δ, E_valence], emit a confined LO;
- emit a per-l HELO at a high trial energy for completeness where the valence is diffuse.

**Ion type falls out for free.** The atomic solve *is* the ion-type discriminator: an anion's
diffuse, shallow valence eigenvalue → the HELO branch (fixes under-capture); a light cation's
deep, contracted eigenvalue (small ⟨r⟩) → the contracted-lever branch (fixes over-capture,
task #48); a transition metal's semicore → the confined-LO branch. The same policy, reading
the same eigenvalues, picks the right lever per element without per-material tuning.

This bucket **depends on task #48 landing** (the light-cation lever is one of the branches the
policy dispatches to) and **feeds Bucket 1** (the `Input` basis block defaults to
`default_basis_spec`; hand specs become an override, not the norm).

Acceptance: the multi-material EFG validation set (TiO₂, corundum, MgF₂, + the light-cation
cases) runs with **zero hand-tuned `los=` specs**, matching the hand-tuned numbers.

---

## Bucket 3 — Shielding-route consolidation (analytic-USPP; retire finite-q)

Per the decision above:
1. **Scope analytic-USPP.** `sigma_shielding_dq` / `ShieldingDq` are deep-NC-locked: dense
   `eigh(H)`, `_resolvent_apply` with S=I, `_dense_velocity_matrices = ∂H/∂k` only,
   `_d2h_mats = ∂²H` only. The shipped S-metric primitives (#349/#351) are the *matrix-free
   batched-CG* path, so generalizing the *dense-eigh* analytic path is genuine new work — it
   needs its own measured scoping (mirror `abs_sigma_scoping.md`): can the dense resolvent take
   the generalized (H−εS) form and the velocity take v = ∂H/∂k − ε∂S/∂k using
   `OverlapVelocity` / `_overlap_kbprojectors`, or does the ∂²/∂q² term need a new
   augmentation-second-derivative? Answer before building.
2. **Reuse the augmentation physics.** `gipaw.para_aug_operator` (angular L_R × radial AE−PS
   ∫[…]/r³, su(2)-validated) and the diamagnetic/core terms carry over unchanged — this is the
   part the finite-q vs analytic choice does *not* touch.
3. **Validation.** NC-limit reduction gate (analytic-USPP on an NC pseudo == plain NC analytic
   to machine precision; the #349 gate pattern), then diamond ²⁹Si ≈ 400 ppm and MgO ¹⁷O ≈
   215 ppm as the quantitative checks.
4. **Remove finite-q.** Once analytic-USPP validates, delete `sigma_shielding` /
   `induced_current_q` and their NC-locked assembly, keeping only the analytic path. Update the
   `gipaw.py` module docstring (currently stale: it claims the S-metric Sternheimer "is not
   built" — #349/#351 shipped it).

Acceptance: one shielding entry point (analytic), USPP+NC, passing the NC-limit gate and the
Si/MgO references; no finite-q code remains.

---

## Bucket 4 — Deduplication audit + housekeeping (last; low-risk, high-tidiness)

**Radial numerics sharing (narrow — the ODE solvers do NOT merge).** `flapw/radial.py`
(Numerov / tridiagonal radial *Schrödinger ODE*) has no PW counterpart; the PW side's
`pseudo/radial.py` is a *transform/quadrature* layer (`sbt`, `simpson`, `sph_jl`) — different
math. The genuine dedup is the shared small numerics:
- `sph_jl` (spherical Bessel, l≤4) — flapw's boundary match needs it too; share one impl.
- composite-Simpson quadrature — flapw integrates on the log mesh with a crude `Σ u²·(r·dx)`;
  it could borrow `pseudo.radial.simpson` for higher-order accuracy on the same mesh.
- log-mesh (`flapw.radial.log_mesh`) vs UPF-mesh (rab weights) — one mesh abstraction.

**Housekeeping.**
- Add the `gradwave.flapw` subpackage to the typed-file list (`[[tool.ty.overrides]]`) once
  it's clean; put `@override` on the flapw `Smearing`/functional subclass methods.
- Tier-mark the heavier FLAPW/NMR tests (Elk-comparison, shielding) so the fast gate stays fast.
- Refresh `experiments/autoapw/STATUS.md` (stale "remaining" section) and point it at this doc.

---

## Not consolidation — new chapters (deferred, need a green-light)

These are new capability, not cleanup, and are out of scope for this plan:
- **Differentiable NMR-crystallography refinement** — cashing in the differentiable-EFG moat
  (FLAPW-DFPT, #346); now more accurate post-#353. Buildable.
- **PW paramagnetic NMR / Fermi-contact hyperfine** (battery / LiFePO₄ destination) — the
  S-metric bridge (#349/#351) is its foundation. ~weeks.
- **Absolute-σ PR-B pieces** that survive the finite-q retirement (X_β cross-density assembly,
  prefactor calibration) fold into Bucket 3's analytic path rather than the finite-q one.

---

## Recommended sequence

1. **Bucket 1 (unification API)** — unblocks everyone; makes the rest testable through `api.run`.
2. **Task #48 (light-cation lever)** — in flight; it is a prerequisite branch for Bucket 2.
3. **Bucket 2 (universal basis policy)** — depends on #48; removes per-material hand-tuning.
4. **Bucket 3 (analytic-USPP shielding, retire finite-q)** — needs its own scoping first.
5. **Bucket 4 (dedup + housekeeping)** — low-risk, do alongside or last.

Buckets 1 and 3-scoping can proceed in parallel (disjoint files: `inputs/`+`api/` vs
`postscf/`). Bucket 2 gates on #48.
