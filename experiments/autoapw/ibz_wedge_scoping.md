# IBZ wedge reduction of the NMR-shielding k-sum — scoping + measured go/no-go

Scope: decide whether reducing the shielding pipeline's k-sum from the full
unreduced BZ mesh to the irreducible wedge of the little group of q is worth
building, and if so, how. This is intelligence for a build decision — one
calibration measurement, no feature implementation.

The shielding pipeline (`src/gradwave/postscf/kgeometry_nmr.py`) currently
forces the full mesh: `_guard` raises unless `res.system.sym is None`
(i.e. `use_symmetry=False`), because the ±q Sternheimer construction folds
k+q across the whole mesh. Both routes sum over the full mesh:

- the production analytic route `sigma_shielding_dq` / `ShieldingDq`
  (exact ∂/∂q at q→0; lives on `origin/main`, PR #336), and
- the finite-q validation route `sigma_shielding` (commensurate ±q).

At 6³ that is 216 unreduced k-points per q-direction; the shielding tensor
needs ≥2 mesh axes, each solved at its own q-direction.

---

## 1. Headline: the little-group reduction factor for the ACTUAL q-directions

A q-shift breaks the crystal point group to the **little co-group of q**,
`G_q = {g : W⁻ᵀq ≡ q (mod reciprocal lattice)}`
(`symmetry.little_cogroup`). The shielding driver puts q along each sampled
reciprocal axis `b_i` (`q_hat = b[i]/|b[i]|`; `sigma_shielding_dq` line ~1215,
`sigma_shielding` line ~774), so the relevant little groups are those of the
reciprocal-axis directions.

Computed with the repo's own symmetry machinery (`find_spacegroup`,
`little_cogroup`, `star_of_q`, `little_group_ibz`) on the exact test configs —
script `experiments/autoapw/ibz_wedge_littlegroups.py`:

### Si — diamond, Fd-3m, point group Oh (48 rotations)

| quantity | q ‖ b1 | q ‖ b2 | q ‖ b3 |
|---|---|---|---|
| \|G_q\| (little co-group) | 6 | 6 | 6 |
| **asymptotic factor \|G\|/\|G_q\| = \|star\|** | **8×** | **8×** | **8×** |
| discrete-mesh IBZ reduction, 2³ | 2.0× (8→4) | 2.0× | 2.0× |
| discrete-mesh IBZ reduction, 4³ | 3.2× (64→20) | 3.2× | 3.2× |
| discrete-mesh IBZ reduction, 6³ | 3.9× (216→56) | 3.9× | 3.9× |

Every reciprocal axis is equivalent under Oh, so all three axes reduce
identically. Asymptotic factor 8×; realized 2–3.9× at 2³–6³.

### Rutile TiO2 — P4_2/mnm, point group D4h (16 rotations)

| quantity | q ‖ b1 (a-axis) | q ‖ b2 (a-axis) | q ‖ b3 (c-axis) |
|---|---|---|---|
| \|G_q\| (little co-group) | 4 | 4 | 8 |
| **asymptotic factor = \|star\|** | **4×** | **4×** | **2×** |
| discrete-mesh IBZ reduction, 2³ | 1.0× (8→8) | 1.0× | 1.3× (8→6) |
| discrete-mesh IBZ reduction, 4³ | 1.8× (64→36) | 1.8× | 2.7× (64→24) |
| discrete-mesh IBZ reduction, 6³ | 2.2× (216→96) | 2.2× | 3.6× (216→60) |

Note the tetragonal inversion vs Si: the c-axis has the *larger* little group
(8 ops, so smaller asymptotic star ×2) yet the *larger* discrete-mesh reduction
(3.6× at 6³), because the extra c-axis stabilizer operations act more
effectively on the finite mesh points. The a-axes give only 2.2× at 6³.

### Reading the two factors

- **Asymptotic (point-group) factor** = star size = the large-mesh limit:
  Si 8×, TiO2 4× (a) / 2× (c). This is the number the vet quoted as "the larger
  prize."
- **Discrete-mesh factor** is what you actually get at a usable mesh, and it is
  materially smaller because the high-symmetry mesh points (Γ, zone-boundary,
  axis-lying) sit on their own stabilizers and don't fold. At the meshes the
  campaign actually runs (4³–6³): **Si 3.2–3.9×, TiO2 1.8–3.6×.**

So the honest per-axis reduction sits in the **2–4× band at practical meshes**
(approaching 8×/4×/2× only asymptotically). That is above the vet's "~1–2× ⇒
not worth" floor but below its "4–8× ⇒ clearly worth" ceiling — a genuinely
marginal factor that measurement 2 (the reducible fraction) must tip.

### Time-reversal interacts, and the two routes differ

`_guard` (both branch and `origin/main`) blocks only `use_symmetry`
(`res.system.sym is not None`); it does **not** block ordinary time reversal.
Two consequences the factors above must be read against:

- **The finite-q route (`sigma_shielding`) requires the fully unfolded mesh.**
  `kpq_map` matches every k+q against a *stored* mesh point; a TR-folded mesh
  keeps only one of each ±k pair, so k+q folds to an absent point and the map
  raises. Its baseline is genuinely the full 64/216, and §1's factors
  (3.2–3.9× Si, 1.8–3.6× TiO2) apply directly.
- **The analytic route (`sigma_shielding_dq`) already tolerates TR folding.**
  It is purely per-k at q=0 (no `kpq_map`), so with `setup_system`'s default
  `time_reversal=True` it runs on the TR-folded mesh: the profile below ran at
  4³ on **nk=36** (= TR-fold of 64, no point group), not 64. TR alone gives
  ~1.8× for free. The little-group wedge for the analytic route is therefore
  *incremental on top of TR* and must **compose** with it — the little group of
  the axis direction q̂ combined with the ±q̂ TR line. That composition (and its
  correctness for the antisymmetric q-derivative summand) is exactly what the
  route-equivalence gate (§5d) settles empirically; it is not free from §1's
  numbers, which are computed against the point-group-only full mesh.

---

## 2. The reducible fraction of the analytic route's wall

`IBZ wedge reduction removes only the per-k work` (the eigh setup + the
Sternheimer/resolvent solves). It cannot touch the fixed tail (the Biot–Savart
G-space sum and the 3×3 least-squares tensor solve). Amdahl caps the speedup at
`1 / ((1−f) + f/r)` with `f` = reducible fraction, `r` = reduction factor.

Profiled one `sigma_shielding_dq` evaluation, Si 12 Ry, 4³ (nk=64), OMP=2 on
asus (script `experiments/autoapw/ibz_wedge_profile.py`), split into:

- PER-K SETUP — `ShieldingDq.__init__`: dense `eigh(H(k))` + velocity matrices
  at every mesh k. **Reducible** (needed only on the wedge reps).
- PER-K SOLVE — `branch_fields_axis`: conduction-resolvent applies per axis.
  **Reducible.**
- FIXED — `_biot_savart_sigma_cols_dq` (G-space Biot–Savart) + `lstsq`.
  **Not reducible.**

Measured (single pass, `min` of one; the per-k parts are seconds each so the
timer noise is <1%):

| phase | wall | fraction | reduces by |
|---|---|---|---|
| PER-K SETUP (`__init__`, eigh+velocity, nk=36) | 25.83 s | 8.11 % | union-of-wedges factor |
| PER-K SOLVE (`branch_fields_axis`, 3 axes) | 292.79 s | 91.89 % | per-axis wedge factor |
| FIXED (Biot–Savart + lstsq) | **0.015 s** | **0.0047 %** | — (untouched) |
| total (excl. SCF) | 318.63 s | 100 % | |

**The fixed tail is 15 milliseconds — five thousandths of one percent.** The
per-k Sternheimer/resolvent work is essentially the entire wall. `reducible
fraction = 1.0000` to four figures. (Ran on the default TR-folded mesh nk=36; the
fraction is independent of nk — both per-k parts scale with nk, the fixed part
does not — so it transfers to 64/216 unchanged.)

The one wrinkle: the two reducible parts reduce by **different** factors,
because the eigh setup is q-independent (one H(k) serves all axes) and so shrinks
only to the **union** of the per-axis wedges, while each axis's solve shrinks to
its **own** wedge:

| | union factor (setup) | per-axis factor (solve) |
|---|---|---|
| Si 4³ | 2.21× (64→union 29) | 3.20× (64→20) |
| Si 6³ | 2.45× (216→union 88) | 3.86× (216→56) |
| TiO2 4³ | 1.25× (64→union 51) | 1.78–2.67× |
| TiO2 6³ | 1.46× (216→union 148) | 2.25–3.60× |

Solve is 92 % of the wall and gets the good (per-axis) factor; setup is 8 % and
gets the weaker union factor.

---

## 3. The Amdahl-bounded honest ceiling

Because the fixed part is 15 ms, **the Amdahl tax is essentially zero: the net
speedup equals the k-point reduction factor.** The only thing that pulls it below
the per-axis factor is the 8 % setup fraction reducing by the weaker union
factor. Combining the measured fractions with the split factors
(`speedup = 1 / (0.081/r_setup + 0.919/r_solve)`):

| system / mesh | net analytic-route speedup ceiling |
|---|---|
| Si 4³ | **3.09×** |
| Si 6³ | **3.69×** |
| TiO2 4³ | **1.91×** |
| TiO2 6³ | **2.42×** |

These are ceilings **relative to the point-group-only full mesh** (the §1
baseline; matches the finite-q route exactly). The product
`reducible-fraction × little-group-factor` the task asked for is, plainly:
reducible fraction ≈ 1, so the ceiling is the reduction factor itself, degraded
only by the setup/solve factor split — landing at **~3.1–3.7× for Si and
~1.9–2.4× for rutile TiO2**.

**The TR bracket (analytic route only).** Because `sigma_shielding_dq` already
runs on the TR-folded mesh (nk 64→36 at 4³ ≈ 1.8× banked for free), part of
these ceilings may already be realized today. If the little-group wedge composes
cleanly with the existing TR fold, the ceilings above stand; if TR and the axis
little group substantially overlap (for Si, `inversion∘TR` fixes q̂, so TR adds
little the point group didn't already offer along an axis), the *incremental*
gain over today's TR-folded behavior could be as low as ~1.7–2× on Si. The
route-equivalence test (§5d) is what pins which end of this bracket is real; it
cannot be settled from the k-counts alone.

---

## 4. GO / NO-GO

**QUALIFIED GO for the analytic (production) route via the low-risk
output-tensor variant; NO-GO (defer) for the finite-q/screening route.**

The two decision numbers land in the middle of the vet's band, not the top:

- **Reduction factor: 2–4× at practical meshes** (Si 3.2–3.9×, TiO2 1.8–3.6×
  per axis), not the 8×/4×/2× asymptotic prize. Above the "~1–2× ⇒ skip" floor,
  below the "4–8× ⇒ obvious" ceiling.
- **Reducible fraction ≈ 1.000** (fixed tail 15 ms). So the factor converts to
  wall almost losslessly — **net ceiling ~3.1–3.7× (Si), ~1.9–2.4× (TiO2)**.

On the factor alone this is marginal. What tips it to GO is **build cost, not
factor**: the reduction is not "a major feature touching `_guard`" — the
k-reduction backbone is **already landed and verified** (`little_group_ibz`,
`star_of_q`, `QFieldSymmetrizer`, and the fully worked `chi0_q_reduced`
star-unfold that reproduces the full-mesh response to <1e-6). For the analytic
route specifically, the response field is a **q=0 vector field**, so it reuses
only *already-shipped q=0 primitives* (`VectorFieldSymmetrizer` /
`symmetrize_atom_tensor`, the dielectric IBZ recipe) — no new symmetry class,
no umklapp, no projective-rep trap. A ~2–4 day, low-risk change buying ~3× on
Si (the campaign's main test system) clears the bar.

It is a **NO-GO for the finite-q/screening route's reduction**: that path needs a
genuinely new q≠0 vector field symmetrizer (~1–2 weeks, medium risk) to speed up
a **validation** route, at the smaller rutile factor. Not worth it until the
analytic reduction is proven and a real need appears.

**Two gates before committing even the analytic build:**

1. **The TR bracket must be resolved** (§3). If the analytic route's default
   TR fold already banks most of the win, the *incremental* gain could be ~1.7×
   and the GO flips to NO-GO. This is a one-script check: run
   `sigma_shielding_dq` on the TR-folded mesh (nk=36) vs the unfolded mesh
   (nk=64, `time_reversal=False`) and compare both σ (correctness) and wall.
2. **Confirm `sigma_shielding_dq` is the route the campaign actually runs.** If
   the campaign has moved to the finite-q route for some systems, re-weight
   toward the NO-GO side.

The coordinator's "shared-with-phonon-DFPT value" point is **real but narrower
than stated**: the k-reduction plumbing phonons need (`chi0_q_reduced`) is
*already built*. The only genuinely-new shared artifact is the q≠0 vector field
symmetrizer — and that belongs to the NO-GO finite-q route, not the GO analytic
one. So the shared-value argument does not itself lift the analytic build; the
analytic build stands on its own cheap ~3× or not at all.

---

## 5. If GO — design sketch (build on the already-landed star-unfold)

**Prior art is much further along than "a major feature touching `_guard`."**
The little-group star-unfold (`docs/design/little-group-star-unfold.md`) is
landed and verified through Phase 3 for the **scalar density** response:

- `symmetry.little_cogroup`, `star_of_q`, `little_group_ibz` — group theory,
  tested (`tests/unit/test_little_group.py`).
- `symmetry.QFieldSymmetrizer` — q-dependent **scalar** field fold (umklapp g0,
  q-shifted band-limit `{|q+G|≤G_cut}`, non-symmorphic translation phase;
  fail-fast on projective small reps).
- `postscf.dfpt_q.chi0_q_reduced` — sums `chi0_q` over the little-group IBZ of q
  (orbit reps × star multiplicities) and folds with `QFieldSymmetrizer`,
  reproducing the full-mesh `chi0_q` to <1e-6. The k↔k+q bookkeeping
  (`kpq_map`, umklapp G0 box phase) is the SAME machinery the shielding pipeline
  already imports (`_g0_phase`, `_reindex_bk`, `kpq_map`).

So the k-reduction backbone exists and is verified. What shielding needs on top
is narrower than a from-scratch build, and it differs by route:

### 5a. The analytic route (`sigma_shielding_dq`) — the easy, safe reduction

The analytic route's response fields (`S⁰`, `∂S/∂s`) are **q→0 (uniform-B)
fields**: plain G-vectors, no e^{iqr} modulation, no umklapp, and q=0 is Γ so
there is **no projective-small-rep trap**. Its k-sum reduces under the little
group of the q̂ *direction* (§1's numbers) using machinery that **already
exists at q=0**:

- Reduce the mesh with `little_group_ibz(mesh, q̂, sg)` per axis; run
  `ShieldingDq.__init__` eigh and `branch_fields_axis` only on the wedge reps,
  weighted by star multiplicity.
- Reconstruct the full-BZ-summed current field by folding with the **existing**
  `symmetry.VectorFieldSymmetrizer` (polar-vector fold, Cartesian
  `S = AᵀWA⁻ᵀ`, `symmetry.py:769`) — the induced current `j` is a **polar**
  vector, so it uses the plain rotation `S` (no determinant). The external-field
  columns `q̂×e_pol` are axial, but they are *inputs* to the least-squares fit,
  not summed over k.
- Equivalently (the dielectric recipe, `postscf/dielectric.py`): accumulate the
  per-site σ contribution over the wedge and symmetrize the **output** rank-2
  tensor with `symmetry.symmetrize_atom_tensor` (`S·T·Sᵀ` + `atom_map`) — no
  field symmetrizer at all. This is the lower-risk variant and is how the
  dielectric/Born-charge IBZ sum is already done.

This route reuses **only already-landed, already-verified q=0 primitives**. The
one genuinely new correctness item is the polar-vs-axial choice for the summed
field (see risks).

### 5b. The finite-q route (`sigma_shielding`, screening path) — needs new work

The finite-q route sums genuine wavevector-q current fields (e^{iqr}-modulated,
complex, with umklapp). Reducing it needs a **q-dependent VECTOR field
symmetrizer** — the combination that neither existing class is:
`QFieldSymmetrizer` (q-umklapp + q-shifted mask + translation phase) × the
Cartesian component mixing of `VectorFieldSymmetrizer`. Both halves are worked
examples in `symmetry.py`; the merge is ~1 new class. The screened path already
routes density through `chi0_q`, which has the `k_indices`/`k_weights`/
`symmetrizer` hooks — but the *current* assembly (`induced_current_q`,
`velocity_perturbation_q`, `_NLPairVelocity`) runs its own per-k loop over the
full mesh and would need the same hooks plus the vector symmetrizer.

### 5c. What `_guard` must become

Today: `raise` unless `res.system.sym is None`. New contract: **accept** a
symmetry-reduced result together with the q-construction's reduction data —
the little group `G_q`, the wedge k-indices, and the per-rep star weights — and
plumb them into `velocity_perturbation_q` / `ShieldingDq` (which currently read
`system.spheres`, `system.kweights` as the full mesh). Keep the full-mesh path
as the default and the validation oracle; the reduced path is opt-in, mirroring
`chi0_q`'s `k_indices`/`k_weights`/`symmetrizer` opt-in and
`apply_chi0`'s `assume_totally_symmetric`.

### 5d. Validation strategy

- **Route-equivalence (the primary gate):** the wedge σ must equal the current
  full-mesh σ to ~`cg_tol` — a dense twin. `sigma_shielding_dq` full-mesh is the
  oracle; assert `‖σ_wedge − σ_full‖ < 1e-6` on Si (4³) and TiO2 (4³). This
  single number catches a wrong determinant, a wrong star weight, or a wrong
  umklapp immediately (as it did for `chi0_q_reduced`).
- **Existing invariants, unchanged:** gauge/longitudinal null
  (`test_gauge_longitudinal_null`), q=0 TR null + inert screening
  (`test_q0_tr_null_and_inert_screening`), cubic isotropy of σ for Si
  (`test_sigma_cubic_isotropy`), and site equivalence (the two Si atoms /
  the symmetry-equivalent O sites in rutile must come out equal).
- **Orbit–stabilizer self-check:** `|star|·|G_q| == |G|` per axis (already in
  `test_little_group.py`); assert the wedge weights sum to 1.

### 5e. Effort estimate

- **Analytic route only (§5a), via output-tensor symmetrization (§5a variant 2):**
  small. `little_group_ibz` + star weights already exist; wire wedge k-indices
  through `ShieldingDq`/`branch_fields_axis`, symmetrize the output σ with
  `symmetrize_atom_tensor`, add the route-equivalence test. Estimate **2–4
  days**, low risk, no new symmetry class.
- **Finite-q + screening route (§5b):** add the q-dependent vector symmetrizer
  and the `k_indices`/weights hooks to the current assembly. Estimate **1–2
  weeks**, medium risk (the projective-small-rep fallback and the q-umklapp
  vector fold).

### 5f. Top-3 risks

1. **Polar-vs-axial (pseudovector) bookkeeping.** The summed field must carry
   the correct Cartesian transformation: the microscopic current `j` is a
   **polar** vector (plain `S`), but the physical shielding relates axial
   `B_ind`↔`B_ext`, and the vet flagged the pseudovector `det(S)·S` case. Get
   the determinant wrong and it is silently wrong for improper operations
   (inversion, mirrors, S4 — all present in Oh and D4h). *Mitigation:* the
   route-equivalence gate on a cell WITH improper ops (Si has inversion) fails
   loudly if the determinant is wrong; the output-tensor variant (§5a v2) sinks
   the choice into `symmetrize_atom_tensor`, whose `S·T·Sᵀ` convention is
   already proven for Born charges.
2. **Degenerate-star / high-symmetry mesh points.** k-points on the little-group
   stabilizers (Γ, axis-lying, zone-boundary) belong to short orbits; the
   star-weight and self-mapping bookkeeping must handle multiplicity <|G_q|
   exactly (this is why the discrete factor < asymptotic). *Mitigation:*
   `_orbit_reduce` / `_little_group_orbits` already handle it for `chi0_q_reduced`;
   reuse verbatim, and assert weight-sum = 1.
3. **Projective small reps at zone-boundary q (finite-q route only).**
   `QFieldSymmetrizer` fails fast at non-symmorphic zone-boundary q (Si
   [1/4,1/4,0]); the shielding q are interior (q=1/n along an axis, n≥3), so
   this should not bite, but the finite-q route must inherit the same fail-fast
   and fall back to the full mesh at any q that trips it.

---

## Appendix — scripts & provenance

- `experiments/autoapw/ibz_wedge_littlegroups.py` — computes §1 (spglib via
  `find_spacegroup`; `little_cogroup` / `star_of_q` / `little_group_ibz`).
- `experiments/autoapw/ibz_wedge_profile.py` — §2 profile (Si 12 Ry 4³, asus).
- Little-group numbers reproduce the orbit–stabilizer identity
  `|star|·|G_q| = |G|` exactly for every axis (printed by the script).
- Cells: Si `si_fcc()` (a=5.43 Å, `tests/helpers.py`); rutile TiO2 from
  `experiments/autoapw/_common.py` (a=8.68083, c=5.59096 Bohr, u=0.3048).
