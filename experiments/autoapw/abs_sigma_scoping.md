# Absolute chemical-shielding (σ) through the USPP/PAW backend — scoping

**Verdict: GO** via the finite-q route (`sigma_shielding` / `induced_current_q`).
The hard #349/#351 work (the S-metric magnetic Sternheimer) is already shipped and
validated on real PAW Si; the remaining lift is wiring + one small new operator-input
assembly + a prefactor calibration. The "gauge / covariant-position" term — feared as
the primary risk — is **not** a research problem: it is already factored into the
su(2)-validated on-site paramagnetic-augmentation operator in `gipaw.py`; only its input
cross density is unwired.

Branch `abs-sigma-scoping` off `origin/main` @ 6992795 (asus). Measured with
`OMP_NUM_THREADS=2`.

---

## 0. Measured facts (de-risk probe)

`probe_abs_sigma.py` on asus, Si, ecut = 12 Ry:

| quantity | value |
|---|---|
| NC bare σ_iso per site, Si (2,2,2) | **34.04 / 59.45 ppm** (smooth valence only) |
| finite-q `sigma_shielding` wall time (NC, 2×2×2) | **217.5 s** (12 solve-sets × 3 μ) |
| PAW smooth continuity `rel_kin_only` | 6.40e-02 (KB term matters) |
| PAW smooth continuity **`rel_full`** | **4.08e-04** (closes: q·mean[j_kin+j_nl] = mean s) |
| PAW `rho_ij_atoms` (becsum) | `[18×18, 18×18]` present |
| PAW `becps` = ⟨p̃\|ψ̃⁰⟩ | `[8×36, 8×36]` present |
| `tests/unit/test_smetric_response.py` on HEAD | **8 passed** |

The NC bare number (tens of ppm) is the *smooth valence* shielding only; the absolute
σ (Si bulk ≈ 300–400 ppm) is dominated by σ_core (≈ 838 ppm) minus the augmentation
corrections — which is exactly why absolute σ needs the on-site pieces, not just the
bare route.

---

## 1. Assembly gap — what is NC-locked

Two shielding routes exist. They lock at very different depths.

### 1a. Finite-q route — `sigma_shielding` → `induced_current_q` (the target)

- **`velocity_perturbation_q` (kgeometry_nmr.py:301) — NOT a gap.** It already carries
  the full S-metric backend: `uspp=USPPResponseCtx`, `s_apply=`, `dsvel=` (generalized
  velocity v = ∂H/∂k − ε∂S/∂k), routed through `_response.cg_sternheimer(s_apply=,
  s_occ=)`. Tested end-to-end on PAW Si (`test_uspp_velocity_perturbation_q_runs`,
  `test_uspp_smooth_continuity_closure`). The δu solve is DONE.

- **`sigma_shielding` (kgeometry_nmr.py:951) — driver gap.** Calls
  `velocity_perturbation_q(res, …)` (L1040) and `induced_current_q(…)` (L1045) with
  **no `uspp` context**; builds `VelocityApply(system)` (L1035) and
  `dense_sternheimer_for(res)` (L1036), both NC-only. Needs a `uspp=ctx` kwarg that
  forwards the ctx and swaps in `ctx.v_h` / `dense=None`.

- **`induced_current_q` (kgeometry_nmr.py:648) — the substantive NC-locks:**
  - L697 `bk = system.batch; assert bk is not None` — `USPPSystem.batch` is `None`.
  - Kinetic current (L775–786) and `_drho_q` use `bk.kpg` / `sol.bk_kq.kpg` — purely
    **kinematic, reusable** once `bk` comes from `ctx.bk`.
  - L787 `j_dia = 2κ·e_pol·res.rho` — for USPP this is the **smooth** valence density
    (the AE−PS diamagnetic correction is the on-site σ_dia_aug, already built). Reuse
    with the smooth ρ; correct by construction.
  - L789 `_NLPairVelocity(system, sol)` reads `system.batch.dij_full` (L526) — the
    **bare** D. For USPP the KB nonlocal current must carry the **screened** D
    (`ctx.dscr`) and the PAW β. This exact pattern is already shipped in
    `uspp_smooth_continuity` (L1223–1258: `_overlap_kbprojectors` β + `ctx.dscr`).
  - `screen=True` branch (L719–771) uses `dfpt_q.chi0_q` / `_k_hxc_q` — NC only. **Not
    needed:** the linear-in-B density response is TR-odd, so K_Hxc screening is inert at
    q→0 (documented + `test_q0_tr_null_and_inert_screening`); leave `screen=False`.

- **Biot–Savart (`_biot_savart_sigma_cols` L913, `_antisym_field_g` L901) — reusable.**
  Pure G-space kinematics + nuclear site phases e^{i(q+G)·r_s}. S-independent.

### 1b. Analytic ∂/∂q route — `sigma_shielding_dq` / `ShieldingDq` (production, but DEEP-locked)

Entirely NC-locked, at a level the shipped S-metric primitives do **not** cover:

- `BlochHK.from_scf` + dense `torch.linalg.eigh(H)` (`_eigh_and_dh`, L1373) — standard
  eigenproblem; needs the **generalized** H u = ε S u.
- `_resolvent_apply` (L1268) — G(ε)P_c with plain ⟨u_m|x⟩ (S = I); needs an S-metric
  resolvent + S-orthogonality.
- `_dense_velocity_matrices` (L806) — ∂H/∂k only, no −ε∂S/∂k.
- `_d2h_mats` (L1279) — second derivatives of H only; would need ∂²S.

The shipped S-metric primitives are the **matrix-free batched CG** path
(`cg_sternheimer`/`_cg_smetric`, `OverlapVelocity`), not the **dense eigh** path the
analytic route is built on. Generalizing `ShieldingDq` is a genuine second build (a
dense generalized eigensolver + S-metric resolvent + ∂S/∂k, ∂²S/∂k²). **Recommendation:
leave `sigma_shielding_dq` NC-only; deliver the first absolute USPP σ through the
finite-q route.** (The finite-q route's 217 s cost is the price; production speed can
come later once the physics is pinned.)

### 1c. On-site augmentation (`gipaw.py`)

- σ_core (`core_lamb_shielding`) — DONE.
- σ_dia_aug (`dia_aug_tensor` / `ground_state_shielding`) — DONE; input `rho_ij_atoms`
  present on the PAW result.
- σ_para_aug (`para_aug_operator` / `para_aug_tensor`) — **operator DONE + su(2)-validated**;
  **missing:** its input cross density X_β and prefactor calibration (see §3).

**Conclusion:** the ±q antisymmetric uniform-B assembly is NOT the only gap — but the
gaps that remain are (i) driver wiring, (ii) a screened-D nonlocal-current variant whose
pattern is already shipped, and (iii) the σ_para_aug cross-density input. None is new
*physics*.

---

## 2. Reuse vs new — coverage of the locked call-sites

| locked piece (finite-q route) | shipped S-metric analogue? | new code? |
|---|---|---|
| Sternheimer δu solve (velocity + resolvent + projector) | **yes** — `velocity_perturbation_q(uspp=ctx)` (#349/#351), tested on PAW | none — invoke it |
| generalized velocity v = ∂H/∂k − ε∂S/∂k | **yes** — `_USPPHVelocity` + `OverlapVelocity`/`_overlap_kbprojectors` | none |
| KB nonlocal current with screened D | **pattern shipped** — `uspp_smooth_continuity` (dscr + `_overlap_kbprojectors`) | refactor into a `_NLPairVelocity` variant (no new physics) |
| kinetic current + `_drho_q` | kinematic (`ctx.bk`) | none — repoint `bk` |
| diamagnetic smooth term | reuse smooth ρ | none |
| Biot–Savart → B_ind | S-independent | none |
| on-site σ_dia_aug | `dia_aug_tensor` (input present) | none |
| on-site σ_para_aug operator | `para_aug_tensor` (built, validated) | **input** X_β assembly (small) + prefactor |

**Roughly 6 of the ~8 locked call-sites in the finite-q path are covered outright by
shipped S-metric primitives.** Genuinely new code: (a) the `uspp=` kwarg plumbing through
`induced_current_q`/`sigma_shielding`; (b) a screened-D `_NLPairVelocity` (lift of an
existing pattern); (c) the X_β cross-density assembly + para_aug wiring; (d) the
prefactor/field-normalization calibration.

---

## 3. Gauge / covariant-position — the key risk, downgraded

The gauge-covariant position (r covariant through the USPP/PAW augmentation) enters in
two places, both already accounted:

1. **Smooth (interstitial) part.** The covariant velocity v = ∂H/∂k − ε∂S/∂k carries
   the [H, r] content with the ∂S/∂k covariance; the e^{iqr} "position" phase enters the
   finite-q route only through the Miller-diagonal sphere transfer T (kinematic, exact).
   Both are shipped (`OverlapVelocity`, `velocity_perturbation_q(uspp=)`), and the
   **smooth continuity closure `rel_full = 4.08e-04`** measured here is the direct proof
   that the smooth current + its divergence are self-consistent — the covariant smooth
   half is complete on its own.

2. **On-site augmentation part.** The augmentation current from the covariant position is
   the AE−PS orbital-current operator L_R/r³, and its OPERATOR is already built and
   su(2)-validated: `gipaw.para_aug_operator` (angular L_α = ⟨Y_I|L_α|Y_J⟩ ×
   radial ∫[(rφ)(rφ) − (rφ̃)(rφ̃)]/r³, with the closed-shell Lamb null realized exactly).
   No new position operator must be derived.

The **only** genuinely open item is feeding σ_para_aug its input and fixing one scale:

- **X_β cross density** X_β[a,b] = Σ_o ⟨ψ̃⁰_o|p̃_a⟩⟨p̃_b|ψ̃^(1),β_o⟩. The ground factor
  ⟨ψ̃⁰|p̃⟩ is the shipped `becps` (measured present, [8×36]); the response factor
  ⟨p̃|δu⟩ is a projection of the now-available S-metric δu onto the PAW β̃. This is a
  short einsum, not a derivation.
- **Prefactor / field-normalization** of σ_para_aug relative to the bare Biot–Savart σ
  (flagged honestly in the `gipaw.para_aug_tensor` docstring as "fixed by the
  PAW-consistent magnetic response … not yet built"). This is the one number to nail.

**Verdict:** the covariant-position term is a **wiring + calibration** task, not research.
Risk is downgraded from "may be a research problem" to "one prefactor to pin, de-risked
by the NC-limit gate and the already-passing continuity closure."

---

## 4. Cheapest first validation

1. **NC-limit reduction (no external reference needed) — the primary gate.** Build a
   `USPPResponseCtx` from a **norm-conserving** pseudo (augmentation q = 0 ⇒ S = I,
   dscr = dij, no on-site term) and confirm the `uspp`-routed `sigma_shielding`
   reproduces the plain NC `sigma_shielding` **to machine precision**. This is exactly
   the gate that validated #349 (`test_nc_limit_reduction_velocity_perturbation_q`,
   passing). Reference in hand from this probe: NC Si (2,2,2) σ_iso = 34.04 / 59.45 ppm.
2. **Smooth continuity closure on PAW** — already passing at `rel_full = 4.08e-04`;
   re-assert per new assembly.
3. **Smallest real quantitative check.** Si PAW (`Si.pbe-n-kjpaw_psl.1.0.0.UPF`,
   ecut 12 Ry) on a **2×2×2** mesh (≥2 axes required; the shipped `si_paw` fixture is
   2×1×1 and insufficient). The probe's PAW SCF converged at 2×1×1; 2×2×2 (8 k) is
   comparably cheap. Full σ = σ_bare + σ_core + σ_dia_aug + σ_para_aug vs bulk-Si ²⁹Si
   (≈ 300–400 ppm absolute). Second material: MgO ¹⁷O (≈ 215 ppm) with the O PAW pseudo.
   Both runnable on asus (single small SCF + one shielding assembly each).

---

## 5. Effort estimate — stepwise

Sizes: **S** ≈ hours, **M** ≈ a day, within one focused agent each.

| # | step | size |
|---|---|---|
| 1 | `uspp=` kwarg through `induced_current_q` + `sigma_shielding` (use `ctx.bk`, `ctx.v_h`, smooth ρ, `dense=None`) | S |
| 2 | screened-D `_NLPairVelocity` variant (lift the `uspp_smooth_continuity` dscr + `_overlap_kbprojectors` pattern) | S |
| 3 | NC-limit reduction gate: `sigma_shielding(uspp=ctx-from-NC)` == NC `sigma_shielding` to machine precision | S (test) |
| — | **PR-A: absolute *smooth* USPP σ + NC-limit gate — clean incremental extension** | — |
| 4 | X_β cross-density assembly (`becps` ⊗ ⟨β̃\|δu⟩) → `para_aug_tensor`; full-σ assembler σ_bare + core + dia_aug + para_aug | M |
| 5 | prefactor / field-normalization calibration of para_aug vs the Biot–Savart smooth term (the one open unknown; de-risked by #3 + continuity) | M |
| 6 | quantitative validation: Si PAW 2×2×2 σ_iso vs bulk Si; MgO ¹⁷O vs 215 ppm | M (compute) |
| — | **PR-B: on-site paramagnetic augmentation wired + real-material validation** | — |

**Measured GO.** The finite-q route is a clean incremental extension of the #349/#351
S-metric foundation: the δu solve, the generalized velocity, the screened-D nonlocal
current pattern, and the covariant-position on-site operator are all already shipped and
validated. The gauge/covariant-position term does **not** make this a research problem —
it is factored into the su(2)-validated `para_aug_operator`, awaiting only its input
cross density and a single prefactor. The analytic `sigma_shielding_dq` production route
stays NC-only for now (its dense-eigh generalization is a separate, larger build).

Housekeeping note: the `gipaw.py` module docstring (§"Status of the end-to-end
paramagnetic number") is **stale** — it states the S-metric Sternheimer "is not built,"
which #349/#351 have since shipped. Update it when wiring PR-B.
