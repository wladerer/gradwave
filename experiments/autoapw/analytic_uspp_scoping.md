# Analytic chemical-shielding (σ) generalized to USPP/PAW — scoping

**Verdict: GO — a clean incremental extension of the #349/#351 S-metric foundation.**
The one term feared as a research blocker — the second q-derivative ∂²/∂q² of the
generalized (H−εS), i.e. the augmentation-second-derivative ∂²S/∂k² — is **measured to
be autograd-generic**: it is produced by the *same nested-`jvp`* the analytic route
already uses for ∂²H, through a ~15-line traceable dense `S(k)` assembled from the
**shipped** `_overlap_kbprojectors`. No new hand-derived augmentation-second-derivative
is needed. The remaining work is a second dense-eigh path (a generalized eigensolver,
an S-metric resolvent, a USPP dense H/S context) whose every physics ingredient is
already shipped and FD-validated. The single genuine open unknown is the σ_para_aug
prefactor calibration — shared with the finite-q route, a calibration not a derivation.

Recommendation on the in-flight finite-q PR-A: **do not land it as a finite-q shielding
product** (we are deleting finite-q); go straight to analytic-USPP. PR-A's only durable
pieces (the frozen-veff + screened-D ground-state ctx, the NC-limit gate pattern) are
already in-tree via #349/#351, and the reduction-gate reference is the plain-NC analytic
σ measured below — not a finite-q number.

Branch `analytic-uspp-scoping` off `origin/main` @ 6992795 (asus). `OMP_NUM_THREADS=2`.
Probe: `experiments/autoapw/probe_analytic_uspp.py`.

---

## 0. Measured facts (de-risk probe)

| quantity | value |
|---|---|
| **NC analytic σ_iso, Si (2,2,2) 6 Ry** (reduction-gate reference) | **20.817879 / 20.819837 ppm** per site |
| analytic σ max off-diagonal (isotropy) | 2.8e-11 ppm |
| **`_d2h_mats` (nested jvp) vs central-FD of ∂H/∂k**, β-nonlocal active | rel **2.5–3.4e-10** (μ=0,1,2) |
| **∂²S/∂k² via nested jvp through traceable `S(k)`** (`_overlap_kbprojectors`, PAW Si) | finite (max ~3–5e-3), matches FD **8e-10–1.2e-9** |
| PAW overlap projectors on the sphere | nproj=36, npw=181, q_int present |

The middle two rows are the crux. `_d2h_mats` already differentiates the KB-nonlocal
term `p.mᵀ (D p*)` twice (dij≠0, matched to FD at 1e-10). The USPP overlap
`S = I + p.mᵀ (q_int p*)` is the **same functional form** (q_int for D, no kinetic
diagonal), so the third row confirms directly: nested `jvp` through a dense `S(k)` built
from the shipped overlap projectors returns a finite, FD-matched ∂²S/∂k². The
augmentation-second-derivative is autograd, not algebra.

---

## 1. The analytic route's NC-locks (enumerated)

`sigma_shielding_dq` → `ShieldingDq.branch_fields_axis` is the production ∂/∂q path. It
is a **dense-eigh** formulation (per mesh-k: dense H, `eigh`, forward-mode velocities,
nested-jvp second derivatives), NOT the matrix-free batched-CG path the S-metric
primitives (#349/#351) live on. Reading `branch_fields_axis` and every helper it calls,
the exact S=I / missing-augmentation locks are:

| # | lock (file:sym) | assumes / lacks |
|---|---|---|
| L1 | `_eigh_and_dh` → `torch.linalg.eigh(H)` (kgeometry.py:398) | standard eigenproblem `H u = ε u` (S=I); needs generalized `H u = ε S u`, S-orthonormal U |
| L2 | `_dense_velocity_matrices` (kgeometry_nmr.py:806) / the `dh` in `_eigh_and_dh` | ∂H/∂k with **bare** D only; needs screened-D ∂H/∂k (current operator) **and** ∂S/∂k (for the generalized velocity) |
| L3 | `_resolvent_apply` (kgeometry_nmr.py:1268) | `a = uc.mH @ x` — plain ⟨u_m\|x⟩ (S=I); needs ⟨u_m\|S\|x⟩ and S-orthogonal P_c |
| L4 | `_d2h_mats` (kgeometry_nmr.py:1279) | ∂²H only; the generalized second q-derivative also needs −ε ∂²S/∂k² |
| L5 | `BlochHK.from_scf` + `.h` (kgeometry.py:263) | builds H from `res.v_eff` + NC `system.upfs`; no S, no screened D, no frozen USPP potential |
| L6 | covariant k-derivative `dtu`/`wmat @ uo` (in `branch_fields_axis`) | `wmat = q̂·v` is bare ∂H/∂k; the generalized covariant velocity is band-dependent `∂H/∂k − ε_n ∂S/∂k` |
| L7 | current-field assembly `dk.v[μ] @ du0`, `κ q̂_μ`, `2κ ρ` diamagnetic | kinetic + dia terms kinematic (reusable); the KB-nonlocal current needs screened D; the on-site augmentation current is absent |

L4 is the one the consolidation-plan Bucket 3 and the finite-q scoping (§1b) both flagged
as the reason "generalizing the dense-eigh path is genuine new work … answer before
building." Measured above: it is **not** a derivation.

---

## 2. Reuse vs new — coverage of the locked call-sites

Every lock's *physics ingredient* is already shipped and FD-validated. What is new is
**densification + assembly glue**, because the dense route needs (npw,npw) *matrices*
where the S-metric primitives expose matrix-free *applies*.

| lock | shipped S-metric ingredient (physics) | genuinely new (glue) |
|---|---|---|
| L1 eigh(H,S) | dense S from `_overlap_kbprojectors` (β, q_int) — measured; Cholesky-whitening is stdlib | dense generalized eigensolver + S-orthonormal back-transform — **S** |
| L2 ∂H/∂k (screened) + ∂S/∂k | `_USPPHVelocity` (∂H/∂k, screened D) + `OverlapVelocity`/`_overlap_kbprojectors` (∂S/∂k), FD-matched 2e-9 | densify the two matrix-free applies into (npw,npw) — **S** |
| L3 S-metric resolvent | dense S (above) | one `S@` insertion + S-orthogonal P_c — **S** (trivial) |
| L4 ∂²S/∂k² | traceable `S(k)` via `_overlap_kbprojectors.p`; nested-jvp pattern of `_d2h_mats` — **measured** rel 1e-9 | ~15-line `S(k)` builder mirroring `BlochHK.h`'s NL line — **S**, *not a derivation* |
| L5 USPP dense H/S ctx | `uspp_frozen.frozen_veff`, `screened_dscr`, β (`_overlap_kbprojectors`); `build_uspp_response_ctx` already assembles these | a `BlochHK`-USPP dataclass (dense H+S, traceable in k) — **M** |
| L6 generalized covariant velocity | same ∂H/∂k, ∂S/∂k as L2 | per-band `−ε ∂S` broadcast in `wmat @ uo` — **S** (trivial) |
| L7 current assembly | kinetic/dia kinematic (reuse smooth ρ); KB-nonlocal current = screened-D ∂H/∂k (L2); on-site `gipaw.para_aug_operator` (route-agnostic, §4) | wire screened-D current + para_aug cross-density input (shared with finite-q PR-B) — **M** |

**~6 of the 7 locks are covered outright by shipped primitives** (L1–L4, L6 ingredients,
L7 kinetic/dia/on-site). The genuinely-new code is (a) a dense generalized eigensolver
+ S-orthonormal path, (b) densifying the two matrix-free velocity applies, (c) a USPP
dense H/S context (mostly `build_uspp_response_ctx` re-used), and (d) the para_aug
cross-density input. No new *physics*; the eigensolve is never differentiated (eigh under
`no_grad`), so the S-orthonormal back-transform does not interact with the forward-mode
∂H/∂k, ∂S/∂k, ∂²H, ∂²S passes.

---

## 3. The ∂²/∂q² term — the flagged risk, downgraded to autograd

The analytic route's second-order term is `½ M_μ`, `M_μ = q̂·∇(∂H/∂k_μ)`, the mixed
second k-derivative computed by nested `torch.func.jvp` through the *traceable* dense
`H(k)`. Two measured facts collapse the risk:

1. **`H(k)` already contains a projector-nonlocal term** `p.mᵀ (D p*)`, and `_d2h_mats`
   already differentiates it twice — matched to central-FD of the velocity at rel
   ~3e-10 (probe §2). So the machinery handling ∂²β/∂k² for H is shipped and correct.

2. **USPP `S(k) = I + p.mᵀ (q_int p*)` is the identical functional form.** Building it
   from the shipped `_overlap_kbprojectors.p` (traceable — it is the same assembly that
   `p_and_dp` forward-differentiates for ∂S/∂k) and pushing it through the same nested
   `jvp` returns a finite ∂²S/∂k² matching FD at rel ~1e-9 (probe §3), on a real PAW Si
   system with the augmentation charge present.

So the generalized `M_μ = q̂·∇(∂H/∂k_μ) − ε q̂·∇(∂S/∂k_μ)` needs **no** new
augmentation-second-derivative: the −ε∂²S piece is `_d2h_mats` re-run on a traceable
`S(k)`, scaled per band. This is the single most important scoping result — the term
that would have made this a NEEDS-DECISION is autograd-generic.

---

## 4. Gauge / covariant position — not a new risk (confirmed)

The on-site paramagnetic-augmentation current operator is
`gipaw.para_aug_operator` = angular L_α × radial AE−PS ⟨1/r³⟩ (su(2)-validated per the
finite-q scoping §3, closed-shell Lamb null exact). Reading `para_aug_tensor`
(gipaw.py:347) confirms it is **route-agnostic**: it consumes a per-field ground×response
*cross density* `X_β[a,b] = Σ_o ⟨ψ̃⁰_o|p̃_a⟩⟨p̃_b|ψ̃^(1),β_o⟩` (shape (3,n,n)) and returns
σ^para; it has no dependence on how δu was obtained (finite-q vs analytic). The diamagnetic
(`dia_aug_tensor`) and core (`core_lamb_shielding`) terms are likewise ground-state-only.
The covariant-position physics carries over to the analytic assembly **unchanged** — its
input cross density is the only wiring, and the prefactor is the one open calibration
(the same one the finite-q route carries; honestly flagged in the `para_aug_tensor`
docstring). Not a new risk for the analytic route.

---

## 5. Cheapest validation + the PR-A decision

**Cheapest first check — NC-limit reduction gate, no external reference.** Build the USPP
dense H/S context from a **norm-conserving** pseudo (augmentation q_int = 0 ⇒ S = I,
dscr = D, no on-site term) and confirm analytic-USPP reproduces the plain-NC analytic σ
**to machine precision**. Smallest runnable system: the existing Si 2-atom, 6 Ry,
(2,2,2), fft (15,15,15) case (`test_sigma_dq_cubic_isotropy_and_site_equivalence`).
Reference measured here: **σ_iso = 20.817879 / 20.819837 ppm**, off-diagonal 2.8e-11 ppm.
This gate needs *no finite-q code* — it self-checks the dense analytic path with
uspp-ctx-from-NC vs uspp=None. Then Si PAW (2,2,2) 12 Ry and MgO ¹⁷O as the quantitative
checks (same systems the finite-q scoping named).

**Is the in-flight finite-q PR-A (`abs-sigma-pra`) worth landing? No — go straight to
analytic-USPP.** Reasoning:
- The finite-q *assembly* (`induced_current_q`, the batched matrix-free current) is a
  **different code path** from the dense analytic route and is slated for deletion. Its σ
  is throwaway.
- PR-A's only durable contributions are (i) the frozen-veff + screened-D ground-state
  ctx (`build_uspp_response_ctx` / `frozen_veff` / `screened_dscr`) and (ii) the NC-limit
  gate pattern. Both are **already in-tree** via #349/#351 (the ctx is exactly what the
  analytic USPP H/S context re-uses for L5), and the reduction reference is the plain-NC
  analytic σ above — not a finite-q number.
- Landing a finite-q shielding entry point we have committed to removing is
  negative-value churn. If `abs-sigma-pra` has already produced reusable ctx/test
  scaffolding, salvage those into the analytic build; do not ship its finite-q σ.

---

## 6. Effort estimate — stepwise

Sizes: **S** ≈ hours, **M** ≈ a day, one focused agent each.

| # | step | size |
|---|---|---|
| 1 | USPP dense H/S context (`BlochHK`-USPP: frozen-veff diagonal + screened-D β-nonlocal for H; I + q_int β-nonlocal for S; both traceable in k). Re-use `build_uspp_response_ctx` pieces | M |
| 2 | generalized eigh(H,S) via Cholesky-whitening + S-orthonormal back-transform, replacing `_eigh_and_dh`; return ε, S-orthonormal U, dense ∂H/∂k (screened D), dense ∂S/∂k | S |
| 3 | S-metric resolvent: `_resolvent_apply` gains a dense-S / `s_apply` arg (⟨u_m\|S\|x⟩), S-orthogonal P_c | S |
| 4 | generalized covariant velocity + current operator: `wmat @ uo` → `∂H_k@uo − (∂S_k@uo)·ε`; current uses screened-D ∂H/∂k | S |
| 5 | ∂²S via traceable `S(k)` (mirror `_d2h_mats`); add `−ε ∂²S` to the M-term — **measured autograd-generic** | S |
| 6 | screened-D KB-nonlocal current + smooth-ρ diamagnetic in the field assembly; **NC-limit reduction-gate test** vs 20.818 ppm | M |
| — | **PR-analytic-A: smooth USPP analytic σ + NC-limit gate — clean incremental extension** | — |
| 7 | on-site para_aug: X_β cross density (`becps` ⊗ ⟨β̃\|δu⟩) → `para_aug_tensor` (route-agnostic); **prefactor calibration** (the one open unknown, shared with finite-q PR-B, de-risked by the NC gate) | M |
| 8 | quantitative: Si PAW (2,2,2) σ vs bulk ²⁹Si; MgO ¹⁷O vs ≈215 ppm | M (compute) |
| — | **PR-analytic-B: on-site paramagnetic augmentation wired + real-material validation** | — |

Roughly a focused week: PR-analytic-A (smooth + gate) then PR-analytic-B (on-site + real
material). The whole build is dense-eigh at npw ~ O(10²–10³) — the analytic route's
existing scaling regime; USPP does not change it.

**Measured GO.** The dense-eigh analytic path *is* a second build (a generalized
eigensolver, an S-metric resolvent, a USPP dense H/S context), but not a research problem:
no new physics, every ingredient shipped and FD-validated, the on-site augmentation
operator route-agnostic, and — the decisive measured result — the ∂²/∂q² augmentation
second-derivative produced by the existing nested-`jvp` through a traceable `S(k)`, no
hand-derivation. The one residual unknown is the σ_para_aug prefactor calibration, which
is a calibration shared with the finite-q route, gated by the NC-limit reduction check.
