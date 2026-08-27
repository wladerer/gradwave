# Analytic-USPP bare shielding diverges with ecut — root cause

**Verdict: the divergence is a numerical instability of the dense-eigh S-metric
analytic route, NOT a missing physical augmentation term.** The dense-eigh
resolvent expands the magnetic perturbation in the S-ORTHONORMAL eigenbasis of
the augmentation overlap `S(k) = I + Σ_ij q_ij|β_i⟩⟨β_j|`. For hard-augmentation
first-row anions (O) `S(k)` becomes progressively overcomplete as ecut grows
(min-eig → 0), the eigenbasis expansion develops large denominator-reweighted
terms, and the bare σ blows up — even though the *physical* velocity coupling
and the shipped matrix-free CG response are ecut-stable, and the smooth-current
continuity closes to machine precision. Soft PAW (Si) is immune (`cond(S) ≈ 2`).

System: MgO rocksalt (a = 4.21 Å, 2-atom), Mg semicore PAW
(`Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF`) + `O.pbe-n-kjpaw_psl.1.0.0.UPF`, PBE,
2×2×2 MP. Runs on asus, OMP_NUM_THREADS=5. Probe scripts:
`experiments/autoapw/mgo_*.py` (committed on this branch).

---

## 0. The defect (reproduced)

`sigma_shielding_dq(res, uspp=ctx)` bare ¹⁷O σ_iso (this branch, symmetry off):

| ecut (wfc/rho) | bare Mg | **bare O** | published GIPAW ¹⁷O anchor |
|---|---|---|---|
| 40 / 160 Ry | +342.1 | **−1912.1** | ≈ +215 ppm |
| 60 / 360 Ry | +328.0 | **−3758.3** | ≈ +215 ppm |

The O bare term nearly doubles for a 1.5× ecut increase at fixed mesh — not a
basis-convergence tail. The core / dia_aug / para_aug terms are ecut-stable and
sane; the sickness is isolated to the smooth analytic-USPP bare route. Diamond
²⁹Si (soft PAW) through the identical code path is fine (bare ≈ −5.7 ppm, total
≈ 398 ppm), so the route is correct for soft datasets.

---

## 1. It is NOT a high-G accumulation tail (shell decay)

Biot–Savart integrand `|contribution|` binned by |G| for the O column
(`mgo_bare_divergence_diag.py`): the integrand DECAYS to ~1e-18 well before the
grid edge at BOTH rungs (peaks near |G| ≈ 8–14, gone past |G| ≈ 40). But the
*entire* response field is ~5× larger at 60 than at 40 Ry across ALL shells,
dominated by the `cross1` (∂S/∂s response) contribution. So the field itself
grows in magnitude; there is no un-decaying high-G tail, no aliasing, no
wrong-grid FT. The current field is bandlimited to 2·G_wfc and the dense grid
(ecutrho) fully contains it.

## 2. It is NOT a missing augmentation current (continuity closes)

`uspp_smooth_continuity` on the +q velocity Sternheimer response
(`mgo_continuity_diag.py`) — the identity
`q·mean[j_kin + j_nl] = mean s + (augmentation current + truncation)`:

| ecut | source | q·(j_kin+j_nl) | **remainder (aug+trunc)** |
|---|---|---|---|
| 40/160 | 1.7e-13 | 4.9e-14 | **1.17e-13** |
| 60/360 | 1.5e-14 | 1.4e-13 | **1.57e-13** |

The smooth+nonlocal CG current is exactly conserved; the augmentation-current
remainder is machine-precision zero at both rungs (×1.35, staying at 1e-13).
There is **no missing physical augmentation term** — the ∂²/∂q² augmentation
second derivative hypothesized in the consolidation plan is not the cause.

## 3. The shipped matrix-free CG response is ecut-stable; the dense-eigh route is not

Same MgO ground state, band-resolved response at k = mesh point 3
(`mgo_probe_numerator.py`, `mgo_probe_norms.py`):

| quantity | 40/160 | 60/360 | scaling |
|---|---|---|---|
| **CG matrix-free total \|δψ\|** (`velocity_perturbation_q`, shipped/#349) | 2.8675 | 2.9071 | **×1.01** |
| CG \|δψ\|, O 2s band | 0.134 | 0.135 | ×1.01 |
| dense-eigh analytic \|du0\|, O 2s band | 0.979 | 2.45 | ×2.5 |
| dense-eigh analytic \|du1\| | 1.385 | 4.365 | ×3.2 |
| resolvent numerator ⟨m\|S·O0\|4⟩ (m: ε≈+21 eV) | 29.68 | 76.54 | ×2.58 |
| **physical velocity coupling ⟨m\|(∂H/∂k−ε∂S/∂k)\|4⟩** | 2.042 | 2.060 | **×1.01** |

The physical operator coupling is ecut-stable to 1%. The matrix-free CG solve of
the SAME Sternheimer equation is ecut-stable to 1%. Only the dense-eigh
resolvent's S-weighted numerator `⟨u_m|S|O0 u⟩` grows — through the `S`
insertion, i.e. the augmentation part `⟨m|Σq|β⟩⟨β|O0 u⟩`, not the bare coupling.
The dominant growing term is a NORMAL conduction state (ε≈+21 eV, euclidean
‖u‖≈0.93), so ghost-state projection alone would not cure it.

## 4. The driver: augmentation-overlap overcompleteness

`min-eig S(Γ)` vs ecut (`mgo_probe_spectral.py`, `overlap_cond_calib.py`):

| dataset | ecut | npw | min-eig S | max-eig | **cond(S)** |
|---|---|---|---|---|---|
| Si PAW (soft) | 12 Ry | 181 | 0.810 | 1.348 | **1.7** |
| Si PAW | 20 | 411 | 0.782 | 1.514 | 1.9 |
| Si PAW | 30 | 749 | 0.775 | 1.564 | 2.0 |
| MgO (hard O) | 40 | 537 | 4.84e-2 | 20.59 | **426** |
| MgO | 60 | 965 | 1.51e-2 | 49.02 | **3245** |
| MgO | 80 | 1471 | 1.38e-2 | 56.36 | 4089 |
| MgO | 100 | 2109 | 6.83e-3 | 57.10 | 8358 |
| MgO | 120 | 2741 | 2.57e-3 | 59.33 | 23100 |

`cond(S)` tracks the bare-σ blow-up 1:1 (426→3245 as −1912→−3758). The
generalized eigenproblem `H U = S U ε` then produces high-euclidean-norm
S-orthonormal "ghost" states (‖u‖₂ = 3.7→7.8 at ε = +803→+2615 eV) and, more
importantly, over/under-weights the S-orthonormal expansion of normal states.
`ShieldingDq`'s `_resolvent_apply_s` (`du = Σ_m u_m⟨u_m|S|x⟩/(ε_n−ε_m)`) breaks
the cancellation the S-completeness relation `Σ_m u_m⟨u_m|S| = I` would enforce,
because each term carries a different energy denominator. The matrix-free CG
solve of `(H−εS)δu = −P_c^S O u` never forms this ill-conditioned eigenbasis and
is immune.

**Control — all-NC MgO** (ONCV Mg + O, plain NC analytic route, `uspp=None`,
`mgo_probe_spectral.py` part 3): O bare = −276 (40) → −294 (60) → −324 (80) Ry.
Mild real basis drift (~48 ppm over 40→80), NOT the ×2 PAW divergence — the
instability is specific to the S-metric dense-eigh path with an ill-conditioned
augmentation overlap.

---

## 5. Deliverable this PR: guard + evidence + fix plan (not the fix)

A correct fix re-architects `ShieldingDq`'s response backend from the dense-eigh
S-orthonormal resolvent to the shipped matrix-free S-metric CG Sternheimer (the
one measured stable above). That is a substantial rebuild of the analytic route
(the whole ∂/∂q `du0`/`du1`/covariant-derivative assembly is built on the
eigenbasis resolvent) and needs full re-validation (NC gate, Si ≈ 398 ppm). An
honest diagnosis + guard beats a rushed re-derivation that risks the S-insertion.

**Shipped here:**
- `uspp_overlap_conditioning(ctx)` — cheap cond(S(k)) probe (one `eigvalsh` per
  mesh k, no SCF/autograd), the measured 1:1 predictor of the divergence.
- `sigma_shielding_dq(..., uspp=ctx)` raises `USPPShieldingConditioningWarning`
  when `max cond(S(k)) > 50` (far above soft PAW ≈ 2, far below the diverging
  hard anions ≥ 426). No-op for the NC route and the S=I NC-limit gate.
- This note + the `mgo_*.py` / `overlap_cond_calib.py` probe scripts.

**Fix plan (next PR):** route `ShieldingDq`'s response through
`velocity_perturbation_q` / `cg_sternheimer` (S-metric, `s_apply=`/`dsvel=`) —
already used by `para_augmentation_shielding` and measured ecut-stable here — in
place of `_geigh_and_dhds` + `_resolvent_apply_s`. The ∂/∂q covariant-derivative
and second-order (`du1`, `M_μ`) assembly must be re-expressed on the matrix-free
solves (differentiate the CG solution implicitly rather than the eigenbasis
resolvent). Re-validate against the NC-limit gate (bit-reduction to the plain-NC
analytic σ), Si ≈ 398 ppm, and the MgO ¹⁷O ecut-stability + ≈215 ppm anchor.
