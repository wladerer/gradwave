# Converged-k EFG validation: the #350/#352/#353 ratios re-taken at matched, converged k

Follow-up to `efg_lightcation_diagnosis.md`, which found that the recurring EFG "cation +20 %
overshoot" was a **k-mesh mismatch artifact** — gradwave's published FLAPW numbers were computed at
**k222** while every Elk all-electron reference was at **k444**, and the l=2 EFG (a p/d-population
anisotropy) is strongly k-dependent in these small covalent/ionic cells. Consequence flagged there:
the entire recent EFG validation set (`efg_multimaterial_validation.md` #350,
`main_group_quadrupolar_validation.md` #352, and the `efg_helo_l1_fix.md` #353 HELO gains) was
**gw-k222 vs Elk-k444**, so all the quantitative ratios were k-unconverged.

This study re-runs the whole set at **matched, converged k** in both codes and re-reads the ratios.
Both codes at the **same isotropic n×n×n mesh** (n = 4, 6, 8), muffin-tin radii forced identical,
LDA, rgkmax matched to gradwave's R_min·G_max(ecut). gradwave: `converge_efg` (muffin-tin → warm
fullpot → newton → one exact `efg=True` pass), aug-lmax 4, fp-lmax 4, kerker 0.7, smearing 0, exact
solves; the **#353 production basis** applied where the campaign applied it — the unconfined l=1
anion HELO (e=90, confine=False) on O/F, a confined l=1 Al LO on corundum Al; light cations (B, Li)
get no cation HELO (the #353 recipe does not add one), so h-BN and Li₃N are run baseline. Elk 11.0.2
tasks 0+115, lmaxi 2, xctype 3. On-site / boundary decomposition via the validated `elk_onsite.py`
STATE.OUT rhomt reader (reproduces corundum Al −6.19 / O +27.09 and every EFG.OUT tensor).

All work on asus, worktree `efg-k666-reread` off `origin/main @ 4dbf2fab`. Runners:
`kconv_efg.py` (MAT, KMESH, HELO), `setup_elk_kconv.py` + `elk_kconv.sh` (matched Elk at any n).

## The decisive fact: Elk is k-converged by k4; the k-drift is entirely gradwave's

Elk moves **< 0.5 %** from k4→k8 at every site (e.g. corundum O on-site 27.091 → 27.093 → 27.093;
rutile O on-site −10.49 → −10.30 → −10.27; hbn B −39.44 → −39.61 → −39.62). So the matched
reference is stable and **every gw/Elk ratio drift below is a gradwave-side k-effect**, exactly as
the light-cation diagnosis argued. gradwave is converged to **≤ ~2 %** by **k6**; k4 is not enough
(rutile and anatase do not even gate cleanly at k4, and several sites still move k4→k6).

## Master table — gw vs Elk at matched converged k (k6 unless noted; k8 confirms)

On-site = interior l=2 sphere-Poisson (`V_zz_valence`); full = observable tensor; boundary =
full − on-site. For the strongly biaxial O sites `V_zz` is the largest-|eigenvalue|, whose **sign is
frame-ambiguous** — magnitude ratios |r| are quoted there. eV/Å²; C_Q in MHz.

| Material | Site | basis | full gw/Elk \|r\| | η gw / Elk (full) | on-site \|r\| | C_Q gw / Elk / exp | gate |
|---|---|---|---|---|---|---|---|
| **Corundum** | O (anion) | HELO | 34.43 / 35.64 = **0.966** | 0.48 / 0.51 | **0.953** | 2.13 / 2.20 / 2.167 | k4,k8 GATED |
| **Corundum** | Al (cation) | Al-LO | −5.60 / −5.91 = **0.949** | 0.00 / 0.005 | **0.951** | 1.99 / 2.09 / 2.38 | k4,k8 GATED |
| **MgF₂** | F (anion) | HELO | 50.79 / 51.55 = **0.985** | 0.40 / 0.39 | **0.978** | — (¹⁹F) | k4 GATED |
| **MgF₂** | Mg (cation) | — | +8.81 / −5.51 = 1.60 (✗sign) | 0.84 / 0.55 | 1.63 (✗) | 4.25 / 2.66 | k4 GATED |
| **Rutile** | O (anion) | HELO | −12.14 / −19.14 = **0.634** | **0.11 / 0.74** | 0.86–0.88 | 0.75 / 1.18 | k6,k8 GATED |
| **Rutile** | Ti (cation) | — | +9.92 / +19.37 = **0.512** | 0.49 / 0.36 | 0.54 | 5.93 / 11.57 | k6,k8 GATED |
| **h-BN** | B (cation) | base | −27.23 / −30.19 = **0.901** | 0.00 / 0.00 | **0.929** | 2.67 / 2.96 / 2.934 | k4,k8 GATED |
| **h-BN** | N (anion) | base | −7.61 / −2.92 = 2.6 | 0.00 / 0.01 | 1.22 | — | k4,k8 GATED |
| **Li₃N** | Li1 (1b) | base | +6.74 / +7.28 = **0.926** | 0.00 / 0.00 | **0.945** | 0.654 / 0.706 | k4,6,8 GATED |
| **Li₃N** | Li2 (2c) | base | −3.03 / −3.48 = **0.870** | 0.00 / 0.00 | **0.937** | 0.294 / 0.338 | k4,6,8 GATED |
| **Anatase** | Ti / O | HELO | see below — **MARGINAL both k** | | | | never gated |

k6-vs-k8 movement (the convergence check): corundum dead-flat (Δ < 0.1 %); MgF₂ F 0.976→0.978;
h-BN B 0.930→0.929 (flat); Li₃N flat to 3 digits; rutile O on-site 0.876→0.861 and full η
0.137→0.114 (~2 %, the slowest site); rutile Ti 0.512→0.509. **k6 is converged to ≤ ~2 %;
k8 only pins the last 1–2 % on the most biaxial anion (rutile O) and the heavy TM cation (rutile Ti).**

## k222 → k444 → converged, for the sites the campaign headlined

| Site | quantity | k222 (published) | k444 | converged (k6/k8) | Elk |
|---|---|---|---|---|---|
| Corundum O | on-site gw/Elk | 0.66 (base) → **0.94** (#353 HELO) | 0.95 | **0.953** | — |
| Corundum Al | on-site gw/Elk | 1.17 (base) → **1.105** (#353) | 0.95 | **0.951** | — |
| Corundum Al | C_Q(²⁷Al) | 2.48 (base) / 2.33 (#353) | 1.99 | **1.99** | 2.09 |
| MgF₂ F | on-site gw/Elk | 0.73 (base, no HELO) | — | **0.978** (HELO) | — |
| Rutile O | on-site η | 0.91 (base) → 0.15 (#353) | — | **0.24** | 0.10–0.11 |
| Rutile O | full \|gw/Elk\| | ~0.73 | — | **0.63** | — |
| Rutile Ti | full gw/Elk | **0.88** | — | **0.51** | — |
| h-BN ¹¹B | on-site gw/Elk | 1.14 | 0.93 | **0.929** | — |
| h-BN ¹¹B | C_Q | 3.60 (+22 %) | 2.66 | **2.67** | 2.96 |
| Li₃N ⁷Li1 | on-site gw/Elk | 1.09 | 0.93 | **0.945** | — |
| Li₃N ⁷Li1 | boundary gw/Elk | — | 1.68 (flagged over-capture) | **0.88 (under)** | — |

## The four questions

### 1. Do the #353 HELO anion/Al gains HOLD at converged k? — YES (and they tighten)

- **Corundum O (anion HELO): on-site gw/Elk = 0.953 at k6 = k8**, full 0.966, C_Q(¹⁷O) 2.13 vs Elk
  2.20 vs exp 2.167 (within 2 % of experiment). The #353 k222 gain 0.66 → 0.94 not only holds, it
  is essentially exact at converged k (0.95). **Decisive.**
- **MgF₂ F (anion HELO): on-site 0.978, full 0.985** — the cleanest anion in the set, up from the
  #350 baseline 0.73. The anion HELO generalizes to the non-oxide anion at converged k. **Decisive.**
- **Corundum Al (confined Al HELO): on-site 0.951, full 0.949, k-flat, C_Q 1.99 vs Elk 2.09.** The
  k222-baseline 1.18 overshoot and k222-HELO 1.105 both collapse to ~0.95 at converged k.
- **Rutile O (anion HELO): the η disaster fix is robust to k** — the baseline on-site η 0.91
  collapses to 0.24 (Elk on-site 0.10–0.11), a large qualitative move, and the full V_zz keeps the
  correct **negative** sign at every gated k. **But it does not land on Elk like corundum-O/MgF₂-F:**
  the converged full tensor is only **0.63× Elk in magnitude** and its **full η is 0.11 vs Elk 0.74**
  — gradwave under-captures rutile-O biaxiality at converged k. So the HELO fixes the on-site
  asphericity blow-up robustly, but rutile O's full observable remains ~⅔ of Elk.

**Verdict: the anion/Al HELO gain holds at converged k — decisively for corundum O (0.95), MgF₂ F
(0.98) and corundum Al (0.95); for rutile O the qualitative η fix holds but the full tensor sits at
~63 % with under-captured biaxiality.**

### 2. Standardized-k recommendation — adopt k6 (k666), matched in both codes

Elk is converged by k4 (< 0.5 % to k8); gradwave is converged to ≤ ~2 % by **k6**. k4 is
insufficient (rutile/anatase do not gate at k4; rutile O, rutile Ti, h-BN N still drift k4→k6).
**Recommendation: standardize the EFG validation on a matched k6 (k666) mesh** — adequate to ~2 %
for on-site and Al/anion full tensors — and use k8 only to pin the last 1–2 % on strongly biaxial
anions (rutile O) or heavy-TM cations (rutile Ti). Every published gw/Elk ratio should be re-quoted
at matched k6; the k222 numbers are not reliable (rutile Ti alone moved 37 percentage points).

### 3. Li₃N boundary/Weinert over-capture — NOT a bug; the small spheres are fine at converged k

At converged, **gated** k (dead-consistent across k4/k6/k8) the Li₃N full tensor is a clean
**undershoot**: Li1 0.926, Li2 0.870 of Elk, with the **boundary term 0.44–0.88× Elk (under, not
over)** — Li1 +1.71 vs Elk +1.96, Li2 −0.21 vs Elk −0.48. Both the on-site (0.94) and the boundary
(< 1) sit *below* Elk, a coherent mild under-capture; there is **no 1.7–2.2× boundary over-capture**
and no small-sphere blow-up. The `efg_lightcation_diagnosis.md` flag (boundary 1.68–2.21× Elk at
k444, full 1.09–1.13×) **does not reproduce** in these gated runs. The FLAPW `src` is byte-identical
between that diagnostic's commit (6992795) and this one (4dbf2fab) — no code changed — so the earlier
over-capture was most plausibly an under-converged-state artifact of that quick k444 diagnostic, not
an intrinsic Weinert/pseudocharge issue. **Verdict: the small close-packed Li spheres behave well at
converged k (full tensor 0.87–0.93× Elk); the "boundary over-capture" is not a bug to chase.**

### 4. Light-cation clean-negative — YES, holds at converged k

At k6 = k8 (flat), the light cations land **slightly under** Elk on-site, as the k444 diagnosis
found: **¹¹B (h-BN) 0.929, ⁷Li1 0.945, ⁷Li2 0.937** (all < 1). The k222 "+14–20 % over-capture" was
entirely a k-artifact — gone at converged k. C_Q confirms: h-BN ¹¹B **2.67 MHz** (k6=k8) vs Elk 2.96
vs experiment 2.934 — the k222 "+22 % overshoot" (3.60) is gone, replaced by a ~10 % *undershoot*
against the measurement. Li₃N ⁷Li C_Q 0.654 / 0.294 (0.93× / 0.87× Elk). **Clean negative confirmed.**

## Which #350 / #352 published numbers change (and how)

| Where | published (k222) | converged (matched k6/k8) | direction of change |
|---|---|---|---|
| #350 Corundum Al C_Q | 2.48 ("within 4 % of exp 2.38") | **1.99** (0.95× Elk 2.09; ~16 % under exp) | the "on experiment" was a k+basis coincidence |
| #350 Corundum O C_Q | 1.63 (0.74× Elk) | **2.13** (0.97× Elk; within 2 % of exp) | improves (this reflects the #353 HELO) |
| #350 Rutile Ti | +17.1 = **88 % of Elk** | **+9.9 = 51 % of Elk** | **large** — rutile Ti is very k-sensitive; the "cation parity" was a k222 illusion, gw under-captures Ti on-site by half |
| #350 Rutile O | ~73 % full, η "fixed" | full \|r\| 0.63, η 0.11 vs Elk 0.74 | full-tensor deficit deeper; biaxiality under-captured |
| #350 MgF₂ F | 73 % (baseline) | **0.98** (with #353 HELO) | improves |
| #352 h-BN ¹¹B C_Q | 3.60 (+22 %, "122 % of Elk") | **2.67** (0.90× Elk; ~10 % under exp) | **overshoot → slight undershoot** |
| #352 Li₃N ⁷Li C_Q | 0.825 / 0.407 (116 % / 120 %) | 0.654 / 0.294 (0.93× / 0.87× Elk) | **overshoot → undershoot** |

Unchanged conclusions: MgF₂ Mg²⁺ still wrong-sign (needs 2p semicore in valence); anatase remains
MARGINAL (never gates at k4 or k6, so its converged numbers are not established — at the best
marginal state Ti's sign does flip to the correct negative, −0.63, but tiny vs Elk −9.55, and O's
on-site is sign-flipped; treat anatase as unresolved pending a converged gate). h-BN N (anion, run
baseline here — no HELO) has a small, strongly k-sensitive full tensor and is not a headline nucleus.

## Headline

**The #353 HELO anion/Al gains survive the k-convergence correction — decisively.** At matched,
converged k the anion HELO puts **corundum O at 0.95, MgF₂ F at 0.98** of Elk on-site (from a 0.73
baseline), and **corundum Al at 0.95** — a tight ~0.95 cluster for the closed-shell oxide/fluoride
sites. The light-cation "over-capture" and the Li₃N "boundary over-capture" were both k-artifacts:
at converged k the light cations sit slightly **under** Elk (¹¹B 0.93, ⁷Li 0.94) and Li₃N's small
spheres show a clean **undershoot**, no boundary blow-up. The two casualties of the correction are
**rutile Ti** (the k222 "88 % parity" was an illusion — 51 % at converged k) and **rutile O** (full
tensor 63 %, biaxiality under-captured despite the robust η fix). **Standardize the set on matched
k6 (k666);** the k222 ratios in #350/#352 are not reliable and the table above lists the corrections.

## Reproducing

- gradwave: `experiments/autoapw/kconv_efg.py` (`MAT`, `KMESH`, `HELO`; reuses `_efgrun.converge_efg`,
  `_mgroup` species injection + Mg/F). `gw_kconv.sh MAT N` wraps it (OMP=2, kworkers=4).
- Elk: `setup_elk_kconv.py` + `elk_kconv.sh MAT N` (rmt forced to gw spheres, rgkmax matched,
  isotropic ngridk n); `elk_run.sh MAT N` wraps it. On-site decomposition: `elk_onsite.py STATE.OUT`.
- Run on asus via pueue groups `efggw`/`efgelk`; logs under `~/efg_kconv/logs/` (each ends `EXIT=`).
- Meshes: k4/k6/k8 for rutile, corundum, h-BN, Li₃N; k4/k6 for anatase, MgF₂ (both codes).
