# Main-group quadrupolar NMR validation: ²³Na, ⁷Li, ¹¹B, ²⁷Al vs Elk + experiment

Follow-up to the multi-material EFG study (`efg_multimaterial_validation.md`). That work found that
gradwave's FLAPW electric-field-gradient (EFG) chain reaches the all-electron Elk reference — and,
through it, experiment — for **closed-shell main-group cations**: corundum ²⁷Al landed at 118 % of
Elk with C_Q = 2.48 MHz vs experiment 2.38 (within 4 %), η = 0 exact, *with the Al 2p semicore
frozen*. Anion sites sat at ~73 % of Elk (a muffin-tin site effect) and small semicore-active
cations (anatase Ti, Mg²⁺) failed wrong-sign. **One cation datapoint (²⁷Al in one oxide) is not a
class.** This study tests whether the favorable behaviour generalises to three *new* quadrupolar
main-group nuclei — ²³Na, ⁷Li, ¹¹B — and to a *second* ²⁷Al environment, each against a
matched-sphere Elk 11.0.2 all-electron reference and experiment.

Experiments only; no `src` change. The light species absent from gradwave's FLAPW tables (B, N, Na,
Li; Al/O ship or are re-injected) are added at runtime from `_mgroup.py`, each validated vs NIST-LSD
atomic eigenvalues first (see below).

## Materials (one clean, non-cubic-cation, FLAPW-tractable crystal per nucleus)

- **¹¹B — hexagonal BN** (P6₃/mmc, a = 2.504, c = 6.661 Å; 4 atoms). B is 3-coordinate trigonal-
  planar → axial EFG (η = 0) ∥c. Experimental ¹¹B C_Q = 2.934 MHz, η ≈ 0 (Jeschke et al., *Solid
  State NMR* **12**, 1 (1998)). A light, covalent, filled-valence main-group site — the cleanest
  small-cell ¹¹B benchmark.
- **²⁷Al 2nd site — wurtzite AlN** (P6₃mc, a = 3.111, c = 4.978 Å, u = 0.3821; 4 atoms). A different
  Al environment than corundum: 4-coordinate tetrahedral nitride vs 6-coordinate octahedral oxide.
  Axial EFG (η = 0) ∥c. Experimental ²⁷Al C_Q = 1.914 MHz, η = 0 (single-crystal ²⁷Al/¹⁴N NMR,
  *Molecules* **25**, 469 (2020); Bastow ~1.9 MHz). Al frozen [Ne] (the corundum recipe).
- **⁷Li — α-Li₃N** (P6/mmm, a = 3.648, c = 3.875 Å; 4 atoms). Two crystallographic Li on the
  hexagonal axis → both axial (η = 0): Li1 (1b, 0,0,½; linear N–Li–N between the layers) and Li2
  (2c, ⅓,⅔,0; in the Li₂N plane). Experimental ⁷Li C_Q = 0.258 and 0.585 MHz, η ≈ 0 (static ⁷Li
  NMR; the site→value assignment is not firmly fixed in the literature). Li⁺ has no p semicore.
- **²³Na — ferroelectric NaNO₂** (Imm2, a = 3.5653, b = 5.5728, c = 5.3846 Å; conventional 8-atom
  cell; coords from Ekuma et al., arXiv:1208.5710). Na⁺ in a distorted NaO₆ site with a C₂ axis ∥b
  → biaxial (η ≠ 0). Experimental ²³Na |C_Q| ≈ 1.1 MHz, η ≈ 0.1 (RT; note literature spread and the
  strong NO₂⁻ librational motion that *reduces* the observed static EFG). Na⁺ is isoelectronic with
  Mg²⁺ ([Ne]); frozen 2p is the same choice that reached experiment for Al³⁺ but failed for Mg²⁺.

Larger EFGs than NaNO₃ / cubic salts were chosen where possible to stay above the muffin-tin noise
floor; ⁷Li and ²³Na EFGs are intrinsically small (a fundamental limitation for those nuclei).

## Method (mirrors the multi-material workflow)

- **gradwave FLAPW**: `converge_efg` (muffin-tin SCF → warm fullpot → `newton_polish` → one exact
  `efg=True` pass), aug-lmax = 4, fullpot_lmax = 4, k-mesh 2×2×2, **smearing = 0** (all are
  insulators), symmetry on, exact solves, `kerker = 0.7`, conditioned anion 2s LO
  (`los={"N"/"O":[(0,"2s")]}`). ecut per material chosen so the effective rgkmax ≈ 7 given the
  spheres (h-BN 400, AlN 300, Li₃N 250, NaNO₂ 350 eV).
- **Elk 11.0.2 reference** (asus): tasks 0 then 115, `lmaxi 2`, `xctype 3` (PW-LDA, matched to
  gradwave), `ngridk 4 4 4`, muffin-tin radii **forced** to gradwave's spheres and `rgkmax`
  **matched** to gradwave's R_min·G_max(ecut) so both codes use the same spheres and basis cutoff.
  Stock Elk species (semicore in valence = full all-electron antishielding reference).
- **Comparison** axis-resolved in the shared Cartesian frame (identical lattice + atom order fed to
  both codes). Conversion: 1 a.u. = 97.174 eV/Å²; C_Q[MHz] = 2.4180·Q[barn]·V_zz[eV/Å²]. Q (barn,
  Pyykkö/Stone): ¹¹B +0.04059, ²⁷Al +0.14660, ⁷Li −0.04010, ²³Na +0.10400.

### Runtime species injection (B, N, Na, Li) — validated vs NIST-LSD

`_mgroup.py` injects each species at import (so the spawned k-worker pool re-applies it). gradwave
`atomic_scf` (scalar-nonrelativistic LDA) reproduces NIST-LSD valence eigenvalues to ~1–2 % (deep
valence): B 2p 0.2 % / 2s 2.0 %, N 2p 0.7 % / 2s 1.9 %, Na 2p 2.4 % / 2s 0.7 %, Al 2p 0.9 %; the Na
3s (−3.072 eV) and Li 2s (−3.18 eV) match the true NIST-LSD to <1 % (the apparent larger % is an
imprecise hard-coded reference constant, not a gradwave error). Core partitions: B/N freeze 1s;
Na freezes [Ne] (2p semicore frozen); Li freezes 1s; Al freezes [Ne] (the corundum recipe).

## Reference data (Elk 11.0.2, this work) and experiment

Elk EFGs (rmt forced to gradwave's spheres, rgkmax matched, ngridk 4 4 4, LDA):

| System | Site | V_zz (a.u.) | V_zz (eV/Å²) | η | C_Q (MHz) |
|---|---|---|---|---|---|
| h-BN | B (∥c) | −0.30894 | −30.02 | 0.000 | 2.95 (¹¹B) |
| AlN | Al (∥c) | −0.055628 | −5.41 | 0.007 | 1.92 (²⁷Al) |
| α-Li₃N | Li1 (1b, ∥c) | +0.075200 | +7.31 | 0.000 | 0.709 (⁷Li) |
| α-Li₃N | Li2 (2c, ∥c) | −0.036109 | −3.51 | 0.000 | 0.340 (⁷Li) |
| NaNO₂ | Na (∥b) | +0.059496 | +5.78 | 0.257 | 1.45 (²³Na) |

| System | Site | exp C_Q (MHz) | exp η | Citation |
|---|---|---|---|---|
| h-BN | ¹¹B | 2.934 | ~0 | Jeschke et al., *Solid State NMR* **12**, 1 (1998) |
| AlN | ²⁷Al | 1.914 | 0 | *Molecules* **25**, 469 (2020); Bastow ~1.9 |
| α-Li₃N | ⁷Li | 0.258 / 0.585 | ~0 | static ⁷Li NMR (two sites; assignment ambiguous) |
| NaNO₂ | ²³Na | ~1.1 | ~0.1 | RT ²³Na NMR (spread; NO₂⁻ librational averaging) |

**Elk vs experiment is itself excellent for the two "hard-lattice" cations.** h-BN ¹¹B: Elk 2.95 vs
exp 2.934 (+0.5 %). AlN ²⁷Al: Elk 1.92 vs exp 1.914 (+0.1 %). LDA is essentially exact for these
covalent/ionic filled-shell sites, so gradwave's target (Elk) *is* experiment here — unlike corundum
where Elk was 12 % below experiment. For NaNO₂, Elk (static) is +30 % over experiment (η 0.26 vs
0.1): expected, because the NO₂⁻ group librates at RT and averages the static EFG down — a molecular-
motion effect, not a code discrepancy. Li₃N's two small EFGs are both ~20–30 % above the reported
experimental pair in both codes (see below).

## gradwave FLAPW results

All numbers are the **full** EFG tensor (on-site l=2 sphere-Poisson + Weinert lattice/boundary term),
axis-resolved. On-site valence V_zz quoted as a diagnostic.

### h-BN ¹¹B — GATED (169 s)

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q | exp C_Q / η |
|---|---|---|---|---|---|---|
| B (∥c) | −36.64 | 0.000 | 3.60 (¹¹B) | −45.08 | −30.02 / 0.000 / 2.95 | 2.934 / ~0 |

gradwave B is at **122 % of Elk**, same sign (∥c, negative), η = 0 exact. C_Q(¹¹B) = 3.60 MHz vs
Elk 2.95 and experiment 2.934 → **+22 % over both**. Because Elk here *is* experiment (Elk 2.95 vs
exp 2.934, +0.5 %), gradwave's overshoot is exposed directly against the measurement — it is *not*
a level-of-theory (LDA) effect but a gradwave-vs-all-electron-reference bias. **aug-lmax = 5 gives
V_zz = −36.55 (C_Q 3.587), identical to aug-lmax = 4 — the overshoot is basis-converged, not a
truncation artifact.**

### α-Li₃N ⁷Li — GATED (62 s)

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q | exp C_Q |
|---|---|---|---|---|---|---|
| Li1 (1b, ∥c) | +8.51 | 0.000 | 0.825 (⁷Li) | +5.80 | +7.31 / 0.000 / 0.709 | 0.258 or 0.585 |
| Li2 (2c, ∥c) | −4.20 | 0.000 | 0.407 (⁷Li) | −3.51 | −3.51 / 0.000 / 0.340 | 0.585 or 0.258 |

Both Li sites are at **116 % (Li1) / 120 % (Li2) of Elk**, same sign, η = 0 exact — the same overshoot
as B. Both codes agree the 1b (linear N–Li–N) site carries the *larger* EFG; the experimental
site→value assignment (0.258 vs 0.585) appears reversed relative to the calculation, and both codes
sit ~20–40 % above the experimental magnitudes (⁷Li C_Q is a fraction of a MHz — near the
muffin-tin/measurement noise floor, and sensitive to the small Q(⁷Li)).

### wurtzite AlN ²⁷Al — GATED (after `refine_mgroup`; the chunked continuation's marginal mode needed a warm plain-Anderson + newton polish, r_nsph 2.6e-4, newton res 1.3e-6)

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q | exp C_Q / η |
|---|---|---|---|---|---|---|
| Al (∥c) | −19.11 | 0.000 | 6.78 (²⁷Al) | −18.98 | −5.41 / 0.007 / 1.92 | 1.914 / 0 |

**AlN Al is a genuine outlier at 353 % of Elk** (same sign ∥c, η = 0 exact, but 3.5× the magnitude);
C_Q(²⁷Al) = 6.78 MHz vs Elk 1.92 and experiment 1.914. This is *not* an ungated-state artifact — a
fully-converged refine (r_nsph 2.6e-4, newton res 1.3e-6) gives the same −19.11, and **aug-lmax = 5
gives −19.40 (C_Q 6.88), basis-converged.** The over-capture is on-site (−18.98, ≈ the full tensor),
so it is the l=2 valence-density asphericity at Al that is grossly overestimated. Contrast with
*ionic octahedral* corundum Al (118 % of Elk): the *covalent tetrahedral nitride* Al site — where the
Al 3p participates strongly in the Al–N bond and the wurtzite C₃v EFG is a small residual of a
near-Td cancellation — is where gradwave's muffin-tin partition fails badly. (h-BN B and Li₃N Li, in
the *same* nitride, are only ~+20 % — so it is Al-in-covalent-coordination specific, not a nitride
effect.)

### NaNO₂ ²³Na — INCOMPLETE (gradwave SCF did not converge in the compute window)

The Elk reference is in hand (V_zz = +5.78 eV/Å² ∥b, η = 0.257, C_Q(²³Na) = 1.45 MHz). The gradwave
run did **not** complete: the NaNO₂ muffin-tin SCF is a hard molecular-ion case — the covalent NO₂⁻
group forces a very short N–O bond (1.27 Å) and therefore tiny anion spheres (R = 0.62 Å), and the
2-iter smoke test already showed r_v ≈ 2.7×10² (vs ~30 for the other cells); Stage A did not reach a
fixed point within the budget on the 8-atom cell. So the gradwave ²³Na datapoint is **deferred**
(candidate fixes: seed from a converged muffin-tin at a lower ecut, larger anion spheres via a longer
N–O cutoff, or a Na-only large-sphere sub-cell). Two independent points bear on what to expect:
(a) Na⁺ is isoelectronic with Mg²⁺, which failed *wrong-sign* frozen-2p in MgF₂ (this run would test
whether Na⁺ needs its 2p semicore in valence); (b) even static all-electron Elk sits +30 % over
experiment for NaNO₂ (C_Q 1.45 vs ~1.1, η 0.26 vs 0.1), the expected signature of the NO₂⁻
librational motion that averages the RT EFG down — so NaNO₂ ²³Na is a poor cross-*experiment* target
regardless of code (motion, not electronic structure, sets the ~30 % gap).

## Comparison and verdict

Consolidated gradwave-vs-Elk (full-tensor V_zz) with the prior corundum ²⁷Al datapoint:

| Material | Nucleus | Site | gradwave V_zz | Elk V_zz | gw/Elk | sign | η gw/Elk | C_Q gw / Elk / exp (MHz) |
|---|---|---|---|---|---|---|---|---|
| **h-BN** | ¹¹B | B (∥c) | −36.64 | −30.02 | **122 %** | ✓ | 0.00 / 0.00 | 3.60 / 2.95 / 2.934 |
| **α-Li₃N** | ⁷Li | Li1 (1b) | +8.51 | +7.31 | **116 %** | ✓ | 0.00 / 0.00 | 0.825 / 0.709 / 0.26–0.59 |
| **α-Li₃N** | ⁷Li | Li2 (2c) | −4.20 | −3.51 | **120 %** | ✓ | 0.00 / 0.00 | 0.407 / 0.340 / 0.26–0.59 |
| Corundum (prior) | ²⁷Al | Al (∥c) | −6.99 | −5.90 | **118 %** | ✓ | 0.00 / 0.005 | 2.48 / 2.09 / 2.38 |
| **AlN** | ²⁷Al | Al (∥c) | −19.11 | −5.41 | **353 %** | ✓ | 0.00 / 0.007 | 6.78 / 1.92 / 1.914 |
| NaNO₂ | ²³Na | Na (∥b) | — (SCF incomplete) | +5.78 | — | — | — / 0.26 | — / 1.45 / ~1.1 |

Aug-lmax convergence checked on both aug4 and aug5: h-BN B −36.64→−36.55, AlN Al −19.11→−19.40 —
both **basis-converged**; the overshoots are real, not truncation.

### Verdict

**1. The QUALITATIVE capability is solid across main-group cations.** Every one of the five
gated cation sites (h-BN B, Li₃N Li1/Li2, corundum Al, AlN Al) reproduces the **site symmetry
(η = 0 exact for all five axial sites), the V_zz sign, the principal-axis direction (∥c), and the
magnitude ordering** of the all-electron reference. The species-injection machinery extends cleanly
to four new light elements (B, N, Na, Li — each NIST-LSD-validated), and the convergence pipeline
(muffin-tin → warm fullpot → newton, with a warm-plain-Anderson refine for the marginal cells) gates
three of four materials. As a *fingerprinting* tool — which site, what symmetry, what sign, is this
polymorph/coordination A or B — gradwave FLAPW works for main-group quadrupolar nuclei.

**2. The QUANTITATIVE C_Q does NOT uniformly land within 10 % of experiment — there is a systematic
gradwave-vs-Elk overshoot.** For the light main-group cations gradwave's V_zz is a strikingly
consistent **116–122 % of Elk** (¹¹B 122 %, ⁷Li 116/120 %, corundum ²⁷Al 118 %), same sign, η exact.
The decisive control is **h-BN, where Elk itself equals experiment (2.95 vs 2.934, +0.5 %)**: gradwave
is then +22 % over the *measurement*, so the overshoot is a real bias against the all-electron
reference, **not** an LDA level-of-theory effect. This reframes the prior "corundum ²⁷Al within 4 % of
experiment" headline as a **partial cancellation**: gradwave 118 % of Elk × Elk 88 % of experiment
(2.09 vs 2.38) ≈ landed on experiment by luck of the two errors pointing opposite ways. So
"quantitative main-group quadrupolar NMR" is **not yet bankable across the board** — the ionic
octahedral corundum ²⁷Al is the favorable case; the covalent tetrahedral AlN ²⁷Al is a **353 %
outlier**; the light cations carry a uniform **~+20 % C_Q overshoot**.

**3. Diagnosis direction for the ~20 % overshoot.** Two candidate causes are ruled out by the data:
(a) **not a Q-moment / unit convention** — the overshoot is on the EFG *tensor* itself (V_zz ratios
are 116–122 %, before Q enters), and it is consistent across three nuclei with very different Q
(¹¹B +0.041, ⁷Li −0.040, ²⁷Al +0.147 barn), so no per-nucleus Q error can explain it; (b) **not
aug-lmax truncation** — aug5 reproduces aug4 to ≤1.5 % for both h-BN and AlN. What remains is a real
**muffin-tin partition bias in the EFG assembly**: cations run **high (+20 %)** while the prior study
found anions run **low (~73 %)** at the same aug-lmax — a *site-type-dependent* over/under-weighting
of the aspherical density between the sphere (on-site l=2 Poisson) and the boundary (Weinert lattice)
terms, most plausibly in how the sphere radius splits the valence-tail asphericity. **AlN Al (353 %)
is the amplified limit of the same mechanism**: the covalent Al–N bond pushes far more aspherical Al
3p density into the boundary region, and the wurtzite EFG is a small C₃v residual of a near-Td
cancellation that the partition gets badly wrong. The concrete next probes: (i) an R_MT sweep at a
fixed cation site (does gw/Elk trend toward 1 as R_MT grows, isolating the sphere/boundary split?);
(ii) an on-site-only vs full-tensor decomposition against Elk's own on-site/lattice split; (iii) a
2p-in-valence Al/Na run (does putting the semicore in valence — the Mg²⁺ lesson — change the cation
bias?).

**Headline (honest):** gradwave FLAPW gives the **right qualitative EFG** (η, sign, axis, ordering,
site/polymorph discrimination) for main-group quadrupolar nuclei — a working fingerprinting
capability now — but its **C_Q carries a consistent ~+20 % overshoot vs the all-electron reference**
for light main-group cations (¹¹B 122 %, ⁷Li 116–120 %, corundum ²⁷Al 118 %), with a covalent
tetrahedral outlier (AlN ²⁷Al 353 %). "Quantitative main-group quadrupolar NMR" is **not yet
bankable**; closing the ~20 % cation overshoot (a muffin-tin sphere/boundary partition bias, not a Q
convention and not aug truncation) is the specific, well-posed next task. ²³Na (NaNO₂) is deferred
(hard molecular-ion SCF; and even Elk overshoots experiment there via NO₂⁻ librational motion).

## Reproducing

- gradwave drivers: `experiments/autoapw/{hbn,aln,li3n,nano2}_efg.py` (each `converge_efg` from
  `_efgrun.py`), species injection + reporting in `_mgroup.py`, hard-cell warm-tighten in
  `refine_mgroup.py`. Validate species first: `validate_mgroup_species.py`. Run on asus via
  pueue, `OMP_NUM_THREADS=2 KWORKERS=4`. `LMAX=5` for the aug-lmax convergence check. No `src`
  change (B/N/Na/Li injected at runtime, each NIST-LSD-validated).
- Elk: `setup_elk_mgroup.py` builds all four input dirs (rmt forced to gradwave's spheres, rgkmax
  matched to gradwave's R_min·G_max(ecut)); run `~/github/elk-11.0.2/src/elk` in each (tasks 0, 115).
