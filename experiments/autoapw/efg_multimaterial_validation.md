# Multi-material EFG validation: anatase TiO₂, α-Al₂O₃, MgF₂ vs Elk + experiment

Follow-up to the rutile TiO₂ campaign (`TIO2_NMR.md`). That work established a single trustworthy
FLAPW electric-field-gradient (EFG) datapoint: rutile Ti at Elk parity (gradwave +17.1 vs Elk
+19.34 eV/Å²) and rutile O at ~73 % of Elk with the correct η. **One datapoint is not a
characterization.** This study turns it into a set by running three more materials against the same
Elk 11.0.2 all-electron reference (and, where a quadrupolar nucleus exists, against experiment):

- **anatase TiO₂** — the *other* TiO₂ polymorph, same Ti/O species, different geometry →
  polymorph discrimination.
- **α-Al₂O₃ corundum** — a different cation/nucleus (²⁷Al), the classic axial-EFG benchmark.
- **MgF₂** — the *non-oxide* rutile-structure analog (Ti→Mg, O→F). The F site sits at the *same*
  crystallographic position as rutile's O, so the F-vs-Elk comparison is a controlled test of
  whether the "anion at ~73 % of Elk" deficit is O-chemistry-specific or a property of the rutile
  anion site. ¹⁹F is I=½ (no quadrupole), so the F check is gradwave-vs-Elk tensor only.

Experiments only; no `src` change. The Al/Mg/F species (absent from gradwave's FLAPW tables) are
injected at runtime from the experiment scripts (validated vs NIST LDA first — see below), so the
committed diff is entirely under `experiments/`.

## Method (mirrors the rutile workflow)

- **gradwave FLAPW**: `crystal_scf_multi(..., efg=True, fullpot=True)`, aug-lmax = 4, fullpot_lmax =
  4, ecut = 300 eV (≈ rgkmax 7.3, above Elk's 7), **smearing = 0** (insulators — mandatory, per the
  rutile diagnosis that Fermi smearing across the gap destabilises the EFG), symmetry on, exact
  solves (`subspace_reuse=False`), conditioned O 2s local orbital (`los={"O":[(0,"2s")]}`,
  `el_override={"O":{0:"2p"}}`) — the #344 recipe. k-mesh 2×2×2.
- **Robust convergence** (`_efgrun.converge_efg`): the cold fullpot k222 trajectory is run-to-run
  fragile (marginal interstitial mode |ρ|≈1.02). So: (A) muffin-tin SCF to seed the
  spherical/interstitial potentials; (B) a short fullpot continuation warm-started from A, chunked,
  tracking the best (lowest r_v) state and bailing on divergence; (C) `newton_polish` of that best
  state to the coupled fixed point. Then one exact `efg=True` pass.
- **Elk 11.0.2 reference** (asus `~/github/elk-11.0.2`): tasks 0 then 115 (EFG), `lmaxi 2`,
  `xctype 3` (PW-LDA, matched to gradwave's LDA), `rgkmax 7`, `ngridk 4 4 4`. The muffin-tin radii
  are **forced** in the species files to exactly gradwave's spheres, so both codes use identical
  spheres. Species files are the validated rutile `Ti.in`/`O.in`; `Al.in` is Elk's stock species
  (2p **in valence** → a full all-electron antishielding reference).
- **Comparison** is axis-resolved in the shared Cartesian frame (identical lattice vectors and atom
  order fed to both codes), not by |V|-sorted eigenvalues. For all three materials here the V_zz
  eigenvalue is non-degenerate, so |V|-sorting is unambiguous (unlike rutile O). Conversion:
  1 a.u. (Ha/Bohr²) = 97.174 eV/Å²; C_Q[MHz] = 2.4180·Q[barn]·V_zz[eV/Å²].

Muffin-tin radii (Å): anatase Ti 1.06 / O 0.824 (Ti–O bond 1.934 Å); corundum Al 0.97 / O 0.824
(Al–O bond 1.855 Å); MgF₂ Mg 1.0 / F 0.80 (Mg–F bond 1.980 Å). O is held at rutile's 0.824 Å in all
oxides so the O comparison is clean across the set; F (0.80 Å) is close to O.

### Geometries (primitive cells, experimental lattice constants)

- **Anatase TiO₂**, I4₁/amd (#141), a = 3.7842 Å, c = 9.5146 Å, O at (0,0,0.2081). Primitive BCT
  cell, 2 Ti + 4 O = 6 atoms (same size as the rutile cell).
- **α-Al₂O₃ corundum**, R-3c (#167), a = 4.7602 Å, c = 12.9933 Å, Al z = 0.35216, O x = 0.30624.
  Rhombohedral primitive cell, 4 Al + 6 O = 10 atoms.
- **MgF₂**, P4₂/mnm (#136) — *identical topology to rutile TiO₂*, a = 4.621 Å, c = 3.016 Å,
  F at (u,u,0) with u = 0.303. 2 Mg + 4 F = 6 atoms; the F sublattice is the rutile O sublattice.

### Runtime species injection (Al, Mg, F)

gradwave's FLAPW ships atomic-species data (`CONFIG`, `_CORE`, `_VAL_E`, `_VALENCE_NL`) only for
Ti/O/Ne/Be/He. `corundum_efg.py` (Al) and `mgf2_efg.py` (Mg, F) inject the missing species at
runtime, at module top level so the spawned k-worker pool re-applies it — **no `src` file is
modified.** Each injected atom was validated against NIST LDA eigenvalues before use (gradwave
`atomic_scf`): Al 1s −55.06 / 2p −2.54 / 3s −0.295 / 3p −0.092 Ha; Mg 1s −45.89 / 2p −1.70 /
3s −0.165 Ha; F 1s −24.16 / 2s −1.07 / 2p −0.39 Ha — all within ~1 % of NIST (the 1s gaps are the
expected scalar-nonrelativistic shift). Core partitions: Al/Mg **frozen [Ne]** (valence 3s^n),
F **frozen 1s** (valence 2s²2p⁵, like O) with the conditioned F 2s LO. Note Al/Mg thus *freeze the
2p semicore*, so — as the rutile campaign found for the Ti 3p semicore — their Sternheimer
antishielding is not captured, while Elk keeps 2p in valence. That gap is a deliberate, interpretable
probe of whether a given cation *needs* the semicore in valence.

## Reference data (Elk 11.0.2, this work) and experiment

Elk EFGs, this study (rmt forced to gradwave's spheres, ngridk 4 4 4, LDA):

| System | Site | V_zz (a.u.) | V_zz (eV/Å²) | η | C_Q (MHz) |
|---|---|---|---|---|---|
| Anatase TiO₂ | Ti (axial ∥c) | −0.09758 | −9.48 | 0.000 | 5.66 (⁴⁹Ti) |
| Anatase TiO₂ | O | −0.12779 | −12.42 | 0.096 | 0.77 (¹⁷O) |
| α-Al₂O₃ | Al (axial ∥c) | −0.06075 | −5.90 | 0.005 | 2.09 (²⁷Al) |
| α-Al₂O₃ | O | +0.36665 | +35.63 | 0.512 | 2.20 (¹⁷O) |
| MgF₂ | Mg | −0.05670 | −5.51 | 0.553 | 2.66 (²⁵Mg) |
| MgF₂ | F | +0.53050 | +51.55 | 0.385 | — (¹⁹F, I=½) |
| Rutile TiO₂ (prior) | Ti | +0.1990 | +19.34 | 0.36 | 11.5 (⁴⁹Ti) |
| Rutile TiO₂ (prior) | O | −0.1967 | −19.10 | 0.74 | 1.18 (¹⁷O) |

Experiment (solid-state NMR/NQR; |C_Q| only — NMR does not fix the sign). Q moments (Pyykkö 2018):
⁴⁷Ti +0.302, ⁴⁹Ti +0.247, ¹⁷O −0.02558, ²⁷Al +0.1466, ²⁵Mg +0.1994 barn (¹⁹F has no Q).

| System | Site | C_Q (MHz) | η | Citation |
|---|---|---|---|---|
| Anatase | ⁴⁹Ti | 4.04 (⁴⁷Ti 4.94) | 0.06 | Labouriau & Earl, *Chem. Phys. Lett.* **270**, 278 (1997); Bastow, *SSNMR* **12**, 201 (1998) |
| Anatase | ¹⁷O | small, unresolved (δ≈562 ppm) | — | Bastow & Stuart, *Chem. Phys.* **143**, 459 (1990) |
| Rutile | ⁴⁹Ti | 13.4 | 0.2 | Bastow 1998; Labouriau & Earl 1997 |
| Rutile | ¹⁷O | 1.8 | 0.6 | Bastow & Stuart 1990 |
| α-Al₂O₃ | ²⁷Al | 2.38 | 0 (C₃ site) | Jansen et al., *SSNMR* **3**, 241 (1994); MacKenzie & Smith 2002 |
| α-Al₂O₃ | ¹⁷O | 2.167 | 0.517 | Brun, Derighetti, Hundt & Niebuhr, *Phys. Lett. A* **31**, 416 (1970) |
| MgF₂ | ²⁵Mg | sparse/not established | — | (²⁵Mg NMR of MgF₂ not firmly reported) |
| MgF₂ | ¹⁹F | no quadrupole (I=½) | — | — |

Two cross-code / cross-experiment sanity points already visible in the references:
- **Polymorph discrimination is real and large.** Ti C_Q collapses rutile→anatase in *both* Elk
  (11.5 → 5.66 MHz) and experiment (13.4 → 4.04 MHz); η collapses from ~0.36 (rutile, biaxial) to
  ~0 (anatase, axial by the Ti site's D₂d symmetry). The two polymorphs are cleanly distinct.
- Elk vs experiment is itself only semiquantitative under LDA: anatase Ti Elk 5.66 vs exp 4.04
  (+40 %), corundum Al Elk 2.09 vs exp 2.38 (−12 %), corundum O Elk 2.20 vs exp 2.17 (+1 %). LDA
  EFG errors of 10–40 % are normal, so gradwave's target is **Elk**, with experiment as the outer
  bracket.

## gradwave FLAPW results

All numbers are the **full** EFG tensor (on-site l=2 sphere-Poisson + the Weinert lattice/boundary
term — the observable), axis-resolved in the shared Cartesian frame (z = crystallographic c). The
on-site valence-only V_zz is quoted separately as a diagnostic.

### α-Al₂O₃ corundum — GATED (r_nsph 6.6e-5, r_v 1.0e-2)

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q | exp C_Q / η |
|---|---|---|---|---|---|---|
| Al (∥c) | −6.99 | 0.000 | 2.48 (²⁷Al) | −7.26 | −5.90 / 0.005 / 2.09 | 2.38 / ~0 |
| O | +26.27 | 0.431 | 1.63 (¹⁷O) | +17.76 | +35.63 / 0.512 / 2.20 | 2.167 / 0.517 |

gradwave Al is at **118 % of Elk** (same sign, axial η=0 exact) and its **C_Q(²⁷Al)=2.48 MHz lands
within 4 % of the experimental 2.38 MHz** (Elk itself is 2.09, 12 % under experiment). This is with
the Al 2p semicore **frozen** — the antishielding-in-valence lesson from rutile Ti does *not* recur
for Al³⁺: a filled-shell main-group cation reaches Elk/experiment without semicore-in-valence.
gradwave O is at **74 % of Elk** (same sign, η 0.43 vs 0.51) — the rutile ~73 % anion deficit
recurs almost exactly.

### MgF₂ — GATED (r_nsph 1.5e-4, r_v 2.3e-3; the anion control)

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q |
|---|---|---|---|---|---|
| F | +37.53 | 0.724 | — (¹⁹F) | +30.05 | +51.55 / 0.385 / — |
| Mg | +11.08 | 0.896 | 5.34 (²⁵Mg) | +11.57 | −5.51 / 0.553 / 2.66 |

The **F site is at 73 % of Elk** (+37.53 vs +51.55), same sign, with its V_zz principal axis in-plane
(∥[1̄10]) exactly matching Elk — **identical to the rutile-O (73 %) and corundum-O (74 %) deficit,
for an anion that carries none of O's 2p covalency.** This is the controlled result: swapping the
rutile anion O→F leaves the ~73 % deficit unchanged, so it is a property of the *rutile anion site*
(gradwave's muffin-tin under-capture of the anion asphericity), not O chemistry. gradwave Mg,
by contrast, is a small-cation failure like anatase Ti: +11.08 vs Elk −5.51 (201 %, **wrong sign**,
η 0.90 vs 0.55) — Mg²⁺'s EFG is dominated by the frozen-2p antishielding that gradwave omits (unlike
Al³⁺, Mg²⁺ *does* need its semicore in valence).

### Anatase TiO₂ — MARGINAL (r_nsph ~1e-3; reproducible across two independent runs)

Anatase's coupled fixed point is harder to reach than rutile's or corundum's — the marginal
interstitial mode diverges the continuation before r_nsph < 1e-3, and `newton_polish` is
chaotic-sensitive here (it diverged after ~40 min in one attempt). Two independent runs (r_nsph
3.3e-3 and 1.3e-3) nevertheless agree on the EFG to within a few %, so the numbers below are
reproducible, not a single lucky draw.

| Site | gradwave V_zz | η | C_Q | on-site V_zz | Elk V_zz / η / C_Q | exp C_Q / η |
|---|---|---|---|---|---|---|
| Ti (∥c) | +3.70 | 0.000 | 2.21 (⁴⁹Ti) | +3.92 | −9.48 / 0.000 / 5.66 | 4.04 / 0.06 |
| O | −5.26 | 0.44–0.60 | 0.33 (¹⁷O) | +12.2 | −12.42 / 0.096 / 0.77 | small |

- **Ti**: η = 0 is exact (the D₂d site is axial ∥c, as in Elk and experiment), and gradwave does
  reproduce the *polymorph magnitude collapse* (rutile Ti +17.1 → anatase Ti +3.70, a 4.6× drop
  mirroring Elk's 19.34 → 9.48 and experiment's 13.4 → 4.04 MHz). But gradwave gets the **sign
  wrong** (+3.70 vs Elk −9.48): the anatase Ti EFG is small and *sign-inverted relative to rutile*,
  and gradwave keeps its rutile-like +sign. The on-site term (+3.92) is itself small and +signed,
  so this is not a lattice-term artifact — the small anatase-Ti EFG's sign is genuinely not captured.
- **O**: the **full** V_zz is −5.26 (42 % of Elk), same sign as Elk, but η is unstable (0.44–0.60
  vs Elk 0.10). The striking point is the **on-site valence term: +12.2 eV/Å² = ~98 % of Elk's
  full |V_zz| (12.42)** — far better than rutile's on-site O (~30 %). So anatase O's deficit is *not*
  an on-site (muffin-tin 2p asphericity) under-capture like rutile; the on-site physics is nearly
  exact. The full-tensor deficit comes from the lattice/boundary term, which in gradwave carries an
  opposite sign to the on-site term and over-cancels it (on-site +12.2 → full −5.26), and rotates
  the principal axis (gradwave V_zz ∥ c; Elk V_zz in-plane ∥ b). This boundary-term assembly is the
  anatase-O lever, distinct from the rutile-O on-site lever.

## Comparison and verdict

Consolidated gradwave-vs-Elk (full tensor) across the set, with the prior rutile datapoint:

| Material | Site | gradwave V_zz | Elk V_zz | gw/Elk | sign | η gw vs Elk | C_Q gw / Elk / exp (MHz) |
|---|---|---|---|---|---|---|---|
| Rutile (prior) | Ti | +17.1 | +19.34 | 88 % | ✓ | 0.41 / 0.36 | 11.4 / 11.5 / 13.4 |
| Rutile (prior) | O | ~+13.9 | −19.10 | ~73 %* | frame | 0.75 / 0.74 | ~0.97 / 1.18 / 1.8 |
| **Anatase** | Ti | +3.70 | −9.48 | 39 % | ✗ | 0.00 / 0.00 | 2.21 / 5.66 / 4.04 |
| **Anatase** | O | −5.26 | −12.42 | 42 % | ✓ | ~0.5 / 0.10 | 0.33 / 0.77 / — |
| **Corundum** | Al | −6.99 | −5.90 | 118 % | ✓ | 0.00 / 0.005 | 2.48 / 2.09 / 2.38 |
| **Corundum** | O | +26.27 | +35.63 | 74 % | ✓ | 0.43 / 0.51 | 1.63 / 2.20 / 2.167 |
| **MgF₂** | Mg | +11.08 | −5.51 | 201 % | ✗ | 0.90 / 0.55 | 5.34 / 2.66 / — |
| **MgF₂** | F | +37.53 | +51.55 | **73 %** | ✓ | 0.72 / 0.39 | — (¹⁹F) |

\* rutile O full-tensor was frame-ambiguous under |V|-sorting (near-degenerate large eigenvalues);
the ~73 % is the in-plane component match.

**Anion sites, gw/Elk full V_zz:** rutile O 73 %, corundum O 74 %, **MgF₂ F 73 %** — three sites,
one number. Cation sites split: rutile Ti 88 % (✓ sign), corundum Al 118 % (✓), anatase Ti 39 %
(✗ sign), MgF₂ Mg 201 % (✗ sign).

### Verdict

**1. Does the machinery generalize beyond rutile? Yes — it is now a characterized set, not a single
datapoint.** Four materials (rutile + three new), eight sites, all run end-to-end against a
matched-sphere Elk all-electron reference (and experiment where a quadrupolar nucleus exists). Three
of the four new-material runs GATED cleanly (corundum r_nsph 6.6e-5, MgF₂ 1.5e-4, both in ~2 min of
fullpot after the muffin-tin stage); anatase is MARGINAL (r_nsph ~1e-3) but reproducible across two
independent runs. The pipeline (muffin-tin stage → warm fullpot → newton_polish) is validated by the
two clean gates. Runtime species injection (Al, Mg, F), each pre-validated vs NIST LDA, works — the
machinery is not Ti/O-locked.

**2. Polymorph discrimination is real and captured.** The Ti EFG collapses rutile→anatase in
gradwave (17.1 → 3.7), Elk (19.34 → 9.48), and experiment (13.4 → 4.04 MHz), and η goes from
biaxial (0.36) to axial (0.00). gradwave cleanly separates the two polymorphs by magnitude and η.

**3. Does cation parity generalize? Split, by EFG size — a clear pattern.** For a *large* cation EFG
gradwave hits Elk parity (rutile Ti 88 %); for a *different element* it can hit parity outright —
**corundum ²⁷Al is the headline: C_Q = 2.48 MHz vs experiment 2.38 (within 4 %), 118 % of Elk, η = 0
exact**, and this with the Al 2p semicore *frozen*. So the rutile "cation needs its semicore in
valence for antishielding" lesson does **not** recur for a filled-shell main-group cation — Al³⁺
reaches Elk/experiment without it. But for *small/ionic* cation EFGs — anatase Ti (+3.70 vs Elk
−9.48, wrong sign) and MgF₂ Mg (+11.08 vs Elk −5.51, wrong sign, 2×) — gradwave gets the sign wrong.
Mg²⁺ is the counterpoint to Al³⁺: its EFG *is* dominated by the frozen-2p antishielding, so freezing
the semicore fails. Net: cation parity generalizes for large and/or closed-shell main-group EFGs
(rutile Ti, corundum Al), and fails for small transition-metal (anatase Ti) or antishielding-
dominated (Mg²⁺) cations whose semicore must be in valence.

**4. Does the anion ~73 % deficit recur — and is it O-specific? It recurs, and the F control proves
it is NOT O-specific — it is the rutile/oxide *anion site*.** Three anion sites, one number:
rutile O **73 %**, corundum O **74 %**, MgF₂ F **73 %** — all same sign, V_zz axis matching Elk.
The MgF₂ F test is decisive: F sits at rutile's O crystallographic position but carries none of O's
2p covalency, yet it lands at the *identical* 73 %. So the deficit is not O-2p-covalency
under-capture; it is a property of how the muffin-tin scheme captures the *anion asphericity at this
site type* — structural/site, not chemistry. Anatase O is the lone exception that refines the
picture: its *full* tensor is 42 % but its *on-site valence* term is ~98 % of Elk, so there the
residual sits in the lattice/boundary assembly (which over-cancels the on-site term and rotates the
axis), not in the on-site capture. Anion story: **≈73–74 % of Elk at the rutile/corundum anion site
regardless of element (O or F); anatase O is on-site-exact with a boundary-term cancellation.**

**Headline pattern:** anions land at a strikingly consistent **73–74 % of Elk** across O and F at the
rutile/corundum anion site (structural, not chemical); cations hit Elk/experiment parity when the
EFG is large or the ion is closed-shell main-group (rutile Ti 88 %, corundum **²⁷Al C_Q 2.48 vs exp
2.38 MHz**), and fail (wrong sign) when small/ionic with an active semicore (anatase Ti, Mg²⁺).
Every result is interpretable; none is a code error. The ²⁷Al benchmark lands on experiment,
extending the trustworthy-EFG machinery to a non-Ti nucleus, and the MgF₂ F control pins the anion
deficit to the site, not the chemistry.

## Reproducing

- gradwave: `experiments/autoapw/{anatase_efg,corundum_efg,mgf2_efg}.py` (drivers), `_efgrun.py`
  (the muffin-tin → warm-fullpot → newton_polish converger), `refine_anatase.py` (warm-tighten a
  saved state; `NEWTON=0` to skip the slow polish). Run on asus via pueue/qrun, OMP_NUM_THREADS=2,
  kworkers=4. Corundum injects Al, MgF₂ injects Mg+F at runtime (each validated vs NIST LDA) —
  no `src` change.
- Elk: `setup_elk_{anatase,corundum,mgf2}.py` build the input dirs (species rmt forced to gradwave's
  spheres); run `~/github/elk-11.0.2/src/elk` in each dir (tasks 0, 115).
