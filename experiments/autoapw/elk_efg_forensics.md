# Elk vs gradwave FLAPW EFG forensics — why rutile TiO2 comes out at −0.44×

Read-only source audit, 2026-08-20. Elk source: `asus:~/github/elk-11.0.2/src/`; run dir
`asus:~/tio2_efg/`; gradwave: `.claude/worktrees/autoapw-differentiable-surface/src/gradwave/flapw/`
(all paths below relative to those roots). No code was modified; no SCF was run. Small local
arithmetic in `scratchpad/pcsum.py`, `fit.py`, `fit2.py`.

## Executive verdict

The −0.44× is not a missing physical term in the two-term formula and not a literal sign
error in the assembly algebra (audited factor-by-factor: both the interior l=2 Poisson
coefficient and the v_bc/R² harmonic continuation are correct and isomorphic to Elk's
construction). It is the **content of the boundary (lattice) term**: gradwave's measured
boundary term equals **β× the true external field with a single negative β** across all
three symmetry-fixed principal axes of the O site. Fitting `ours = v_val + β·x`,
`Elk = λ·v_val + x` on the three O components gives an *exact* three-component fit at
(β, λ) = (−2.16, 3.07), and β = −1 fits within 5–10% at λ ≈ 1.5; no positive-β solution
exists once the sign of the external field x is pinned (independently, by a point-ion
lattice sum with Elk's own sphere charges: x ≈ [−13.3, +15.9, −2.6] eV/Å² on
[110]/[1̄10]/[001]). The mechanism that can produce a *coherently* inverted boundary term
without a visible `sign()` bug: the surface projection `v_bc` is dominated by the
**fictitious plane-wave-continuation density ρ_I inside the muffin-tin spheres** — (i) the
own sphere's inside-R continuation is *not* subtracted (only the own L>0 pseudocharge's
field is), and (ii) the neighbor Ti sphere's inside-R continuation is moment-corrected only
to L ≤ aug-lmax while the O surface sits 0.024 Å (support ratio 0.98) from the Ti sphere,
where uncorrected high-L moments radiate unattenuated — plus interstitial-XC contamination
(`v_grid = v_hart + vxc(ρ_I)`, absent from Elk's Coulomb-only `vclmt`). The fictitious
in-sphere electron density is concentrated along exactly the bond axes where the true
external field is dominated by the *negative* net Ti spheres, so the artifact field is
anti-parallel to the true field axis-by-axis — a natural, uniform negative β, then amplified
self-consistently by the fullpot SCF (the same self-field leak that produced the documented
runaway fixed point). **Cheapest decisive test (no SCF):** from a stored gated state,
(1) print the raw Cartesian O tensor — one line; Elk's O1 at (u,u,0) has T_xy = −0.1838 a.u.,
so gradwave T_xy > 0 confirms the inversion, T_xy < 0 kills it (and the story reverts to a
uniform ~0.44 magnitude deficit with a z-axis sign anomaly); (2) re-assemble
`_efg_from_multipoles` three ways: with `v_hart` instead of `v_grid` (isolates XC), with
`field(ρ_I·mask_ownR)` added to `vbc_own` (isolates the own self-term), and with the Ti-region
sources replaced by an analytic −3.17·E2/d monopole (isolates the neighbor near-field).

---

## Q1. Term-by-term accounting

### Elk's V_ij at the nucleus

`writeefg.f90`: EFG = ∂²(vclmt)/∂r_i∂r_j at r→0, with the l=m=0 rows of `vclmt` zeroed
(the strided `rfmt(i)=0` loops over `lmmaxi`/`lmmaxo` blocks), differentiated twice by
`gradrfmt` (spline), evaluated as `grfmt2(1,j)*y00` (the l=0 component of the second
gradient at the first radial point). Since v_lm(r) ~ r^l near the origin, only the **l=2
channel** of `vclmt` survives two derivatives at r→0 (l=0 removed; l=1 → linear → 0; l≥3 →
0). Requires `lmaxi ≥ 2` (the inner MT region carries only l ≤ lmaxi); the run used lmaxi=2.

`vclmt` is assembled in `potcoul.f90` → `zpotcoul.f90` and contains, in the l=2 channel:

1. **Own-sphere full density, particular solution** — `genzvclmt` applies the radial Green's
   function (the equation in `zpotcoul.f90`'s header:
   `V_lm(r) = 4π/(2l+1)[r^-(l+1)∫₀^r ρ_lm r'^(l+2) dr' + r^l ∫_r^R ρ_lm r'^(1-l) dr']`)
   to the *entire* `rhomt`: valence with all APW+LO cross terms to l ≤ lmaxo = 6, **plus the
   core** — but the core is strictly spherical (`gencore.f90`: `rhocr = Σ occ (f²+g²)·r⁻²·y00`,
   solved in the spherical `vsmt`; INFO.OUT core leakage 2.4×10⁻⁸ e), so the core contributes
   **zero** to l=2 directly. Semicore Ti 3s/3p are *valence* here (LOs, see Q4) and DO enter ρ_2M.
2. **Own nucleus** — `potcoul.f90` adds `vcln` (from `potnucl.f90`, point charge `zn=spzn=−Z`)
   to the **l=0 rows only** → zero l=2 → zero EFG contribution. Confirmed: own-nucleus term
   is identically zero by symmetry/structure, not approximately.
3. **Homogeneous (lattice/boundary) term** — `zpotcoul.f90`: sphere multipoles
   `qlm = (2l+1)/(4π)·R^(l+1)·zvclmt(R)` (read off the particular solution's surface value;
   because `vcln` was already added, the l=0 moment includes the nuclear charge — the doc's
   `z_ij Y_00 δ_l0`, "should be taken as negative"); the interstitial continuation's moments
   are subtracted (the `jlgprmt(l+1,ig)`/`zlm` loop); a Weinert pseudocharge of order
   `lnpsd = 6` (INFO.OUT "Constant for pseudocharge density: 6" ≈ ¼·R·Gmax) with those moment
   deficits is added in G-space; Poisson `V^P(G) = gclgp·ρ^P(G)`; then the homogeneous
   coefficient is `z1 = (zlm − zvclmt(R,lm))/R^l` continued inward as `z1·r^l`
   ("match potentials at muffin-tin boundary" block). The subtraction `− zvclmt(R,lm)` removes
   the **entire own-sphere multipole field analytically and exactly** (particular solution at
   R = 4π q_lm^MT/((2l+1)R^(l+1)), and the own-region grid sources — ρ_I continuation + own
   pseudocharge — sum to exactly q^MT by construction). What remains in z1 is the field of
   everything outside the sphere: other spheres' total charge (electrons + nuclei, via their
   pseudocharge moments, all l ≤ lmaxo=6), and the true interstitial density.
4. **Not present:** XC (vclmt is pure Coulomb); constant shifts (l=0 removed).

### gradwave's v_M^full

`efg.py:335-344` (`efg_tensor_full`): `v_M^full = v_M^val + v_bc_2M/R²`, with

1. **Valence particular term** `_valence_v` (`efg.py:254-257`):
   `v_M = (4πE2/5)∫ρ_2M/r dr` from the l=2 multipoles of the augmented in-sphere ψ
   (`sphere_density_multipoles_bands`, cross terms limited to l ≤ aug-lmax = 3–4; frozen
   spherical core → 0, same as Elk).
2. **Boundary term** (`scf.py:1686-1689`): `v_bc = Π_2M[v_grid at surface] − vbc_own`, where
   - `v_grid = v_hart + vxc_lda(ρ_I)` (`scf.py:758`) — **includes interstitial XC**;
   - `v_hart = fft_poisson(ρ_I + Σ spheres [monopole pseudocharge (net = q_sph − Z − q_i_in,
     `scf.py:726`) + Gram-exact L>0 pseudocharges with moments q^MT_L − q^I_L
     (`scf.py:741-748`)])`;
   - `vbc_own` = the surface projection of the own **L>0 pseudocharge's** field only
     (`scf.py:765-771`).
3. Own nucleus: inside `v_sph` as −Z·E2/r, l=0 only → no EFG term (`scf.py:785`). Other
   nuclei: inside the monopole nets → included. Same as Elk in effect.

### The difference list (the deliverable)

| # | Item | Elk | gradwave | Effect on the l=2 boundary term |
|---|------|-----|----------|-------------------------------|
| D1 | XC in the projected grid | `vclmt` is Coulomb-only (`potcoul.f90`) | `v_grid = v_hart + vxc_lda(ρ_I)` (`scf.py:758`) feeds both the SCF C_LM and the EFG (`scf.py:1686`) | Spurious. vxc is most negative along the bond channels → *negative*-on-bond l=2, i.e. same sign as the true external field: masks, does not cause, the inversion — but it is flatly wrong vs the EFG definition (second derivative of the **Coulomb** potential). |
| D2 | Own-field subtraction completeness | Analytic and exact: `z1 = (zlm − zvclmt(R))/R^l` removes the *total* own-region multipole field q^MT (ρ_I continuation + pseudocharge together) | Subtracts only the own L>0 pseudocharge's band-limited field (moment q^MT − q^I). The **fictitious ρ_I continuation inside the own sphere (l=2 moment q^I_2M,in) is retained** and continued inward as r² | Spurious self-term ∝ q^I_2M(in): the in-sphere PW density is enhanced along the bonds → *positive*-on-bond — anti-parallel to the true external field. Est. +2–4 eV/Å² on the O bond component alone. |
| D3 | Own l=0 monopole aliasing | Covered by the same analytic surface subtraction | `vbc_own` is L>0 only; the own monopole ball's grid-anisotropy l=2 leakage is retained | Small (~sub-eV/Å²) but nonzero at a low-symmetry site. |
| D4 | Neighbor-sphere near field | Sphere moments matched to l ≤ lmaxo = 6, pseudocharge order lnpsd = 6, Gmax = 12 bohr⁻¹ (grid 35×35×24, spacing 0.125 Å) | Moments matched only for L present in `rho_2m` (L ≤ aug-lmax = 3–4), npow = 4 ball, coarser grid | The O surface sits 0.0457 bohr = **0.024 Å** from the Ti sphere (support ratio 1.098/1.122 = 0.98). Multipole equivalence at that point requires *all* moments; each uncorrected high-L moment of the bond-lumped fictitious ρ_I inside the Ti sphere radiates onto the O patch attenuated only by (0.98)^L ≈ 1. Positive-on-bond again. Elk shares the geometry but corrects to L=6 with a smoother order-6 pseudocharge on a finer grid. |
| D5 | ρ_2M richness (valence term) | lmaxapw=8, density/potential to lmaxo=6, 5 LOs on Ti (incl. 3s/3p semicore), 3 LOs on O | aug-lmax 3–4, optional LOs | Elk's on-site valence term is λ ≈ 1.5–3× gradwave's (from the fit, Q3) — a magnitude deficit, not a sign issue; consistent with aug-lmax 3→4 growing |O| by 6% and fixing Ti's η. |
| D6 | Nuclei | Own l=0 (zero EFG); others via monopole moments | identical in effect | none |
| D7 | Core | Strictly spherical (`gencore.f90`), leakage 2×10⁻⁸ e | frozen spherical | none directly; semicore polarization only via LO-valence (both codes can) |

## Q2. Analytic check of gradwave's two factors

**(a) Interior l=2 Poisson — CORRECT.** `lx_sphere_poisson` (`efg.py:219-231`) is the
standard (2L+1)-denominator Green's kernel, identical to Elk's `zpotclmt` formula. As r→0
the inner integral term scales as r⁴ (ρ_2M ~ r² at the origin) and the r²-coefficient is
`(4πE2/5)∫₀^R ρ_2M/r dr` — exactly `_valence_v`. The multipole construction
(`sphere_density_multipoles*`, projection with conj(Y_LM) on a Gauss–Legendre grid exact for
the band-limited integrand) is convention-consistent with the projection used in
`interstitial_boundary_multi` and with `_tensor_from_v` (V_zz = √(5/π)·v₀ etc. — re-derived,
correct; both terms live in the same complex-Y coefficient space). The r<²/r>³ kernel is
handled by the two cumulative sums; no lower-limit defect.

**(b) Harmonic continuation — the FORM is correct; the CONTENT is the problem.** For an l=2
field whose sources are all outside R, the interior solution matching surface value v_bc is
v_bc·(r/R)²·Y_2M and the r²-coefficient is v_bc/R² — same as Elk's `z1·r^l` with
`z1 = (·)/R^l`. Both codes agree. The requirement is that v_bc contain *only* external
sources. Elk guarantees this analytically (exact q^MT surface subtraction, D2). gradwave's
numerical replacement subtracts the own L>0 pseudocharge's *actual band-limited* field —
which correctly kills the ~20% pseudocharge-aliasing residue the log documented — but the
subtracted object has moment content q^MT − q^I, not q^MT: **the fictitious inside-R ρ_I
(moment q^I_2M,in) stays in v_bc** and is then continued inward with the (r/R)² profile,
which is doubly wrong (fictitious source, and interior sources don't continue as r²). Error
sign at O: positive-on-bond (electron density in the bond channels), i.e. **opposite** to
the true external field; magnitude grows with the covalency of the converged density and is
amplified by the SCF (the aspherical density polarizes in whatever C_LM field it is given).

## Q3. The −0.44× hypothesis test

**Axes.** The O site (Wyckoff 4f, C2v) has symmetry-fixed principal axes [110] (the in-plane
Ti–O bond), [1̄10], [001]. Elk's EFG.OUT tensor for O1 at (u,u,0) is
[[−0.01283, −0.18383, 0], [−0.18383, −0.01283, 0], [0, 0, +0.02566]] a.u. → eigenvalues
**[110]: −0.1967 (−19.1 eV/Å²), [1̄10]: +0.1710 (+16.6), z: +0.0257 (+2.5)**. The gradwave
logs never recorded eigenvectors — only |·|-sorted eigenvalues — so the componentwise
pairing (and hence the "−0.44") was an assumption. Everything below treats both assignments.

**The true external field x (point-ion lattice sums, `pcsum.py`, converged to <0.1% at
Rc=80 Å, electron-potential convention).** With Elk's own sphere populations
(net Ti = 18.83−22 = −3.17 e, net O = 7.67−8 = −0.33 e, uniform-background neutralized):
x(O) = [**−13.3** on [110], **+15.9** on [1̄10], −2.6 on z] eV/Å²; formal ±4/∓2 charges give
[−15.0, +18.8, −3.7]. x(Ti) is small: [−3.4 on z, +1.8, +1.6] (nets) or [+0.78, −0.52,
−0.26] (formal). Two consequences: (i) the external term at O is comparable to Elk's total —
Elk_total ≈ x + (gradwave's own valence-only MT value [−4.31, +3.73, +0.58]) on the two
large components (−17.6 vs −19.1; +19.6 vs +16.6); (ii) **sign(x_bond) < 0 is pinned**.

**The fit (`fit2.py`).** Model: ours = v_gw + b (b = the boundary term actually applied),
Elk = λ·v_gw + x, b = β·x; v_gw = the MT-only valence [−4.31, +3.73, +0.58]
(axis assignment justified by O-2p physics: σ-depletion along the bond → negative on [110],
lone-pair excess on [1̄10] → positive — the same pattern as Elk's total and the external field).

- Assignment A1 (+8.33 on the bond): b = [+12.64, −11.10, −1.55]. Best fit:
  **β = (−2.16, −2.16, −2.16) at λ = 3.07 — an exact three-component fit with one β**
  (x = [−5.85, +5.13, +0.72], sign-compatible with the screened external field; the z sign
  flip vs the bare point-ion sum is exactly where smooth-density screening matters — Elk's
  total z is +2.5). The constrained β = −1 fits at λ = (1.50, 1.47, 1.64), i.e. within
  5–10%. The observed ratio follows: with x ≈ 2.9·v_gw componentwise (ratios 2.93/2.98/2.67
  — itself strikingly uniform), ours = v − x·|β|′ and Elk = λv + x give −0.43 for
  (β,λ)=(−1,1.5) — the measured −0.436/−0.444/−0.388.
- Assignment A2 (−7.37 on the bond, signs matching Elk axis-wise): **no uniform β exists**
  (best relative spread 4.4; the z component alone would need β = −5 while the in-plane need
  +1.1/+1.9). A "uniform 44% deficit" story is internally inconsistent — the z component
  (−0.97 vs Elk +2.5, on the axis that pairs unambiguously) already requires a sign-inverted
  boundary contribution.
- No positive-β solution exists for any λ < 4.4; λ ≥ 4.4 would require the external field to
  be *positive* along the bond, contradicting the point-ion sum. **β < 0 is robust.**

**Candidates evaluated:**
- Own nucleus: identically zero in both codes (l=0 channel only). Confirmed, not a candidate.
- Other nuclei missing from v_bc: **no** — they are in the monopole nets (`scf.py:726`), and
  the far-field sign of v_hart around Ti sites is correct (negative, attractive to electrons).
  A point nucleus needs no band-limited representation *of itself*; only its sphere's net
  moment matters outside R — which is present. The *near-field delivery* of that moment at a
  0.024 Å clearance is D4, a real error channel, but not an omission.
- Ti 3p semicore: real for Ti's magnitude (D5) but cannot touch O; the clean −0.44 lives on O.
- XC contamination (D1): wrong direction (it *adds* Elk-signed l=2); masks, does not cause.
- **Sign-inverted boundary content: the winner.** Not as a literal code sign error — the
  assembly algebra audits clean (projection conventions, Poisson sign, min-image direction,
  fft conventions all verified) — but as D2 + D4: the retained fictitious in-sphere ρ_I
  fields are anti-parallel to the true external field on every axis (positive electron lumps
  in the bond channels vs negative net cation spheres along the same axes), so
  b ≈ (own valence-tail physics) + (anti-parallel artifact) ≈ −|β|·x coherently. The SCF then
  polarizes the aspherical density in this inverted field (the LO/aspherical coupling exists
  precisely to let it respond), which is how a messy artifact becomes an *exactly
  proportional* −2.2× — and why the same loop previously exhibited a structure-preserving
  runaway fixed point (log: "LEAK CONFIRMED", "runaway", gain > 1).
- Consistency with the new data point: aug-lmax 3→4 grows |O| by 6% without a flip — the
  valence term (and the artifact, both density-driven) grow together; growth-away-from-zero
  is exactly what a proportional-inversion story predicts (and what a
  cancellation-through-zero story forbids).
- Ti: x(Ti) is small (±2–3), v_gw(Ti) small (~−3 MT-only, ~1 fullpot), so the inversion
  contributes proportionally little; Ti's gap to +19.3 is D5 (on-site semicore/valence
  richness), consistent with η snapping to Elk's 0.38 at aug-lmax=4 while the magnitude stays.

**Test-suite blind spot (why this survived):** every existing gate is sign-blind — cubic/Ne
nulls are zero by symmetry regardless of the boundary term's sign; the single-atom test
measures own-field *cancellation*, not external-field *sign*; η is scale- and sign-invariant.
No test compares the boundary term against an analytic external field.

**Decisive experiments (no SCF; from a stored gated state or the next run):**
1. Print `i["efg"]["a2"]["tensor"]` (one line). Prediction if the inversion story is right:
   T_xy(O1 at (u,u,0)) **> 0** (Elk: −0.1838 a.u.). If T_xy < 0, A2 was the real pairing and
   the story becomes uniform-deficit + a z-anomaly — then D4/D5 lead instead.
2. Unit test: one sphere, ρ_2m = 0, plus a smooth external charge q at d in a box (grid
   density, no SCF): `efg_tensor_full` boundary part vs analytic 2qE2/d³. Prediction: passes
   with correct sign at large d (the algebra is fine), degrades/inverts as d → R_sphere gap
   scales (D4), which localizes the error to near-field content.
3. Three re-assemblies of `_efg_from_multipoles` on the same state: (a) `v_hart` instead of
   `v_grid` → measures D1; (b) `vbc_own += Π[fft_poisson(ρ_I·mask_own-R)]` → measures D2;
   (c) replace the Ti-region sources by an analytic point −3.17 monopole field → measures D4.
   Prediction: (b)+(c) move the O boundary term from ≈ +12.6 to ≈ −6…−13 on the bond axis
   and the total to Elk's side of zero; (a) is an eV-scale correction on top.

## Q4. Elk basis inventory (the match checklist)

From `~/tio2_efg/elk.in`, `INFO.OUT`, `Ti.in`, `O.in`:

- **Cell/geometry:** a = 8.68083, c = 5.59096 bohr, u = 0.3048; R_MT: Ti 2.0754 bohr
  (1.0983 Å), O 1.5566 bohr (0.8237 Å); point nuclei; xctype 3 (PW92 LDA); ngridk 4×4×6
  (24 IBZ k-points); no smearing specified (defaults).
- **Basis/resolution:** rgkmax = 7 (average-radius rule → |G+k|max = 4.047 bohr⁻¹);
  lmaxapw = 8; lmaxo = 6 (density & potential); lmaxi = 2 (inner MT); Gmax(ρ,V) = 12 bohr⁻¹,
  G-grid 35×35×24 (12305 G-vectors, spacing ≈ 0.125 Å); pseudocharge order lnpsd = 6;
  46 local orbitals total, 49 valence states.
- **Ti (spzn = −22):** core = 1s, 2s, 2p (10 e); valence = 3s, 3p, 3d, 4s (12 e). APW:
  order 1 (single u_l at E = 0.15 Ha, no l-exceptions, l = 0…8). LOs (5):
  l=0 [u+u̇ @0.15]; l=1 [u+u̇ @0.15]; l=2 [u+u̇ @0.15]; l=0 semicore [u@0.15 ⊕ u@−2.2872 Ha
  (fixed E)]; l=1 semicore [u@0.15 ⊕ u@−1.4164 Ha]. **Ti 3s/3p are valence via dedicated
  semicore LOs.**
- **O (spzn = −8):** core = 1s only (2 e); valence = 2s, 2p (6 e). APW order 1. LOs (3):
  l=0 [u+u̇ @0.15]; l=1 [u+u̇ @0.15]; l=0 [u@0.15 ⊕ u@−0.8727 Ha] (a 2s-anchored LO).
  **O carries no l=1 extra-energy LO** — the running gradwave probe's O l=1 LO at +5 eV has
  no Elk counterpart; Elk's O LOs sit at the same 0.15 Ha linearization energy plus the 2s
  binding energy, not above the valence window.
- **Charge state (for the point-ion model):** Ti sphere 18.83 e (net −3.17), O sphere 7.67 e
  (net −0.33), interstitial 7.67 e.
- What gradwave would need to match: density/potential harmonics to L=6 (aug-lmax 4–5 in
  flight is still short, though the EFG's direct need is the l=2 density and the l≤6 moments
  for the *neighbor* fields, D4); APW l to 8; Ti 3s/3p LOs (mode exists); Gmax ≈ 12 bohr⁻¹
  grid for the pseudocharge/boundary work; Weinert order ~6 pseudocharges.

## Q5. External tiebreaker (literature)

- **⁴⁷/⁴⁹Ti, rutile (experiment):** C_Q(⁴⁹Ti) = 13.4 MHz, η ≈ 0.19–0.2 (single-crystal NMR:
  |V_zz| = 2.2±0.1 ×10²¹ V/m² ≈ 22 eV/Å², η = 0.19±0.01 — Kanert/Kolem-type single-crystal
  data as summarized in the ⁴⁷,⁴⁹Ti hyperfine-interactions literature); ⁴⁷Ti scales by
  Q(47)/Q(49) = 1.223 → ≈ 16.4 MHz. Elk's Ti (19.34 eV/Å² → C_Q(⁴⁹Ti) = 11.5 MHz) is ~86% of
  experiment with the right order and axes — **Elk is a sane target**; gradwave's Ti (~1
  eV/Å²) is far off both. η(exp) = 0.19 vs Elk 0.36: Elk overestimates η but not magnitude.
- **¹⁷O, rutile (experiment):** detectable by MAS (Bastow & Stuart, "¹⁷O NMR in simple
  oxides", Chem. Phys. 1990 — detection limited to C_Q < 4 MHz); the campaign log's
  C_Q(¹⁷O) = 1.8 MHz, η = 0.6 could not be independently re-verified in this session's
  searches (flagged, not contradicted). Elk gives C_Q(¹⁷O) = 1.18 MHz, η = 0.74 — same
  order as the quoted experiment; gradwave's 44%-scale O value gives ~0.5 MHz.
- **Sign:** NMR measures |C_Q| only; no experimental sign determination for host Ti/O in
  rutile was found (TDPAC exists only for Cd/Ta impurity sites). The sign question is
  therefore settled *between calculations*: Elk's O V_zz negative-along-bond in the
  electron-potential convention is what the point-ion lattice field, the O 2p σ-depletion
  argument, and WIEN2k-family rutile EFG studies (e.g. the WIEN2k-based temperature-dependence
  study of the Ti EFG in rutile, arXiv:2011.05629) all support. Nothing in the literature
  supports a positive-along-bond O EFG of gradwave's sign.

Sources: [VASP wiki EFG](https://www.vasp.at/wiki/index.php/Calculating_the_electric_field_gradient),
[EFG database, Sci. Data](https://www.nature.com/articles/s41597-020-00707-8),
[⁴⁷,⁴⁹Ti QCPMG](https://www.sciencedirect.com/science/article/abs/pii/S1090780705003332),
[⁴⁷,⁴⁹Ti hyperfine interactions in oxides and metals](https://www.sciencedirect.com/science/article/abs/pii/S0926204098000666),
[TiO₂ ssNMR review](https://www.oaepublish.com/articles/cs.2024.12),
[Ti EFG in rutile, ab initio T-dependence](https://arxiv.org/pdf/2011.05629),
[¹⁷O NMR in simple oxides](https://www.sciencedirect.com/science/article/abs/pii/0301010490870257),
[TDPAC Cd/Ta in TiO₂](https://arxiv.org/pdf/0809.0647),
[⁴⁷,⁴⁹Ti NMR of TiO₂ gels](https://pubs.acs.org/doi/10.1021/cm990248r).

## Appendix: verified sign conventions (both codes identical)

Electron-potential-energy convention throughout both codes: electron density positive,
`fft_poisson`/`genzvclmt` give positive Hartree for positive ρ; nuclear terms negative
(`potnucl` with zn = spzn = −Z; gradwave `−Z·E2/r`). Elk's qlm nuclear inclusion is negative
("should be taken as negative", `zpotcoul.f90` doc). gradwave conventions:
fftn/ifftn pairing in `interstitial_boundary_multi` reconstructs V(c+RΩ) with the correct
+iG·r phase; `_min_image_vec` points center→grid-point; multipole and boundary projections
both use conj(Y_LM); `_tensor_from_v` verified against ∂²(r²Y_2M). No algebraic sign slip
found anywhere in the assembly chain — which is precisely why the content-level inversion
(D2/D4) is the operative hypothesis.

---
## ADDENDUM (post-decisive-test, asus job 148): executive verdict FALSIFIED, difference list stands
The proposed T_xy test came back -7.915 eV/A^2 — same sign as Elk (-17.86), ratio 0.443.
The boundary term is NOT anti-parallel; the beta=-2.16 fit was built on the eigenvalue
pairing that the raw tensor now disproves (the "alternate axis pairing" this report
rejected is reality — its inconsistency argument failed because the [001] component is
small and trace-forced, not independently informative). What survives: the term-by-term
difference list (D1-D5), now read as MAGNITUDE mechanisms for the ~2-2.5x in-plane
deficit, with D4 (near-field moment truncation at the 0.024 A gap) and D5 (density/basis
richness) promoted to prime suspects; and the basis checklist + literature anchors
(Elk remains the right target vs experiment).
