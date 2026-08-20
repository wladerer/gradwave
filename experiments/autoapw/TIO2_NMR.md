# TiO2 rutile solid-state NMR (EFG / C_Q) — validation target

## Structure (experiment)
P4_2/mnm (#136). a=b=4.5937 Å (8.68083 Bohr), c=2.9587 Å (5.59096 Bohr), u=0.3048.
Ti (2a): (0,0,0), (1/2,1/2,1/2).  O (4f): (u,u,0), (-u,-u,0), (1/2+u,1/2-u,1/2), (1/2-u,1/2+u,1/2).

## Elk 11.0.2 reference (my run on asus, LDA/xctype 3, ngridk 4 4 6, rgkmax 7, tasks 0 115, lmaxi 2)
run dir asus:~/tio2_efg . EFG in a.u. (Ha/bohr^2); ×97.174 -> eV/Å^2.
| site | V_zz (a.u.) | V_zz (eV/Å^2) | eta | C_Q (MHz) |
|------|-------------|---------------|-----|-----------|
| Ti   | 0.1990      | 19.34         | 0.36| 11.5 (49Ti, Q=0.247 b) |
| O    | 0.1967      | 19.1          | 0.74| 1.18 (17O, Q=-0.02558 b)|
Both large V_zz (~19 eV/Å^2) — well above the muffin-tin noise floor (unlike BeO 1.67).

## Experiment (solid-state NMR/NQR)
49Ti: C_Q 13.4 MHz, eta ~0.2.  47Ti: C_Q ~16.4 MHz (=1.223×49Ti).  17O: C_Q 1.8 MHz, eta 0.6.
Q (Pyykko 2018): 47Ti +0.302 b, 49Ti +0.247 b, 17O -0.02558 b.
C_Q[MHz] = 2.4180 · Q[barn] · V_zz[eV/Å^2].

## Known gotcha
Ti 3p semicore contributes ~comparably to valence for the Ti EFG; frozen-3p LAPW gives a poor
Ti EFG. Elk treats 3p as valence (local orbitals) by default. gradwave currently freezes [Ar]
(incl. 3p) -> expect the Ti EFG to be OFF until 3p is moved to valence (local orbitals = TODO).
The O EFG (2p valence) is the more reachable first match.

## gradwave call (a_bohr in Bohr; radii in Å)
a_bohr = [8.68083, 8.68083, 5.59096]; radii Ti 0.95, O 0.80 Å (Ti-O min bond 1.946 Å, no overlap).

## gradwave results (progress log)
Elk ref: Ti V_zz=19.34 eta=0.36 ; O V_zz=19.1 eta=0.74 (eV/Å^2).

- frozen [Ar] (incl 3p), ecut250 k(2,2,3): Ti V_zz=-3.93 eta=0.33 ; O V_zz=+13.65 eta=0.39. TiO2
  converges (~83s). O correct sign/order but low; Ti wrong sign.
- **3p in valence** (l=1 linearized at 3p), same settings: O V_zz=+16.06 eta=0.63 (Elk 19.1/0.74)
  -> O clearly improved by the Ti-3p-in-valence field. Ti V_zz=-3.61 still wrong sign.

Physics: the O EFG is directly O-2p-valence-driven -> muffin-tin captures it (converging toward Elk
with ecut/k + the 3p-in-valence Ti field). The Ti EFG (Ti4+ ~d0) comes from the 3p semicore
POLARIZING in the non-spherical crystal field (Sternheimer antishielding), which exists only in a
FULL-POTENTIAL run (spherical muffin-tin has no l=2 field for 3p to respond to). => Ti needs
fullpot=True (+ 3p in valence). O is the reachable muffin-tin NMR result; Ti needs fullpot.

## Full results table (gradwave vs Elk: Ti +19.34/0.36, O -19.1/0.74 eV/Å^2)
| variant | ecut/k/smear | Ti V_zz/eta | O V_zz/eta | O C_Q(17O) |
|---|---|---|---|---|
| frozen [Ar]       | 250 / 2,2,3 / 0.15 | -3.93 / 0.33 | +13.65 / 0.39 | 0.85 |
| 3p-valence        | 250 / 2,2,3 / 0.15 | -3.61 / 0.03 | +16.06 / 0.63 | 0.99 |
| 3p-val muffin-tin | 400 / 4,4,6 / 0.15 | -3.67 / 0.44 | -13.81 / 1.00 | 0.85 |
| fullpot 3p-val    | 250 / 2,2,2 / 0.15 | -3.16 / 0.11 | +15.95 / 0.72 | 0.99 |
| fullpot+antishield| 250 / 2,2,2 / 0.05 | -3.23 / 0.07 | +12.83 / **0.745** | 0.79 |
Elk R_MT: Ti 2.075 bohr (1.098 Å), O 1.557 bohr (0.824 Å); mine were Ti 0.95/O 0.80 (O nearly matched).

## Findings
1. **O (17O) EFG is a genuine result**: fullpot + the lattice-field (antishielding) term gives eta=0.745,
   matching Elk's 0.74 exactly; V_zz ~13 eV/Å^2 (~70% of Elk 19.1), C_Q ~0.8 MHz (Elk 1.18, exp 1.8),
   right (negative) sign. The ~30% magnitude gap is NOT R_MT (O radii nearly matched) — it is the
   simplified muffin-tin/lmax=2/LDA-selfconsistency vs Elk's full implementation.
2. **Ti EFG stays wrong** (~-3.2 vs Elk +19.3) across muffin-tin, fullpot, and fullpot+antishielding.
   Ti4+ is d0, so its EFG is Sternheimer antishielding of the lattice field, dominated by the 2p CORE
   polarization. The frozen 2p core (computed spherically) CANNOT respond -> the dominant term is
   missing. 3p-in-valence + the l=2 lattice field capture only the (small) 3p response. The remaining
   piece is a core-Sternheimer / non-spherical-core response (DFPT for the frozen core) — the known
   reason Ti EFG is a hard FLAPW case. Groundwork (3p-valence, fullpot, lattice field) is in place.

## Next step (designed, not yet implemented): core-Sternheimer for the Ti EFG
The missing piece is the frozen 2p (and 3p) core's l=2 polarization response to the crystal field.
Per core state u_c=rR_{n l_c} (energy E_c) and each l=2 potential V_{2M}(r), the first-order radial
response delta-u_{l'} (l' in {l_c-2, l_c, l_c+2}, dominant l'=l_c for p->p) solves the inhomogeneous
Sternheimer equation
    [H_{l'} - E_c] delta-u_{l'}(r) = -G^{l'm'}_{2M,l_c m_c} V_{2M}(r) u_c(r),
    H_{l'} = -hbar^2/2m (d^2/dr^2 - l'(l'+1)/r^2) + V_sph(r),  (Gaunt G).
Solve by the radial Green's function G(r,r')=u_reg(r_<) u_irr(r_>)/W from two homogeneous solutions of
(H_{l'}-E_c)u=0 (numerov). The response l=2 density delta-rho_{2M} = sum_core 2 occ_c u_c delta-u_{l'}
(Gaunt-weighted)/(4 pi r^2) adds to rho_2m before the EFG. Validation gates before trusting it:
(a) cubic Ne fullpot null stays ~0; (b) Ti V_zz moves toward Elk +19.34 with correct sign; (c) the O
result is not degraded. This is the standard FLAPW/Blaha-Schwarz antishielding; it is the reason a
d0-cation EFG needs full-potential + core response, and is a research-grade addition to build+validate
carefully (not overnight without a reference for intermediate values).

## Headline run + HONEST convergence caveat (must not over-claim)
fullpot + Elk-matched R_MT (Ti 1.098/O 0.824 Å), ecut350, k(3,3,3), 40 iters, sm0.05:
  Ti V_zz=-3.95 eta=0.016 ; O V_zz=+8.59 eta=0.658 C_Q(17O)=0.53 MHz.
This is WORSE than the k222 run (-12.8/0.745/0.79) I first reported. Across all settings the O V_zz
ranges +8.6..±16 eV/Å^2 with UNSTABLE SIGN (its two large eigenvalues are near-equal magnitude, like
Elk's -19.1/+16.6, so the |V_zz| assignment flips), eta 0.66..0.75, C_Q 0.5..1.0. => the O EFG is NOT
k-converged/stable in this scheme. Honest status: gradwave's fullpot FLAPW RUNS end-to-end on a TM
oxide and yields EFG tensors of the RIGHT ORDER with eta in Elk's range (0.74), but it is a
qualitative-to-semiquantitative result (|V_zz| ~45-70% of Elk, sign unstable), NOT a tight match. Needs
proper k-convergence (k444+, each fullpot run ~35 min) + likely higher lmax before any quantitative
claim. Ti remains wrong (core-Sternheimer). Do not claim "matches Elk"; claim "working pipeline,
right order, eta in range, not yet converged".

## KEY DIAGNOSIS (k-sweep, 2026-08-19): smearing was the instability; sm=0 gives correct SHAPE
With smearing>0 on TiO2 (marginal LDA gap) the EFG was unstable/garbage (O evals jumped e.g.
k222 [-5.3,+3.1,+2.2] vs k223 [+12.5,-6.3,-6.2], e_fermi jumping 6.4->4.5 eV) — partial occupation
across the gap differs per k-mesh. WITH smearing=0 (exact insulator fill, 22 bands), muffin-tin k446
gives O evals [-4.31,+3.73,+0.58] **eta=0.730 (=Elk 0.74)** — the SHAPE is right and stable. BUT the
MAGNITUDE is only |V_zz|=4.3 vs Elk 19.1 (~22%). So the real deficit is MAGNITUDE not shape: muffin-tin
under-captures the O 2p asphericity + antishielding (likely over-ionizes O2- toward spherical 2p6, cf
BeO). Fullpot adds response (|V_zz| ~7-12) but must also use smearing=0 to be stable. ALWAYS use
smearing=0 for TiO2 (insulator); the earlier "eta=0.745" fullpot number was a smeared partial-occ fluke.
Remaining question: does fullpot+sm0 give correct eta AND larger (toward 19) magnitude, and does the
muffin-tin magnitude converge up with k.

## sm=0 sweep results (2026-08-19) — the proper (stable) numbers
All with Elk R_MT (Ti 1.098/O 0.824 A), smearing=0 (essential; TiO2 is an insulator).
| variant           | ecut/k     | O evals (eV/A^2)          | eta   | C_Q(17O) |
|-------------------|------------|---------------------------|-------|----------|
| muffin-tin        | 350/2,2,3  | [-5.33,+4.54,+0.78]       | 0.706 | 0.330 |
| muffin-tin        | 350/4,4,6  | [-4.31,+3.73,+0.58]       | 0.730 | 0.267 |
| **fullpot+antishield** | 300/3,3,3 | [-14.09,+13.52,+0.57] | 0.919 | 0.872 |
| Elk 11.0.2        | 350/4,4,6  | [-19.10,+16.60,+2.50]     | 0.740 | 1.18  |
KEY: muffin-tin gets eta right (~0.73=Elk) but |V_zz| only ~4-5 (22% of Elk) — under-captures the O
2p asphericity/covalency. FULLPOT + the lattice-field antishielding lifts |V_zz| to ~14 (74% of Elk),
C_Q 0.87 vs Elk 1.18 (74%) — a genuine semiquantitative result — BUT eta rises to 0.92 (too axial: the
smallest principal value +0.57 vs Elk +2.5; the response amplifies the two large components more than
the third). So current honest status: O C_Q ~74% of Elk with sm=0+fullpot, right order + right sign;
eta too axial. Remaining: k-convergence (fp446 running), and the response-anisotropy (eta) gap.

## CONVERGENCE TOWARD ELK (fullpot+antishielding, sm=0, Elk R_MT, ecut300~rgkmax7.3) — the real result
| k-mesh | O evals (eV/A^2)        | eta   | C_Q(17O) | % of Elk C_Q |
|--------|-------------------------|-------|----------|--------------|
| 3,3,3  | [-14.09,+13.52,+0.57]   | 0.919 | 0.872    | 74%          |
| 4,4,6  | [+15.72,-14.35,-1.37]   | 0.826 | 0.972    | 82%          |
| Elk    | [-19.10,+16.60,+2.50]   | 0.740 | 1.18     | 100%         |
ALL THREE components move MONOTONICALLY toward Elk with k: |V_zz| 14.1->15.7 (->19.1), eta 0.92->0.83
(->0.74), C_Q 0.87->0.97 (->1.18), 3rd eigenvalue |0.57|->|1.37| (->2.5). Systematic convergence to the
reference = real validation. Muffin-tin (no response) stuck at |V_zz|~4 (k223 5.3, k446 4.3, k668 4.0)
= 21% -> the antishielding response is ESSENTIAL and correct in direction. At Elk's own mesh (4,4,6):
82% of Elk on |V_zz| AND C_Q, still improving. ecut300 already ~rgkmax7.3 > Elk 7, so not ecut-limited;
residual is higher k + method (lmax=2, frozen O core response). HONEST HEADLINE: gradwave 17O C_Q(TiO2)
= 0.97 MHz at Elk's k-mesh (Elk 1.18, exp 1.8), converging toward Elk as k increases. sm=0 is essential.

## WALK-BACK (2026-08-19): the fullpot O magnitude is lmax-UNSTABLE — "82%" was likely an lmax=2 artifact
CRITICAL: fullpot O |V_zz| at k333 is 14.1 (lmax=2) but 4.4 (lmax=3) — a 3x collapse, NOT convergence.
The lmax=3 value (4.4) matches the muffin-tin (~4.0). So the k-convergence toward Elk I reported was
converging in k at FIXED lmax=2, but lmax itself is not converged: the lmax=2 fullpot amplification to
~14 is very likely a TRUNCATION ARTIFACT of the l=2-only non-spherical potential coupling only l<=2
channels. The RELIABLE, lmax-consistent result is the MUFFIN-TIN: |V_zz| ~4-5 (stable across k: 5.3,
4.3, 4.0), eta ~0.61-0.73 (right shape), = ~21-26% of Elk 19.1. So the honest O result is: correct
SHAPE (eta~0.7=Elk), magnitude ~1/4 of Elk (large deficit), and the fullpot "response" that lifted it
was an lmax artifact. Testing lmax=4 (decisive: if ~lmax3 then fullpot converges to ~4-5 and lmax2=14
was the artifact; if different, fullpot is just lmax-broken). LESSON: always check lmax convergence for
a full-potential EFG; a single lmax is not enough. My non-spherical potential is l=2-only (Elk uses
lmax~8); mismatched potential-lmax vs augmentation-lmax is the likely instability source.

## lmax-CONVERGED verdict (decisive): fullpot O |V_zz| = lmax2 14.1 / lmax3 4.4 / lmax4 4.5 -> ~4.5
lmax3 ~= lmax4 => fullpot CONVERGES in lmax to |V_zz|~4.5, eta~0.76 (=Elk 0.74). lmax2=14 was the
artifact. FINAL HONEST O RESULT: eta correct (0.76 vs Elk 0.74) but MAGNITUDE ~4.5 = 24% of Elk 19.1,
C_Q(17O) ~0.28 MHz (Elk 1.18, exp 1.8). The fullpot antishielding does NOT lift the magnitude at
converged lmax (~4.5 = muffin-tin ballpark). The ~4x magnitude deficit is real: under-captured O 2p
asphericity/covalency (my LAPW over-ionizes O2- toward spherical 2p6, cf BeO; Elk's LOs+high-lmax
capture more covalency). muffin-tin is itself lmax-sensitive too (k446: lmax2 4.31, lmax3 7.41). NET:
gradwave FLAPW reproduces the EFG SHAPE (eta) of a reference code but under-estimates the MAGNITUDE ~4x
for this oxide — a genuine method/basis limitation, not a convergence knob. Ti also fails (core response
missing). This is the honest state; earlier "82% converging to Elk" was an lmax=2 artifact, retracted.

## Accuracy build-out (2026-08-19, session 2): high-L potential + LAPW+LO
Implemented + committed (all gates green so far):
1. **General-L non-spherical potential** (3f1f18b): sphere_pseudocharge_lm (moments exact for
   L=1,3,4), lx_sphere_poisson, interstitial_boundary_multi, nonspherical_potential(lset),
   _weinert_multi matches every L>0, crystal_scf_multi(fullpot_lmax=). Odd L included (O 4f site
   has no inversion -> L=1,3 fields polarize O 2p; these were entirely missing before).
2. **LAPW+LO local orbitals** (f23ee2b): confined phi=a·u+b·u̇+c·u(E2), phi(R)=phi'(R)=0; secular
   extension + LO-aware density/multipoles; crystal_scf_multi(los=, val_e=, core=, el_override=).
   Gates: Ne no-LO bit-consistent (22.6235), redundant-LO shift -0.001 eV, no ghosts, null ok.
3. **LO aspherical coupling** (0a6faf8): _nonspherical_augment_lo — the semicore's response channel
   to the crystal field (the antishielding). Null preserved (-1.4e-6 @ fp_lmax=4).
Ti LO treatment (runner "lo" mode): 3s/3p semicore LOs, core=1s2s2p, val_e=12, El(l=1)->3d region.
In flight on asus: hl_l3/hl_l4 (is the fullpot O EFG now lmax-STABLE with fp_lmax=4?), lo_fp/lo_mt
(semicore-LO A-side, launched pre-coupling). Next: B-side full treatment (fullpot fp4 + LO + coupling)
-> does Ti move toward Elk +19.34 and O toward -19.1? Then core-Sternheimer only if Ti still short
(gate: internal sum-over-states check with the same tridiagonal operator).

## Fullpot-path optimizations (340a2c1, bit-exact): 2.7x measured
Gaunt memoization + k-independent aspherical-integral hoist + ylm-per-k hoist + masked/shared-geometry
pseudocharges. TiO2 fullpot+LO (ecut250 k222 aug-lmax3 fp_lmax4): 606 -> 226 s per 3 iters = 2.68x.
Nulls and the muffin-tin span reproduce to the last digit; flapw suite green through the LO work.

## B-side FULL TREATMENT result (fullpot fp_lmax=4 + Ti 3s/3p LOs + LO-crystal-field coupling)
ecut300 k222 sm0 Elk-R_MT aug-lmax3, 35 iters (4391s pre-pool):
  Ti evals [+4.47,-4.30,-0.17] eta 0.92 C_Q 2.67 | Elk [+19.34,-13.16,-6.18] 0.36 11.5
  O  evals [+3.90,-3.64,-0.26] eta 0.87 C_Q 0.24 | Elk [-19.10,+16.60,+2.50] 0.74 1.18
**Ti V_zz has the CORRECT SIGN (+) for the first time** — every variant without the LO coupling gave
~-3. The semicore antishielding channel (LO + aspherical coupling) works qualitatively. Both sites now
~20-25% of Elk magnitude UNIFORMLY -> suspect one common factor (k-convergence at k222, or residual
response) rather than site physics. Muffin-tin+LO for reference: Ti [-3.02,+1.98] eta 0.31.
Next: k-convergence of the B config (k222->k333->k444) with the new fast stack (pool+IBZ+warm start);
if magnitude climbs with k it is convergence, if flat ~23% it is structural.

## hl2_l3: general-L potential ALONE (no LOs), fullpot fp_lmax=4 aug-lmax3 k222 sm0 ecut300
  Ti [+6.79,-3.61,-3.18] eta 0.06 C_Q 4.05 | Elk [+19.34,-13.16,-6.18] 0.36 11.5  (35%, RIGHT SIGN)
  O  [+12.19,-7.58,-4.62] eta 0.24 C_Q 0.75 | Elk [-19.10,+16.60,+2.50] 0.74 1.18 (64%)
The high-L (incl odd-L) potential alone fixes Ti's sign and reaches 35%/64% of Elk — BETTER than the
B-side with LOs (Ti 4.5 / O 3.9)! => the LO config (O 2p LO at +5eV, Ti semicore bookkeeping) may be
hurting; needs an A/B isolating Ti-LOs-only vs O-LO. Odd-L crystal fields (missing before this
session) are a first-order ingredient at the O 4f site.

## ROOT CAUSE FOUND (2026-08-19): spontaneous symmetry breaking from winner-take-all occupations
The metamorphic suite caught it on first execution: in a bct 2-O cell (translation-equivalent atoms,
full mesh, NO symmetrization) the two atoms' EFG tensors came out axially identical (eta=0) but with
DIFFERENT magnitudes (-11.28 vs -15.96) — charge disproportionation between identical sites.
Mechanism: sm=0 filled the lowest nbands winner-take-all; when the filling boundary cuts a DEGENERATE
manifold (O 2p4!), the diagonalizer's arbitrary in-subspace basis decides where charge goes and SCF
feedback locks in a spuriously broken, floating-point-path-dependent state. This ONE mechanism
explains: the smeared-TiO2 instability, the IBZ-vs-full 8 eV drift (symmetry machinery itself exact
to 1e-15 — verified orbit cover, RhoSymmetrizer idempotency+invariance, Wigner-D unitarity/rotation/
projector), the aug-lmax "instability" (different basis -> different arbitrary choice -> different
basin), and the historic BeO breakdown. The BeO IBZ "validation" was vacuous (2-op group: IBZ==full).
FIX: _occ_degenerate_aware — spread electrons EQUALLY over a boundary degenerate group (within 1e-3
eV): the T->0 smearing limit = subspace-trace density = symmetry-invariant. Clean-gap insulators
unchanged; insulator path now solves +4 pad bands so boundary degeneracy is visible.
PREVENTION: (1) metamorphic MR suite (tests/integration/test_flapw_metamorphic.py) — equivalent-atom
agreement, IBZ==full at a SHARED warm-started fixed point, warm-restart stability, pool bit-equality,
EFG frame equivariance, resolution-knob stability; (2) runtime info["symmetry_dev"] (raw-density
asymmetry across equivalent atoms + unfolding projector residual); (3) protocol: path comparisons
only at shared fixed points; symmetry features validated only on groups with REAL reductions.
ALL PRIOR COLD-START TiO2/BeO NUMBERS at coarse k are basin-suspect; the campaign must be re-run on
the fixed stack (deg-aware occ + warm-started chains + IBZ + pool).

## CAMPAIGN POINT 1 (degeneracy-fixed stack, k222 cold, fullpot fp4 aug3, IBZ+pool, 18 min):
  O  [+17.75,-14.37,-3.39] eta 0.619 | Elk [-19.10,+16.60,+2.50] 0.74  -> |components| at 93%/87%!
  Ti [-1.68,+1.52,+0.15]   eta 0.818 | Elk [+19.34,-13.16,-6.18] 0.36  -> still small (LO probe next)
The spurious symmetry breaking WAS the O magnitude deficit: every basin-broken run gave 20-64%; the
degeneracy-aware occupations recover ~90% at the COARSEST mesh. (symdev printout in this run is a
normalization bug, fixed in e169633+1; physics unaffected.) k333/k444 warm points next.

## TI SOLVED (campaign k333 warm, fullpot fp4 aug3, NO LOs, NO core-Sternheimer):
  Ti [+19.05,-13.41,-5.64] eta 0.408 | Elk [+19.34,-13.16,-6.18] 0.36  -> V_zz 98.5% of Elk,
  C_Q(49Ti) 11.4 vs Elk 11.5 MHz (exp 13.4). The "missing deep-core Sternheimer" hypothesis was
  WRONG: Ti needed (a) the general-L (odd-L incl.) crystal field and (b) the SYMMETRIC SCF basin
  (degeneracy-aware occupations). All prior "Ti impossible" conclusions were broken-basin artifacts.
  O at this point: [+13.17,-10.82,-2.35] (69%), still moving with k (k222 gave 93%) — k444 + LO
  probe pending. symdev printout still the pre-fix normalization (cosmetic).

## SHARPENED DIAGNOSIS (telemetry, 2026-08-19): the fullpot aspherical loop was NEVER converged
New per-iteration telemetry (verbose=True: span, r_v, r_nsph, symdev, beta_nsph) shows r_nsph —
the v_nsph relative residual, previously UNMEASURED — decaying slowly (~10%/iter even for Ne
fullpot) while the span-only criterion "converges" within 3 iterations. ALL fullpot TiO2 endpoints
(k333 Ti 98.5% match, k444 collapse, aug4 300% O, TiLO probe) were snapshots of a still-moving
aspherical self-consistency. Fixes landed: adaptive-damped v_nsph mixing (halve step on residual
growth), r_nsph < 0.05 REQUIRED for convergence, telemetry to see it. The k333 Ti/Elk match must be
re-established with an r_nsph-converged run (more iters and/or Anderson on v_nsph if slow).
Campaign complete; all its fullpot numbers are provisional pending converged-loop reruns.

## Session close state (2026-08-19 late): staged-start + collapse landed; TiO2 blow-up isolated
- Aspherical dH collapse (8e0af35): 109x kernel / 3.9x per iteration (129->33 s local, 11 s on asus).
  Elk gap now ~15-20x per unit work (was ~100x).
- Staged MT->fullpot start: Ne r_nsph 10%/iter (cold) -> 4e-5 (staged). TiO2 staged run converged
  smoothly to d~6e-3/r_nsph~0.08 then BLEW UP at it34 on a [subsp] iteration (span 24->61, r_v 100).
  HYPOTHESIS: electronic level crossing under the growing aspherical potential slips through the
  subspace gate's blind spot (gate validates returned states; a NEW state entering the window
  between forced exact solves is invisible). Ne (no crossings) stages cleanly.
- DISCRIMINATOR RUNNING: same staged gated run with subspace_reuse=False (exact solves only) —
  if it converges, the fix is fullpot-aware solve cadence (e.g. exact solves while r_nsph>1e-3, or
  a window-entry check in the gate); if it still blows up, the instability is occupation/crossing
  physics (next: larger degen tol, tiny smearing ramp, or occupation tracking across iterations).
- FLAPWRecorder (3c86336) now captures every iteration (info["recorder"]) — no more lost histories.

## FIRST TRUSTWORTHY CONVERGED TiO2 FULLPOT EFG (staged, exact solves, all gates green, 13 iters ~2min):
  Ti [-1.76,+1.26,+0.50] eta 0.43 | Elk [+19.34,-13.16,-6.18]  -> Ti small/wrong sign (REAL, converged)
  O  [+13.55,-11.29,-2.27] eta 0.67 | Elk [-19.10,+16.60,+2.50] -> O ~70%, shape good
The subspace-blind-spot hypothesis CONFIRMED (exact-only converges cleanly; cadence fix 495d245:
exact solves during fullpot). The earlier k333 "Ti 98.5%" was a transient. Ti search matrix running
(aug4/fp6/k444/TiLO from the converged baseline, ~3-6 min per config with the full stack).

## EXACT-SOLVE MATRIX VERDICT (2026-08-19 late): the aspherical loop has a RUNAWAY FIXED POINT
fp6 CONVERGED (r_nsph 3.5e-4) to O [+115.3,-100.4,-14.9] eta 0.74 — Elk's exact SHAPE at 6x the
magnitude; TiLO converged in 6 iters to the same runaway (O +129); base/aug4/k444 hit the cap
partway to it. The earlier clean 13-iter run (O +13.6) and these are TWO fixed points of the same
loop. HYPOTHESIS: self-interaction in the antishielding chain — a sphere's own l=2 field leaking
into its own lattice term C_LM=(v_bc-V_part(R))/R^L (pseudocharge moment/normalization/sign
breaking the own-field cancellation) -> loop gain >1 -> saturated structure-preserving runaway.
UNIT TEST DESIGNED: single atom in a tetragonal box has no other spheres -> its C_LM must be ~0;
any nonzero = the leak, isolated from SCF dynamics. Check sphere_pseudocharge_lm moments on the
REAL grid (not the unit-test grid), the interstitial-inside subtraction (dv**bl term), and the
v_bc evaluation at exactly R vs the pseudocharge support edge. FIX gain<1 -> single physical fixed
point -> then rerun the matrix. All current TiO2 fullpot magnitudes are basin-dependent until then.

## LEAK CONFIRMED + FIX DESIGN (handoff): single-atom lattice term = 3.19 eV/A^2 (20% of V_zz, must
be 0) -> self-interaction verified. Naive per-m grid normalization made it WORSE (5.17): on the
coarse grid the Y_LM are NOT orthogonal, so per-m moments pick up cross terms. CORRECT FIX: solve
the (2L+1)^2 grid Gram system G c = q per (atom, L) in sphere_pseudocharge_lm, with
G[m',m] = sum(radial_in * d^L * conj(Y_m') * Y_m) * dvol — then the pseudocharge's MEASURED grid
moments equal Q_LM exactly and the own-field cancellation in C_LM holds by construction. Acceptance:
single-atom leak -> ~0, TiO2 fullpot converges to ONE fixed point from any start, magnitudes stable
across fp_lmax/k. coulomb.py reverted to analytic normalization pending this fix.

## FIXES LANDED + REFRAME (6a5c0fc): Gram-solve grid moments (1e-16 exact, L=1..4) + NUMERICAL
own-field subtraction (own pseudocharge's band-limited surface projection removed from C_LM; the
analytic V_part cancellation failed ~20% from aliasing of the ~5-grid-point pseudocharge shape).
REFRAME of the single-atom test: with own-field subtracted, its remaining "lattice" term is the
atom's OWN WAVEFUNCTION TAILS outside R_MT — real physics, not a leak; and its r_nsph
non-convergence is plausibly PHYSICAL LDA orbital-polarization multistability of the degenerate p4
shell (equal-occupation rule fights the potential's symmetry breaking) — a bad acceptance vehicle.
DECISIVE ACCEPTANCE = the TiO2 matrix (closed-shell insulator, Elk ref): (a) r_nsph gates green,
(b) base/fp6/k444 magnitudes mutually consistent (no 6x runaway), (c) O near the physical basin
(~14-19 vs Elk 19.1). Matrix running on the full fixed chain (Gram + own-field + joint Anderson +
staged + exact solves).

## SLEDGEHAMMER (de93344): true full-potential interstitial
User called the incrementalism; the audit agreed. Landed together: (1) warped-interstitial H
matrix elements FT[(V_I-v_i0)·Θ_I](G-G') (Toeplitz-indexed, one FFT/iter) — the standard FLAPW
term we NEVER had ("constant in the interstitial" was in the docstring all along); (2) interstitial
XC (previously ZERO — a potential discontinuity at every R_MT); (3) unified metric-weighted
Anderson over (v_sph ⊕ v_nsph ⊕ interstitial grid) in the L2 measure (√r²dr, √dvol) — the
principled scale-weighting; the interstitial was previously entirely UNMIXED and drove boundary
conditions at full amplitude; (4) r_v < 0.1 eV added to the convergence gate. MR suite green;
nulls at a new 3e-5 floor (warp band-limitation); dilute Ne unchanged (empty interstitial).
Pre-sledgehammer matrix (for the record): no config reached the gate, TiLO still found the
runaway — incremental fixes necessary but insufficient. ACCEPTANCE = the TiO2 matrix now running:
gates green + config-consistent magnitudes + O toward Elk 19.1.

## PHYSICS-BLIND KERNEL ANALYSIS (sanitized-spec agent, no domain vocabulary) — speed roadmap
Validated the method: it independently rediscovered the measured PW small-cell result (dense
blocked applies beat FFT applies at N<~3000 = the gated ToeplitzApply finding). Ranked attacks:
1. Iterative block pencil solver + p-mode B-deflation (small LOBPCG on B's low end -> projected
   pencil; reproduces canonical orth on troubled modes only, NO full eigendecomposition): 10-40x
   on the solve, N^3 -> N^2 n_b. Caveats: deflation-tol consistency with tol*max, safeguarded
   ortho, B-positivity monitoring, iterate n_b+5-10 wide for clusters.
2. AUGMENTED Rayleigh-Ritz warm start in span[V_prev, U]: inter-iteration perturbation is
   U dM U^T with FIXED U -> augmenting with U captures it to ALL orders (the principled fix for
   the subspace-reuse blindness that corrupted our runs); 1-2 refinement sweeps mop up the small
   Toeplitz leak. 3-10x, composes with #1.
3. Single-process K-batching + batched/fused scatter-FFT reductions (3-10x on density pass).
4. Apply-only operators + once-per-run Woodbury factors for B (assembly -> ~0).
5. Eisenstat-Walker tolerance forcing + widened regularized Anderson (history 15-25 + Tikhonov;
   history 6 truncates the low-rank Jacobian).
Compound: 5-15x wall now, ~10x N-ceiling; the apply-only representation is exactly the
differentiable-torch substrate -> speed and autograd roadmaps CONVERGE. H structure that enables
it all: H = Diag + 3-level-Toeplitz(G-G') + sum_atoms B_a M_a B_a^T (Gram/addition theorem),
S identical; between iterations only ~200 scalars in M + the Toeplitz symbol change.

## PW PHYSICS-BLIND WORKFLOW RESULT (wf_2810c676-295; full text in the run journal)
Vetted, code-mapped, ranked (gain x p / effort), all honoring the measured-negatives list:
1. Coarse-operator draft slaved to adaptive-tau (S effort, 1.15-1.35x CPU, p.5): run the local
   apply on a coarse box while tol_eff is loose (existing smooth= plumbing!); exact operator at
   the convergence gate -> fixed point AND implicit-diff gradients unchanged. Kill signal: +1
   outer iteration on Fe/Cr/Al in the bench_scf battery.
2. CholQR2 for _orthonormalize_b (S, all devices, p.6): Gram+Cholesky x2 GEMMs replace batched
   tall-skinny QR (the documented consumer-GPU round bottleneck) + removes the CPU-offload
   roundtrip; span-equivalence physics argument; jitter/fallback rank safety.
3. Sphere-aware pruned 3D FFT (M, 1.1-1.4x, p.4): 3x batched 1D passes pruning to the wfc-sphere
   sub-box; bit-equality unit test vs fftn; benchmark-gated per size class.
4. fp64-certified fp32-expansion Davidson (M-L, 1.5-3x consumer GPU, p.4): fp32 expansion applies,
   fp64 RR/Gram + fp64-certified residuals (dodges the measured fp32-metal regression).
5. Convergence-cohorted lockstep: PROBE FIRST via existing history_out spread histogram.
6. Toeplitz auto-gate (gating half only).
The sanitize phase's spec was faithful incl. the AD constraint; workflow validated twice now.

## ACCEPTANCE MATRIX FINAL (asus, 2026-08-19)
base  k333 aug3 fp4:  n_it=40 r_nsph=9.7e-2 (NOT gated)  Ti [-0.75,+0.58,+0.17] eta .55 | O [+8.10,-7.15,-0.95] eta .77
aug4  k333 warm:      n_it=30 r_nsph=1.7e-1 (NOT gated)  O [-5.53,+5.53,0] eta 1.00
fp6   k333 warm:      r_nsph=4.0 DIVERGED — confirms unresolvable L=5,6 pseudocharges; cap matched L at 4
k444  aug3 warm:      n_it=30 r_nsph=4.2e-3 GATED        Ti [-1.09,+0.80,+0.29] eta .47 | O [+8.33,-7.22,-1.11] eta .73
TiLO  aug3 k333:      n_it=21 r_nsph=5.7e-4 GATED        Ti [-1.04,+0.89,+0.15] eta .70 | O [+8.58,-7.46,-1.12] eta .74

Reading: the two GATED runs agree with each other (O 8.3-8.6, eta .73-.74) and O's eta matches
Elk (0.74) exactly; O magnitude is ~44% of Elk's 19.1. Ti is ~1 eV/A^2 vs Elk 19.3 — the Ti
site is dominated by semicore/core polarization the current lmax=3 aug + no-O-LO basis doesn't
capture; TiLO barely moves it, so the gap is basis (aug-lmax / LO set / fullpot_lmax), not
convergence. Next lever: fullpot_lmax=4 with matched-L capped at 4, higher aug-lmax, O LOs.

## MEASURED SPEEDUPS (asus uncontended, 8 threads CPU, 2026-08-19; benchmarks/results/asus/)
Toeplitz auto-gate: si2 1.20x c2 1.25x gaas 1.21x al 1.30x cu 1.69x mgo 1.38x | si8/si64 1.00x
(gate correctly declines) — strictly-positive vs the old always-on flag's 0.79-0.80x regressions.
CholQR2: CPU-neutral (0.94-1.03x) as expected — its target is the consumer-GPU QR round trip; GPU
check pending. Energies bit-match in all 32 runs; metal iteration parity holds (kill signal clear).

## NEWTON PROBE, EASY CONFIG (asus): honest split verdict
Mini-TiO2 gamma lmax=2 fp2: warm state already r_nsph 2.9e-2; Anderson gate=3/tight=7 iters vs
NK 29 F-evals — ANDERSON WINS easy fixed points. NK quality itself confirmed (r_nsph 7.5e-9 in 3
Newton steps). Decisive test = production-hard config (lmax=3 fp4 k222) queued (newton_probe_hard).

## NEWTON PROBE, HARD CONFIG (asus job 109) — SLEDGEHAMMER VALIDATED
TiO2 ecut300 lmax3 fp_lmax4 k222, 590k dof, from a 24-iter warm state (r_v 2.6, r_nsph 0.23):
  Anderson baseline: 50 MORE iterations -> r_v STUCK at 4.6e-1, r_nsph 9.6e-3; tight NEVER reached
  Newton-Krylov:     32 F-evals (3 Newton steps) -> r_v 1.2e-7, r_nsph 1.5e-9
Wall even in the naive probe embedding: 549s (NK, converged) vs 722s (Anderson, NOT converged).
On the production-hard config the damped/Anderson iteration effectively STALLS (limit-cycle-like
r_v plateau) while Newton with FD JVPs converges quadratically in 3 steps. Production embedding
(hoist setup out of F, autograd JVPs later, Anderson steps as Krylov preconditioner) is now the
top FLAPW convergence task; expected effect: the 40+-iteration tails of the acceptance matrix
collapse to ~1 Newton solve from any gate-adjacent warm state.

## DMD MODE ANALYSIS (asus jobs 112/113) — WHY Anderson stalls, measured
Easy config: unstable complex pair |rho|=1.24 (sph+interstitial mix) + a few slow modes; 13
modes |rho|>0.5. Anderson's m=5 window handles the low-dimensional instability -> converges.
HARD config: unstable complex pair |rho|=1.02 that is 94% INTERSTITIAL, plus a second
interstitial-dominated pair at |rho|=0.92 (27 it/decade) and a cluster of 0.68-0.75 modes; 12
modes |rho|>0.5. The bare damped map is UNSTABLE via long-wavelength interstitial sloshing, and
the slow/unstable cluster exceeds Anderson's window -> the measured r_v 0.46 stall.
Implications: (a) Newton polish (shipped, flapw/newton.py) handles it outright — consistent with
its 3-step convergence; (b) a targeted interstitial-channel Kerker screen G^2/(G^2+k0^2) in the
joint Anderson is the cheap complementary fix — NOTE this is the FLAPW interstitial channel with
direct evidence of sloshing, distinct from the measured-closed PW small-cell Kerker avenue.

## KERKER A/B, HARD CONFIG (asus job 117) — DRAMATIC WIN AT k0=0.7
Same warm state, 40-iteration continuations (ecut300 lmax3 fp4 k222):
  plain:      r_v ~2e+2, r_nsph ~3-7   -> the plain continuation actively DIVERGES (worse than the
                                          earlier 0.46 stall; trajectory-dependent blowup)
  kerker=0.7: r_v ~2-9e-2, r_nsph 3.4e-4..2e-3  -> GATED and near-tight within 40 iterations
  kerker=1.5: diverges like plain — NOT monotonic in k0; over-screening (floor-dominated
              interstitial, 10x slower channel) appears to destabilize the coupled system through
              the sphere channels. k0~0.7 A^-1 is the working window; treat k0 as a knob to A/B
              per system class, do not blindly increase.
Production recipe now: Anderson + kerker=0.7 for the trajectory, newton_polish to machine
precision. Default remains opt-in until the easy-config matrix is re-validated with it.

## CLEAN GPU VERDICTS + WARM-GATE (2026-08-19 late)
CholQR2 RTX3050 whole-SCF: ~0.95-1.0x (neutral-negative) -> default flipped to OFF; retest on
datacenter GPU only. Pruned FFT GPU: 0.27-0.67x SLOWER (torch strided-pass overhead persists on
cuFFT) -> measured-closed, branch stays unmerged. TiO2 warm gate: k222+kerker GATED at r_nsph
1.8e-3 (best production point yet) but the k333 continuation DIVERGED even warm+screened — k333
needs newton_polish from the k222 state, not more Anderson. (Diverged-state EFG printed O
[-23.4,+18.3,+5.0] — Elk-like sign pattern but untrustworthy; do not quote.)

## OVERNIGHT PERF+GIPAW (2026-08-20 early)
Setup hoist (setup/init/iterate split, bit-identical x59 arrays) + surface-phase memo in efg.py
(the profiled 67% -> 6%): per-F 2.09->0.72s (2.89x); warmup 3.2x. Next levers per fresh profile:
nonspherical_potential angular XC (46%), radial integrations (~30%). GIPAW M6-M8 shipped on
qgt-autograd-k: sign CALIBRATED (Peierls-flux anchor), k+-q velocity Sternheimer (1e-9 vs dense
incl. umklapp), screened induced current (f-sum pinned, TR null 2e-13). Remaining for shielding:
KB nonlocal current (div j=0 closure) + q->0 antisymmetric assembly — one physics step, fresh
session.

## TRUSTWORTHY EFG (pipeline, asus job 128, 2026-08-20) — FIRST FULLY-CONVERGED PRODUCTION POINT
k222+kerker: gated r_nsph 2.2e-4 in 32 it. newton_polish k333: CONVERGED (33 F-evals, res 1.5e-5)
— Newton succeeds at the k-point where every Anderson variant diverged; production-validated.
EFG k-stable across k222->k333:
  O : [+8.33,-7.37,-0.97] eta 0.768   (k222: [+8.43,-7.39,-1.04] eta 0.754) | Elk 19.1, eta 0.74
  Ti: [-1.01,+0.81,+0.20] eta 0.607                                          | Elk 19.3, eta 0.36
VERDICT: convergence error is now ELIMINATED as an explanation. O eta matches Elk (0.75-0.77 vs
0.74); magnitude ~44% of Elk, k-stable; leading-component sign opposite to Elk's listing (check
Elk's sign convention before calling it a discrepancy). The remaining gap is BASIS (aug-lmax,
O LOs, fullpot_lmax) — next lever: aug-lmax 4-5 + O LO A/B from these gated states (task #7).

## ELK SIGN CONVENTION RESOLVED (2026-08-20) — the flip is REAL, and it is a clean -0.44x
Elk's EFG.OUT is d^2(vclmt)/dr_i dr_j (writeefg.f90) where vclmt is the ELECTRON Coulomb
potential (species files store spzn=-Z, confirmed O.in: -8.0) — i.e. -1x the electrostatic
(NQR) convention. gradwave's _valence_v is +(4*pi*E2/5) Int rho_2M/r — the SAME
electron-potential sign. Same convention => the opposite leading sign is NOT a convention
artifact. Sharper: componentwise ours = -0.44x Elk almost exactly
([+8.33,-7.37,-0.97] vs [-19.1,+16.6,+2.5]; ratios .436/.444/.388) — which is WHY eta
matches while everything else disagrees. Two candidate explanations the basis A/B
discriminates: (a) a missing opposite-sign contribution ~1.44x the current valence term
(semicore/core polarization -> aug-lmax/LO variants should grow magnitude and flip the
sign through a cancellation); (b) a global sign slip in one term of our chain (then the
A/B magnitude grows but never flips). Elk raw a.u. eigenvalues, O: [-0.1967,+0.0257,+0.1710]
(1 Ha/Bohr^2 = 97.17 eV/A^2).

## BASIS A/B ROUND 1 (asus 129) — CONTROL CAUGHT TWO REAL DEFECTS; queue killed+requeued
The base control (== job 128 config) REPLICATED the gated-state physics (k222 EFG O
[+8.40,-7.37,-1.03] eta 0.755 vs job 128's [+8.43,-7.39,-1.04] eta 0.754) but exposed:
(1) The surface-phase memo (e42729e) THRASHED at production scale: ~1 GiB per PER-CENTER
    entry x 6 spheres cycling vs a 1.5 GiB budget = zero hits + ~1 GiB alloc churn per
    call on a 14 GiB box -> k222 263s (job 128, pre-memo) vs 1633s (with memo, plus
    fp32-agent fast-tier contention). FIX: factor the center phase out (E = e^{iGc} * E0(R)),
    cache per SPECIES radius (2 entries, not 6), budget 4 GiB; validated 2.7e-15 vs inline.
    Rounding association changes at the ulp level — which matters because:
(2) The cold k222 trajectory is RUN-TO-RUN FRAGILE (the measured |rho|~1.02 marginal mode
    amplifies ulp/BLAS nondeterminism): job 128 gated at 32 it / 2.2e-4, the identical
    config this round hit the 40-cap at 4.9e-3 — and newton_polish k333 DIVERGED (residual
    2.3e3) from 4.9e-3 where it converged (33 F-evals) from 2.2e-4. Newton's k333 basin
    needs r_nsph ~<1e-3. FIX: the A/B runner now continues k222 in warm chunks until
    r_nsph < 1e-3 (cap 120 it) and refuses to polish from a marginal state.
Moral for the recipe: "Anderson+kerker then Newton" is production-valid only with an
explicit basin gate between the legs; do not trust a single lucky trajectory.

## NEWTON DISCRIMINATOR (asus 143) — CHAOS CONFIRMED, MEMO EXONERATED, FIX SHIPPED
From the saved base_k222 state (healthy: k222 1-it resume r_v 4.0e-2, r_nsph 1.7e-3):
  A: newton k222            -> res 5.0e+3 (diverged, 25 F-evals)
  B: newton k333 (memo on)  -> res 25.3   (diverged, 25 F-evals)
  C: newton k333 (memo OFF) -> res 1.4e-3 (CONVERGING, exhausted maxiter, 41 F-evals)
B vs C differ at most at the ulp level (cached vs freshly-recomputed phase matrix; matmul
alignment nondeterminism) — outcome spread 5e+3..1.4e-3 across ulp-different maps means
FD-Newton-Krylov on this map is CHAOTIC-SENSITIVE, not bugged. Job 128's k333 success and
the probe's k222 success were good draws (2/6 overall). The memo is exonerated (round-1's
thrashing memo was provably bit-identical to job 128's F anyway). Also measured: the k333
map kicks to r_v 1.67 on the FIRST iterate from a k222-converged state — k333 Newton always
starts in rough territory. FIX (newton.py): newton_polish now runs up to `rounds`(=3)
newton_krylov restarts with MONOTONE ACCEPTANCE (each round starts from the best iterate so
far; never returns worse than its input) and re-rolls the FD step rdiff per round — C's
1.4e-3 near-miss becomes round 2's start, the regime where quadratic convergence re-engages.
Possible future lever (not built): 5-10 k333 Anderson+kerker pre-iterations to contract the
smooth channel before Newton.
