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
