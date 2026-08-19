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
