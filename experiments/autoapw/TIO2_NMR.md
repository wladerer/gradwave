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
