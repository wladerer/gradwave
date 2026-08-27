# FLAPW core-level (XPS) shift validation

Validation of gradwave's all-electron FLAPW initial-state core-level shifts
(`src/gradwave/flapw/core_levels.py`, shipped in #378) against Elk 11.0.2 and
experiment. Two questions:

1. **Within-cell** (site vs site, one cell): does gradwave reproduce Elk?
2. **Cross-cell** (Si 2p in bulk Si vs SiO2, ~4 eV experiment): can an explicit
   reference make the comparison meaningful despite the wandering interstitial zero?

All FLAPW SCFs and Elk runs were executed on the asus peer; the gradwave driver is
`experiments/autoapw/xps_core_levels.py`. Elk was **actually run** (11.0.2, built at
`~/github/elk-11.0.2`); core eigenvalues are read from `EVALCORE.OUT`, the Fermi
level from `EFERMI.OUT`.

## Method and reference frame

FLAPW freezes the core but re-solves it every SCF step in the converged spherical
muffin-tin potential to build rho_core; `core_levels` keeps those eigenvalues. Both
codes reference core eigenvalues to their own flat interstitial zero, which **wanders
between cells**. So the physics is in differences that share one reference:

* **Within a cell** every muffin tin sits on the same interstitial zero, so a
  same-element site-vs-site difference `Delta = e_core(B) - e_core(A)` is physical
  (equivalent sites give exactly 0; the systematic radial-mesh error cancels too).
* **Across cells** the absolute levels are incomparable. We reference each cell's
  core level to a physical marker OF THAT CELL — the valence-band maximum (VBM) —
  and compare binding energies `BE = VBM - e_core`. The SCF shifts the cores and the
  VBM together under any interstitial change, so `BE` is invariant to the wander
  (unit-tested: `test_referenced_binding_energy_wander_invariance`).

Sign convention: `e_core < 0` (bound); `BE > 0`; a MORE positive `Delta` / larger
`BE` means a MORE bound core (higher XPS binding energy).

## Settings

LDA (PW92; Elk `xctype 3`), `lmaxi 2`. gradwave: muffin-tin FLAPW, ecut 220-250 eV,
2x2x2 k-mesh, symmetry on. Elk: `tasks 0`, `rgkmax` matched to gradwave's
`R_min * Gmax(ecut)`, 2x2x2 k-mesh. Muffin-tin radii (converted; gradwave takes
radii in Angstrom, Elk in Bohr):

| system | species | R_MT (Angstrom) | R_MT (Bohr) |
|---|---|---|---|
| rutile TiO2 | Ti / O | 1.098 / 0.824 | 2.075 / 1.557 |
| bulk Si | Si | 1.10 | 2.079 |
| beta-cristobalite SiO2 | Si / O | 0.80 / 0.75 | 1.512 / 1.417 |

## Task 1a — real-crystal equivalent-site null (rutile TiO2)

Rutile's four O (Wyckoff 4f) and two Ti (2a) are each symmetry-equivalent, so every
same-element within-cell shift must be exactly 0. This is the real-crystal analogue
of the two-Ne null unit test, on a close-packed cell.

| level | Elk (Ha, all sites) | Elk spread | gradwave within-cell shift |
|---|---|---|---|
| O 1s | -18.20176413 | 0.0 (8 digits) | 0.0 (4 O identical) |
| Ti 1s | -178.0096670 | 0.0 (8 digits) | 0.0 (2 Ti identical) |
| Ti 2p | -15.9/-16.1 (SO split) | 0.0 | 5.7e-7 eV (numerical) |

Both codes: equivalent-site shift = 0. gradwave O 1s = -497.008 eV and Ti 1s =
-4807.184 eV are identical across all equivalent sites; the residual Ti 2p spread is
6e-7 eV (solver noise). (Elk data from `~/tio2_efg/EVALCORE.OUT`; gradwave from
`xps_core_levels.py rutile`.) Absolute levels differ between codes (wander + radial
mesh) — only within-cell shifts are compared.

## Task 1b — nonzero within-cell shift (synthetic Ti + 2 O)

A controlled inequivalent-O case: one Ti with two O at DIFFERENT Ti-O distances
(1.75 and 2.30 Angstrom) in a 13-Bohr cubic box — the system behind the module's
`-1.67 eV` demo. Same geometry in both codes; Elk run at several O R_MT.

| O R_MT (Bohr) | Elk e(short) Ha | Elk e(long) Ha | Elk Delta(long-short) eV |
|---|---|---|---|
| 0.70 | -18.74369747 | -18.70873174 | +0.951 |
| 1.00 | -18.59966385 | -18.54894916 | +1.380 |
| 1.40 | -18.64178795 | -18.53667882 | +2.860 |

gradwave (O R_MT 0.70 Angstrom = 1.32 Bohr): **Delta(long-short) = -1.667 eV**.

**Finding — this open geometry is a poor benchmark and it maps the method's limit.**
Elk robustly makes the O NEARER the (net-positive) Ti MORE bound (Delta > 0, growing
with R_MT as more of the site-distinguishing potential falls inside the sphere); the
standard core-level "potential model" (BE rises with a more positive Madelung
potential, i.e. nearer a cation) agrees with Elk. gradwave gives the OPPOSITE sign.
The mechanism is specific: here the two O sites are chemically identical and differ
ONLY by a long-range Madelung difference that lives in the region BETWEEN the spheres.
gradwave represents its spherical muffin-tin potentials on a FLAT shared interstitial
zero, so a purely inter-site interstitial potential difference is discarded; only the
small in-sphere residual survives, and it comes out wrong-signed. Elk carries the full
interstitial potential in plane waves and keeps it. Note this is the opposite regime
from the Si/SiO2 headline below, which gradwave gets RIGHT: there the shift is an
ON-SITE chemical-state change (Si0 -> Si4+, valence charge leaving the Si sphere),
captured inside the muffin tin. The takeaway: gradwave's core-level shifts are
trustworthy when the distinguishing potential is on-site (close-packed crystals,
oxidation-state changes) and unreliable when it is a bare interstitial Madelung term
(open cells, tiny spheres). The bundled `test_core_level_shift_inequivalent_oxygen`
guards reproducibility of the -1.67 eV number but is NOT a physical-correctness check
(its sign disagrees with Elk); its comment has been corrected to say so.

## Task 2 — cross-cell headline: Si 2p, bulk Si vs SiO2

Textbook XPS: Si 2p in SiO2 is ~4 eV higher binding energy than in bulk Si (Si0 ->
Si4+). Bulk diamond Si (a = 5.431 Angstrom) and ideal beta-cristobalite SiO2 (6-atom
FCC-primitive cell, a_conv = 7.40 Angstrom, tetrahedral SiO4, formal Si4+). Both O in
SiO2 come out equal in both codes (another equivalent-site null: gradwave O 1s =
-501.640 eV x4; Elk O 1s = -18.40637 Ha x4). Elk's Si 2p is the (1:2)-weighted
2p1/2/2p3/2 average (gradwave is scalar, no core SOC).

| quantity (eV) | bulk Si (gw / Elk) | SiO2 (gw / Elk) |
|---|---|---|
| Si 2p (interstitial-zero frame) | -84.963 / -83.828 | -89.340 / -89.285 |
| VBM (Gamma) | 7.339 / 6.128 | 0.454 / -0.923 |
| BE(potential-aligned) = -e(2p) | 84.963 / 83.828 | 89.340 / 89.285 |
| BE(VBM-referenced) = VBM - e(2p) | 92.302 / 89.956 | 89.795 / 88.362 |

Cross-cell Si-2p shift Delta BE(SiO2 - Si):

| reference | gradwave | Elk | experiment |
|---|---|---|---|
| **potential alignment (interstitial zero, e_ref=0)** | **+4.38** | **+5.46** | **~+4** |
| VBM-referenced (bracket) | -2.51 | -1.59 | (see below) |

**Both codes get the right sign and magnitude under potential alignment** (gradwave
+4.4, Elk +5.5 vs ~+4 eV experiment) and **both over-correct to the WRONG sign under
VBM referencing** (-2.5 / -1.6). That the two independent all-electron codes agree on
this pattern shows it is a methodological property, not a gradwave bug.

### Referencing scheme — choice and honesty

Candidates: (a) the average electrostatic potential; (b) each cell's VBM/E_F; (c) both,
to bracket. **We use (a), realized as the FLAPW interstitial zero** (`e_ref = 0`): the
core-level chemical shift is, in initial-state theory, the difference of core
eigenvalues on a common potential reference, and the interstitial zero is FLAPW's proxy
for the average electrostatic potential. The core levels are already in that frame, so
the shift is just their raw difference. Scheme (b) was the intuitive first choice but
is WRONG for a core-level shift: the VBM position relative to the mean potential differs
by ~7 eV between narrow-gap Si and wide-gap SiO2, and referencing to it folds that
band-structure/gap difference into the number, flipping the sign. The code-to-code
table above is the evidence — (b) fails identically in Elk. We report (b) only as an
uncertainty bracket. The helper `flapw.core_levels.cross_cell_binding_shift` takes an
explicit `e_ref`, so it does both: pass `0.0` for the physical shift, the VBM for the
bracket.

### Error assessment

Two approximations bound the result:

1. **Cross-cell alignment (the ~1 eV code-to-code spread).** Potential alignment via the
   interstitial zero is exact only if both cells' interstitial zeros are the same
   physical reference; the mean potential of an infinite crystal is defined only up to a
   surface term, and the two codes use different R_MT/basis. The gradwave-vs-Elk spread
   (+4.4 vs +5.5) is the honest uncertainty of the cross-cell number. The WITHIN-cell
   nulls (rutile, SiO2 O) are exact by construction and carry none of this.
2. **Initial-state (Koopmans) approximation.** Computed from the ground state, no core
   hole, so it omits **final-state screening** — valence relaxation around the hole,
   larger in polarizable Si than in ionic SiO2, which REDUCES the measured shift. For
   Si/SiO2 the initial-state shift already captures the bulk of the ~4 eV (dominated by
   the on-site loss of Si valence charge to O); the final-state correction is typically
   <= 1 eV and would move gradwave's +4.4 toward experiment, not away.

Minor caveats: ideal beta-cristobalite is a higher-symmetry, slightly expanded proxy
for real silica (the Si 2p shift is a short-range SiO4 property, so this is small);
Gamma-only VBM (exact for Si, a good proxy for wide-gap SiO2); muffin-tin (not
full-potential) FLAPW; LDA.

## Reproduce

```bash
# gradwave (asus, OMP modest):
uv run python experiments/autoapw/xps_core_levels.py rutile 2 250
uv run python experiments/autoapw/xps_core_levels.py headline 2 220
```

Elk input decks were generated inline (LDA, tasks 0, matched rgkmax/R_MT) and are
not committed; the settings above regenerate them.
