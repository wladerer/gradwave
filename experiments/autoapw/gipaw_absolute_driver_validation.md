# Absolute GIPAW shielding through the driver — anchor validation

Validation runs for the Phase-1 ssNMR PR (`docs/plans/ssnmr-spectrum-plan.md`):
the absolute GIPAW σ assembly (`sigma_shielding_gipaw`) reached through
`api.run_nmr` (`nmr.task='shielding'`, `nmr.shielding_level='auto'` on PAW
pseudos), checked against the two published GIPAW anchors. Runs on asus
(OMP_NUM_THREADS=8), PBE, `symmetry=False` (full spatial mesh), `pref = R_E`
(no free scale anywhere).

## Anchors

- Diamond-structure Si, ²⁹Si: absolute σ_iso ≈ 400 ppm (GIPAW literature value;
  the shipped slow-tier test `tests/unit/test_kgeometry_nmr.py::
  test_sigma_shielding_gipaw_si_absolute` reproduces ≈398 ppm at the same
  settings by direct function call — this note checks the *driver* path).
- MgO rocksalt, ¹⁷O: absolute σ_iso ≈ 215 ppm.

## Runs

### Si diamond (a = 5.43 Å, 2 atoms)

Pseudo `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (tests/fixtures/qe/pseudos). 12 Ry /
48 Ry ecutrho, 2×2×2 MP, nbands 8, etol 1e-8 / rhotol 1e-7 / diago 1e-9.

| site | σ_bare | σ_core | σ_dia_aug | σ_para_aug | σ_iso (total) |
|------|--------|--------|-----------|------------|----------------|
| Si 0 | −5.68  | 837.79 | 2.24      | −436.02    | **398.32 ppm** |
| Si 1 | −5.68  | 837.79 | 2.24      | −435.86    | **398.49 ppm** |

vs the ≈400 ppm ²⁹Si anchor: within ~2 ppm. Equivalent-site spread 0.17 ppm.
Matches the shipped direct-call slow-tier test (≈398 ppm at identical settings),
confirming the driver wiring adds nothing.

### MgO rocksalt (a = 4.21 Å, 2 atoms)

Pseudos `Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF` (semicore 2s2p-in-valence Mg PAW,
fetched from the QE pslibrary — not committed; download via
`https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF`)
+ `O.pbe-n-kjpaw_psl.1.0.0.UPF`. 40 Ry / 160 Ry ecutrho, 2×2×2 MP, nbands 12.

**Anchor NOT met — recorded as a limitation, not hidden.** Two runs (2×2×2 MP,
nbands 12/14, SCF converged to |drho| < 6e-9):

| run | Mg: bare / core / dia / para → σ_iso | O: bare / core / dia / para → σ_iso |
|-----|----------------------------------------|--------------------------------------|
| 40 Ry / 160 Ry | 342.1 / 412.8 / 109.4 / 528.6 → **1392.8** | −1912.1 / 271.1 / 18.4 / 62.6 → **−1560.1** |
| 60 Ry / 400 Ry | 328.0 / 412.8 / 96.4 / 616.4 → **1453.6** | −3758.3 / 271.1 / 18.5 / 69.9 → **−3398.8** |

vs the ≈215 ppm ¹⁷O anchor (and ≈560 ppm ²⁵Mg): the O smooth bare term is
wildly wrong and **diverges with ecut** (−1912 → −3758 going 40 → 60 Ry at
fixed mesh), so this is not a basis-convergence tail. The core/dia_aug terms are
stable and sane; the sickness is isolated to the smooth analytic-USPP
`sigma_shielding_dq` bare route (and the para_aug that consumes its response) on
the hard-augmentation O dataset with semicore Mg. The same assembly on the soft
Si PAW hits its anchor, and the driver calls the identical functions in both
cases — i.e. this is the pre-existing analytic-USPP generalization gap flagged
in `docs/plans/flapw-nmr-consolidation.md` (Bucket 3: does the ∂²/∂q² term need
an augmentation second derivative?), now with a concrete failing material.
Follow-up belongs to the shielding-route consolidation, not the driver wiring.

## Error bars / honesty

- The 2×2×2 mesh and 12 Ry ecut for Si are the *shipped-test* settings, chosen
  so the driver number is directly checkable against the validated direct-call
  number; they are not converged production settings. The finite-k/ecut error on
  Si at these settings was previously measured at the ~10–20 ppm level (the
  slow-tier test's acceptance band is 360–440 ppm around the ≈400 anchor).
- Absolute σ carries the frozen-core Lamb term from the PAW dataset's core
  density; core relaxation is not included (consistent with standard GIPAW).
- The G = 0 macroscopic shape term is omitted (bulk convention).
- The MgO failure above bounds the current trust region: the absolute assembly
  is validated for soft PAW datasets (Si-like); hard-augmentation first-row
  anions (O) through the analytic-USPP bare route are NOT yet trustworthy and
  diverge with ecut. Treat any `gipaw_absolute` number on O/N/F sites as
  unvalidated until the Bucket-3 scoping lands.
