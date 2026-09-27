# MgO ¹⁷O bare shielding: the CG resolvent closes the ecut divergence

**Deliverable that was "pending the run" in `quartz_shielding_profile.md`
(Deliverable 2) and `gipaw_absolute_driver_validation.md` (the MgO limitation):
the recorded σ_iso(¹⁷O) ecut ladder for the OLD dense-eigh resolvent vs the
matrix-free S-metric CG resolvent, on one fixed MgO ground state.**

## Question

The analytic-USPP bare ¹⁷O magnetic shielding (`sigma_shielding_dq` through the
S-metric context) DIVERGED with plane-wave ecut on the hard-augmentation O
dataset. `gipaw_absolute_driver_validation.md` documented the O total σ_iso
running −1560 → −3399 ppm (bare −1912 → −3758) going 40 → 60 Ry at fixed mesh —
not a basis-convergence tail. `analytic_uspp_ecut_divergence.md` traced the root
cause to the dense-eigh S-orthonormal conduction resolvent (`_resolvent_apply_s`)
expanding the ∂/∂q response in the ill-conditioned S-orthonormal eigenbasis of
the augmentation overlap S(k), whose condition number climbs to 10³–10⁴ as ecut
grows. The scoped fix — the matrix-free S-metric CG Sternheimer resolvent
(`_SMetricResolventCG`, `kgeometry_nmr.py`), selected by
`response_backend="auto"` above cond(S) > 50 — never forms that eigenbasis. Its
corrected ¹⁷O numbers were never recorded. This note records them.

## Configuration (pinned)

- **Cell**: MgO rocksalt, a = 4.21 Å, 2-atom fcc primitive. Mg at (0,0,0), O at
  (½,½,½).
- **Pseudos**: PAW route (the divergence and the fix live on the S-metric path;
  the norm-conserving/ONCV route has S = I and does not exhibit this divergence).
  Mg `Mg.pbe-n-kjpaw_psl.0.3.0.UPF` (SSSP-efficiency PAW, z_val = 2), O
  `O.pbe-n-kjpaw_psl.1.0.0.UPF` (committed under `tests/fixtures/qe/pseudos/` —
  the hard first-row anion that drives cond(S)). The semicore
  `Mg.pbe-spnl-kjpaw` of the original §0 report is not on the box; this Mg PAW
  reproduces the same ill-conditioned O-overlap regime (cond(S) ≈ 2186 at 40 Ry,
  matching `analytic_uspp_ecut_divergence.md` §6).
- **k-mesh**: 2×2×2 Monkhorst–Pack, `symmetry=False` (finite-q response needs the
  full spatial mesh with ≥2 axes > 1).
- **ecut ladder** (fixed ecutrho = 4× ratio, fixed mesh, so only the wavefunction
  cutoff varies): 40/160, 50/200, 60/240, 70/280 Ry. The 40/160 rung matches the
  documented divergence rung.
- **SCF**: PBE, nbands = 12, etol 1e-8, rhotol 1e-7, diago_tol 1e-9.
- **Shielding**: `sigma_shielding_gipaw` (bare + core + dia_aug + para_aug),
  `use_symmetry=False`, `cg_tol=1e-10`, `max_iter=600`. `out["bare"]` is the
  `sigma_shielding_dq` term (the diverging quantity); `out["total"]` the anchor.
- **Backends**: `response_backend="dense"` (the old dense-eigh resolvent, to
  reproduce the divergence) and `response_backend="auto"` (routes to the
  `_SMetricResolventCG` matrix-free CG resolvent because cond(S) ≫ 50).
- Driver: `experiments/autoapw/mgo_17o_cg_convergence.py`.

## Results — σ_iso(¹⁷O), dense vs CG (ppm)

One MgO ground state per rung; both backends run on the identical SCF result and
S-metric context, so only the resolvent differs.

| ecut/rho (Ry) | grid | npw | cond(S)max | **dense** bare O | **dense** total O | **CG** bare O | **CG** total O |
|---|---|---|---|---|---|---|---|
| 40 / 160 | 24³ | 537  | 2186 | −1871.95 | −1519.44 | **−70.50** | **+282.01** |
| 50 / 200 | 25³ | 749  | 4233 | −3237.23 | −2878.09 | **−72.02** | **+287.13** |
| 60 / 240 | 27³ | 965  | 5588 | −3779.03 | −3418.69 | **−61.80** | **+298.54** |
| 70 / 280 | 30³ | 1243 | 6247 | −3395.33 | −3035.02 | **−57.37** | **+302.95** |

`auto` routed to the CG backend at every rung (cond(S) ≫ 50). Mg is ecut-stable
on both backends (dense/CG bare Mg −16 to −20, total Mg +575 to +579); the
sickness is isolated to the hard-anion O site, as expected.

## Verdict — UNBLOCKED

The CG resolvent PLATEAUS where the dense resolvent DIVERGES.

- **Dense (old)** bare ¹⁷O swings over a ~1900 ppm range (−1872 → −3779 → −3395)
  with no convergence and a non-monotonic turnover at 70 Ry — the signature of an
  ill-conditioned numerical instability that tracks cond(S) (2186 → 6247), not a
  basis-convergence tail. This reproduces the documented divergence (−1912 →
  −3758 over 40 → 60 Ry with the semicore Mg of the original report; −1872 →
  −3779 here). The 60 → 70 Ry step alone moves the dense bare (and total) O by
  **+384 ppm**.
- **CG (auto)** bare ¹⁷O is flat: −70.50 → −72.02 → −61.80 → −57.37, a **14.6 ppm**
  spread over the entire 40 → 70 Ry ladder and **+4.4 ppm** across the top two
  rungs (60 → 70). The total ¹⁷O is likewise stable and smoothly convergent
  (+282.0 → +303.0, **+4.4 ppm** over 60 → 70). The ×2-per-rung dense blow-up is
  gone; a few-ppm ecut convergence is recovered. The 40 Ry CG numbers reproduce
  the partial values recorded in `analytic_uspp_ecut_divergence.md` §6 (bare
  −70.5, total +282.0) exactly.

**The bare ¹⁷O ecut divergence is closed by the CG backend.** Oxides (hard
first-row anions) that were untrustworthy on the dense-eigh route are ecut-stable
on `response_backend="auto"`.

**Honest caveat (not a divergence).** The CG total ¹⁷O plateaus near +303 ppm,
~90 ppm above the ≈ +215 ppm literature GIPAW anchor. That residual is *stable*
with ecut (the divergence is gone) and reflects the pseudo choice here — a
non-semicore Mg PAW (`Mg.pbe-n-kjpaw_psl.0.3.0`, z_val = 2, the semicore
`Mg.pbe-spnl` of the original report is not on the box) plus the hard
`O.pbe-n-kjpaw`, at test-grade 2×2×2 / 40–70 Ry settings — together with the
standard frozen-core GIPAW approximation. This note demonstrates ecut-STABILITY
(the blocker), not absolute-anchor agreement, which is a separate convergence
(k-mesh, ecut, pseudo) exercise.

## Provenance

- Host: **asus** (22 cores), `OMP_NUM_THREADS=8`, routed through
  `./scripts/gwq --host asus run --group bench` (pueue slot-capped).
- gradwave `main` @ **e23c7768** (canonical checkout; the `_SMetricResolventCG`
  backend is shipped on main).
- Pseudos: Mg `Mg.pbe-n-kjpaw_psl.0.3.0.UPF` (SSSP-efficiency PAW, md5
  `24ecedc7f3e3cbe212e682f4413594e4`), O `O.pbe-n-kjpaw_psl.1.0.0.UPF` (committed).
- Driver `experiments/autoapw/mgo_17o_cg_convergence.py`; raw results
  `experiments/autoapw/data/mgo_17o_cg_convergence.json`.
- Wall time: ~2.95 h for the 8-eval ladder (dense 746/1052/1455/1860 s, CG
  746/1063/1485/1934 s per rung at 40/50/60/70 Ry).
