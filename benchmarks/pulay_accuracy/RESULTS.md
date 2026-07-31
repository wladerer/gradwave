# Pulay pressure-error estimator accuracy (issue #227)

Truth harness `measure.py` on the sheared-Si cell from
`tests/integration/test_stress_error.py`. Ratio is estimate / true, where

    true  = P_raw(45 Ry) - P_raw(ecut),  P_raw = -tr(sigma)/3  [GPa]
    est   = estimate_pressure_error(res_ecut, ...)  [GPa]

`P_raw` uses `postscf.stress.stress(symmetrize=False)`, converted with
1 eV/A^3 = 160.2176634 GPa. The estimate uses the default annulus
(factor*ecut, factor=2.5) so the reference at 45 Ry is a genuine target the
estimator does not see. A ratio of 1.0 is exact; the estimator is a correctly
signed first-order indicator that historically under-estimates.

Reproduce:

    uv run python benchmarks/pulay_accuracy/measure.py \
        --variants baseline,extrap,cg,extrap+cg \
        --out benchmarks/pulay_accuracy/RESULTS.md

## Baseline (rung 0)

Ratios are estimate / true; 1.0 is exact.

| kmesh | ecut (Ry) | P_true (GPa) | baseline |
|---|---|---|---|
| 2x2x2 | 10 | 5.304 | 0.435 |
| 2x2x2 | 12 | 3.496 | 0.478 |
| 2x2x2 | 14 | 1.806 | 0.523 |
| 2x2x2 | 16 | 0.843 | 0.599 |
| 3x3x3 | 10 | 6.395 | 0.434 |
| 3x3x3 | 12 | 3.799 | 0.478 |
| 3x3x3 | 14 | 2.111 | 0.524 |
| 3x3x3 | 16 | 1.085 | 0.582 |

## Rung 1: annulus-truncation extrapolation (off by default)

`extrapolate=True` samples the frozen-state energy error at annulus factors
(1.8, 2.2, 2.6, 3.0) and fits the tail `dE(ecl) = dE_inf + c*ecl**(-p)`, using
`dE_inf` in the volume derivative. Finding: the annulus tail is nearly
volume-independent on silicon, so it barely enters the pressure (the volume
derivative of the tail is small). Extrapolating a nearly-flat, noisy tail does
not move the ratio toward 1 at every ecut. It is worse at ecut 10 to 12 (the
independent minus/plus fits amplify sampling noise into the small pressure
difference) and only marginally better at 14 to 16. The feature is shipped but
off by default. The under-estimate comes from the diagonal resolvent, not
annulus truncation, which motivates rung 2.

| kmesh | ecut (Ry) | P_true (GPa) | baseline | extrap |
|---|---|---|---|---|
| 2x2x2 | 10 | 5.304 | 0.435 | 0.343 |
| 2x2x2 | 12 | 3.496 | 0.478 | 0.407 |
| 2x2x2 | 14 | 1.806 | 0.523 | 0.617 |
| 2x2x2 | 16 | 0.843 | 0.599 | 0.615 |
| 3x3x3 | 10 | 6.395 | 0.434 | 0.328 |
| 3x3x3 | 12 | 3.799 | 0.478 | 0.413 |
| 3x3x3 | 14 | 2.111 | 0.524 | 0.621 |
| 3x3x3 | 16 | 1.085 | 0.582 | 0.597 |
