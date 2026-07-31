# Pulay pressure-error estimator accuracy (issue #227)

Truth harness `measure.py` on the sheared-Si cell from
`tests/integration/test_stress_error.py`. Ratio is estimate / true, where

    true  = P_raw(45 Ry) - P_raw(ecut),  P_raw = -tr(sigma)/3  [GPa]
    est   = estimate_pressure_error(res_ecut, ...)  [GPa]

`P_raw` uses `postscf.stress.stress(symmetrize=False)`, converted with
1 eV/A^3 = 160.2176634 GPa. The estimate uses the default annulus
(factor*ecut, factor=2.5) so the reference at 45 Ry is a genuine target the
estimator does not see. A ratio of 1.0 is exact. The estimator is a correctly
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

## Rung 1: annulus factor is flat (measured negative)

The pressure estimate is nearly independent of the annulus factor, so there is no
tail worth extrapolating. Sweeping the factor at fixed ecut (2x2x2 Si, default
strain) gives (P_est in GPa):

| ecut (Ry) | f=1.8 | f=2.0 | f=2.2 | f=2.6 | f=3.0 | f=3.5 | f=3.9 |
|---|---|---|---|---|---|---|---|
| 10 | 2.361 | 2.421 | 2.393 | 2.282 | 2.247 | 2.281 | 2.318 |
| 16 | 0.479 | 0.477 | 0.488 | 0.508 | 0.519 | 0.524 | 0.525 |

The spread is a few percent and non-monotone. The frozen-state energy error
itself grows with the factor and saturates, but its volume derivative (the
pressure) does not, because the high-G tail responds weakly to the strain. This
matches the asus quartz ladder at 65 Ry, flat to +/-0.3% across factors 2.0 /
2.5 / 3.0 (+3.809 / +3.783 / +3.796 GPa, issue #227). Annulus truncation is not
the source of the under-estimate, so no extrapolation feature is warranted.

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

## Rung 2: iterative annulus Sternheimer solve (solver="cg")

`solver="cg"` replaces the diagonal kinetic-only resolvent (T_G - eps)^-1 with a
preconditioned conjugate-gradient solve of the annulus-projected operator
P_annulus (H - eps) P_annulus, capturing the local and nonlocal potential
coupling the diagonal drops. This is the decisive improvement. The ratio rises
from 0.43-0.60 to 0.61-0.77 at every (ecut, kmesh) point, monotone toward 1 as
ecut grows, for 110 to 224 extra H-applies (about 0.1 to 0.4 s per estimate).
The diagonal baseline column is bit-identical, so the default path is untouched.

Full comparison (ratio = estimate / true; 1.0 exact):

| kmesh | ecut (Ry) | P_true (GPa) | baseline | extrap | cg | extrap+cg |
|---|---|---|---|---|---|---|
| 2x2x2 | 10 | 5.304 | 0.435 | 0.343 | 0.611 | 0.636 |
| 2x2x2 | 12 | 3.496 | 0.478 | 0.407 | 0.689 | 0.526 |
| 2x2x2 | 14 | 1.806 | 0.523 | 0.617 | 0.710 | 0.814 |
| 2x2x2 | 16 | 0.843 | 0.599 | 0.615 | 0.770 | 0.881 |
| 3x3x3 | 10 | 6.395 | 0.434 | 0.328 | 0.608 | 0.631 |
| 3x3x3 | 12 | 3.799 | 0.478 | 0.413 | 0.688 | 0.607 |
| 3x3x3 | 14 | 2.111 | 0.524 | 0.621 | 0.713 | 0.775 |
| 3x3x3 | 16 | 1.085 | 0.582 | 0.597 | 0.750 | 0.825 |

`extrap+cg` is sometimes higher but not monotone (it drops to 0.526 at 2x2x2 /
12 Ry, below cg alone at 0.689), so extrapolation stays off by default even on
top of the CG solver. The robust recommendation is `solver="cg"` with
`extrapolate=False`.

## Rung 3: self-consistent (Schur-coupled) dressing (no-go, not shipped)

Adding the Harris-like double-counting term (1/2)<drho|K_Hxc|drho> to the frozen
energy error to restore self-consistency moves the silicon ratio by only one to
two percent, far short of the remaining gap, on both solvers:

| point | diagonal | diagonal+sc | cg | cg+sc |
|---|---|---|---|---|
| 2x2x2 / 10 Ry | 0.435 | 0.441 | 0.611 | 0.624 |
| 2x2x2 / 16 Ry | 0.599 | 0.602 | 0.770 | 0.772 |

The double-counting term is a positive penalty that can only shrink the estimate,
so self-consistency is not the missing lever. Gated off and not shipped. The
measured no-go is recorded in docs/ideas.md.

## Quartz validation (relaxed alpha-quartz, 65 Ry)

Relaxed geometry from benchmarks/results/sio2-elastic/out_relax/relax.json, Si/O
ONCV PBE, kmesh (3,3,3), use_symmetry=False, smearing none. Ground-truth error
7.653 GPa (P_raw(130 Ry) - P_raw(65 Ry), given). The harness reproduces the
given reference exactly: P_raw(65 Ry) = -3.779 GPa and the diagonal estimate
3.783 GPa (ratio 0.494).

| solver | P_est (GPa) | ratio | cost |
|---|---|---|---|
| diagonal | 3.783 | 0.494 | 0 extra H-applies |
| cg | 4.048 | 0.529 | 168 H-applies, ~81 s |

The CG annulus solve improves quartz from 0.494 to 0.529, a smaller lift than on
silicon (0.49 to 0.69). At 65 Ry the annulus sits at very high kinetic energy
where the diagonal resolvent is already close, so the potential coupling the CG
restores is a smaller relative correction. The remaining quartz gap is in the
annulus-only restriction and the second-order truncation, which rung 1 (flat) and
rung 3 (no-go) do not close cheaply.
