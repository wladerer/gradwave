# ecut recommender probe

Feasibility probe (not a product). Question: can the whole error-vs-cutoff CURVE
be read off a single converged SCF by binning the Cances complement correction by
its per-plane-wave kinetic energy T_G and cumulatively summing from the top, and
does inverting that curve give a usable settings recommender ("energy to 1
meV/atom -> use ecut = X")?

## Method

The shipped estimator (`postscf.discretization_error.estimate_density_error`)
returns one scalar, the second-order energy lowering summed over the complement
annulus `ecut < T_G <= ecut_large`:

    dE = sum_k w_k sum_i f_i sum_G Re[ dpsi_i(G)* R_i(G) ],
    dpsi_i(G) = -R_i(G) / (T_G - eps_i)  on the annulus, 0 elsewhere.

Every summand carries its own T_G, and its denominator depends only on T_G (not
on where the basis was truncated). So bin the per-G summand by T_G and sum only
the part with `T_G > ecut'`: that is the energy still sitting above a virtual
cutoff ecut', i.e. the predicted remaining error at ecut', for every ecut' in
[ecut, ecut_large], from one probe SCF.

`curve.py::energy_error_curve` implements this. It reuses the shipped estimator's
private helpers (no change to shipped behaviour). Sanity check, in both systems
below: the full-annulus curve value reproduces the shipped `denergy` to round-off
(rel diff 2e-16 for Si, 0e0 for Al), confirming the binning is exact.

The annulus top `ecut_large` was set equal to the sweep reference cutoff, so
"predicted error rel to ecut_large" and "true error rel to reference" measure the
same thing and are directly comparable. Constraint from the shipped code:
`ecut_large <= 4*ecut`.

Compute: `torch.set_num_threads(2)`, CPU, small cells, runs sequential.

## Result 1: Si insulator (energy)

2-atom Si, PBE, Si_ONCV_PBE-1.2 (4 e), 4x4x4, use_symmetry, smearing=none.
Probe ecut = 15 Ry, ecut_large = 40 Ry. Reference = 40 Ry sweep point.

Sweep energies [eV]: 12->-213.859607, 15->-214.141396, 18->-214.228263,
22->-214.277086, 26->-214.293577, 32->-214.305687, 40->-214.314152.

Predicted vs actual remaining error (rel to 40 Ry), meV/atom:

| ecut (Ry) | actual | predicted | ratio pred/act |
|---|---|---|---|
| 15 | 86.378 | 63.687 | 0.74 |
| 18 | 42.945 | 25.945 | 0.60 |
| 22 | 18.533 | 10.842 | 0.59 |
| 26 | 10.288 |  6.670 | 0.65 |
| 32 |  4.233 |  2.368 | 0.56 |

Recommended ecut [Ry] for targets:

| target (meV/atom) | predicted | true |
|---|---|---|
| 10 | 23.0 | 26.5 |
|  3 | 31.0 | 34.5 |
|  1 | 36.0 | 38.5 |

The curve tracks the sweep across the WHOLE 15-32 Ry window with a ratio of
0.56-0.74: a consistent underestimate near 0.6x, always inside a factor of 2. The
recommender is therefore optimistic by ~3 Ry at every target (it recommends a
cutoff whose true error is ~1.5x the target). Same direction and size as the
manual's pressure-error finding (0.5-0.75x, a consistent under-estimate).

## Result 2: Al metal (energy)

1-atom fcc Al, PBE, Al_ONCV_PBE-1.2, 6x6x6, use_symmetry, gaussian 0.14 eV.

Note: Al_ONCV_PBE-1.2 is an 11-electron pseudo (2s2p semicore), converged only by
~80-100 Ry. A 15 Ry probe (as first attempted) is deeply pre-asymptotic and
produces garbage (140 eV/atom "errors", ratio collapsing to 0.3). The annulus
diagonal-dominance assumption needs the probe to be in the asymptotic regime.
Re-run with probe ecut = 40 Ry, ecut_large = 100 Ry, reference = 100 Ry (which is
converged to 0.04 meV/atom vs 120 Ry).

Predicted vs actual remaining error (rel to 100 Ry), meV/atom:

| ecut (Ry) | actual | predicted | ratio pred/act |
|---|---|---|---|
| 40 | 756.192 | 701.755 | 0.93 |
| 45 | 301.369 | 261.937 | 0.87 |
| 50 | 122.325 |  97.719 | 0.80 |
| 60 |  20.012 |  12.629 | 0.63 |
| 70 |   2.489 |   1.185 | 0.48 |
| 85 |   0.084 |   0.272 | 3.23 |

Recommended ecut [Ry] for targets:

| target (meV/atom) | predicted | true |
|---|---|---|
| 10 | 61.5 | 66.0 |
|  3 | 66.5 | 70.0 |
|  1 | 71.0 | 79.5 |

Smearing / partial occupation does NOT degrade the curve. Over the meaningful
window (40-70 Ry, true error 2.5-756 meV/atom) the ratio is 0.48-0.93, again
inside a factor of 2, and closer to 1 at the probe cutoff than Si was. The 3.23
ratio at 85 Ry is a divide-by-near-zero artifact: the true error there is 0.084
meV/atom, at the reference/noise floor, so the ratio is meaningless. Recommender
is again optimistic, by 4-8 Ry.

## Result 3: displaced Si (force) -- the cumulative trick FAILS here

2-atom Si, atom 1 displaced +0.10 A along x, 4x4x4, no symmetry. Probe 15 Ry,
ecut_large 40 Ry. Force curve built by raising the annulus floor to ecut' (mask
dpsi/drho to T_G > ecut', `curve.py::density_error_at_virtual_cutoff`) and running
the shipped `estimate_force_error`.

| ecut (Ry) | act max\|dF\| | pred max\|dF\| | ratio | act Fx1 | pred Fx1 | corr |
|---|---|---|---|---|---|---|
| 15 | 0.0170 | 0.0089 | 0.53 | +0.0170 | +0.0089 | +1.00 |
| 18 | 0.0077 | 0.0063 | 0.82 | -0.0077 | -0.0063 | +1.00 |
| 22 | 0.0049 | 0.0022 | 0.46 | +0.0049 | -0.0022 | -1.00 |
| 26 | 0.0043 | 0.0001 | 0.03 | +0.0043 | +0.0001 | +0.95 |
| 32 | 0.0006 | 0.0011 | 1.72 | -0.0006 | +0.0011 | -1.00 |

Only the probe cutoff and one step beyond (15, 18 Ry) track in sign and rough
magnitude. By 22 Ry the predicted sign flips (corr -1.00). The reason is visible
in the sweep itself: the true force error is tiny (max 0.017 eV/A) and
NON-monotonic in ecut (Fx1 goes -1.41933, -1.39465, -1.40727, -1.40663, -1.40172,
-1.40235) -- egg-box / k-sampling noise dominates the real basis signal above ~22
Ry, so the "ground truth" force error is itself unreliable there and the
first-order prediction has nothing coherent to track. The single-shot force
estimate AT the probe cutoff is fine (right sign, ~0.5x magnitude, matching the
manual); extending it to a curve is not.

## Findings

Usable extrapolation range (energy): the one-shot curve stays within a factor of
2 of truth over essentially the ENTIRE annulus [ecut, ~0.8*ecut_large], not just
near the probe. It only breaks in the last ~20% of the window, where the true
error has already dropped below ~1-2 meV/atom (at/under the reference noise
floor). The bias is a consistent underestimate, ratio ~0.6 (Si) to ~0.6-0.9 (Al),
never an overestimate in the meaningful regime -- so it fails safe on magnitude
but the recommender it drives is OPTIMISTIC (too-loose ecut) by 3-8 Ry unless the
predicted error is scaled up by ~1.5x (a calibration constant, stable across both
systems tested) or a safety margin is added.

Precondition: the probe cutoff must be in the asymptotic regime. For a semicore
pseudo (Al 11-e) that means a high probe (40 Ry, not 15). A too-loose probe gives
nonsense; the estimator has no way to know it is pre-asymptotic, which is a real
gotcha for an automated recommender.

Force curve: no. Only the probe point is trustworthy; the true force error hits
the egg-box noise floor within one or two steps and the prediction sign-flips.

## Go / no-go

GO for an ENERGY ecut recommender; NO-GO for a force ecut recommender.

The energy curve is nearly free (one probe SCF, one extra pass already done for
`denergy`, plus a cheap bin-and-cumsum), reproduces the shipped scalar exactly,
and tracks the true sweep within 2x over the whole annulus. That is enough to turn
"energy to X meV/atom -> ecut = Y" into a single-calculation answer, with the
caveats that (a) it needs a calibration factor (~1.5x on the predicted error) or a
safety margin to stop being optimistic, and (b) it must refuse / warn when the
probe is pre-asymptotic.

Proposed API (if built):

    from gradwave.postscf.discretization_error import ecut_recommendation

    rec = ecut_recommendation(
        res,                       # a probe SCF, ideally in the asymptotic regime
        targets_meV_per_atom=(10, 3, 1),
        ecut_large=None,           # default 2.5*ecut, capped at 4*ecut
        calibration=1.5,           # scale predicted error up (fail-safe margin)
    )
    rec.curve         # (ecut'[eV], dE[eV]) arrays -- the whole error-vs-cutoff curve
    rec.ecut_for      # {target_meV_per_atom: recommended_ecut_eV}
    rec.asymptotic    # bool + reason; False warns the probe is too loose

It is a thin wrapper: `energy_error_curve` for the curve, `recommend_ecut` for the
inversion, an `int_drho`-style asymptotic-regime check, and a calibration knob.
Restrict the first cut to the energy channel and to NC nspin=1 (what this probe
covered); the same binning extends to USPP/PAW and nspin=2 energy (same scalar
`denergy` structure) but that was not measured here.

## Files

- `curve.py` -- `energy_error_curve`, `recommend_ecut`,
  `density_error_at_virtual_cutoff` (force-curve helper). No shipped code changed.
- `run_si.py`, `run_al.py`, `run_si_force.py` -- the three experiments.
- `si.log`, `al.log`, `force.log` -- raw captured output.
