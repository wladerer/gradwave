# USPP/PAW nspin=2 mixing default, johnson vs pulay

## The question

The USPP/PAW SCF defaulted `nspin=2` to `pulay` while defaulting `nspin=1` to
`johnson` (`_resolve_uspp_mixing_scheme` in `scf/uspp_loop.py`). The magnetic
half of that split rested on one measurement, namely bcc Fe blowing up from 29
iterations under pulay to 93 under forced johnson, from johnson discarding a
becsum step-damping that a robust ferromagnet was thought to need. The
2026-07-30 case-study campaign (`research/convergence-case-studies`) could not
reproduce that blowup, measuring forced johnson at 16 iterations against pulay's
30 on the same cell, and `docs/manual/wisdom.md` carries a dated note saying the
default now rests on a stale number and should be re-benchmarked. This is that
re-benchmark.

## What the prior nspin=1 work carried

The local branch `perf/scf-uspp-johnson-default` (tip `2ffcf02`, remote deleted)
defaulted the PAW/USPP path to johnson for `nspin=1`, with the evidence Si
18→12, Cu 19→13, Pt 21→12, and 8-atom Si 20→13, all at a bit-identical fixed
point. That branch itself was never merged. The same change landed on main as
PR #150 (`f836d01`), so the nspin=1 half of the question is already settled and
in production. That commit deliberately kept `nspin=2` on pulay, citing the same
bcc Fe 29→93 blowup this note reopens. The nspin=2 half is what remained open.

## Method

Each `(system, scheme, start_mag)` runs `scf_uspp` with identical SCF settings
across schemes, so iteration counts compare directly. Settings match the
case-study campaign, which matched the wisdom.md-era numbers. The oracle is the
converged fixed point, namely the free energy and the absolute moment. A scheme
that lands on a different energy or moment picked a different branch and is
recorded as such, not averaged with the others.

- **bcc Fe PAW**, `Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`, a = 2.87, 45/360 Ry, 8x8x8,
  nbands 12, gaussian 0.1 eV, etol 1e-8, rhotol 1e-6, `use_symmetry=True`.
  start_mag 0.7 and 0.3.
- **fcc Ni PAW**, `Ni.pbe-spn-kjpaw_psl.1.0.0.UPF`, a = 3.52, same cutoffs and
  mesh. start_mag 0.6, 0.3, and 0.02, so branch behavior near the Stoner
  boundary is visible across seeds rather than one lucky seed.
- **AFM Fe 2-atom (B2)**, two Fe in a simple-cubic cell seeded opposite
  (+0.7, -0.7), 45/360 Ry, 4x4x4, `use_symmetry=False`. This is a two-sublattice
  stand-in for the documented AFM Cr case. There is no PAW or USPP Cr or Co
  pseudo in the repo, only norm-conserving ones (`Cr_ONCV`, `Co_ONCV`, and the
  norm-conserving `benchmarks/delta_gauge/pseudos/Cr.upf`), so AFM Cr cannot run
  on this path. Collinear `nspin=2` cannot use magnetic (magmoms) symmetry, and
  the plain crystallographic spacegroup of B2 would symmetrize the two Fe equal
  and erase the AFM order, so this runs on the full BZ.

Ran on asus, 8 threads. Reproduce with

```bash
uv run python experiments/uspp_mixing_default/run_bench.py fe ni --threads 8
uv run python experiments/uspp_mixing_default/run_bench.py afm_fe --threads 8 --out results_afm.json
```

## Results

Iterations to the standard tolerances, wall time on asus, converged free energy,
and absolute moment. `conv` is whether the run reached tolerance inside the cap
(120 for Fe, 80 for Ni, 60 for AFM Fe). Tags are the flight-recorder diagnosis.

| system | start_mag | pulay | johnson | broyden | E (eV) | m (muB) | fixed point |
|---|---|---|---|---|---|---|---|
| bcc Fe PAW | 0.7 | 30 | **16** | 32 | -4479.034400 | 2.4974 | shared |
| bcc Fe PAW | 0.3 | 26 | **16** | 32 | -4479.034400 | 2.4974 | shared |
| fcc Ni PAW | 0.6 | 27 | **18** | 29 | -5836.773436 | 0.7897 | shared |
| fcc Ni PAW | 0.3 | 25 | **15** | 31 | -5836.773436 | 0.7897 | shared |
| fcc Ni PAW | 0.02 | 19 | **13** | 21 | -5836.773436 | 0.7897 | shared |
| AFM Fe (B2) | +0.7/-0.7 | 58 | **31** | 60 (no conv) | -8957.126342 | 3.6434 | see below |

Wall times track the iteration counts. On bcc Fe at start_mag 0.7 johnson took
15.7 s against pulay's 23.8 s and broyden's 29.5 s. On AFM Fe johnson took
158 s against pulay's 260 s.

## Branch stability

Johnson reached the pulay fixed point on every converging case, to the printed
precision on both the free energy and the moment, and it did so in fewer
iterations every time. The underseeded fcc Ni at start_mag 0.02 is the case
built to expose a branch flip, and every scheme held the moment at 0.79 muB
rather than collapsing to the nonmagnetic branch. Johnson caused no collapse and
no flip anywhere.

The one non-convergence is broyden on AFM Fe. It hit the 60-iteration cap
without reaching tolerance, wandered to a different energy (-8958.003173 eV) and
moment (4.30 muB), and tripped charge-sloshing and level-crossing tags. That is
a broyden failure, not a johnson one. Broyden is not the proposed default and
was carried only as a third column. If anything the AFM case sharpens the
argument for johnson, since johnson converged there cleanly where broyden did
not.

No tag fired on any Fe or Ni run under any scheme.

## Decision

Flip the USPP/PAW `nspin=2` default to johnson. Johnson wins every magnetic PAW
pair here at the same fixed point, with no blowup and no branch flip, so the
mirror-inversion the default encoded is now unjustified and the whole USPP/PAW
path defaults to johnson, matching the norm-conserving path's magnetic choice
rather than inverting it. `_resolve_uspp_mixing_scheme` now returns johnson for
both nspin channels, an explicit `mixing_scheme` still wins, and pulay stays
available for anyone who wants it.

The becsum step-damping the pulay default leaned on was a crutch tuned for
pulay, described in the crutch re-audit note in `docs/manual/wisdom.md`. Matching
QuantumESPRESSO's unscaled becsum mix removed the johnson penalty, which is the
likely reason the 93-iteration blowup no longer exists. The #199 Stoner
`spin_precond` change is the other candidate named in the case-study note. I did
not isolate which one closed the gap, and for the default it does not matter,
since the blowup is gone under current code either way.

## Files

- `run_bench.py`, the runner. Case keys `fe`, `ni`, `afm_fe`.
- `results.json`, the fe and ni rows. `results_afm.json`, the AFM Fe rows.
- `traces/`, one flight-recorder trace per run.
