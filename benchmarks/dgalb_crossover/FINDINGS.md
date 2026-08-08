# DG-ALB crossover study — findings

Does the DG-ALB (differentiable adaptive-local-basis / DGDFT) reformulation buy
**wall-time**, not just memory, and above what atom count? Measured on asus
(8 threads, fp64), Si supercells, ecut 20 Ry, ALBs M=8/core-atom, DG element =
1 conventional cell with an ext=2 (buffer) extended element, davidson fixed at
12 iterations. See `bench_dgalb.py` for the full REAL-vs-MODEL accounting.

## Results

```
atoms  n_elem   npw_pw  nb_pw   npw_e   M_el     D   npw/D    t_pw_s   t_alb_s  t_glob_s  t_dg_s   PW/DG
    8      1      1647     20   13133    64      64    26x      0.07      3.22     0.00      3.22   0.02x
   64      8     13133    154   13133    64     512    26x      9.71     26.02     0.02     26.04   0.37x
  216     27     43867    519   13133    64    1728    25x   OOM(est)    92.93     0.31     93.24     —
  512     64    104415   1229   13133    64    4096    25x   OOM(est)   219.95     4.13    224.08     —
 1000    125     —        —     13133    64    8000    —      skip       442.46    (gated)  442.46     —
```

`t_pw` = full plane-wave `davidson_batched` (REAL, the object DG replaces).
`t_dg` = ALB build (REAL) + reduced global `eigh` (MODEL, dense upper bound).
`OOM(est)` = the plane-wave davidson's estimated peak exceeds the 9 GB budget,
so it is skipped rather than tripping the OS OOM-killer (the 14 GB box cannot
hold the global sphere + projector table — the wall DG-ALB removes). At 1000
atoms plane waves cannot even be *set up*: `setup_system` builds the dense
projector table (nproj x npw ~ 32 GB) and OS-OOM-kills before any peak gate can
act, so that baseline is skipped (`--no-pw`) and the dense global eigh gated off
— the row isolates the ALB build.

## The ALB build is linear — the O(N) signature

Wall time per element is flat across two decades of atom count:

```
atoms   n_elem   t_alb (s)   s / element
    8       1        3.22        3.22
   64       8       26.02        3.25
  216      27       92.93        3.44
  512      64      219.95        3.44
 1000     125      442.46        3.54
```

~3.5 s/element with only a slight upward drift — the ALB regeneration is O(N)
while the plane-wave davidson it replaces is ~N^2.4 (and un-runnable past
~128–216 atoms). The crossover made concrete: a linear curve overtaking a
superlinear one that also hits a hard memory wall.

## What the numbers say

1. **There is a genuine speed win, and the crossover sits at ~150–200 atoms.**
   Below it plane waves win (8 atoms 0.02x, 64 atoms 0.37x — one big efficient
   dense solve beats many small element solves). Plane-wave cost scales ~N^2.4
   here (0.07 s → 9.71 s over 8x atoms); the ALB build scales **linearly**
   (~3.5 s per element: 3.2 → 26 → 93 → 224 s over 1 → 8 → 27 → 64 elements).
   Two curves of different order cross: extrapolating the plane-wave N^2.4 fit,
   PW would need ~184 s at 216 atoms vs DG's 93 s (~2x DG win) and ~1130 s at
   512 vs 224 s (~5x), the margin widening with size.

2. **The memory wall arrives first and is absolute.** The plane-wave baseline
   is already un-runnable at 216 atoms on 14 GB (global sphere + projector
   table), while DG-ALB completes 216 (93 s) and 512 (224 s) atoms — because
   the ALB representation never forms a global npw-sized object. So above the
   crossover plane waves first lose on speed, then simply cannot run.

3. **The ALB basis is ~25x smaller than plane waves, flat across sizes**
   (npw/D ≈ 25–26x). That constant dimension reduction is the lever behind
   both the memory win and the small global solve.

4. **At 10^2–10^3 atoms the DG cost is the ALB *build*, not the global solve.**
   `t_glob` (dense global eigh) stays tiny (0.02–4 s) next to `t_alb`
   (26–224 s). The per-SCF ALB regeneration dominates; the global
   diagonalization is not yet the bottleneck at these sizes.

## Reading the result honestly

- The crossover atom count depends on the buffer (ext) and M/atom. This run
  uses ext=2 (each extended element is itself a 64-atom box — a heavy,
  conservative buffer); a lighter halo moves the crossover left. The prototype
  exposes `--core/--ext/--m-per-atom` to map that sensitivity.
- `t_glob` is a **dense** eigh — an O(N^3) upper bound with a small prefactor.
  It is cheap here but is the term that eventually re-binds at much larger N.
  The path past it is a linear-scaling density-matrix solve (Chebyshev FOE,
  reusing `solvers.chebyshev_filtered_batched`, or purification) on the
  block-sparse ALB Hamiltonian — the natural follow-up, and the second
  crossover that actually reaches 10^4+ atoms.
- Timing study, not a converged SCF: `v_eff` is a smooth placeholder and
  davidson runs a fixed iteration count, so the cost centres (FFT/GEMM/RR) are
  set by npw/nb/M/D, not by potential values. DG interior-penalty surface
  assembly is not included (O(N_elem), small next to the solves).

## Bottom line

DG-ALB's wall-time win is real and asymptotic, with a crossover near
~150–200 atoms for this (conservative) buffer — the *same shape* of result the
PhaseFold NUFFT-projector study found, but here the crossover lands right at
the ceiling where plain plane waves hit the memory wall, so DG-ALB is what lets
the calculation exist at all past ~200 atoms. The next lever to reach *enormous*
(10^4+) is swapping the dense global eigh for an O(N) density-matrix solver on
the block-sparse ALB Hamiltonian.

---

# Follow-up: global-solver scaling (dense eigh vs O(N) density-matrix)

`bench_dgalb_solver.py` asks the forward question toward enormous N: does the
dense reduced global `eigh` (O(D^3)) eventually dominate, and does an O(N)
density-matrix solver on the block-sparse ALB Hamiltonian push that ceiling out?
asus, 8 threads, fp64, M_elem=64, avg_deg=7, n_cheby=500, purify_iters=25.

```
atoms  n_elem     D    t_alb_s  t_dense_s  t_foe_s  t_purify_s  dense>build?
   64      8    512      28.0      0.01      47.3       4.7          no
  216     27   1728      94.5      0.28     172.4      17.2          no
 1000    125   8000     437.5     16.37     831.3      83.1          no
 4000    500  32000    1750.0   OOM(est)  3316.6     331.7          no
10000   1250  80000    4375.0   OOM(est)  7858.2     785.8          no
30000   3750 240000   13125.0   OOM(est) 23144.2    2314.4          no
```

`t_dense` = measured `eigvalsh(D)` (OOM(est) past the 9 GB budget). `t_foe`,
`t_purify` = block-sparse density-matrix cost, grounded in a measured batched
`bmm` at the DG block count (~N_elem*avg_deg^2, linear in N) times the iteration
count. `t_alb` = measured-linear ALB build (3.5 s/element, from bench_dgalb.py).

## Findings — two of them against the naive expectation

1. **The "second crossover" is a MEMORY wall, not a compute one.** Dense `eigh`
   is so LAPACK-efficient that it stays *cheaper* than the O(N) density-matrix
   solvers right up to where it OOMs: at D=8000 (1000 atoms) dense is 16 s vs
   purification 83 s. The O(N) solver is not a global-step *speed* win — it is a
   *memory* enabler that lets the global solve exist past ~1000-1500 atoms
   (where the dense D x D no longer fits 14 GB).

2. **The global solve is never the DG bottleneck — the ALB build is, at every
   size.** Even at 30,000 atoms the linear ALB build (3.6 h) dwarfs the best
   global solve (0.6 h). DG-ALB cost is gated by ALB regeneration (linear), so
   the global-solver choice is second-order. `dense>build?` is "no" throughout.

3. **Purification beats FOE ~10x** (25 iters x 2 matmuls << 500 Chebyshev terms)
   and is cleanly O(N) (332 -> 786 -> 2314 s for 4000 -> 10000 -> 30000 atoms).
   FOE's advantage is the metals/finite-T case (purification needs a gap), not
   cost — so the choice is physics-driven, not speed-driven.

## Practical guidance (flips the naive prescription)

Use dense `eigh` for the reduced global solve as long as it fits (~1000-1500
atoms on 14 GB); past that switch to **purification** purely to dodge the memory
wall, NOT for speed; reach for **Chebyshev-FOE** only on metals (finite-T, no
gap). And regardless of global solver, the thing to optimise for enormous N is
the ALB *build* — that is where the wall-time actually goes.

## Caveats

n_cheby/purify_iters are inputs (f(H) accuracy not computed — a cost study);
cold metals inflate n_cheby (FOE worse). The block-sparse step is a conservative
batched-bmm model of a truncated sparse-sparse product with fixed pattern (the
standard O(N) locality assumption). Dense `eigh` gated at 9 GB.
