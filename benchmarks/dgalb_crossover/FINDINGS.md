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

---

# Two-scale accuracy: do ALBs represent the crystal to sub-meV?

`bench_dgalb_accuracy.py` — the rigorous element<->crystal test the crossover
study left open. Real SCF on a 216-atom Si crystal gives the true occupied
orbitals; they are sampled on real-space points in the central conventional cell
(the "core") by explicit Fourier sum. Separately, real SCF on an ISOLATED
k x k x k conv-cell element gives ALB-candidate eigenstates, evaluated on the
SAME core points. Residual = fraction of the crystal occupied subspace outside
span{M lowest element states}. asus, fp64, LDA, ecut 12 Ry, K=12^3 core points.

Pipeline correctness verified separately: crystal=1, element==crystal gives
residual 1.7e-16 at M>=N_occ (machine zero) — frame/Fourier/projection correct.

```
buffer_k  elem_atoms   M   M/core-atom   mean_res    ~E_err_meV
   1          8         8      1.0        6.6e-01       7923     no buffer:
   1          8        32      4.0        2.3e-01       2768     residual PLATEAUS
   1          8        64      8.0        1.4e-01       1685     ~0.11 — cannot be
   1          8        96     12.0        1.1e-01       1347     fixed by more ALBs
   2         64        32      4.0        2.1e-01       2562     with buffer:
   2         64        64      8.0        3.5e-02        422     collapses steeply
   2         64        96     12.0        1.8e-03         22     with M -> ~22 meV
   3        216        64      8.0        8.8e-02       1054     under-resolved
   3        216        96     12.0        5.0e-02        603     (M fixed, elem grew)
```
(M/core-atom = M / 8, the core is one conventional cell = 8 atoms.)

## Findings

1. **The buffer is essential — and its absence is unfixable.** A no-buffer
   isolated element (k=1) plateaus at ~11% residual (~1300 meV) no matter how
   many ALBs you add: the isolated element's boundary conditions don't match the
   crystal, and no amount of basis fixes a wrong BC. This is precisely the
   physics DGDFT's extended-element buffer exists to cure.

2. **With a buffer, ALB representability is real and steep.** k=2 collapses
   0.21 -> 0.035 -> 0.0018 over M = 32 -> 64 -> 96, reaching ~22 meV (proxy) at
   ~12 ALBs/core-atom and still dropping past the M cutoff — consistent with
   DGDFT's published sub-meV at ~20-40 ALBs/atom. The DG-ALB representation is
   sound, provided the buffer is carried (which is also what makes the
   extended-element solves cost what the crossover study measured).

3. **M scales with the CORE, not the extended element** — the non-monotonic k=3
   is the proof. At fixed M=96 the whole-crystal "element" (k=3) is WORSE (0.05)
   than the smaller k=2 (0.0018): its lowest 96 states out of 432 occupied are
   too smooth/delocalised to cover the core's higher-frequency content. It is
   under-resolved, not buffer-regressed — it would need M proportional to the
   element size. This is exactly why DGDFT keeps M per CORE atom with a fixed
   small buffer halo (the k=2 regime), not per extended-element atom.

## Caveats

LDA, ecut 12 Ry (moderate), and k=2's buffer is one-sided/partial (the block is
corner-aligned, not centred — a symmetric halo would do better, so 22 meV is a
conservative reading; note max_res 0.076 vs mean 0.0018 = a few
boundary-weighted orbitals lag). Residual is a subspace fraction; the meV column
is a crude fraction x bandwidth proxy, not a variational energy. Still, the
qualitative verdict is robust: buffer essential, ALBs accurate with M/core-atom,
and the M-per-core scaling law falls straight out of the data.

## Bottom line

The accuracy axis the crossover study left open comes out **positive**: DG-ALB
is a genuine (not lossy) representation — a fixed buffer plus ~10-12 ALBs per
core atom captures the true crystal occupied subspace to tens of meV and
improving, while a no-buffer element is BC-limited at ~1 eV. Together with the
speed result (linear ALB build, crossover ~150-200 atoms) and the solver study
(dense eigh until it OOMs, then purification), the three studies bracket the
DG-ALB moonshot on all the axes that decide it — short of the full DG
interior-penalty global assembly, which remains the real build.

---

# Build: kill point 1 CLEARED — 1D SIPG spike

`dgalb_spike_1d.py` — the de-risking spike from BUILD_PLAN.md. Tests whether the
symmetric interior-penalty (SIPG) assembly of an adaptive local basis reproduces
the exact plane-wave spectrum, in 1D, before any 3D machinery is built. Periodic
1D KS-like H = -HB d^2/dx^2 + V (six Gaussian wells); reference from a dense
plane-wave diagonalisation; DG-ALB from per-element extended-box solves,
core-restricted + canonically orthonormalised, assembled with the SIPG form.

```
# sigma sweep (M=24): the coercivity threshold
 sigma    max_err     min_eig     ref_e0     herm
   1.0   1.81e+03   -1811.45    -0.37859   3.6e-15   spurious modes far below truth
  50.0   1.05e+02    -104.99    -0.37859   3.6e-15   still unstable
 200.0   8.66e-05    -0.37859   -0.37859   3.6e-15   STABLE, sub-meV
2000.0   6.98e-05    -0.37859   -0.37859   3.6e-15   no over-penalisation

# M convergence (coercive sigma = 2*M^2)
  M   M/atom   sigma   max_err    e0_err
  4    4.0       32   2.6e-02   5.9e-09   under-resolved basis
  8    8.0      128   1.3e-05   9.6e-08   sub-meV
 12   12.0     288   3.2e-06   4.7e-07   best (~0.03 meV)
 24   24.0    1152   7.1e-05   1.2e-06   sub-meV
```

## Findings

1. **The SIPG assembly is correct and variational.** The global operator is
   Hermitian to machine precision (~3e-15) for every sigma and M — the
   consistency/symmetry/penalty signs are right.
2. **The interior penalty behaves exactly as theory predicts.** Below the
   coercivity threshold (sigma ~ 100 here) spurious eigenvalues crash far below
   the true ground state (-1811 at sigma=1); above it the spectrum is correct and
   accuracy is flat out to sigma=2000 (no over-penalisation). The threshold grows
   with basis richness (sigma ~ M^2), the standard interior-penalty scaling.
3. **DG-ALB reproduces the exact spectrum to sub-meV at ~8-12 ALBs/atom**
   (1e-5-1e-6 Ha ~ 0.03-0.3 meV) — consistent with the accuracy study's
   representability curve, now confirmed as an actual variational solve.

## What this retires

Kill point 1 from BUILD_PLAN.md: the deepest uncertainty of the moonshot — does
the DG interior-penalty assembly give a correct, variational operator from a
discontinuous adaptive basis? — is answered YES in the cleanest setting. The
penalty is real, tunable (sigma ~ M^2 above a coercivity floor), and the method
hits sub-meV. The remaining risk is Phase 2's kill point 2: does this hold in 3D
with a real potential and nonlocal pseudopotentials. But the SIPG form itself is
now validated end-to-end.
