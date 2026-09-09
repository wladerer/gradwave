# Probe: RR-subspace GEMM vs true H-apply wall split in `davidson_batched`

**Verdict: NO-GO.** Deep-band locking is not worth building for large-N slab SCF.

## Question

A prior moonshot found that on a heavy Cu slab the subspace `eigh` is only ~2.3%
of wall (so speeding up the diagonalization is dead), but suspected that the
*other* Rayleigh-Ritz subspace GEMMs — the Gram build `s = v.conj() @ hv.mT`,
`eigh(s)`, and the Ritz reconstruction `x = u·v` / `hx = u·hv` in
`davidson_batched` — were buried undifferentiated inside a heuristic profiler's
~47% "h-apply matmul" bucket. Deep-band locking (RR-subspace deflation of
converged deep semicore bands) only pays if that RR-GEMM share is large **and**
a large fraction of bands are lockable early.

## Workload (reproduces the moonshot config, on asus, OMP_NUM_THREADS=8)

Cu(100) 2×2×3 = 12 atoms, `Cu_ONCV_PBE-1.2` (19e semicore, Z=19), 150 bands,
kmesh (2,2,1) → **3 IBZ k**, ecut 40 Ry, nspin 1, FFT grid 40×40×120, gaussian
smearing 0.15 eV. `eigensolver=davidson` forced (the production `auto` gate
also picks davidson here: nb=150 < CheFSI threshold 640). Ran one SCF to
convergence (26 iterations, F = −59170.9498 eV, E_F = 3.24 eV).

Instrumentation: env-gated (`GRADWAVE_RR_PROFILE=1`) `time.perf_counter` around
each op class inside `davidson_batched`, summed over all rounds and all 26 SCF
steps. Probe branch `probe/rr-vs-apply-split`, not for merge.

## Measured wall split (summed over the whole SCF; total wall 1465.7 s)

| bucket | seconds | % wall |
|---|---:|---:|
| true H·ψ apply (nonlocal KB + local FFT V·ψ) | 570.7 | 38.9% |
| Gram `s = v.conj()@hv` | 205.6 | 14.0% |
| `eigh(s)` | 73.0 | 5.0% |
| Ritz reconstruction `x=u·v`, `hx=u·hv` | 136.4 | 9.3% |
| ortho / restart QR + solve_triangular | 235.7 | 16.1% |
| **RR-GEMM (Gram + eigh + Ritz) — the lockable set** | **415.1** | **28.3%** |
| davidson-internal total | 1221.4 | 83.3% |
| (remainder: SCF mixing, density build, XC, host glue) | ~244 | ~16.7% |

**RR-GEMM is 28.3% of wall** — above the ~20% GO bar, and **72.7% the size of
the true-apply bucket**. The moonshot's suspicion was correct: a large, real
slice of RR-subspace GEMM work was hiding in the heuristic profiler's "h-apply
matmul" bucket. `eigh` alone is 5.0% (consistent with the moonshot's ~2.3%
figure being for a different/lighter accounting), but the Gram + Ritz
reconstruction that scale with it are 4.7× larger than `eigh`.

## Lockable fraction (bands with residual < diag_tol=1e-9 by round 2)

**0.000 — at every SCF step (cold step 1, mid, and fully-warm step 26).** No
bands, deep semicore included, are converged to the final 1e-9 tol by round 2.
The premise that Cu's deep 3s/3p semicore converges fast enough to lock early is
refuted at the tolerance a frozen-deflation locking scheme requires.

A confirmatory run (probe extended to dump the full by-round, by-threshold lock
curve — `PROBE_SCF_MAX_ITER=3` for the cold long solve) was armed but
deprioritized: asus was memory-saturated by concurrent agent jobs for the
duration, and the by-round curve cannot change the verdict — it can only locate
*when*, within the ≤32%-of-bands lockable envelope (below), the deep bands
reach 1e-9. That envelope already caps the ceiling regardless of the round.

## Ceiling estimate

The deep-band-locking ceiling ≈ lockable_fraction × RR-GEMM-share (locking
removes converged bands from the RR GEMMs; Gram scales (nb_active/nb)², Ritz
reconstruction ~linearly, eigh³ but eigh is tiny):

- **Measured:** lockable × RR-share = 0.000 × 0.283 = **0.00 → ceiling ≈ 1.00×**.
- **Maximally optimistic analytic bound:** the truly-deep lockable bands are
  3s (1/atom) + 3p (3/atom) = 4/atom × 12 = 48 of 150 bands (32%). The Cu 3d
  (5/atom) is the transition-metal d-band at/near E_F — the *hardest*-converging,
  charge-sloshing bands, not lockable early; 4s + buffer likewise. Even if all 48
  deep bands locked from round 1 (they do not), the Gram (∝ dim²) drops to
  0.68²=0.46 and Ritz to ~0.68, saving at most ~13% of wall → ceiling ~**1.15×**,
  the boundary. Every real-world discount (deep bands do not reach 1e-9 by round
  2; the mandatory final full-width RR; the (active/nb)² model ignores that the
  unconverged expansion block persists; partition/re-activation bookkeeping) cuts
  below that.

## Verdict

**NO-GO.** The RR-GEMM share (28.3%) clears the 20% bar — that half of the gate
is genuinely met, and it is a real, previously-mis-attributed cost — but the
**estimated net ceiling does not clear 1.15×**: the measured lockable fraction is
0.00 (nothing locks to the final tol by round 2), and even the maximally
optimistic analytic bound only *touches* 1.15× before any implementation
discount. Deep-band locking is a corpse for this workload: the lockable set (deep
3s/3p semicore) is small (≤32% of bands) and does not converge early, while the
bands that dominate the band count and converge slowly (3d near E_F, 4s, buffer)
are exactly the ones you cannot lock. Do not build it.
