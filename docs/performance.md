# Performance campaign (2026-07-15)

Measured on the 8-core laptop unless noted. Every number below comes
from a committed test or a script run this session, at the commits
following 5b9ae45.

## Corrections to the starting assessment

Two planned items already existed and needed no work. IBZ k-reduction
with G-space density symmetrization has been in place since the
symmetry milestone (`use_symmetry=True`, gated by the IBZ-equals-full-
mesh tests, used by the Δ-factor benchmarks). Mixed precision existed
on the NC path and only needed a port to the USPP/PAW batched solver.

## Landed

### Irreducible phonon displacements (postscf/phonons.py)

`HessianSymmetry` selects the minimal displacement set whose group
orbit spans every atom, and reconstructs the full Γ Hessian from the
computed columns. Diamond Si needs one column of six, zincblende two.

| quantity | 6 columns (before) | 1 column (now) |
|---|---|---|
| Si example, Hessian stage | ~6 response solves | 31 s total |
| optical triple | 585.91 / 585.99 / 586.32 | 586.32 exactly degenerate |
| vs ph.x 586.093 | 0.03% (mean) | 0.04% |

The reconstruction also removes the column-to-column numerical spread,
so degeneracies and the acoustic zeros are exact by construction.

A grid lesson came out of the gate test. The discretized energy surface
itself breaks non-symmorphic symmetry when FFT dims are incommensurate
with the fractional translations, because the XC quadrature is only
invariant under whole-grid-spacing shifts. Si 15/60 at the auto 18³
grid shows 2.6e-2 relative column anisotropy in the directly computed
Hessian, and 20³ restores 7e-5. `gamma_hessian` warns on incommensurate
grids. The showcase grids (32³, 40³) were commensurate by luck.

### Kerker preconditioning of the adjoint outer solve

`kerker_q0` filters the grid blocks of the outer residual by
G²/(G²+q0²). The fixed point is unchanged (gated at 2.6e-6 relative on
Si). Measured behavior is the opposite of the textbook expectation at
these system sizes. On the 1-atom Al metal it slows the solve (3 → 7
outer iterations) because Anderson with history is already near-exact
on so small a linear problem. On vacuum-cell O₂ it converts the
long-standing stagnation floor into genuine convergence.

| O₂ spin adjoint (35/280) | outer its | final residual |
|---|---|---|
| no preconditioner | 22 | 1.4e-4, floored |
| kerker_q0 = 1.5 | 27 | 1.4e-5, converged |

The training script now sets it for O₂. This addresses the wandering-
floor failure that ended training run 6 at epoch 13; `floor_tol` stays
as the safety net.

### Mixed precision on the USPP/PAW batched Davidson

fp32 draft solves while the adaptive diago tolerance is above 1e-5,
with the generalized subspace reduction always in fp64 (an fp32
Cholesky of the near-singular USPP overlap is where garbage rotations
come from) and fp64 S-normalization. Opt-in `mixed_precision=True`.

| system | fp64 | mixed |
|---|---|---|
| O₂ 35/280, 8 CPU cores | 56.6 s (20 it) | 46.0 s (18 it) |
| Si kjpaw 30/120 6³, RTX 3050 | 44.5 s (21 it) | 42.9 s (22 it) |

Free energies agree to 8e-9 eV (O₂) and exactly (Si). The CPU gain is
~1.2×. The GPU gain is only ~4%, which falsifies the fp64-throughput
motivation for this card and workload. The USPP iteration on the 3050
is FFT-bandwidth and launch-latency bound with CPU-resident one-center
work in every iteration, so cutting fp64 FLOPs barely moves it. The
option is harmless and may matter on genuinely GEMM-bound sizes, but
the measured recommendation is CPU molecular workloads.

### One-center HVP factory

Profiling killed the planned optimization and found a better one. The
ρ_lm assembly (the visible Python loops) costs 11 ms per call through
the existing dense maps; 85% of `hvp_becsum` is the autograd double
backward through the angular quadrature. The adjoint evaluates that HVP
at the same frozen converged becsum every outer iteration, so
`hvp_factory` builds the first-order graph once per atom and each call
pays one retained second backward. Ni kjpaw spin measures 870 → 524 ms
per call (1.66×), bit-identical results. The density-loss adjoint, the
position response and newton all inherit it.

### GPU-clean adjoint

`apply_chi0` and the outer solve now allocate on the system's device,
with the CPU-anchored one-center quadrature bridged explicitly in both
directions. RTX 3050 smoke (Si kjpaw 30/120, 4³, TR-reduced): free
energies bit-identical across cpu, cuda and cuda+mixed; the cuda
adjoint gradient matches cpu to 1e-9 relative. Honest caveat on speed
at this size, where the cuda adjoint is slower than the asus CPU (16.7
vs 8.3 s) — the per-k solves are latency-bound and the one-center
bridge crosses to the CPU every outer iteration. The plumbing is a
correctness enabler; the payoff arrives with many-k or big-box systems,
the same routing rule the SCF already follows.

## Relaxation timing vs QE (2026-07-15)

Displaced diamond (C ONCV, 50 Ry, 4³, fmax 0.01 eV/Å), identical
geometry and pseudo in both codes on the same 8 laptop cores; all three
land the same minimum (final coordinates agree to 1e-4 Å).

| run | ionic steps | wall |
|---|---|---|
| QE 7.5, BFGS | 5 (6 SCFs) | 14.0 s |
| gradwave, BFGS | 3 | 47.6 s |
| gradwave, FIRE | 25 | 405 s |

Two separate factors made the original run slow. The optimizer default
was FIRE (25 steps where BFGS needs 3 — an 8.5× penalty; the default is
now bfgs). The remaining 3.4× against QE is per-ionic-step cost.

The obvious fix — warm-starting each step's SCF — was built (NC
start_from with atomic-density extrapolation and orbital reuse in the
calculator) and helps the fixed-geometry case: same-position restarts
drop 9 → 2 iterations, which is what checkpoint restarts and parameter
scans want. Ionic moves stay at ~8 iterations from any seed.

That turned out to be parity, not a deficit. Reading the per-cycle
counts out of the same pw.x run: QE takes 7/6/6/5/4/4 iterations per
ionic step, resets its Broyden mixer every SCF cycle, and starts warm
steps at ethr 1e-6 — the same rule implemented here. An earlier note in
this file blamed "mixer state QE keeps between steps"; that was wrong,
QE keeps none. The remaining 3.4× is per-ITERATION throughput: QE ~0.4
s/iteration against ~1.4 s here on the same 8 cores for C at 50 Ry
(threaded FFT + band solver maturity), plus the per-step autograd force
backward. Closing it is solver-kernel work, not SCF-logic work.

Wall-clock deltas between warm-start variants measured back-to-back on
the laptop were dominated by thermal throttling — iteration counts are
the trustworthy metric for solver-logic questions.

## Deferred with measurement: Γ-point real wavefunctions

The O₂ 35/280 SCF profile (53 s total) breaks down as 22 s Davidson H
applies (13.5 s of it FFTs), 5.8 s XC potential assembly, 4 s one-center
ddd, 4 s mixing, and the remainder in density build and occupations.
Half-basis real algebra at Γ can at best halve the H-apply share, so the
end-to-end ceiling is roughly 1.3–1.5× for the most invasive change in
the stack (spheres, projectors, augmentation, density build and the
Sternheimer solver all touch the storage convention). Mixed precision
already banked 1.2× on the same system at a fraction of the risk.
Revisit if Γ-only molecular workloads become the dominant cost.
