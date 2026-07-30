# Plan: fitted mixing preconditioners, distilled back to analytic form

## Status

Proposed, not started (2026-07-30). Make the density-mixing preconditioner a
fitted object trained on flight-recorder trajectories by unrolled differentiation,
then distill each fitted kernel by symbolic regression to recover or improve the
analytic form. The learned multi-pole charge-channel prototype
(`scf/learned_precond.py`) is the starting point, and its measured win on fcc Cu,
10 iterations down to 8, is the existence proof. This plan extends it to the spin
channel, adds the distillation step, and states where each rung stops.

## The idea, and the constraint that makes it tractable

The mixing preconditioner generalizes Kerker and local Thomas-Fermi. Bare Kerker,
`R(G)*G^2/(G^2+q0^2)`, is the single-pole long-wavelength approximation to the
exact response inverse `(1 - v_c chi0)^-1`, and `MultipoleKerkerPrecond` already
replaces the one pole with a learned sum fitted by unrolling the real Pulay
recurrence (`fit_multipole`). The distillation step rests on one property of these
families. The learnable objects are low-dimensional functions, a radial filter
`K(|G|)`, a density-dependent screening length `k0(n(r))`, and a spin gain
`g(chi0, I)`, not deep networks. Symbolic regression over a handful of scalar
inputs is tractable by construction, whereas the same regression against a
multilayer network's weights is not. The plan is built around keeping every fitted
object inside a family a symbolic search can read back.

## The ladder

Each rung is a gate. A rung that produces a measured negative is a result, and the
substrate it built stays for the next rung.

### 1. Benchmark the in-repo Stoner preconditioner against arXiv:2606.26693

`scf/spin_precond.py` builds the Stoner magnetization-channel operator
`(I - chi0^diag K_mm)^-1`, inverted by Woodbury, and its docstring already follows
arXiv:2606.26693. Run it head-to-head against that paper's own formulation on the
paper's regime, collinear ferromagnets near the magnetic transition (fcc Ni near
Stoner, and the near-critical cases the paper reports). Either outcome is a result.
If the in-repo operator matches, that validates the implementation against an
external anchor. If the paper's formulation wins, that names the gap to close
before any learned extension is worth attempting.

### 2. Extend the learned prototype to the spin channel

Extend `MultipoleKerkerPrecond` to the magnetization block, trained on the
flight-recorder corpus by unrolled differentiation (5 to 10 steps, chunk
checkpointing borrowed from the Hvp work if memory demands), with the objective the
energy-metric error from the stopping-rule plan
(`docs/plans/energy-metric-stopping.md`). The fitted approach differs from the
earlier hand-shaped attempt. A prior measured negative found no single constant
filter weight `w0` that
helps the near-critical Stoner mode, because damping the uniform mode collapses the
moment and boosting it does not beat johnson (recorded in `docs/ideas.md`). The
difference here is that the filter shape is fitted end to end through the real mixer
rather than swept by hand. The gate is out-of-family, the learned spin filter must
beat the analytic Stoner preconditioner from rung 1, or the negative is recorded.

### 3. Symbolic-regression recovery calibration

Fit a free-form charge-channel kernel on bulk-metal trajectories, then symbolically
regress it and require recovery of the Kerker form `G^2/(G^2+k^2)` with a
Thomas-Fermi-like density dependence in `k`. This rung is a pipeline validation
rather than a discovery. The answer is known, and the requirement is that the search
finds it. Tooling is PySR where the nixpkgs and uv toolchain supports it. PySR is the
harder install, it pulls a Julia runtime through juliacall and neither PySR nor its
Julia backend is in nixpkgs. The fallback is SINDy-style sparse regression over a
physics-informed library seeded with `G^2`, `n`, `chi0`, and `I`, which is pure
Python (numpy, scipy, scikit-learn) and installs cleanly under uv. Passing this rung
is the license to trust rung 4.

### 4. Distill the fitted spin-channel kernel near the Stoner transition

Symbolically regress the rung-2 spin filter on trajectories near the Stoner
transition. Recovering the `chi0/(1 - I*chi0)` structure of arXiv:2606.26693 from
fitted data is an independent, data-driven derivation of that paper's result.
Finding a better out-of-family form that a symbolic search can still write down is
the discovery this ladder is aimed at.

### 5. Open territory, the transverse channel

The transverse (noncollinear) magnetization channel has no analytic preconditioner
in the literature, so a fitted-then-distilled kernel there would be a first. This
rung is contingent on the noncollinear campaign's verdict
(`research/noncollinear-convergence`). If that campaign finds the transverse floor
is not preconditioning-limited, which the m-channel measurements so far suggest, the
payoff is iteration count and nothing more, stated as such rather than as a
convergence-limit fix.

## Hardware and standing rule

Training is fp64 on the asus CPU (the RTX 3050 fp64 rate is about 1/64 of fp32), so
model size is capped by wall time. That cap is the same property that keeps the
symbolic step viable, a filter small enough to train in fp64 on a CPU is a filter
small enough to regress into a closed form. The owner's standing rule holds
throughout, prototypes live in `experiments/`, and production wiring into
`scf/learned_precond.py` or `scf/spin_precond.py` follows only after a validated
win, never on the strength of a prototype alone.

## Deferred siblings

Two adjacent learned-solver directions were considered and deferred by the owner on
2026-07-30, recorded here so the decision is on file. Solver-aligned learned
initialization for solids (SAIL-style, arXiv:2604.21657) would train an initial
density or wavefunction that lands inside the SCF convergence basin, but the
published work is molecular only and the wisdom.md finding that iteration count is
set by mixing rather than the initial guess makes it the lower-value direction here.
Differentiable spin constraints (DeltaSpin-style, arXiv:2208.04551) would add a
learned penalty that fixes moment magnitude and direction during the SCF, which is a
constraint scheme rather than a preconditioner and belongs to the noncollinear
convergence work, not this plan.

## References

- Barat, Levitt, Torrent, arXiv:2606.26693 (June 2026), Stoner-susceptibility
  preconditioner for collinear ferromagnets near the magnetic transition, no
  noncollinear or SOC coverage.
- SAIL, solver-aligned initialization, arXiv:2604.21657, molecular only (deferred
  context).
- DeltaSpin, differentiable spin constraints, arXiv:2208.04551 (deferred context).
