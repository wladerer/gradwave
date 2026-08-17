# Learned exchange functional — scope, data, and an honest value assessment

Use gradwave's differentiability to learn a **transferable** exchange enhancement
factor for solids — a *generalizable* improvement to the method, not a one-off fit.

## Novelty & honest value (read this first)
- **Parametric (κ,μ) fitting is NOT a result** — it is PBEsol (Perdew 2008) with
  less data. Its only role is validating the differentiable-training machinery.
- **SR-for-XC already exists**: SyFES (Ma et al., *Sci. Adv.* 2022) evolved a
  symbolic functional beating its parent; agentic discovery (2026) beat ωB97M-V.
  We are not inventing symbolic functional discovery.
- **The defensible gap** (confirmed by literature survey): nobody has assembled
  *constrained SR + differentiable-DFT-in-the-loop + solids*. The two real
  contributions inside it: (1) analytic `dE/dθ` through the SCF makes the inner
  constant-fit cheap and lets us fit **through the potential** (Burke's
  KS-as-regularizer, PRL 126, 036401) — the proven antidote to reward-hacking;
  (2) a **solids** focus (most SR/ML functional work is molecular).
- **Honest risks:** solid-state ground truth is thin (~55 lattice constants, ~13
  QMC systems); and a learned functional *absorbs the host code's numerical
  errors* — gradwave has observed rough edges (Cu pseudo anomaly, Pt/Pd/W
  Δ-gauge outliers, near-zero-gap convergence). The realistic outcome is a
  *PBEsol/SCAN-class* functional whose contribution is the **pipeline**, not a
  better functional. **This will not dethrone SCAN.** The value is methodological.

## Ground truth (data agent findings)
- **Targets:** ZPAE-corrected 0 K experimental lattice constants + cohesive
  energies for ~55 solids (Peng et al., PRX 6, 041005 (2016); tabulated in
  Mejía-Rodríguez & Trickey, PRB 98, 115161 (2018)).
- **Coverage ranking:** lattice constants ≫ cohesive ≈ bulk moduli ≫ 0 K
  formation energies. QMC "true" ground truth is only ~13 solids (Shulenburger,
  PRB 88, 245117 (2013)).
- **Recipe:** fit primarily to lattice constants; cohesive energies as a
  down-weighted second channel; reserve the 13 QMC solids **untouched** as
  held-out. The Δ-gauge AE data is a numerical *constraint*, not an accuracy
  target (it is PBE-AE — the answer for the same functional).

## Pipeline (implemented: `benchmarks/functional_learning/train_fx.py`)
- **Loss** (pressure-at-experimental-volume, avoids differentiating an argmin):
  `L(θ) = Σ P(V_exp; θ)²`, `P = −dE/dV`; `dP/dθ` from `energy_param_grads`
  (dE/dθ FREE at convergence by variational stationarity).
- **Held-out** elements + QMC solids are the generalization test.
- Serial now; the per-element EOS spokes are SeedPool-parallelizable.

## Results — parametric Stage A (measured, asus)
The differentiable-training loop works and **rediscovers known physics**: fitting μ
against experimental lattice constants (train Al, Si) drives μ 0.2195 → ~0.13
(≈PBEsol) and collapses train V0-error 2.18% → 0.65%. But it **does not
generalize** — held-out **Ge** V0-error *rises* 2.66% → 3.26% as μ falls:

| μ | train MAE (Al, Si) | held-out Ge |
|---|---|---|
| 0.2195 (PBE) | 2.18% | +2.66% |
| 0.1302 (≈PBEsol) | 0.65% | +2.98% |
| 0.0907 | 0.99% | +3.19% |

Confirmed **real, not k-point noise**: at converged kmesh 10 / ecut 40 the Ge
trend is identical (+2.66 → +2.98 → +3.19% as μ drops). Lowering μ genuinely
*expands* Ge (wrong direction) while it contracts Al/Si — a concrete demonstration
of GGA non-transferability (why PBEsol is a compromise and SCAN needed τ). The
pipeline caught it via the held-out set, which is exactly what the methodology is
for. **Takeaway:** the machinery is validated; the parametric GGA ceiling is real;
generalization requires the richer constrained form (`ConstrainedFx`) → meta-GGA τ.

## The learnable object (staged, always constrained)
Keep `LearnableX`'s philosophy — UEG limit and Lieb-Oxford bound enforced BY
CONSTRUCTION ("weird PBE, never garbage") — and lift it from parameters to forms:
- **Stage A** (machinery check): μ only → should recover a PBEsol-like μ.
- **Stage B**: constrained (κ, μ).
- **Stage C** (this doc's build): a constrained GGA `F_x(s)` — skeleton fixes the
  exact limits, SR searches a *bounded interior* correction. Then meta-GGA
  `F_x(s, α)` (τ), where the transferable accuracy lives.

## The SR layer (the ambition)
- **Engine:** genetic programming / PySR (SyFES-style). NOT SISSO (it regresses a
  scalar per row, has no native constraints, caps at GGA algebra).
- **Ansatz:** `E_x = ∫ n·ε_x^unif·F_x(s,α)`. The `(s,α)` coordinates make uniform
  scaling exact for free. Skeleton pins: `F_x(0,1)=1` (UEG), small-`s` coefficient
  `μ=10/81` (GE2), Lieb-Oxford envelope `F_x ≤ 1.804` (`≤1.174` at α=0), large-`s`
  tail `~s^{-1/2}`, α-switch for one-electron self-correlation-freedom. SR searches
  only the bounded interior term.
- **Inner evaluation = gradwave**: fit each candidate's free constants THROUGH the
  differentiable SCF (exercises the potential, not just energies) — the
  anti-reward-hack mechanism.
- **Selection & validation:** accuracy-vs-complexity Pareto front (parsimony buys
  transfer); constraints as hard equality/inequality terms *in* the loss;
  generalization proven on held-out elements + the 13 QMC solids.

## Exact constraints (to bake into the grammar)
Exchange: negativity `F_x>0`; spin-scaling; uniform-scaling (free via (s,α));
UEG `F_x(0,1)=1`; GE2 `F_x→1+μs²`, `μ=10/81`; large-`s` `~s^{-1/2}`; two-electron
bound `F_x(s,0)≤1.174`; Lieb-Oxford `F_x≤1.804`. Correlation: `E_c≤0`;
one-electron self-correlation-free (α=0); high/low-density scaling; GE2
`β(0)=0.066725`. (Sun/Ruzsinszky/Perdew PRL 115, 036402; Kaplan/Levy/Perdew,
Annu. Rev. Phys. Chem. 74, 193 (2023).)
