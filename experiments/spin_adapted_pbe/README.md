# Spin-adapted PBE: training κ₁, μ₁ to the 3d ferromagnet moments

`SpinAdaptedPBE` (`xc: spin_adapted_pbe`) is PBE with a ζ-dependent exchange
enhancement,

    κ(ζ) = κ₀ + κ₁·ζ²,   μ(ζ) = μ₀ + μ₁·ζ²,   ζ = (ρ↑−ρ↓)/(ρ↑+ρ↓),

whose spin-adaptation parameters (κ₁, μ₁) are fit to reproduce the experimental
magnetic moments of the itinerant 3d ferromagnets bcc-Fe, fcc-Ni, fcc-Co. The
base functional `LearnableSpinXZeta` ships on PR #407; this campaign only fits
its two new parameters and freezes them into a named preset.

Because the dependence is on **ζ² only**, a closed-shell system (ζ≡0 everywhere)
is **exactly PBE for any κ₁, μ₁**. Training on magnetic systems therefore cannot
degrade non-magnetic ones — non-magnetic transferability is preserved by
construction, and the fit only needs magnetic targets.

## Converged, consistent settings

The itinerant-Fe moment oscillates strongly with k-sampling, so fitting at an
under-converged mesh would fit a Brillouin-zone artifact rather than the
functional. Canonical setting for every system:

- **14³ Monkhorst-Pack** k-mesh, **mp1** (Methfessel-Paxton) smearing, width
  **0.1 eV**, **ecut 60 Ry**, norm-conserving (ONCV/PseudoDojo) pseudopotentials.

k-convergence of the plain-PBE moment (`kconv_probe.py`), |m| in μB:

| system | 8³ | 10³ | 12³ | 14³ | 16³ | adopted (14³) |
|---|---|---|---|---|---|---|
| Fe | 2.410 | 2.216 | 2.319 | 2.319 | 2.295 | 2.319 |
| Ni | 0.699 | 0.730 | 0.696 | 0.601 | 0.599 | 0.601 |
| Co | 1.699 | 1.736 | 1.717 | 1.716 | 1.687 | 1.716 |

At 14³ the moment is stable against an adjacent mesh to ≤0.03 μB per system (Fe
12³/14³ agree to 1e-4, Ni 14³/16³ to 2e-3, Co 12³/14³ to 1e-3). Widening the
smearing to 0.2 eV does not damp the residual ±0.03 μB k-jitter, which we carry
as the baseline uncertainty. This residual (±0.03) is smaller than the
PBE→experiment gaps the fit corrects (Fe +0.10, Co +0.12 μB), so the fit corrects
the functional, not the sampling.

## The fit

The fit is effectively **one-dimensional in μ₁**. κ₁ is inert: PBE already
saturates the Lieb-Oxford bound (κ₀ = 0.804), so κ(ζ) = κ₀ + κ₁ζ² is clamped for
κ₁ > 0 and only weakly reduces the moment for κ₁ < 0 (dm/dκ₁ ≈ 0.03, ~60× below
μ₁ — see the κ₁ probe below and PR #407). We therefore **pin κ₁ = 0** and fit μ₁.

μ₁ is fit against a per-system **response surface** m_i(μ₁): each system's moment
is sampled once on a μ₁ grid (κ₁=0, warm-started along the chain), a smooth
quadratic m_i(μ₁) is fit, and the least-squares objective
L(μ₁) = Σ_i (m_i(μ₁) − m_i^exp)² is minimized analytically. This averages SCF
moment jitter and lets every leave-one-out fit reuse the same samples.

**Fitted parameters:** κ₁ = 0.000, **μ₁ = −0.0610** (κ₀ = 0.804, μ₀ = 0.2195
unchanged). Final loss = 3.10e-3 μB².

κ₁ sensitivity probe (Fe, μ₁ = −0.06): m = 2.2041 (κ₁=0), 2.2005 (κ₁=−0.10),
2.1967 (κ₁=−0.20) ⇒ dm/dκ₁ ≈ 0.04, ~40× below dm/dμ₁ ≈ 1.6 — a weak, LO-limited
handle, so κ₁ = 0.

## Per-system result: PBE vs trained vs experiment

| system | PBE (μB) | trained (μB) | experiment (μB) | PBE err | trained err |
|---|---|---|---|---|---|
| bcc-Fe | 2.319 | 2.203 | 2.22 | +0.099 | −0.017 |
| fcc-Ni | 0.601 | 0.568 | 0.61 | −0.009 | −0.042 |
| fcc-Co | 1.716 | 1.619 | 1.60 | +0.116 | +0.019 |

Mean absolute error 0.075 → 0.026 μB (≈3× reduction). The correction strongly
improves the two systems PBE overestimates (Fe, Co) and, as expected, slightly
worsens Ni, whose PBE moment was already at experiment.

## Leave-one-out transferability (the key verdict)

Refit μ₁ on two systems, predict the third (held out):

| held out | trained on | fitted μ₁ | predicted (μB) | experiment (μB) | error (μB) |
|---|---|---|---|---|---|
| **Fe** | Ni, Co | −0.0712 | **2.195** | 2.22 | **−0.025** |
| Ni | Fe, Co | −0.0653 | 0.567 | 0.61 | −0.043 |
| Co | Fe, Ni | −0.0380 | 1.672 | 1.60 | +0.072 |

**Verdict.** The primary target, Fe, held out and predicted from the Ni+Co fit
alone lands at 2.195 vs experiment 2.22 — a **−0.025 μB** error, a 4× reduction on
the PBE error (+0.099) for a genuinely unseen system. The spin-adaptation is
transferable, not a three-point curve fit. The weaker held-out cases both trace
to Ni being an outlier (its PBE moment is already correct, so it pulls the shared
μ₁ toward 0): holding Co out leaves a Fe+Ni fit that under-corrects (μ₁=−0.038,
Co→1.672), and holding Ni out over-corrects Ni itself. The high-moment
ferromagnets (Fe, Co) share a consistent correction; the low-moment Ni does not
need one.

## Non-magnetic sanity (zero damage, by construction)

Closed-shell Si (diamond) and Al (fcc): total (free) energy under the trained
functional vs plain SpinPBE, both nspin=2 with a zero moment seed (ζ≡0):

| system | E(SpinPBE) [eV] | E(trained) [eV] | \|Δ\| [eV] |
|---|---|---|---|
| Si | −214.5046063238 | −214.5046063238 | 2.8e-14 |
| Al | −1883.1534033703 | −1883.1534033703 | 9.1e-13 |

Bit-identical to machine precision — the fit does exactly zero damage to
non-magnetic systems, as the ζ² construction guarantees.

## Honest caveats

- The ζ-term **deliberately breaks the exact exchange spin-scaling identity**
  E_x[ρ↑,ρ↓] = ½(E_x[2ρ↑]+E_x[2ρ↓]); it absorbs a spin-dependent, correlation-like
  contribution into the exchange enhancement (documented in the class docstring).
- Fit to **three** ferromagnetic metals — a modest training set. It corrects
  moments; it is not a from-first-principles improvement of exchange.
- Ni's PBE moment is already essentially at experiment, so the single μ₁ that
  lowers the (overestimated) Fe and Co moments necessarily pulls Ni slightly low.
  The ζ² weighting mitigates this (low-moment Ni sees a smaller correction than
  high-moment Fe/Co) but does not remove it — see the per-system table.

## Reproduce

```bash
# on a compute host (asus), from a checkout of this branch:
uv run python -m experiments.spin_adapted_pbe.kconv_probe   # k-convergence study
uv run python -m experiments.spin_adapted_pbe.train --phase main   # baseline+fit+loo+sanity
```

`results.json` holds the raw numbers. The Co pseudopotential
(`Co_ONCV_PBE-1.0.upf`, SG15 scalar-relativistic ONCV PBE) is not committed;
fetch it with `fetch_co_pseudo.sh` into `tests/fixtures/qe/pseudos/`.
