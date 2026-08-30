# FSM training campaign: E(M) curves for the spin-adapted PBE

This campaign multiplies #408's training signal (3 moments, 1 parameter, one
kind of physics) with fixed-spin-moment E(M) curves and broader chemistry — an
ordered alloy (B2 FeCo) and a half-metallic Heusler (Co₂MnSi) — then refits μ₁
and asks whether the residual structure justifies a second parameter.
Data: `results.json` (all curves, fits, LOO); raw per-curve JSON in `raw/`.

## The smeared FSM mode (Phase 0, the one src/ change)

`tot_magnetization` with a smearing now solves TWO per-channel Fermi levels —
μ↑ for N↑=(N_e+M)/2, μ↓ for N↓ — with per-channel smeared occupations and
summed entropy (`scf.common.fsm_smeared_occupations`). The result carries
`fermi_spin=(μ↑, μ↓)`; the constraining field is h = (μ↑−μ↓)/2 = ∂F/∂M, which
vanishes at the E(M) minimum. Consistency gate
(`tests/integration/test_fsm_smeared.py`): an FSM run at the unconstrained
moment reproduces the free SCF (fermi-dirac: ΔF ~ 5e-12 eV, μ↑−μ↓ ~ 1e-7 eV).

Root-selection subtlety (found the hard way): mp1's counting function is
locally non-monotone, so a channel's N(μ)=N_target can have several roots on
coarse meshes (bcc-Fe 3³: two μ↓ roots 0.10 eV apart redistributing 0.13 of a
state, landing the FSM SCF on a fixed point 6.7e-4 eV above the free one);
`core.occupations.find_fermi_near` follows the branch nearest the shared Fermi
level. On the production meshes below, N_σ(μ) is dense enough that this is a
non-issue — verified by the argmin-vs-unconstrained sanity gate on every curve.

## Systems, settings, targets

ecut 60 Ry, mp1 0.1 eV, `LearnableSpinXZeta(κ₁=0, μ₁)` for
μ₁ ∈ {0, −0.06, −0.10} — κ₁ stays 0 (dead, LO-clamped handle, #408).

| system | cell | k-mesh | exp M (μB/cell) | source |
|---|---|---|---|---|
| bcc-Fe | 1 at., a=2.87 | 14³ | 2.22 | as #408 |
| fcc-Ni | 1 at., a=3.52 | 14³ | 0.61 | as #408 |
| fcc-Co | 1 at., a=3.54 | 14³ | 1.60 | as #408 |
| B2 FeCo | 2 at., a=2.857 | 12³ | 4.60 | neutron site moments μ_Fe→~3.0, μ_Co≈1.7–1.8: Collins & Forsyth, Phil. Mag. 8, 401 (1963); mean-moment magnetization ~2.3 μB/at.: Bardos, JAP 40, 1371 (1969). Literature spread 4.5–4.7 — see the sensitivity note below. |
| Co₂MnSi L2₁ | 4 at., a=5.654 | 10³ | 5.00 | Slater-Pauling integer; polarized neutron 5.0: Brown et al., JPCM 12, 1827 (2000) |

k-convergence of the new cells (unconstrained PBE moment): FeCo 12³ 4.5794 vs
14³ 4.5785 (Δ 0.0009 μB — 12³ converged); Co₂MnSi 10³ 5.0000 vs 12³ 5.0000
(half-metal integer, mesh-insensitive by construction).

## E(M) curves (Phase 1)

8–9 grid points per curve plus two refinement points at m_free±0.05 (the
0.15–0.3 μB production grids limit field-root interpolation to ~0.05 μB on the
stiffer curves without them). Minimum extraction: the constraining field
μ↑−μ↓ = 2·∂F/∂M crosses zero at the minimum, and the root of its PCHIP
interpolant is far better conditioned than the argmin of a shallow asymmetric
F(M) well (Ni: field root agrees with the unconstrained moment to 5e-4 μB
where the F-spline argmin missed by 0.06). Stiffness d²F/dM² is the second
difference over the ±0.05 refinement points (SCF-exact energies).

Sanity gate — field-root argmin vs unconstrained SCF moment, all 15 curves:
max |m_free − m*| = 0.011 μB (≤ 0.02 required). PASS.

PBE (μ₁=0) minima and FSM spin stiffness:

| system | m_free (μB) | m* field root | d²F/dM² (eV/μB²) |
|---|---|---|---|
| Fe | 2.3187 | 2.3187 | 0.40 |
| Ni | 0.6012 | 0.6012 | 0.55 |
| Co | 1.7157 | 1.7061 | 1.49 |
| FeCo | 4.5794 | 4.5862 | 1.39 |
| Co₂MnSi | 5.0000 | 5.0000 | kink (one-sided ~7.6) |

Stiffness caveat: the ±0.05 μB local curvature carries k-resolution noise
(band crossings sweep the Fermi surface as M varies) — values are good to
~±50%, and the well is visibly anharmonic (left/right slopes differ ~2–3×).

Co₂MnSi is a textbook half-metal here: E(M) has a KINK at the Slater-Pauling
integer (∂F/∂M jumps −0.58 → +0.18 eV/μB across M=5.00 at μ₁=0), the free
moment is 5.0000 exactly, and the plateau survives the entire μ₁ range
(m_free = 5.0000 at μ₁=−0.10 too): the minority gap absorbs the ζ² correction
without moving the moment. Its FSM points inside the gap show a benign
smearing-tail sensitivity (F at M=m_free ~1e-4 eV above F_free at μ₁≠0 —
μ↓'s position inside the gap shifts mp1 occupation tails), which does not
affect the minimum.

Free moments m_free(μ₁) — the SCF-exact response samples the refit uses:

| system | μ₁=0 | μ₁=−0.06 | μ₁=−0.10 |
|---|---|---|---|
| Fe | 2.3187 | 2.2041 | 2.1618 |
| Ni | 0.6012 | 0.5680 | 0.5563 |
| Co | 1.7157 | 1.6213 | 1.5613 |
| FeCo | 4.5794 | 4.5330 | 4.4851 |
| Co₂MnSi | 5.0000 | 5.0000 | 5.0000 |

(Fe at μ₁=−0.06 reproduces #408's trained 2.203, Ni 0.568 — full consistency
with the 3-system campaign.)

## Refit (Phase 2)

Quadratic m*(μ₁) per system through the three samples; uniform-weight least
squares on the 5 experimental moments:

**μ₁\* = −0.0475** (#408: −0.0610 — moved +0.0135, a material shift; the
FSM-field-root cross-check interpolants give −0.0432). MAE 0.049 → 0.027 μB.

| system | PBE | refit (interp) | refit (direct SCF) | exp | err (interp) |
|---|---|---|---|---|---|
| Fe | 2.3187 | 2.2230 | 2.2142 | 2.22 | +0.003 |
| Ni | 0.6012 | 0.5734 | 0.5716 | 0.61 | −0.037 |
| Co | 1.7157 | 1.6405 | 1.6537 | 1.60 | +0.041 |
| FeCo | 4.5794 | 4.5451 | 4.5435 | 4.60 | −0.055 |
| Co₂MnSi | 5.0000 | 5.0000 | 5.0000 | 5.00 | −0.000 |

Direct unconstrained SCFs at μ₁=−0.0475 (residual.py) confirm the quadratic
interpolants to ≤0.013 μB (direct-SCF MAE 0.031 vs interpolated 0.027 — the
difference is interpolation error of the response model, not SCF physics).

Leave-one-out (5 systems, 1 parameter — now meaningful):

| held out | fitted μ₁ | predicted | exp | err | PBE err |
|---|---|---|---|---|---|
| Fe | −0.0462 | 2.225 | 2.22 | **+0.005** | +0.099 |
| Ni | −0.0502 | 0.572 | 0.61 | −0.038 | −0.009 |
| Co | −0.0335 | 1.662 | 1.60 | +0.062 | +0.116 |
| FeCo | −0.0585 | 4.535 | 4.60 | −0.066 | −0.021 |
| Co₂MnSi | −0.0475 | 5.000 | 5.00 | −0.000 | −0.000 |

LOO MAE 0.034 vs PBE 0.049. The structure is stark: the two systems PBE
OVERestimates (Fe +0.10, Co +0.12) improve dramatically; the two already at or
below experiment (Ni −0.01, FeCo −0.02) get worse — a single global μ₁ can
only pull every moment down. Co₂MnSi is insensitive (half-metal plateau): it
contributes no gradient but anchors the fit against |μ₁| large enough to
collapse the gap.

**FeCo target sensitivity** (its experimental value has a genuine 4.5–4.7
spread): μ₁\* = −0.066 / −0.0475 / −0.036 at FeCo exp = 4.50 / 4.60 / 4.70.
The refit's shift from #408's −0.0610 is therefore comparable to the
experimental-target uncertainty of the one alloy system; −0.061 sits inside
the band at FeCo exp ≈ 4.52. We ship the point estimate at the adopted 4.60
(μ₁ = −0.0475) and carry this band as the honest parameter uncertainty.

## 2nd-parameter verdict (Phase 3)

For each system at its equilibrium state under the fitted functional
(`raw/residual.json`): spatial means of the reduced gradient s, |ζ|, and the
iso-orbital indicator α = (τ−τ_W)/τ_unif (τ from the converged orbitals via
`core.metagga.tau_b`), weighted by the spin density |ρ↑−ρ↓| ("mag") and by
the exchange-correction magnitude |e_xc^fit − e_xc^PBE| ("Δexc"):

| system | refit err (μB) | ⟨s⟩_mag | ⟨\|ζ\|⟩_mag | ⟨α⟩_mag | ⟨s⟩_Δexc | ⟨\|ζ\|⟩_Δexc | ⟨α⟩_Δexc |
|---|---|---|---|---|---|---|---|
| Fe | +0.003 | 0.533 | 0.176 | 0.911 | 0.763 | 0.210 | 0.947 |
| Ni | −0.037 | 0.539 | 0.042 | 0.832 | 0.712 | 0.051 | 0.839 |
| Co | +0.041 | 0.530 | 0.127 | 0.900 | 0.763 | 0.151 | 0.949 |
| FeCo | −0.055 | 0.537 | 0.187 | 0.913 | 0.770 | 0.229 | 0.960 |
| Co₂MnSi | −0.000 | 0.563 | 0.166 | 0.872 | 0.728 | 0.230 | 0.903 |

**Verdict: a second parameter is NOT justified by this data.**

1. **No ingredient separates the residual signs.** The decisive pair is
   Fe (+0.003) vs FeCo (−0.055): indistinguishable in every measured local
   ingredient (Δ⟨s⟩ = 0.004, Δ⟨ζ⟩ = 0.011, Δ⟨α⟩ = 0.002, and equally so under
   the Δexc weighting). Any semilocal cross-term f(s, ζ, α) acts nearly
   identically on both, so it cannot pull FeCo up without pushing Fe off its
   near-exact value. ⟨s⟩ is flat across all five systems (0.53–0.56 mag-
   weighted) — an s-ζ cross-term has nothing to grab. The ζ² correction acts
   entirely at |ζ| < 0.5 in all five systems (dexc_frac_zeta>0.5 = 0.000), so
   higher powers of ζ modulate the same regions with the same ordering.
2. **The Ni residual is a response property, not a local one.** Ni does have
   the most distinct fields (lowest ζ 0.042, lowest α 0.832), and the ζ²
   weighting already suppresses its local correction ~15× below Fe's — yet its
   shallow well (stiffness 0.55 eV/μB², argmin flat to ±0.1 μB within 4 meV)
   makes the moment respond to even the suppressed correction. Sparing Ni is a
   susceptibility problem a static semilocal ingredient cannot address.
3. **The remaining MAE is at the target noise floor.** 0.027 μB (interp) /
   0.031 (direct) against: the ±0.03 μB k-jitter baseline carried since #408,
   the FeCo experimental spread (4.5–4.7 μB/cell, and saturation measurements
   include ~0.1–0.2 μB/cell of orbital moment a spin-only DFT moment cannot
   reproduce), and the ±0.01 interpolation error of the response model. A
   second parameter fit to these five targets would fit noise.

What WOULD move the needle: more targets of genuinely new physics classes
(antiferromagnets — FSM E(M) generalizes; itinerant weak magnets like ZrZn₂
where the μ₁ response is huge), or a per-channel response-aware form —
both beyond a static ζ-local second parameter.
