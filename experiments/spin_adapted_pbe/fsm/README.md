# FSM training campaign: E(M) curves for the spin-adapted PBE

*(results note — numbers filled by fit.py / the campaign run; see results.json
for the full data)*

This campaign multiplies #408's training signal (3 moments, 1 parameter) with
fixed-spin-moment E(M) curves and broader chemistry (an ordered alloy and a
half-metallic Heusler), then refits μ₁ and asks whether the residual structure
justifies a second parameter.

## The smeared FSM mode (Phase 0, the one src/ change)

`tot_magnetization` with a smearing now solves TWO per-channel Fermi levels —
μ↑ for N↑=(N_e+M)/2, μ↓ for N↓ — with per-channel smeared occupations and
summed entropy (`scf.common.fsm_smeared_occupations`). The result carries
`fermi_spin=(μ↑, μ↓)`; the constraining field is h = (μ↑−μ↓)/2 = ∂F/∂M, which
vanishes at the E(M) minimum — the consistency gate
(`tests/integration/test_fsm_smeared.py`): an FSM run at the unconstrained
moment reproduces the free SCF (fermi-dirac: dF ~ 5e-12 eV, μ↑−μ↓ ~ 1e-7 eV).

Root selection subtlety (found the hard way): mp1's counting function is
locally non-monotone, so a channel's N(μ)=N_target can have several roots on
coarse meshes; `find_fermi_near` follows the branch nearest the shared Fermi
level. On the production meshes below N_σ(μ) is dense enough that this is a
non-issue; the argmin-vs-unconstrained sanity gate (below) is the check.

## Systems, settings, targets

ecut 60 Ry, mp1 0.1 eV, LearnableSpinXZeta(κ₁=0, μ₁) — κ₁ stays 0 (dead,
LO-clamped handle, #408).

| system | cell | k-mesh | exp M (μB/cell) | source |
|---|---|---|---|---|
| bcc-Fe | 1 at., a=2.87 | 14³ | 2.22 | as #408 |
| fcc-Ni | 1 at., a=3.52 | 14³ | 0.61 | as #408 |
| fcc-Co | 1 at., a=3.54 | 14³ | 1.60 | as #408 |
| B2 FeCo | 2 at., a=2.857 | 12³ | 4.60 | Collins & Forsyth, Phil. Mag. 8, 401 (1963), neutron site moments μ_Fe→~3.0 + μ_Co≈1.7–1.8; mean-moment magnetization ~2.3 μB/at. (Bardos, JAP 40, 1371 (1969)). Literature spread 4.5–4.7. |
| Co₂MnSi L2₁ | 4 at., a=5.654 | 10³ | 5.00 | Slater-Pauling integer; polarized neutron 5.0 (Brown et al., JPCM 12, 1827 (2000)) |

k-convergence: TODO (FeCo 12³ vs 14³, Co₂MnSi 10³ vs 12³ moment stability).

## E(M) curves

TODO: per-system m*, curvature (FSM spin stiffness d²F/dM²), sanity gate
(argmin vs unconstrained ≤ 0.02 μB).

## Refit + LOO

TODO: fitted μ₁ vs #408's −0.0610; 5-system PBE/refit/exp table; MAE; all 5
held-out errors.

## 2nd-parameter verdict (Phase 3)

TODO: s/ζ/α distributions of the magnetization-carrying regions vs residual
signs; verdict.
