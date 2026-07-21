# Derivative accuracy

gradwave's distinguishing claim is that **every derivative it produces is
validated**. Reproducing Quantum ESPRESSO is a separate claim, which the Δ-gauge
covers. Each derivative is checked either against a finite difference of its own
energy / SCF re-runs (implementation exactness, which floors near the FD noise),
or against the specialized QE response module (ph.x, hp.x), which mixes
pseudization with implementation and so agrees at the ~0.1–1 % cross-code level.
This benchmark consolidates those checks into one table. Each row cites the
passing test gate it comes from, so `pytest <test>` reproduces it.

    uv run python benchmarks/derivatives/accuracy.py   # table + accuracy.json
    uv run python benchmarks/derivatives/make_fig.py   # derivative_accuracy.png

## What it covers

**19 validated derivatives** spanning the whole feature set — 13 checked against
finite difference / gradcheck (median relative agreement 1e-4, most below the
1e-5 first-derivative FD floor), 6 against a QE response module (median 2e-3):

- **Geometry**: atomic forces −dE/dτ (FD 1e-4, QE egg-box 8.6e-6), fixed-basis
  stress dE/dε (FD 1e-7, QE ≤0.006 kbar over four decades of magnitude), the PAW
  position Hessian ∂F/∂τ (2e-5), and Γ phonon force constants (0.003–0.15 % vs ph.x).
- **Functional learning**: XC-parameter dE/dθ by stationarity (~1e-8), and the
  density-loss adjoint dL/dθ through the SCF fixed point (2e-4), the object that
  makes learnable functionals train.
- **Hybrid parameters**: dE/dα, dE/dω by stationarity (1e-3, 5e-3), and the
  frozen-orbital band-gap gradient dGap/dα (1 %) that the gradient-designed-hybrid
  benchmark uses.
- **DFT+U**: dE/dU by Hellmann–Feynman (1e-4), the +U force gradcheck, and the
  linear-response U by analytic Sternheimer vs hp.x DFPT (0.3 %).
- **E-field DFPT**: ε∞ (0.002 % vs ph.x) and Born charges Z* = ∂²E/∂E∂τ (<2e-3),
  the mixed second derivative through the τ-differentiable pseudopotential.
- **USPP/PAW**: the composite (ρ, becsum) density-loss adjoint (1.2e-6), its
  Fermi-surface variant for metals (2e-4), and the one-center HVP (2e-6).

![derivative accuracy](derivative_accuracy.png)

The table is a static consolidation (the numbers are the passing tests' asserted
tolerances or observed agreements, not a fresh run), because reproducing the QE
comparisons needs the committed QE fixtures and the heavier SCF re-runs. The
gates themselves — `tests/gradcheck/`, `tests/integration/test_*_vs_qe.py`,
`test_uspp_implicit.py`, `test_uspp_position.py`, `test_paw_onsite_hvp.py`,
`test_learned_hybrid.py`, `benchmarks/hybrid_design/validate.py` — are the
reproducible source.
