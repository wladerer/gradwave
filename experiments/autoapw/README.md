# AutoAPW — differentiable augmented (LAPW-style) basis

Investigating whether gradwave's autograd-native design can build the first end-to-end
differentiable all-electron augmented-plane-wave code, so the notoriously hand-derived
Pulay / muffin-tin-surface force and stress terms fall out of autodiff. Longer-term aim
(the user's north star): use autodiff to fit parameters in a pseudoized code with an
all-electron code as the accuracy oracle, and test whether gradwave's existing performance
tricks carry over to an all-electron basis.

See `STATUS.md` for the gate-by-gate build plan and current state.

## GATE A — the moving-boundary surface term (`surface_term_toy.py`) — **PASSED**

The single make-or-break claim under the whole idea: on a fixed FFT grid, a moving muffin-tin
boundary's surface (Pulay) term is invisible to naive grid-based autograd, but is recovered
*exactly* when the region indicator is expressed through its analytic Fourier transform
Θ(G) = W(G)·e^{-iG·τ}, so the atom position τ enters only through a smooth phase — the same
structure gradwave already uses for the USPP/PAW compensation charge (`postscf/uspp_frozen.py`).

The toy builds a two-region energy `E(τ) = ∫_ball(τ) g(r) dV` for a multi-mode field
`g(r) = Σ_j a_j cos(k_j·r + φ_j)`, which has the closed form `E = Σ_j a_j W(k_j) cos(k_j·τ+φ_j)`
and analytic gradient `dE/dτ = -Σ_j a_j W(k_j) sin(k_j·τ+φ_j) k_j` (this *is* the surface
integral). It then checks:

| quantity | result |
|---|---|
| autograd ∂E_Θ/∂τ  vs  analytic surface integral | **≤ 1e-14** (machine precision), all N, CPU + CUDA |
| autograd ∂E_Θ/∂τ  vs  central finite difference | ~2.5e-8 (FD-step-limited) |
| **control** — naive grid-mask autograd force | **exactly 0** (surface term invisible) |
| **control** — grid-mask finite difference vs analytic | off by ~7.8 (biased staircase) |
| gradwave `core.fftbox.g_to_r_box` vs the band-limited indicator | **0.0** (bit-identical) |

Run:

```bash
uv run python experiments/autoapw/surface_term_toy.py --device cpu
uv run python experiments/autoapw/surface_term_toy.py --device cuda   # asus
```

Writes `results_gate_a.json`. The reusable Θ(G) primitive lives in
`src/gradwave/core/sphere_ff.py` (`ball_ff`), regression-tested in
`tests/unit/test_sphere_ff.py`.

## What this does and does not establish

- **Does:** proves the differentiation strategy for a two-region (interstitial + sphere)
  energy is sound and exact in gradwave, and delivers the analytic sphere-indicator Θ(G) the
  full build needs (previously absent from the tree).
- **Does not:** build a mixed-basis Hamiltonian, an on-the-fly radial Schrödinger solve, or
  the APW boundary match. Those are GATES B and C (see `STATUS.md`) and remain the bulk of the
  "second code."
