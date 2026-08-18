# AutoAPW build — overnight status

Goal (user's north star): use autograd/autodiff to fit parameters in a pseudoized code,
using an all-electron (augmented / LAPW-style) code as the accuracy oracle; and test whether
gradwave's existing tricks/optimizations carry over to an all-electron basis.

AutoAPW is a "second code" — three genuinely-absent subsystems:
  (S1) on-the-fly differentiable radial Schrödinger solve (u_l, u̇_l at a linearization energy)
  (S2) APW/lo value+derivative boundary matching at r=R_MT
  (S3) interstitial two-region partition via the analytic sphere-indicator FT Θ(G) + mixed-basis H

The whole build hangs on one make-or-break claim (the deep-dive's "smallest de-risk step"):
  > On a fixed FFT grid, a moving muffin-tin boundary's surface (Pulay) term is INVISIBLE to
  > naive grid-based autograd, but is EXACTLY recovered when the region partition is expressed
  > through the analytic sphere-indicator Fourier transform Θ(G), so τ enters only via exp(iG·τ).

## Sequence (this branch)
- [x] GATE A — Θ(G) single-atom surface-term prototype (decisive; no SCF). **PASSED.**
      experiments/autoapw/surface_term_toy.py
      autograd force via Θ(G) == analytic surface integral to ≤1e-14 (CPU+CUDA), FD-consistent,
      and control: naive grid-mask autograd force == 0 (blocker reproduced exactly).
- [x] GATE A' — Θ(G) route cross-checked vs gradwave core/fftbox.g_to_r_box: bit-identical (0.0).
- [x] Promoted the Θ(G) primitive to src/gradwave/core/sphere_ff.py (ball_ff); filled the
      deep-dive "interstitial Θ(G) — absent" gap. Regression test tests/unit/test_sphere_ff.py (6/6).
- [ ] GATE B — differentiable radial Schrödinger solve (S1): u_l(r;E)+u̇_l inside the MT via a
      Numerov/shooting integrator, autograd through the linearization energy E_l and R_MT,
      checked vs an analytic case (infinite spherical well / hydrogen l=0). NEXT.
- [ ] GATE C — single-sphere APW value+derivative boundary match (S2); differentiable
      R_MT / E_l gradients vs finite difference.
- [ ] Mixed-basis Hamiltonian assembly + single-atom all-electron energy (S3 full).
- [ ] Then #1 DualBasis oxygen gate; then benchmark; then SlepianCore (slabs/molecular crystals).

Prototypes/tests run on asus per user request (`ssh asus`), heavy runs via ./scripts/gwq.

## Result log
- 2026-08-17 GATE A PASSED. thinkpad-CPU, asus-CPU, asus-CUDA all ≤1e-14 vs the analytic surface
  integral; grid-mask control force == 0.0; fftbox cross-check max|Δ| == 0.0. ruff clean, 6/6 unit
  tests green. Reusable primitive src/gradwave/core/sphere_ff.ball_ff. Results in results_gate_a.json.
