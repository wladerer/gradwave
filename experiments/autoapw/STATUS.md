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
- [x] GATE B — differentiable radial Schrödinger solve (S1): u_l(r;E) inside the MT via a
      Numerov integrator (experiments/autoapw/radial_solve.py). **PASSED.** Matches the analytic
      free-particle u_l = C·r j_l(kr) to 1e-9..1e-12 (l=0,1,2); torch.autograd.gradcheck confirms
      ∂u(R_MT)/∂E and ∂u/∂R_MT are exact; Coulomb hydrogen-1s outward shape to 2.5e-4. CPU+CUDA.
      (Atomic units, uniform mesh. TODO for promotion: gradwave eV/Å units via HBAR2_2M, UPF log
      meshes via the x=ln r transform, u̇_l = ∂u/∂E_l for the LAPW linearization.)
- [x] GATE C — single-sphere (L)APW value+slope boundary match (S2)
      (experiments/autoapw/boundary_match.py). **PASSED.** Matches interstitial PW Rayleigh
      component j_l(qr) to a_l u_l + b_l u̇_l at R_MT; C¹ continuity residual ~1e-17 (by
      construction); (a_l,b_l) differentiable in E_l and R_MT (autograd vs FD rel ≤1e-6). u̇_l
      obtained free as autograd ∂u_l/∂E_l through the GATE-B solver. CPU+CUDA.
- [ ] Mixed-basis Hamiltonian assembly + single-atom all-electron energy (S3 full): assemble
      H over {interstitial PWs, augmented sphere channels}, solve the generalized eigenproblem,
      integrate a real UPF/all-electron radial potential (log mesh + eV/Å units). The remaining
      "second code"; gates A–C validate its differentiable primitives.
- [ ] Then #1 DualBasis oxygen gate; then benchmark; then SlepianCore (slabs/molecular crystals).

Prototypes/tests run on asus per user request (`ssh asus`), heavy runs via ./scripts/gwq.

## Result log
- 2026-08-17 GATE A PASSED. thinkpad-CPU, asus-CPU, asus-CUDA all ≤1e-14 vs the analytic surface
  integral; grid-mask control force == 0.0; fftbox cross-check max|Δ| == 0.0. ruff clean, 6/6 unit
  tests green. Reusable primitive src/gradwave/core/sphere_ff.ball_ff. Results in results_gate_a.json.
- 2026-08-17 GATE B PASSED. radial_solve.py Numerov solver matches analytic j_l to 1e-9..1e-12,
  gradcheck-exact ∂u(R_MT)/∂E and ∂u/∂R_MT, hydrogen-1s to 2.5e-4. asus-CPU + asus-CUDA. ruff clean.
- 2026-08-17 GATE C PASSED. boundary_match.py LAPW value+slope match: C¹ residual ~1e-17,
  (a_l,b_l) differentiable in E_l and R_MT (autograd vs FD rel ≤1e-6), u̇_l free from autograd.
  asus-CPU + asus-CUDA, ruff clean. ALL THREE net-new differentiable primitives now validated:
  Θ(G) interstitial surface term (A) + radial muffin-tin interior (B) + the boundary match tying
  them (C). Remaining for a full code: assemble the mixed-basis Hamiltonian + real UPF radial
  potential on a log mesh (S3). The differentiable-by-construction thesis is de-risked end to end.
