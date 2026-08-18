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
- [x] GATE D — a WORKING differentiable all-electron atom (experiments/autoapw/allelectron_atom.py).
      **PASSED.** Bound-state eigenvalues via batched-Numerov shooting + root-finding reproduce the
      analytic hydrogen spectrum (1s -0.499998, 2s/2p -0.125000); exact autograd dE_nl/dZ == -Z/n²;
      and for a screened (Yukawa) potential dE/dλ autograd == finite-diff (1.382428) — the exact
      gradient to fit a pseudopotential/functional parameter against. CPU+CUDA. This is the "full
      potential" demo: a differentiable all-electron ORACLE with exact parameter gradients.
- [x] GATE S3 — periodic mixed-basis (single-atom LAPW) secular equation
      (experiments/autoapw/mixed_basis.py). **PASSED the FLAPW empty-lattice correctness gate.**
      Assembles S and H over {interstitial PWs, augmented sphere channels} — interstitial Θ(G)
      step (gate A) + augmentation from u_l,u̇_l (gate B) + value/slope matching (gate C) +
      weak-form muffin-tin kinetic — and solves H c = ε S c. Empty lattice (V=0) reproduces the
      free-electron bands ½|k+G|² to **5.5e-6 Ha**, converges with lmax (5.2e-4→5.5e-6, FLAPW rule
      lmax≈R·Gmax), and is R_MT-independent (max 3.5e-5). CPU+CUDA. Two real bugs the empty-lattice
      gate caught and fixed: (1) match the u-functions (u=r·R) to r·j_l(qr), not j_l(qr); (2) use
      the WEAK-form kinetic in the sphere for consistency across the C¹ boundary.
- [x] GATE S3b — REAL bands (non-empty muffin-tin well) vs a converged plane-wave reference.
      **PASSED.** LAPW small augmented basis (ecut=8,lmax=6) == converged PW (ecut=20) on the SAME
      well to 2.5e-4–6.7e-4 Ha per band across Γ,X,M; the attractive well pulls the lowest band to
      -0.01 Ha. Concrete AutoAPW payoff (LAPW hits PW-ecut-20 accuracy at ecut=8).
- [x] GATE S3c — DIFFERENTIABLE bands via a torch autograd assembly (`torch_bands`). **PASSED.**
      dε/dV0 (band response to the potential) autograd == finite-diff to ~1e-7 for all bands at a
      general k, flowing through numerov → radial integrals → matching → generalized eigensolve.
      Extends the gate-D atomic oracle to the solid-state mixed-basis solver. (General k lifts the
      cubic degeneracies that make eigh's backward diverge.)
## Production track (NOT to be merged until genuinely production-worthy)

- [x] PROD-A — production radial solver: gradwave eV/Å units + UPF constant-dx log meshes via the
      x=ln r Numerov transform (`radial_log.py`). Verified: hydrogen spectrum exact to <0.01 meV;
      the solver on oxygen's real native log mesh reproduces the analytic Coulomb spectrum to meV
      for contained states.
- [x] PROD-B — robust radial eigensolver (inward-outward matching at the classical turning point)
      (`radial_eigen.py`). Verified: hydrogen exact; DEEP hydrogenic Z=8 1s at -870.76 eV to 8 meV
      (outward-only shooting gets this wrong), 2s/2p to ~1 meV, node counts correct. asus+thinkpad.
- [x] PROD-C — atomic KS self-consistent solve → the real screened all-electron potential
      (`atomic_scf.py`). Tridiagonal radial eigensolve (`radial_eigen.radial_eigs_tridiag`, ~7 ms,
      ~7000× over shooting, robust deep+diffuse) + type-II Anderson mixing + radial-Poisson Hartree
      + LDA (Slater-X + PW92-C). Verified vs NIST LDA: Be 1s 0.002 eV, He 1s 0.13 eV, valence
      ≤0.4 eV; deep cores ~1 eV (O(dx²) stencil, mesh-limited). ~0.1 s per atom. Delivers "real
      potential" + atomic self-consistency. (Perf refactor per the dev-speed discussion.)
- [x] PROD-D — LAPW assembly in production units (eV/Å + log mesh) on a REAL self-consistent
      atomic potential (`prod_lapw.py`). Empty-lattice port correct (free-electron bands to
      2.4e-3 eV); and fed Ne's PROD-C self-consistent all-electron potential, the Γ LAPW valence
      bands reproduce the atom's own KS eigenvalues: 2s −35.51 vs −35.67 (0.17 eV), 2p −13.14 vs
      −13.20 (0.06 eV). Real bands on a real potential. CPU+asus.
- [x] PROD-E — multi-atom cells (`prod_lapw.build_matrices_multi`): per-atom structure phases
      e^{i(k_G'-k_G)·τ_a} on interstitial + augmentation, per-species radial channels, complex-
      Hermitian S/H. Verified: TWO spheres at arbitrary positions (V=0) still give free-electron
      bands to 2.3e-3 eV (decisive structure-factor check); two Ne atoms 3.5 Å apart give
      near-degenerate valence pairs at the atomic KS levels (2s Δ 0.17 eV, 2p Δ 0.03 eV). CPU+asus.
- [~] PROD-F — crystal self-consistency. PARTIAL:
      [x] density-side foundation — the crystal charge decomposition from the LAPW Bloch states
          (`prod_lapw.crystal_charges_check`, population analysis on the overlap sub-blocks). Two
          Ne (16 val e): each sphere 7.912 e (equivalent atoms exactly equal), interstitial 0.177 e,
          total 16.000 — physically correct (Ne valence ~99% inside R_MT). CPU+asus.
      [ ] the full self-consistent LOOP remains — the hard part: interstitial+sphere Coulomb via
          Weinert's method, XC on the crystal density (interstitial FFT grid + sphere radial),
          density mixing, and multi-k Brillouin-zone integration with a Fermi level. This is the
          major remaining subsystem (weeks); the density decomposition above is its prerequisite.
- [ ] PROD-G — promote validated modules into src/gradwave (types, tests, import contracts). LAST.
- [ ] Then #1 DualBasis oxygen gate; then benchmark; then SlepianCore (slabs/molecular crystals).

Prototypes/tests run on asus per user request (`ssh asus`), heavy runs via ./scripts/gwq.

## Result log
- 2026-08-17 GATE A PASSED. thinkpad-CPU, asus-CPU, asus-CUDA all ≤1e-14 vs the analytic surface
  integral; grid-mask control force == 0.0; fftbox cross-check max|Δ| == 0.0. ruff clean, 6/6 unit
  tests green. Reusable primitive src/gradwave/core/sphere_ff.ball_ff. Results in results_gate_a.json.
- 2026-08-17 GATE B PASSED. radial_solve.py Numerov solver matches analytic j_l to 1e-9..1e-12,
  gradcheck-exact ∂u(R_MT)/∂E and ∂u/∂R_MT, hydrogen-1s to 2.5e-4. asus-CPU + asus-CUDA. ruff clean.
- 2026-08-18 GATE S3 PASSED (empty-lattice FLAPW correctness gate). mixed_basis.py single-atom
  LAPW secular equation reproduces free-electron bands to 5.5e-6 Ha, lmax-convergent (5.2e-4→5.5e-6),
  R_MT-independent (≤3.5e-5). asus CPU+CUDA, ruff clean. Empty-lattice gate caught two real bugs:
  the u=r·R factor in the matching target (match to r·j_l, not j_l), and weak- vs strong-form
  muffin-tin kinetic across the C¹ boundary. This is a genuinely working periodic augmented-basis
  solver — the "second code" the earlier gates were foundations for. numpy assembly (not yet
  autograd); differentiable-bands + non-empty potential are the follow-ons.
- 2026-08-17 GATE C PASSED. boundary_match.py LAPW value+slope match: C¹ residual ~1e-17,
  (a_l,b_l) differentiable in E_l and R_MT (autograd vs FD rel ≤1e-6), u̇_l free from autograd.
  asus-CPU + asus-CUDA, ruff clean. ALL THREE net-new differentiable primitives now validated:
  Θ(G) interstitial surface term (A) + radial muffin-tin interior (B) + the boundary match tying
  them (C). Remaining for a full code: assemble the mixed-basis Hamiltonian + real UPF radial
  potential on a log mesh (S3). The differentiable-by-construction thesis is de-risked end to end.
