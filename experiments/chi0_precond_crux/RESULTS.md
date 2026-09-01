# M1 — cheap subspace-χ₀ Woodbury preconditioning: results & verdict

**Branch:** `research/woodbury-chi0-precond-m1`  ·  **Compute:** asus (22-core CPU,
fp64), one job at a time  ·  **Convergence:** etol 1e-9, rhotol 1e-8, entol 1e-9.

## The question

The crux (`research/chi0-precond-crux`) showed the EXACT dielectric
preconditioner ε_ρ⁻¹ = (1 − χ₀·K_Hxc)⁻¹ — a full Sternheimer χ₀ each SCF step,
frozen at the converged reference — cuts bcc-Fe FM iterations ~2.2-2.6× but at
~7.6× the FFTs (it runs a conduction-space χ₀ solve every step). M1 asks: does
the CHEAP student — χ₀ restricted to the in-window Adler-Wiser sum + δμ term the
eigensolver already returns, inverted by Woodbury with **zero FFTs per apply** —
keep most of that iteration cut at a fraction of the FFT overhead, and converge
to the IDENTICAL fixed point?

## What was built

- `src/gradwave/scf/subspace_chi0.py` — `WoodburyPrecond` (generalises
  `StonerSpinPrecond`'s Woodbury LU to the coupled charge+spin window) +
  `build_woodbury_subspace` (freezes the low-rank codensity factors from a
  converged reference; one `K_Hxc` apply per column, once) +
  `apply_chi0_subspace` (matrix-free reference).
- `tests/unit/test_woodbury_chi0_precond.py` — Woodbury identity vs dense solve;
  the frozen low-rank M_ρ = χ₀_sub·K_Hxc reproduces the matrix-free subspace χ₀
  **to 9.2e-16** on a physical (band-limited) residual.
- `experiments/chi0_precond_crux/stage2_cheap_student.py` — the 4-arm A/B
  harness (production / pulay-control / exact teacher / cheap Woodbury).

## A/B tables

<!-- FE_TABLES -->

## Fixed-point identity

<!-- IDENTITY -->

## Verdict

<!-- VERDICT -->
