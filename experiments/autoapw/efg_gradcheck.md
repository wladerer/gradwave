# Diagnostic: is the FLAPW EFG autograd-differentiable through the SCF?

Enabling question for differentiable EFG "NMR-crystallography" (refining structure by
gradient descent on measured quadrupolar couplings): does `torch.autograd` deliver
`dV_zz/du` (and `dV_zz/d(c/a)`) through the full self-consistent FLAPW chain, matching a
finite-difference reference?

Base branch: `fix-rho2m-radial` @ 40705de (the corrected `/r²` multipole density — the FD
runs below use the physically-scaled EFG, not the pre-fix buggy magnitudes).

## VERDICT: (B) implicit-diff needed — but nothing transfers from the PW pattern; effort is weeks, not a day

Not (A). Autograd through `crystal_scf_multi` returns **nothing** — not `None`, not zero: the
call would raise, because **no torch autograd graph is ever constructed anywhere in
`gradwave.flapw`.** The entire forward SCF and the entire EFG evaluation are numpy + scipy.
`V_zz` is a Python `float` produced by `numpy.linalg.eigvalsh`; `u` never was a tensor. There
is no graph to break — one is never built.

It is (B) rather than (C) only in the *abstract* sense that the implicit-function theorem
applies at the converged fixed point (it always does) and FLAPW *does* retain the pieces an
adjoint would need (the augmented eigenvectors `c_prev`, the secular `H`/`S` build). But the
plane-wave implicit-diff code (`gradwave.scf.implicit`) is **not a usable template**: it
differentiates *XC functional parameters*, not *structure*, and every operator in it
(`apply_chi0`'s PW-basis Sternheimer, `HamiltonianK`, `g_to_r`/`r_to_g` fftbox, `apply_k_hxc`)
is plane-wave-native with **zero FLAPW analog**. Wiring FLAPW EFG into an implicit-diff
structural gradient is a from-scratch FLAPW-DFPT-scale subsystem (details + effort below),
closer to a multi-week feature than a day.

**Practical corollary that matters more than the verdict:** rutile has only **two** structural
degrees of freedom (u, c/a), so finite-difference refinement is *buildable today* — 4-5 SCFs per
gradient step, no autograd work. The one caveat (see "Convergence fragility" below) is that plain
Anderson on this fullpot config is chaotically fragile, so each SCF must use the stabilised recipe
(`kerker=0.7`, cold per point, smearing=0). The FD numbers below are exactly `dV_zz/du` and
`dV_zz/d(c/a)`; they enable a first refinement demo now.

---

## Level 1 — Finite-difference reference (ground truth)

Rutile TiO₂ P4₂/mnm, `experiments/autoapw/_common.py` geometry
(a=b=8.68083 Bohr, c=5.59096 Bohr, u=0.3048), Elk R_MT (Ti 1.098 Å / O 0.824 Å).
Config: **ecut 250 eV, aug-lmax 3, fullpot_lmax 4, k 2×2×2, smearing 0 (essential — TiO₂ is
an insulator; any smearing makes the EFG unstable), tol 1e-6, `kerker=0.7`.** Every point is
converged **cold and independently** (a full-state warm start skips the muffin-tin staging and
diverges) to r_v ~ 1e-5 / r_nsph ~ 1e-6, so V_zz is reproducible well below the FD signal (the
base reproduces to all printed digits across independent launches). u perturbed by ±δ in the
fractional coordinate (moves all four O consistently, preserves the space group); c/a
perturbed at fixed a (c → c ± δ·a). Central differences at δ=1e-3 and δ/2=5e-4 (linear-regime
check), plus a Richardson (4th-order) extrapolate. Probe script:
`experiments/autoapw/efg_fd_gradient.py` (run on asus).

Converged base EFG (this config, post-fix): **Ti V_zz = +18.11, O V_zz = +11.44 eV/Å²**
(Elk 11.0.2: Ti 19.34, O 19.1 — Ti now within ~6% of the reference).

### Convergence fragility (a first-order caveat for FD *and* autograd refinement)

The fullpot TiO₂ k222 fixed point is **chaotically convergence-fragile under plain Anderson**:
the same cold run reached r_v=7e-6 in 40 iterations on one launch and *stalled at r_v=1.5
after 90* on another (identical config; BLAS-level nondeterminism decides). This is the
"CHAOTIC-SENSITIVE" behaviour `flapw/newton.py` documents ("ulp-level rounding differences in F
flipped outcomes between residual 5e+3 and 1.4e-3"). Warm-starting a perturbed point from the
base `__full_state__` makes it *worse* — it skips the muffin-tin staging and diverges from
iteration 0.

Consequently a *trustworthy* FD reference requires the campaign's stabilised recipe — the
interstitial **Kerker screen `kerker=0.7`** (validated cure for the interstitial long-wavelength
sloshing, mode_analysis.py) and/or a per-point **Newton polish** (`flapw.newton.newton_polish`,
which drives r_nsph→1e-9). The FD numbers below use `kerker=0.7`, cold per point. **This same
fragility is a first-order obstacle for the eventual autograd/implicit-diff path: the adjoint is
evaluated *at* the fixed point, so the fixed point must first be reached reliably — the
refinement loop needs the stabilised solver either way.**

### FD gradient (the reliable subset) and why the full sweep is blocked

Two geometries converged *tightly* (r_v ~ 1e-5, r_nsph ~ 1e-6, cold + `kerker=0.7`) — the base
and u+δ — giving a first-order **forward-difference** `dV_zz/du`:

| site | V_zz base | V_zz(u+δ), δ=1e-3 | **dV_zz/du** (forward) |
|------|-----------|-------------------|------------------------|
| Ti (a0) | +18.1142 | +20.6242 | **+2510 eV/Å² per unit u** |
| O  (a2) | +11.4395 | +12.9247 | **+1485 eV/Å² per unit u** |

These are trustworthy (both points at r_v ~ 1e-5; the base reproduces to all printed digits
across launches). The magnitudes are large because δu=1e-3 in fractional coordinates ≈ 0.006 Å of
O displacement and the O sub-lattice EFG responds strongly to its own coordinate.

**The central differences, the δ-halving linearity check, and `dV_zz/d(c/a)` could not be
obtained reliably at this config.** Of the perturbed geometries, the −δ and ±δ/2 points
repeatedly fell into the non-convergent basin — measured r_v = 5.5 (u−δ) and 140 (u+δ/2) even
with `kerker=0.7` at 70–80 iterations, and on some launches the *base itself* ran to the
iteration cap. This is the chaotic multistability described above, not a coding error in the
probe (the two converged points bracket a clean, physical response). A trustworthy full FD (both
DOF, central, δ-halved) needs a per-point **Newton polish** (`flapw.newton.newton_polish`, which
drives r_nsph→1e-9) rather than plain Anderson — a bounded follow-up (~a day of scripting +
compute), not a research problem. **This is the same fragility the autograd/implicit-diff path
would face** (the adjoint is evaluated *at* the fixed point), so it is a shared prerequisite, not
an artefact of using finite differences.

Probe: `experiments/autoapw/efg_fd_gradient.py` (asus). It runs the full 9-point sweep and prints
per-point convergence flags + the absolute V_zz table; the central-difference block it also
prints is only meaningful for the points that reach `CONV=True`.

---

## Level 2 — EFG-evaluation differentiability at frozen density

**Does not run, and would be a physically minor partial even if implemented.** The evaluation
path `_efg_from_multipoles → _valence_v/_tensor_from_v/interstitial_l2_boundary/efg_tensor_full`
(`src/gradwave/flapw/efg.py`) is pure numpy + scipy: `np.cumsum` radial Poisson,
`np.linalg.eigvalsh`, `scipy.special.sph_harm_y`, `np.fft.fftn`. `torch.autograd.grad(V_zz,
u_tensor)` cannot be called — `V_zz` is a float with no tensor lineage.

Empirical (`experiments/autoapw/efg_autograd_probe.py`, asus): at a converged state
`type(info["efg"]["a0"]["V_zz"]) == float`, `isinstance(V_zz, torch.Tensor) == False`,
`grad_fn` is `None`, and the tensor is an `ndarray`. There is no torch object to differentiate.

Conceptually, at *frozen* density the only explicit u-dependence in the EFG evaluation is the
interstitial boundary phase `exp(iG·τ(u))` in `interstitial_boundary_multi` (the lattice/
antishielding term) and the own-sphere subtraction geometry. The dominant on-site valence term
`_valence_v` is a radial integral of the (frozen) `ρ_LM` with **no** explicit position
dependence. So the frozen-density partial is essentially just the lattice-term Jacobian — a
real but small slice — and it deliberately omits the physically dominant piece (the density
response `dρ_LM/du`). It is not a useful approximation to the true `dV_zz/du`.

## Level 3 — Full self-consistent gradient: where the graph "breaks"

**Empirically pinpointed** (`efg_autograd_probe.py`): passing a `u` that is a
`torch.Tensor(requires_grad=True)` into `crystal_scf_multi` raises at the **very first line of
`_multi_setup`**, `scf.py:1027`:

```
atoms_cart = [(np.asarray(f, dtype=float) @ A, sym) for f, sym in atoms]
→ RuntimeError: Can't call numpy() on Tensor that requires grad.
```

So autograd does not silently return `None`/zero — it fails loudly at input marshalling: the
fractional→Cartesian position conversion is a `numpy` op, before any physics runs. Even if that
line were made torch-aware, every downstream site below is numpy and would re-sever the graph.
The load-bearing sites, in pipeline order:

- **Eigensolve** — `scf._lapw_multi_k` builds `H`, `S` as numpy arrays (scipy `sph_harm_y`,
  `np.outer`, `eval_legendre`) and solves via `lapw.solve_geneig` → `numpy.linalg.eigh`
  (`scf.py:596`). Eigenvectors are numpy.
- **Density** — `_interstitial_density` (`np.fft.ifftn`), `_sphere_valence_density`/`_bands_amps`
  (numpy GEMMs), `efg.sphere_density_multipoles_bands` (numpy + scipy Gaunt). All numpy.
- **XC** — `vxc_lda(torch.tensor(rho_sph)).numpy()` (`scf.py:1419`, `:842`): a tensor is built
  *from* a numpy array and immediately `.numpy()`'d. Even `vxc_lda`'s internal
  `torch.autograd.grad` (`functionals.py:32`) is a self-contained numpy-in/numpy-out V_c
  derivative, not part of any end-to-end graph.
- **Weinert Coulomb** — `_weinert_multi` (`np.fft`, numpy). This is `K_Hxc`'s forward.
- **Mixing** — `anderson_next` runs on torch tensors but is fed by `torch.from_numpy(...)` and
  read back with `x_next.numpy()` (`scf.py:1433-1483`) — round-trips sever any graph.
- **State round-trip** — `_full_state` does `st.v_by_key[k].numpy().copy()` (`scf.py:1527`);
  the whole `info["state"]` / `info["efg"]` payload is numpy/floats.

The authors already know this: `flapw/newton.py:18` polishes the fixed point with
**finite-difference** directional derivatives and notes "an autograd JVP would be exact and
cheaper — planned once the residual map is torch-clean end to end." It is not torch-clean; it
is numpy. (STATUS.md, gate S3: "numpy assembly (not yet autograd)". The gate-A/C toys proved
individual *primitives* — the Θ(G) surface term, the boundary match — are autograd-exact in
isolation, but they were never assembled into a differentiable crystal SCF.)

---

## Effort to reach (B), and the top risk

To get `dV_zz/du = ∂V_zz/∂u|_explicit + (∂V_zz/∂ρ)·(dρ/du)` with the full self-consistent
response `dρ/du = χ₀·(dV_ext/du)` screened by `K_Hxc` at the fixed point, one must build,
essentially from scratch:

1. **A differentiable (or hand-Jacobian) EFG evaluation** — port `efg.py`'s multipole→Poisson→
   tensor to torch for `∂V_zz/∂ρ_LM` and `∂V_zz/∂u|_explicit`. Small/mechanical.
2. **The displacement perturbation `dV_ext/du`** — the FLAPW analog of the ionic force
   derivative, *including the moving muffin-tin boundary Pulay term*. This is the Θ(G)
   surface-term physics the whole AutoAPW thesis hinges on (gate A): as an atom moves, the
   region partition moves, and the derivative is invisible to naive grid autograd but exact
   through `exp(iG·τ)`. The primitive is validated in isolation; it is **not** wired into the
   crystal augmentation/matching/interstitial split.
3. **χ₀ in the augmented LAPW basis** — a Sternheimer/Dyson solve over the augmented
   eigenvectors with the two-region (muffin-tin ρ_LM + interstitial grid) density split and the
   LO channels. **None** of `scf.implicit`'s PW χ₀ transfers. This is the bulk of the work and
   is FLAPW-DFPT in scope (cf. STATUS.md's separate, still-unbuilt "core-Sternheimer for Ti").
4. **A JVP for `K_Hxc`** — the Weinert Poisson + vxc kernel, currently numpy.

**Effort: multiple weeks** (a genuine FLAPW linear-response subsystem), not a day.

**Top risk:** getting χ₀ right in the augmented basis with the two-region split *and* threading
the moving-boundary Pulay term (2) through the position-perturbation of the augmentation
matching coefficients and the interstitial partition — the exact ingredient that is "invisible
to naive autograd." That coupling is the make-or-break, and there is no reference for the
intermediate quantities to validate against (only the final FD `dV_zz/du` below).

## Bottom line

- **(A) day-away: refuted.** No autograd graph exists in `gradwave.flapw` at all.
- **(B) implicit-diff: the correct target, but a weeks-long from-scratch FLAPW-DFPT build** —
  the PW `scf.implicit` pattern is the right *idea* and the wrong *code* (XC-parameter, not
  structural; PW-basis operators with no FLAPW analog).
- **Buildable now, expensively:** FD refinement on rutile's 2 DOF (u, c/a) works today — the
  forward map is deterministic and, with the stabilised recipe (`kerker=0.7`, smearing=0, cold
  per point), tightly and reproducibly convergent, and the FD gradients below *are* `dV_zz/du`
  and `dV_zz/d(c/a)`. Plain Anderson alone is too fragile (above), so the demo needs the screen.
