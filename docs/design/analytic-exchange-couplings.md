# Analytic exchange couplings J_ij / D_ij via the response kernel — Gate A verdict

**Status: BLOCKED at de-risk Gate A (measured 2026-09-11). The scoped analytic
route needs a subsystem gradwave does not have — a metallic *spinor* χ₀
(transverse spin susceptibility). This note records exactly what was checked, why
the campaign stops, and what a build would actually require, so the next attempt
starts from the real scope instead of the optimistic one in `docs/ideas.md`.**

## The goal

`postscf/spin_exchange.py` extracts the spin-Hamiltonian couplings

    𝒥_IJ^{ab} = -∂²W/∂ê_I^a ∂ê_J^b = ∂T_I^a/∂ê_J^b

(Heisenberg J, DMI D, anisotropic Γ) from constrained-moment SCFs. The per-atom
torque T_I = -dW/dê_I is already exact (the Ma–Dudarev penalty's constraining
field equals minus the internal field at convergence; validated against FD at
ratio 1.000). The *couplings* are its site-to-site derivative, currently taken by
an **outer finite difference**: tilt ê_j by δ along two transverse axes, run two
extra constrained SCFs per (pair, axis), read the induced analytic torque on i.
That ÷δ amplifies SCF noise — the documented reason DMI needs `etol ~1e-9` (at
`1e-5` the D's are 1–3 meV irreproducible noise; the bcc-Fe symmetry null D=0.0000
only passes tight), at a cost of ~5 h per DMI reference.

The campaign proposed to compute the mixed second derivative analytically:

    dT_i/dê_j = ∂T_i/∂ê_j|_ρ  +  (∂T_i/∂ρ)·(dρ/dê_j),

where the implicit term `dρ/dê_j` is one linear-response solve — χ₀ applied
through the constrained path with the constraint-field derivative as RHS — instead
of a tight re-converged SCF.

## Gate A (decisive): does the response kernel run through the noncollinear path?

**No.** Probe: `probes/gate_a_response_nc_path.py`, run on asus on the O2 triplet
(the small genuinely-magnetic single-k system from `tests/integration/test_moment_config.py`).
Two facts, both CONFIRMED empirically:

1. **The constrained-moment torque carries no autograd graph.**
   `scf.noncollinear.scf_noncollinear` is decorated `@torch.no_grad()`. So
   `constrained_moment_scf`'s `info["torque"]` and the atomic moments `M` it is
   built from come back with `requires_grad=False` / `grad_fn=None`. There is no
   graph from ê_j to T_i to differentiate. => the "double-backward / HVP through
   the unrolled last SCF iterations" fallback the campaign flagged **has no
   substrate**: nothing on the spinor SCF path is differentiable, and the
   implicit-diff adjoint machinery (`scf/implicit.py`, `scf/soft_mode.py`) is built
   for the *collinear* `scf.loop.scf`, not for `scf_noncollinear` (neither file
   references `NCResult`).

2. **`apply_chi0` is collinear-only and structurally cannot represent the response
   a tilt induces.** `scf.implicit.apply_chi0` takes an `SCFResult` and handles
   nspin=1 (a scalar grid field) or nspin=2 (a per-spin field, χ₀ **block-diagonal
   over spin**). It handles *metals* fine (`_chi0_channel_metal`, the partial-
   occupation window path). But it maps a scalar/per-channel *local* field to a
   scalar/per-channel *density* response — it has no (ρ, m⃗) spinor contract. The
   probe confirms it rejects the `NCResult` outright (`AttributeError: 'NCResult'
   object has no attribute 'v_eff'`). More fundamentally: tilting ê_j at a
   collinear reference (all moments along ẑ) is intrinsically a **transverse**
   perturbation of the spin axis — it adds a constraining field with an m_x/m_y
   component and the physical response is transverse magnetization. That is exactly
   the block a collinear (spin-diagonal) χ₀ discards. It cannot be faked with the
   collinear kernel.

Because the FD-free route reduces to `dρ/dê_j = χ₀_spinor · (∂B_c/∂ê_j)` and the
double-backward route reduces to differentiating the (no_grad) spinor SCF fixed
point, **both analytic routes require the same missing object**: a metallic spinor
χ₀. Gate B (the tight-etol control) and Gate C (the loose-etol payoff) were not
run — there is nothing to validate until that object exists.

## What exists, and what is missing

Present toward the goal:

- **The K side is done.** `postscf/_response._fxc_hvp_noncollinear` is the coupled
  (ρ, m⃗) f_xc Hessian-vector product at a *nonzero* moment (double-backward through
  `core.xc.noncollinear.energy_with_grid`, the SCF's own linearization point). Plus
  the Hartree kernel. So `K_Hxc` on a spinor (ρ, m⃗) field is available.
- **The metallic occupation machinery is done, but only collinear.**
  `_chi0_channel_metal` + `occupation_derivative` + `divided_difference_weights`
  (Adler–Wiser) handle the Fermi-surface / partial-occupation response for scalar
  channels.
- **A spinor Sternheimer exists, but insulator-only and nonmagnetic.**
  `postscf.dielectric._dielectric_born_soc` solves a spinor conduction-projected
  Sternheimer via `cg_sternheimer` (which accepts a `SpinorHamiltonian`), but it is
  restricted to insulating occupations (`insulator_window`) at m⃗ ≡ 0 and uses the
  nonmagnetic kernel `fxc_hvp_noncollinear_nonmagnetic`. Its own docstring states a
  nonzero moment needs the coupled response, "which this does not provide."

Missing — **a metallic spinor χ₀ (transverse spin susceptibility of a magnetic
metal)**. Concretely, to build the analytic path one would need:

1. **A spinor χ₀ apply** that maps a spinor perturbation (δV, δB⃗) to the spinor
   response (δρ, δm⃗), summed over spinor bands with **partial occupations** (a
   Fermi surface). `cg_sternheimer` is conduction-projected → insulator-only; the
   metallic response needs the full window path (`_chi0_channel_metal`-analog:
   intraband occupation-derivative term + interband Adler–Wiser + the δμ
   number-conservation term) but over **spinor** bands, with the response
   accumulated into all four (ρ, m_x, m_y, m_z) components rather than a scalar
   density. This is the bulk of the work and does not exist anywhere in the tree.
2. **The transverse susceptibility is the numerically delicate part.** χ_⊥ of a
   magnet carries the Goldstone/magnon soft modes; near a spin instability the
   screened response `u = v̄ + K χ₀ u` has gain > 1 (the NiO lesson that already
   forces Anderson mixing in the collinear `solve_adjoint`). The transverse block
   is *softer* than anything the collinear adjoint has had to tame, so convergence
   of the Dyson/adjoint fixed point is an open risk, not a given.
3. **Either** a differentiable adjoint through `scf_noncollinear` (today
   `@torch.no_grad`), **or** a bespoke assembled response solve that never touches
   autograd through the SCF. Given (1) is a hand-assembled χ₀ anyway, the bespoke
   solve is the more honest path — but it is a second implementation of the whole
   implicit-diff stack for the spinor formalism.
4. **The explicit constraint term** ∂T_i/∂ê_j|_ρ (the penalty's own ê_j-dependence
   at fixed density) is cheap and analytic from `scf.moment_penalty` — not a
   blocker, but it must be assembled alongside the implicit term, and its
   correctness is exactly what Gate B would have validated.

This is a second-code-project-sized subsystem (a new spinor response kernel plus
its adjoint), squarely the "large scope → STOP and report" condition the campaign
set. It is not wired here, and no half-built production path was left behind.

## Recommendation

- **Keep the FD extractor as the production path.** It works at tight etol; the
  cost (`~5 h`/DMI reference) is the price of the ÷δ noise, not a correctness gap.
- **If the analytic route is revived, the prerequisite is the metallic spinor χ₀,
  not a "second autograd pass over the torque."** `docs/ideas.md`'s "second
  derivatives of the penalty scalar are within reach of the autograd path" is
  refuted by Gate A: the penalty scalar's second derivative *through the density*
  is precisely χ₀_spinor, and the SCF that produces the density is not
  differentiable. Budget the spinor-χ₀ build (items 1–3 above) as the real work.
- **A cheaper interim** for individual-shell J and the DMI q→0 slope is the
  reciprocal-space route already sketched in `docs/ideas.md` (the spin-spiral
  E(+q) − E(−q) asymmetry / J(q) sweep), which stays on the existing forward SCF
  and sidesteps the response kernel entirely.

## Reproduce

    ssh asus 'cd /tmp/gw-anexch && PYTHONPATH=. uv run python probes/gate_a_response_nc_path.py'
