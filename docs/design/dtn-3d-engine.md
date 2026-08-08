# Design: 3D open-boundary (DtN) plane-wave DFT for surfaces

Taking the validated 1D jellium prototype (`experiments/dtn_1d/`, PR #265) to a real
3D capability: surfaces/slabs/adsorbates in a slab-sized box with no vacuum tax, no
dipole correction, and differentiable surface forces. This is the plan.

## What the 1D prototype settled (and what it did not)

Settled, in a reduced model: (a) open-BC electrostatics makes surface observables
box-independent by construction; (b) it resolves the surface dipole a periodic box
collapses; (c) autograd gives exact surface forces through the boundary fixed point
(0.74% vs finite difference); (d) the energy-exact DtN self-energy is implementable
via a Green's-function contour density (matched the eigensolver to 5.7e-12).

Not settled: any of the 3D machinery. The prototype is 1D, jellium (no atoms, no
pseudopotentials), single `G∥=0` channel. Everything below is new.

## The core difficulty

A plane-wave basis is **periodic in z by construction** — `ψ = Σ_G c_G e^{i(k+G)·r}`
tiles the box in all three directions. The vacuum boundary we want (semi-infinite
open space in z) breaks that periodicity. So there are two *separable* problems, with
very different cost:

1. **Electrostatics.** The Hartree/Poisson solve is periodic (`v_H(G)=4πe²ρ(G)/G²`),
   so a slab feels its z-images (spurious dipole field, image interaction). This can
   be fixed **without touching the wavefunction basis** — only the Poisson solve.
2. **The wavefunctions themselves.** The orbitals are periodic in z; imposing the
   *exact* evanescent vacuum tail requires a non-periodic z representation — a
   different basis, a different Hamiltonian apply, a different SCF engine.

Problem 1 is tractable and captures most of the value. Problem 2 is the multi-quarter
research build. The plan phases accordingly.

---

## Phase 1 — Differentiable open-boundary electrostatics (ESM)

**This is the recommended first build and the 80% win.** It is the
Otani–Sugino *Effective Screening Medium* method (PRB 73, 115407 (2006); implemented
in QE's `esm`), which gradwave would add — and, uniquely, make **differentiable**.

**Mechanism.** In-plane periodicity makes `G∥` a good quantum number, so the Poisson
equation is diagonal in `G∥`: for each in-plane reciprocal vector,
`(∂²_z − |G∥|²) v_H(G∥, z) = −4πe² ρ(G∥, z)`, solved with the *open* 1D Green's
function `e^{−|G∥||z−z'|}/(2|G∥|)` (and the linear/neutral solution for `G∥=0`) instead
of the periodic `1/(G∥²+G_z²)`. The wavefunction basis is untouched — the slab still
needs enough vacuum for the orbitals to decay (ESM fixes the *field*, not the tail),
but the electrostatics is now exact and box-independent, and asymmetric slabs need no
dipole correction. Sub-modes (from ESM): vacuum/vacuum (bare open), vacuum/metal, and
metal/metal (a biased capacitor — this is what unlocks constant-potential
electrochemistry).

**gradwave module map.**
- `core/energies/hartree.py` — the heart. Add an `open_z` Poisson path beside the
  periodic `hartree_potential_r`: in-plane rFFT → per-`G∥` 1D open solve in z → inverse
  in-plane FFT. The 1D prototype's `poisson(..., mode="open")` is the `G∥=0` special
  case; generalize the screened `|G∥|>0` Green's function.
- `core/energies/total.py` — the `G=0` divergence bookkeeping changes (ESM handles the
  average potential / net charge differently; a charged cell becomes well-defined).
- `postscf/stress.py` — the in-plane stress is unchanged; the z "stress" is replaced by
  a force on the boundary planes (ESM has no z-periodicity to strain). Repurpose or
  gate the `e_h` Hartree-stress term (stress.py:231/433) for the open-z case.
- `inputs.py` / `scf/loop.py` — a `boundary: periodic | open_z` config, `esm_mode:
  vacuum | metal | capacitor`, and the boundary z-positions / applied bias.
- `postscf/forces.py` — surface forces pick up the ESM boundary term; autograd through
  the differentiable Poisson gives them for free (the differentiable-ESM novelty).

**What it unlocks.** No dipole correction; asymmetric adsorbates (CO/Au!) done right;
charged surfaces and field-effect; constant-potential electrochemistry; box-independent
work functions and surface energies; **differentiable surface energetics** (inverse
adsorbate design, learned embedding) — none of which any non-AD ESM code offers.

**Cost / risk.** Weeks-to-two-months. Self-contained (one new Poisson path + energy/
stress/force bookkeeping + a config). Main risks: the `G∥=0` / net-charge term and the
z-stress reinterpretation (both well-documented in the ESM literature); getting the
autograd to flow cleanly through the per-`G∥` z-convolution (it is plain torch — should
be fine).

**Validation ladder.** (1) reproduce a periodic result in the `G∥=0`, large-vacuum
limit; (2) box-independence of the work function vs vacuum thickness (the 1D result, in
3D); (3) an asymmetric slab (e.g. a metal slab with an adsorbed dipole) — the two-face
work-function split, no dipole correction, vs a periodic+dipole-correction reference;
(4) benchmark surface energy / work function against QE-ESM; (5) `autograd dE/dz` on an
adsorbate vs finite difference (the Step-2 result, in 3D).

---

## Phase 2 — Mixed basis for the exact wavefunction z-BC

**The hard part, and decision-gated (see below).** To impose the *exact* evanescent
tail on the orbitals — not just the field — the basis must be non-periodic in z.

**Mechanism.** A mixed basis: 2D plane waves in-plane (`G∥`, per `k∥`) × a real-space
grid (or B-spline / DVR / finite-element) in z. The Hamiltonian is, per `k∥`, a matrix
over `(G∥, z)`: kinetic `|k∥+G∥|² − ∂²_z` (block-diagonal in `G∥`), plus the potential
`V(G∥−G∥', z)` which couples in-plane channels at each z (an in-plane convolution). The
open BC is imposed in the z representation (a decaying boundary condition, exact only in
the energy-dependent form — Phase 3).

**gradwave module map (mostly new infrastructure).**
- `grids.py` — a 2D `G∥` sphere per `k∥` + a z-grid; replaces the 3D G-sphere.
- `core/fftbox.py` / a new `core/mixed.py` — in-plane FFT + z real-space transforms.
- `core/hamiltonian.py` — a new H-apply: in-plane FFT to real space, potential multiply,
  back; z-kinetic via a finite-difference / spectral operator; the in-plane
  channel-coupling as an in-plane convolution.
- `solvers/davidson.py` — **reusable as-is**: Davidson only needs an H-apply. This is
  the one big piece that ports for free.
- `pseudo/` — the nonlocal projectors need the mixed-basis form (KB projectors as
  `β(G∥, z)` per atom); a real rebuild of the projector apply.

**Cost / risk.** Multi-quarter. It is effectively a second SCF engine (a
plane-wave-in-plane / real-space-in-z code). The projector apply and the mixed-basis
Hamiltonian are the bulk of the work.

**Decision gate — is Phase 2 worth it over Phase 1?** Honest answer: often no. For
adsorption energetics, surface energies, work functions, and electrochemistry, ESM
(Phase 1) + adequate vacuum captures the physics — the *electrostatics* was the real
artifact; the wavefunction tail decays fast and cheaply. Phase 2 earns its cost only
for: (a) memory-bound cases where you cannot afford the vacuum for the orbitals to
decay, (b) tunneling / STM / field-emission / transport, where the exact tail *is* the
observable, or (c) proving the method exactly for a paper. Recommendation: build Phase 1,
measure whether the residual wavefunction-tail error matters on the target science, and
only then decide on Phase 2.

---

## Phase 3 — Energy-exact DtN via a per-`G∥` Green's-function density

**On top of Phase 2, the full moonshot.** Replace Phase 2's approximate wavefunction
BC with the exact energy-dependent one.

**Mechanism.** Each in-plane channel is a 1D lead with `κ(G∥, E) = √(|G∥|² + 2(V_vac−E))`.
Apply the exact lead self-energy `Σ(G∥, E) = t²·g_surface(G∥, E)` per `(k∥, G∥)` and build
the density by contour integration — the validated `experiments/dtn_1d/greens.py`
machinery, now one lead *per in-plane channel per k-point*. No eigensolver; the
energy-dependent `Σ(E)` is trivial on the contour.

**gradwave module map.**
- a new `scf/` density-construction path (contour integral + `linalg.solve` per contour
  energy) replacing the eigensolver density on the open engine;
- `greens.py`'s `lead_sigma` / `greens_density`, generalized over `(k∥, G∥)`.

**Cost / risk.** Multi-quarter *on top of Phase 2*. The dense complex solves per contour
point are expensive (the 1D cost warning, cubed); a real implementation needs a sparse /
block solver and a pole/Matsubara contour (few tens of energies). Reserve for when the
exact tail genuinely matters (Phase 2's decision gate applied one level deeper).

---

## Differentiability — threaded through, not a phase

gradwave is differentiable-first, so autograd is a *requirement at each phase*, not a
finale. Phase 1's ESM Poisson is plain torch → autograd surface forces come free (the
headline novelty). Phase 2/3's SCF differentiates through the fixed point via the
existing IFT adjoint (`scf/implicit.py`) — the same machinery already shipping for the
periodic SCF, which the 1D prototype confirmed composes with a nested boundary fixed
point. Surface forces reuse `postscf/forces.py` (Hellmann-Feynman) plus the boundary
term. Inverse surface design = the parked 1D demo (`docs/ideas.md`), in 3D.

## Recommendation

1. **Build Phase 1 (differentiable ESM).** It is a real, cited method with a bounded
   scope, it delivers charged surfaces / electrochemistry / no-dipole-correction /
   differentiable surface energetics, and "differentiable ESM" is itself a paper. This
   is the deliverable.
2. **Gate Phase 2/3 on measured need.** After Phase 1, quantify the residual
   wavefunction-tail error on the target science. Only pursue the mixed-basis + exact-Σ
   engine if that error matters (transport/tunneling, or the exact-method paper).
3. **Sequence within Phase 1:** open-z Poisson (vacuum/vacuum) → energy/stress/force
   bookkeeping → the box-independence + asymmetric-dipole + autograd-force validations →
   then the metal/capacitor ESM sub-modes for electrochemistry.

The 1D prototype is the de-risking spike for all of this: its `open` mode is Phase 1 in
miniature, and its `greens.py` is Phase 3 in miniature. Both are validated, so each 3D
phase has a working reference to test against.
