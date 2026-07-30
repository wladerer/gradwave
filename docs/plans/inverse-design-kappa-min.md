# Plan: gradient-based inverse design of a minimum-thermal-conductivity descriptor

## Status

Proposed, not started (2026-07-29). A methods proof of concept that couples the
alchemical composition channel (`scf/alchemical.py`) to the mechanical-properties
modules (`postscf/elastic.py`, `postscf/stress.py`) to get exact composition
gradients of an elastic minimum-thermal-conductivity descriptor on small glass
models.

## Scope, stated honestly up front

This is a methods POC for exact composition gradients of an elastic kappa_min
descriptor on periodic glass models. It is not a thermal-conductivity-of-glasses
study, and no section below should be read as one. The target quantity is the
Clarke / Cahill-Pohl minimum thermal conductivity, an elastic screening
descriptor built from density and sound velocities. Real glass thermal transport
is a different physics and a different (and far larger) calculation, set out in
the next section as the reason this plan is a reframing and not the original
ambition.

## Why not real glass kappa

The original ambition was low or anisotropic thermal conductivity of glasses.
That is out of reach here, for reasons worth recording so the reframing is not
mistaken for the goal.

Glass thermal conductivity is Allen-Feldman diffuson transport plus anharmonic
corrections. It needs the full dynamical matrix of a 200+ atom amorphous cell,
and it needs the amorphous cell in the first place. Against this repo:

- **No thermal-transport code of any kind exists.** A grep of `src/` finds no
  Allen-Feldman, no Boltzmann transport, no Green-Kubo. The only `kappa` in the
  tree is the Ewald and PBE screening parameter, unrelated to heat.
- **The memory ceiling is far below 200 atoms.** `docs/ideas.md`
  ("Raising the system-size ceiling past the dense-allocation cliff") measures a
  practical ceiling of about 96 to 110 atoms on the 6 GB RTX 3050, with a hard
  cliff at 128 atoms from a single roughly 37 GB O(npw^2) dense allocation.
- **Supercell phonons are expensive and NC-only.** `postscf/phonons_supercell.py`
  costs 6N full SCF re-convergences for a P1 cell with no symmetry reduction, and
  `api.run_phonons` raises `NotImplementedError` for USPP/PAW pseudopotentials.
- **There is no melt-quench capability**, so the amorphous model has to come from
  the literature rather than be generated here.

Hardware sets the same limit from the other side. The workhorse is CPU fp64 at 8
threads (the RTX 3050 fp64 rate is about 1/64 of fp32), and a 10-atom hematite
SCF already costs 2275 s on CPU against 594 s on GPU (`benchmarks/minerals/README.md`).
A 200-atom Allen-Feldman calculation is not a scoping question, it is off the
table.

## The surviving plan

Drop the transport physics and keep the descriptor. kappa_min from density and
sound velocities is an established screening quantity in the thermal-barrier-coating
literature (Clarke, Cahill-Pohl), it needs only the elastic tensor and the
density, and both are reachable here. The differentiable content is the composition
gradient of that descriptor, which the alchemical channel makes exact rather than
finite-difference. Five phases, each gated.

### Phase 1, the kappa_min module (post-processing, no SCF cost)

A new `postscf` module implementing the Clarke and Cahill-Pohl minimum-conductivity
models from density and VRH sound velocities. The VRH moduli come from the existing
`moduli_from_cij` (`postscf/elastic.py`). That function is NumPy, so the
descriptor is reimplemented in torch here rather than called through, because
autograd has to reach it in Phase 3. Direction-resolved velocities come from the
Christoffel equation on the full C_ij, giving an anisotropic kappa_min(n). The whole
module is torch end-to-end for that reason, not NumPy. Validate on Si and
alpha-quartz against literature kappa_min values.

### Phase 2, crystal validation (days of `gwq` queue on asus)

Elastic tensor and kappa_min for alpha-quartz (9 atoms) and alpha-GeO2. Gate,
kappa_min within roughly 20 to 30 percent of literature and correct orderings
between the two.

The known caveat to measure here, `postscf/elastic.py` is clamped-ion. Its
docstring states the clamped-ion C44 overestimates the relaxed value (PBE Si
about 98 GPa clamped against about 76 relaxed) with no internal-strain / Kleinman
correction. C11/C12 and the bulk modulus are unaffected, but the shear velocity
feeds kappa_min, so this error matters for open network structures. Quartz and
GeO2 are exactly where the answer is known, so this phase quantifies the
clamped-ion error rather than assuming it is small.

### Phase 3, the differentiable link dC_ij/dlambda (about 1 to 2 weeks)

Today C_ij is an outer finite-difference loop over 12 re-converged strained SCFs
(`elastic_tensor` central-differences 6 strains, `postscf/elastic.py`), so autograd
does not reach the elastic tensor. Three pieces close the gap:

- stress within a single SCF is differentiable (`postscf/stress.py`),
- the alchemical lambda channel threads into the SCF (`scf/alchemical.py`:
  `alchemical_charges`, `blend_local_table`, `blend_projector_data`,
  `setup_alchemical_system`, `alchemical_energy_gradient`), with phases 1 to 4 of
  `docs/design/differentiable-composition.md` implemented and FD-validated in
  `tests/unit/test_alchemical_composition.py`, including heterovalent Si-to-Ge,
- the implicit SCF adjoint exists for NC insulators (`scf/implicit.py`).

So dC/dlambda is the same 12-strain central differences applied to adjoint-computed
dsigma/dlambda, and dkappa_min/dlambda follows by the chain rule through the torch
kappa_min module. Oracle, match a full finite difference of kappa_min over lambda.
Fallback if the adjoint route fails, pure-FD gradients at twice the elastic cost
per composition point.

This closes two promises `docs/design/differentiable-composition.md` makes and has
not delivered, the Phase 3 property-response fold (dB_modulus/dlambda) and a Phase 5
non-energy inverse-design demo.

### Phase 4, glass-model inverse design (about 1 to 2 weeks of asus queue)

A 24 to 36 atom published periodic a-SiO2 model, well under the memory ceiling,
Gamma-only or 2x2x2, NC ONCV pseudos for Si, O, and Ge all in-repo, insulating
nspin=1 so inside every implicit-adjoint scope limit. Per-site lambda on the Si
sites, per-site dkappa_min/dlambda_i ranks the Ge substitution sites, and the
existing `CompositionSurrogate` / `optimize_composition` machinery
(`postscf/composition_design.py`) drives the fraction. Snap to integer and verify
with real SCFs. Each composition point costs one elastic run of 12 SCFs, single-digit
hours on asus through `scripts/gwq`.

External validation, the sign of dkappa_min/d(Ge content) against the known kappa
reduction of GeO2-doped silica from the fiber-optics literature.

### Phase 5, stretch, one shot (decided only after Phase 3 lands)

A Gamma-point Allen-Feldman diffusivity on the final optimized model, 6N = 150 to 220
P1 SCFs, about 2 days queued on asus, plus a new AF module. Validation color only, no
gradients through it. A 36-atom AF kappa is not size-converged, and the writeup says
so plainly.

## Anisotropy tempering

An isotropic glass model with composition disorder shows kappa_min anisotropy at the
noise level, so anisotropy is not a Phase 4 goal. The Christoffel machinery is nearly
free in Phase 1 and gets exercised on quartz, where real elastic anisotropy exists.
Anisotropic glass targets (layered or chain silicates, densified models) are pursued
only if a measurable signal above the FD noise floor appears.

## Kill gates

Any one of these ends the project, with the negative result recorded in `docs/ideas.md`:

- Phase 2, kappa_min misses literature by more than roughly 30 percent, or the
  clamped-ion elastic error dominates the descriptor.
- Phase 3, the adjoint dsigma/dlambda disagrees with the finite-difference oracle.
- Phase 4, the per-site gradients are smaller than the FD noise floor.

## References

- Clarke, *Surf. Coat. Technol.* **163-164**, 67 (2003) — minimum thermal conductivity
  from elastic moduli.
- Cahill, Watson, Pohl, *Phys. Rev. B* **46**, 6131 (1992) — lower limit to thermal
  conductivity.
- Allen, Feldman, *Phys. Rev. B* **48**, 12581 (1993) — thermal conductivity of
  disordered harmonic solids (Phase 5 context, not a target of this plan).
