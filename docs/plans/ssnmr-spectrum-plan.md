# Solid-state NMR spectrum plan

Status: **in progress** (written 2026-08-26). Phase 1 is this PR; Phases 2 and 3
are being built in parallel on their own branches; the final wiring lands once
all three are in.

The goal is a driveable path from a crystal structure to a solid-state NMR
spectrum: absolute chemical shieldings and quadrupolar couplings out of the DFT
stack, referenced to a chemical-shift scale, then synthesized into a lineshape a
spectroscopist can compare to an experiment. The physics primitives already
exist and are validated per-material (see `docs/plans/flapw-nmr-consolidation.md`);
what is missing is the driver wiring, the referencing convention, and the
spectrum-synthesis layer. This document is the forward plan.

Only paths that already exist are written as repo-relative references here (the
`scripts/check_doc_refs.py` doc guard fails on a reference that does not resolve).
Modules that a later phase creates are named without their `src/gradwave/` root so
the guard does not flag them before they land.

---

## Phase 1 — absolute GIPAW shielding in the driver + referencing (this PR)

Before this PR the NMR driver `src/gradwave/api/flapw.py` reached only the bare
valence term (`sigma_shielding_dq` in `src/gradwave/postscf/kgeometry_nmr.py`).
The full absolute assembly `sigma_shielding_gipaw` — σ = σ_bare + σ_core +
σ_dia_aug + σ_para_aug, built on an all-PAW ground state, using the augmentation
physics in `src/gradwave/postscf/gipaw.py` — existed and was validated but was not
reachable from `api.run`.

This phase wires it in:

- **Driver.** `nmr.shielding_level` (`auto` | `bare` | `gipaw`) on the NMR params
  in `src/gradwave/inputs/models.py`. `auto` selects `gipaw` when the ground state
  is USPP/PAW and `bare` for norm-conserving. The gipaw path emits the per-site
  per-term breakdown (σ_bare / σ_core / σ_dia_aug / σ_para_aug / total, ppm)
  alongside the Haeberlen CSA quantities (σ_iso, anisotropy, η, span). The bare
  path is unchanged for existing norm-conserving users.
- **Referencing.** δ_iso = σ_ref − σ_iso. `nmr.sigma_ref` maps a species or
  isotope to a user-supplied reference shielding σ_ref (ppm); each matching site
  reports `delta_iso_ppm`. `api.reference_sigma_iso` runs the same-level shielding
  on a reference solid and returns its σ_iso, so σ_ref can be produced in-code
  rather than transcribed.
- **Validation.** The two published GIPAW anchors — diamond-structure Si ²⁹Si
  σ_iso ≈ 400 ppm and MgO ¹⁷O ≈ 215 ppm — on PAW pseudopotentials, with the mesh
  and error bars recorded in the PR.

Deferred out of this phase: the noncollinear/hybrid shielding path, and any
change to the bare-route numerics.

---

## Phase 2 — spectrum synthesis (`postscf/nmr_spectrum.py`, branch `feat/nmr-spectrum`)

A new post-SCF module that turns the σ tensors (and, for quadrupolar nuclei, the
EFG Cq/η) into a lineshape:

- static CSA powder pattern from the Haeberlen (σ_iso, Δσ, η);
- magic-angle-spinning sideband manifolds (Herzfeld–Berger style) at a given
  spinning rate;
- second-order quadrupolar central-transition lineshapes for half-integer spins;
- Gaussian/Lorentzian broadening and a powder-average integrator.

This is being built in parallel; it consumes the Phase 1 σ output and the Phase 3
EFG output and owns its own file. Phase 1 does not create or edit it.

---

## Phase 3 — PW-side Petrilli–Blöchl PAW EFG (postscf, branch `feat/paw-efg`)

A plane-wave PAW electric-field-gradient path in the post-SCF layer, computed from
the ground state alone (no response solve), reusing the on-site reconstruction
(`PAWOnSite`) already carried by the shielding stack. It is cross-validated against
the all-electron FLAPW EFG (`src/gradwave/flapw/nmr.py`, reached today via
`nmr.task='efg'`), which is the trusted oracle. This gives Cq and η on the same PW
PAW ground state that produces the shieldings, so a single SCF feeds both halves of
the spectrum.

---

## Final wiring (after Phases 1–3)

The spectrum block is fed by the σ tensors (Phase 1) and the EFG Cq/η (Phase 3)
through the synthesis module (Phase 2). The headline validation is a spin-½ case
where the shielding alone sets the lineshape: ²⁹Si MAS of α-quartz and cristobalite,
against the published isotropic shifts and CSA. Quadrupolar cases (¹⁷O, ²³Na, ²⁷Al)
follow once the EFG path and the second-order CT lineshape are both in.

---

## Sequence

1. **Phase 1 (this PR)** — absolute σ in the driver + referencing.
2. **Phases 2 and 3** — in parallel on `feat/nmr-spectrum` and `feat/paw-efg`;
   disjoint files from Phase 1 and from each other.
3. **Final wiring** — spectrum fed by σ + EFG; ²⁹Si MAS headline validation.
