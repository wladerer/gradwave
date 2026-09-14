# Plan: differentiable equilibrium isotope β-factors (Bigeleisen–Mayer RPFR)

Status: **queued** (behind the PAW-COHP PR `feat/paw-cohp-ae-reconstruction`, which
touches the same differentiable-post-processing area). Scoped 2026-09-13.

## Why (and why NOT ICOHP)

Isotopes are electronically identical (Born–Oppenheimer: same Z), so every
electronic descriptor — ICOHP, charges, band structure — is byte-identical across
isotopologues and useless for isotope separation. The isotope effect is entirely
*nuclear-mass* through vibrations: force constants (the Hessian) are electronic and
isotope-independent, but the mass-weighted dynamical matrix `M^{-1/2} H M^{-1/2}`
gives isotope-dependent frequencies. Equilibrium fractionation between two phases/
sites is set by the **reduced partition function ratio** (RPFR, "β-factor",
Bigeleisen–Mayer 1947 / Urey 1947), which depends ONLY on those frequencies. This
is the physically correct DFT handle for (chemical-exchange / geochemical) isotope
fractionation — Li, Mg, B, Ca, O, … stable-isotope science. (Mechanical enrichment
— centrifuge, diffusion — is mass-only and has no DFT content.)

## The physics

For a molecule/cluster (3N−6 modes) or a crystal (phonon modes, per q or a DOS
integral), with `u_i = ℏω_i/(k_B T) = c₂·ω̃_i/T` (c₂ = hc/k_B = 1.43877 cm·K for
ω̃ in cm⁻¹), the RPFR of a light (unstarred) vs heavy (starred) isotopologue is

    β = (RPFR) = Π_i  [ (u_i*/u_i) · e^{-(u_i*-u_i)/2} · (1 − e^{-u_i})/(1 − e^{-u_i*}) ]

reported as 1000·ln β (per mil). The classic Bigeleisen–Mayer form factors the same
quantity as Π (u_i*/u_i)(sinh(u_i/2)/sinh(u_i*/2)). Equilibrium fractionation between
phases A and B is 1000 ln α_{A-B} ≈ 1000 ln β_A − 1000 ln β_B.

**Key efficiency:** H is isotope-independent, so one DFT Hessian → frequencies for
BOTH isotopologues by re-diagonalizing with two mass vectors. All isotope pairs are
near-free post-processing on a single (expensive) Hessian.

## What already exists (reuse, do not rebuild)

- `postscf.phonons.gamma_hessian(res, xc, ...)` — Γ Hessian [eV/Å²], built from
  `postscf.uspp_position.hessian_column` (the differentiable response/adjoint
  kernel). Isotope-independent. THE expensive piece, already here.
- `postscf.phonons.gamma_frequencies(hess, masses_amu)` — frequencies from a Hessian
  and a per-atom mass vector. Masses are already an explicit argument — swapping in a
  heavy isotope's mass is a one-liner.
- `task: qha` / phonon DOS machinery — for the crystal DOS-integral β-factor.
- `constants` (KB_EV; add c₂ / ℏ as needed), `dtypes`.

## Gaps to build

1. **The RPFR/β-factor kernel.** A function `beta_factor(freqs_light, freqs_heavy, T)
   -> 1000 ln β`, with the Bigeleisen–Mayer product, careful handling of the 3
   translational (molecule) / 3 acoustic (crystal, ω→0) modes — these must be
   excluded/limited correctly (u*/u → 1 as ω→0; the standard treatment drops the
   zero modes). Likely `postscf/isotope.py` (new) or an addition to `postscf/thermo`.
2. **Site/element projection.** Site-specific 1000 ln β (the observable for
   fractionation between crystallographic sites or between a solid and a fluid) needs
   the modes weighted by the target atom's displacement — i.e. eigenVECTORS, which
   `gamma_frequencies` does not currently return. Add an eigenvector-returning
   variant and a projected-RPFR (weight each mode by |e_{i,atom}|² for the isotope-
   substituted site). Cross-check against the phonon-DOS route for crystals.
3. **Differentiability end-to-end.** `gamma_frequencies` is numpy today; the RPFR
   must be torch. Build a torch path: mass-weight `H`, `torch.linalg.eigh` (backward-
   differentiable), analytic RPFR. Confirm `gamma_hessian`'s output stays on the
   autograd graph (it composes `hessian_column`, which is differentiable — verify no
   `.cpu().numpy()` detour breaks the graph). Then d(1000 ln β)/d(x) is available for
   x = atomic positions/strain and, via `alchemical.py`, d/dZ — the differentiable
   design signal for MAXIMIZING equilibrium fractionation (a material/ligand that
   most enriches a target isotope). This is the analog of the alchemical-ICOHP idea,
   but the *physically correct* one for isotopes.
4. **API surface.** Ride `run_phonons` the way harmonic `thermo` does (harmonic
   thermo already rides the phonon task), or a small `api.run_beta_factors` /
   `isotopes:` block: inputs = the element/site + isotope pair(s) + T (or T grid);
   output = 1000 ln β per site (and α between sites if >1). Numbers-in, no extra SCF
   beyond the Hessian.

## Validation

- **Analytic diatomic** first: a molecule where the RPFR is a single-mode closed
  form (e.g. an isotopologue of a diatomic in a box) — exact check of the kernel and
  the u = c₂ω̃/T units, independent of DFT error.
- **Literature β-factors**: published DFT RPFRs exist for standard systems (e.g. O
  isotopes in quartz/calcite, Li in minerals, Mg). Compare 1000 ln β at 300–1000 K;
  β-factors are a mature, well-tabulated geochemistry observable, so there IS an
  external oracle here (unlike COHP's near-absence).
- **Differentiability**: finite-difference d(1000 ln β)/d(position) vs autograd.

## Notes / subtleties

- Harmonic RPFR is the standard; anharmonic corrections (and the QHA volume
  dependence via the existing `qha` stack) are a later refinement.
- Γ-only Hessian captures a molecule/cluster exactly; for a crystal use a supercell-Γ
  Hessian or the q-mesh phonon DOS so acoustic contributions are represented.
- Scope stays on legitimate stable-isotope science (geochemistry, medical/stable
  isotopes). Not a weapons-enrichment tool — the sensitive U methods are mechanical
  and have no DFT content.
