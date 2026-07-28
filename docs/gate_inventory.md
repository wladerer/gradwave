# Capability-gate inventory

Tracking doc for issue #147 (umbrella). gradwave gates unsupported
formalism/feature combinations with a loud `raise NotImplementedError` rather
than returning a silently-wrong number. That is the right default, but the
density of gates means the feature matrix has holes off the happy path. This
table enumerates every gate so the holes are visible and progress is measurable.

**Metric of success is the number and user-impact of gates removed, not zero
gates.** Some gates are permanent feature boundaries (e.g. fully-relativistic
DFT+U stress, #142), not gaps — they stay, documented as such.

An ungating = remove the `raise`, thread the missing (nspin / formalism) index
through the real computation, and land a Tier-0/Tier-1 self-oracle test **in the
same commit** (the recipe from PR #45 nspin=2 post-SCF and PR #58 nspin=2 stress).

Regenerate the raw list with:

```bash
grep -rn "raise NotImplementedError" src/gradwave --include=*.py
```

Current count: **67** `raise NotImplementedError` sites, of which **4 are
abstract-method stubs** (interface contracts on base classes, not capability
gaps) — see the last section. So **63 real capability gates** remain.

## Axis legend

| axis | meaning |
|---|---|
| nspin2 | collinear spin-polarization not threaded through this operation |
| NC-only | requires norm-conserving pseudos (USPP/PAW route absent) |
| USPP/PAW | ultrasoft / PAW S-operator gap on an otherwise-supported op |
| +U | DFT+U (Hubbard) term absent from this operation |
| SOC/NC-spinor | noncollinear / fully-relativistic spinor path (often a feature boundary, #142) |
| metaGGA | τ-dependent functional not wired |
| insulator | fixed-occupation (insulator) assumption; metals unsupported |
| symmetry | needs `use_symmetry=False` (perturbation breaks the crystal symmetry) |
| data | reference data not vendored for the requested element |
| validation | argument guard (not a capability gap) |

## Done

| file:line | axis | operation | resolution |
|---|---|---|---|
| ~~api.py:876~~ | nspin2 / NC | elastic constants (`run_elastic`) for nspin=2 needed PAW/USPP | **removed, this PR** — NC stress already sums per spin channel (`postscf.stress`, PR #58); the driver gate was stale. Oracle: `test_run_elastic_nspin2_nc_matches_nspin1` (nonmag-limit == nspin=1) |
| ~~postscf/stress.py~~ (nspin=2 sum) | nspin2 | fixed-basis stress tensor | already ungated in PR #58 (per-spin kinetic/nonlocal + spin-resolved E_xc) |
| ~~postscf/paw_stress.py:36~~ | USPP/PAW +U | stress with DFT+U on USPP/PAW (strained S-dressed projections) | **removed, this PR** — added the Dudarev E_U strain term to `_energy_strained_uspp`: the atomic-orbital projectors go on the strain graph like the base PAW augmentation, then S-dressed (Sφ = φ + Σ\|β⟩q⟨β\|φ⟩) so both the φ and the β strain enter n^{Iσ} (`_hub_sproj_strained`). Also ungated the USPP/PAW +U path through `calculator._calculate_uspp` (and forced symmetry off there, matching the NC +U path). Oracle: `test_paw_stress_hubbard_autograd_vs_fd` (autograd σ == central FD of the strained +U energy; ε=0 == SCF total) + a U=0 inertness bit-for-bit check |
| ~~postscf/uspp_position.py:262,326~~ | USPP/PAW +U | position (Berry) response with DFT+U | **removed, this PR** — threaded the occupation-matrix channel n^{Iσ} through the position (displacement) response, the S-dressed analogue of #156's strain path (Sφ = φ + Σ\|β⟩q⟨β\|φ⟩ now moves with its atom): the bare perturbation gains δH_U = \|∂(Sφ)⟩D_U⟨Sφ\| + h.c. (∂(Sφ) via a jvp through the same `build_uspp_hubbard` S-dressing — `PositionPerturbation._build_dsphi`), `bare_map_derivative` returns the bare δn, and `_self_consistent_response` carries the Dudarev kernel δD_U = −(U−J)·herm(δn) through the existing `apply_chi0`/`k_hub` adjoint machinery. Oracle: `test_position_density_response_hubbard_vs_scf_fd` (analytic dρ*/dτ, dbecsum*/dτ == central FD of +U SCF re-runs, ~3e-5) |
| ~~postscf/uspp_position.py:399~~ | USPP/PAW +U | `hessian_column` (Γ-phonon mixed second derivative) with DFT+U | **removed, this PR** — added the in-graph Dudarev E_U(c, τ) term (the same `hubbard_e_channel` expression `forces_uspp` differentiates for the +U force) so the double backward picks up ∂²E_U/∂τ∂τ′, fed by the +U-total orbital response (V_U feedback threaded through `_self_consistent_response`/`_total_orbital_response`). Oracle: `test_hessian_column_hubbard_vs_fd_of_forces` (analytic +U column == central FD of the +U-aware `forces_uspp`, ~3e-6) + `test_hessian_column_u0_is_inert` (U=0 == non-+U column, ~1e-13) |
| ~~postscf/discretization_error.py:379~~ | nspin2 | Dyson-dressed disc. density error for nspin=2 | **removed, this PR** — spin-resolved coarse-space Dyson (`_dyson_dress_spin`): per-spin χ₀ (conduction-projected Sternheimer, block-diagonal in spin, `_apply_chi0_spin`) dressed through the spin Hxc kernel K_Hxc^{σσ'} (Hartree on total δρ + spin f_xc HVP), reusing `dielectric._k_hxc_spin` and the `_response` primitives (`cg_sternheimer`, `insulator_window`, `sternheimer_shift`). The density loop now keeps the per-spin first-order δρ so the dressing has both channels. Still `use_symmetry=False` + insulating occupations (the χ₀ solve is conduction-projected — same requirement as nspin=1). Oracle: `test_nspin2_dyson_nonmagnetic_limit_matches_nspin1` (nonmag insulator: spin-summed dressed δρ == the nspin=1 dressing to the shared fixed-point tol, ~1e-6 rel) |
| ~~postscf/discretization_error.py:312~~ | nspin2 / symmetry | symmetric disc. error nspin=1 only | **resolved, this PR** — the density/energy error already threaded nspin=2 per spin channel (`use_symmetry=False`); with the Dyson dressing now spin-resolved too, nspin=2 is complete for this module. The remaining raise (now :392) only guards nspin=2 **with** crystal symmetry (the magnetic/AFM IBZ fold), a documented boundary — not a gap. Oracle: `test_nspin2_nonmagnetic_limit_matches_nspin1` (+ the dyson and force-error nspin=2 nonmag variants) |
| ~~postscf/stress_error.py:90~~ | nspin2 | pressure (stress) disc.-error estimate nspin=1 only | **removed, this PR** — the frozen strained rebuild now builds a per-spin v_eff from the per-spin densities (`effective_potentials` on `[ρ↑,ρ↓]`, stacked into `res_s.v_eff`) and `estimate_density_error` sums both channels' energy error, exactly as the fixed-basis stress does. `use_symmetry=False` still required (frozen rebuild needs the full k-set, permanent). Oracle: `test_pressure_error_nspin2_nonmagnetic_limit_matches_nspin1` (nonmag == nspin=1 to ~1e-11 rel) |

## Open — nspin=2 (next tranches)

| file:line | axis | operation blocked |
|---|---|---|
| scf/implicit.py:53 | nspin2 | implicit (adjoint) SCF backward for nspin=2 |
| postscf/newton.py:60 | nspin2 | `newton_polish` raw-map plumbing for nspin=2 |
| postscf/dielectric.py:121 | nspin2 + symmetry | dielectric response with IBZ symmetry (nspin=2 magnetic-group vector fold) |
| postscf/hubbard_u.py:227 | nspin2 | Sternheimer linear-response U is implemented for nspin=2 only; nspin=1 raises (reverse gap) |

## Open — USPP/PAW response & error operators (lower-priority tranche)

| file:line | axis | operation blocked |
|---|---|---|
| postscf/uspp_position.py | insulator | position response: fixed occupations only (metals need occupation derivatives) |
| postscf/uspp_position.py | insulator | `hessian_column`: insulators only |
| postscf/uspp_implicit.py:166 | validation | USPP adjoint: nspin must be 1 or 2 |
| postscf/uspp_implicit.py:193 | insulator | USPP adjoint: non-prefix band occupations |
| postscf/discretization_error.py:278,1215 | USPP/PAW SOC-spinor | disc. error / eigenvalue error for USPP/PAW spinor result |
| postscf/discretization_error.py:283,290 | USPP/PAW | Dyson dressing on the USPP/spinor path |
| postscf/discretization_error.py:467,598,669 | symmetry | USPP density/eig/force error requires `use_symmetry=False` |
| postscf/discretization_error.py:465,673 | +U | USPP density/force error with DFT+U |
| postscf/stress_error.py:86 | symmetry | pressure error requires `use_symmetry=False` |
| postscf/uspp_bands.py:28 | +U | USPP bands with DFT+U (V_U missing from frozen band H) |

**Γ-point Hvp-phonons cross-validated (#141 step 2).** The Γ dynamical
matrix built on `hessian_column` (`postscf.phonons.gamma_hessian`) is now
cross-validated against the finite-displacement path
(`postscf.phonons_supercell` folding of `paw_forces.forces_uspp`) on one PAW
diamond-Si cell — `tests/integration/test_gamma_phonons_self_oracle.py`. The
gates above are unchanged (still nspin=1 / no +U / insulators). The two paths
agree to ~0.01 cm⁻¹ (~1e-5), matching the low-symmetry P1 column check and the
total-energy second difference (the exact BO-surface curvature).

*Degenerate-window systematic — FIXED (#141 follow-up).* As first landed the
analytic Hessian ran ~0.5 % (≈1.6 cm⁻¹ optical) high at the high-symmetry
(band-degenerate) ideal geometry vs BOTH the FD-of-forces Hessian and the
energy second difference (which agreed to 1e-4); the gap was h- and
ecut-independent and vanished at P1. Root cause: at a band degeneracy
`uspp_position.window_response` substituted the off-diagonal S-metric
coefficient with the −½⟨m|δS|n⟩ density limit. That limit is right for the
density/normalization response but leaves a spurious DISCONTINUITY in the
second-derivative assembly — as the degeneracy is lifted by any ε_n≠ε_m the
non-degenerate coefficient `⟨m|δH−ε_nδS|n⟩/(ε_n−ε_m)` gives the exact Hessian
(2e-5), so its continuous limit, the full metric coupling −⟨m|δS|n⟩, is the
correct exactly-degenerate value. `hessian_column` now sets that limit
(`PositionPerturbation.deg_full=True`); the density-response public functions
keep the −½ limit (their own continuous limit), so their gates are unchanged.

## Open — PAW/NC stress & +U

| file:line | axis | operation blocked |
|---|---|---|
| postscf/stress.py:146 | SOC + +U | DFT+U stress on the spin-orbit path (**feature boundary, #142**) |
| postscf/stress_error.py:92 | SOC | pressure error for fully-relativistic pseudos |
| postscf/stress_error.py:95 | +U | pressure error with DFT+U |
| postscf/hubbard_u.py:183 | +U | linear-response U for two Hubbard sites of different species/l |

## Open — NC-only / formalism routing

| file:line | axis | operation blocked |
|---|---|---|
| api.py:236 | NC-only | hybrid functionals need norm-conserving pseudos |
| api.py:872 | SOC | elastic constants for noncollinear/spin-orbit runs |
| api.py:970 | nspin2 / SOC | supercell phonons (forces path is NC, nspin=1) |
| api.py:975 | NC-only | supercell phonons need norm-conserving pseudos |
| calculator.py:575 | USPP/PAW nspin2 | nspin=2 through the ASE calculator is NC-only (USPP/PAW collinear spin not wired) |
| postscf/cohp.py:456 | NC-only | COHP `basis='iao'` needs the NC operator route |
| postscf/cohp.py:550 | NC-only | `projection_rmsp`: NC SCFResult only |
| postscf/pdos.py:312 | SOC/NC-spinor | projected DOS: NC + USPP only, noncollinear is a separate path |
| postscf/volumetric.py:266 | NC-only | ELF: collinear NC only (noncollinear/USPP later) |
| postscf/forces.py:42 | metaGGA / NC | meta-GGA NLCC force needs the batched-k geometry (collinear NC only) |
| opt/joint.py:278 | metaGGA | meta-GGA joint (strain+orbital) minimization (τ rebuild) |

## Open — noncollinear / SOC (mostly feature boundaries, #142)

| file:line | axis | operation blocked |
|---|---|---|
| scf/noncollinear.py:639 | SOC/NC-spinor + metaGGA | noncollinear meta-GGA band structure (τ operator) |
| scf/paw_noncollinear.py:49 | SOC/NC-spinor | noncollinear one-center XC is LDA-only (GGA rejected) |
| scf/uspp_noncollinear.py:193 | SOC/NC-spinor | noncollinear USPP/PAW is LDA-only |
| checkpoint.py:76 | SOC/NC-spinor | checkpointing a `scf_uspp_noncollinear` result (no restart consumer) |
| postscf/dielectric.py:115 | SOC | dielectric response scalar-relativistic only |
| postscf/discretization_error.py:841 | SOC/NC-spinor | force-error estimate NC-collinear only (spinor force terms unassembled) |
| postscf/discretization_error.py:847 | NLCC | NLCC force term in the error estimate (blocked on the g.s. NLCC force) |
| postscf/discretization_error.py:1010 | SOC/NC-spinor + symmetry | noncollinear disc. error requires `use_symmetry=False` |
| postscf/cohp.py:623 | validation | `cohp_noncollinear` expects a noncollinear NCResult |
| postscf/cohp.py:662,665 | validation | `cohp_soc` expects a fully-relativistic (SOC) NCResult/pseudo |
| postscf/pdos.py:387 | validation | `projected_dos_noncollinear` expects a noncollinear NCResult |
| postscf/pdos.py:543,547 | validation | `projected_dos_soc` expects a fully-relativistic NCResult/pseudo |

## Open — symmetry / occupations / data / validation

| file:line | axis | operation blocked |
|---|---|---|
| scf/implicit.py:55 | symmetry | implicit SCF backward requires `use_symmetry=False` |
| scf/implicit.py:66 | insulator | implicit SCF backward supports insulators only (occ = 2) |
| postscf/_response.py:168 | insulator | response builder rejects metallic (partial) occupations |
| postscf/discretization_error.py:396 | symmetry | Dyson dressing requires `use_symmetry=False` (permanent — response solve needs the full k-mesh, both nspin) |
| postscf/newton.py:64 | +U | `newton_polish` +U raw-map plumbing |
| symmetry.py:180,297 | symmetry | shifted meshes not reduced here (caller reduces unshifted) |
| postscf/dispersion.py:134 | data | D3(BJ) reference C6 not vendored for requested element(s) |
| postscf/dispersion_d4.py:171 | data | D4(BJ) reference data not vendored for requested element(s) |
| postscf/dielectric.py:113 | validation | dielectric response: nspin must be 1 or 2 |

## Not gates — abstract-method stubs (interface contracts)

These `raise NotImplementedError` are abstract-method placeholders on base
classes, overridden by every concrete subclass. They are not capability gaps and
should not be counted or "removed".

| file:line | base class / method |
|---|---|
| core/occupations.py:44 | `Smearing.occupation` |
| core/occupations.py:48 | `Smearing.entropy` |
| core/xc/base.py:169 | `XCFunctional.energy_density` |
| core/xc/spin.py:63 | `SpinXC.energy_density` |
