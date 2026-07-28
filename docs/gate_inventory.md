# Capability-gate inventory

Tracking doc for issue #147 (umbrella). gradwave gates unsupported
formalism/feature combinations with a loud `raise NotImplementedError` rather
than returning a silently-wrong number. That is the right default, but the
density of gates means the feature matrix has holes off the happy path. This
table enumerates every gate so the holes are visible and progress is measurable.

**Metric of success is the number and user-impact of gates removed, not zero
gates.** Some gates are permanent feature boundaries, not gaps — they stay,
documented as such. Fully-relativistic DFT+U stress (#142) was originally
filed as one such boundary but turned out not to be: see the Done row below —
+U and the SOC nonlocal term are orthogonal, so the collinear +U stress term
generalizes directly. Treat "#142" tags elsewhere in this doc as a hint to
re-audit, not a guarantee of permanence.

An ungating = remove the `raise`, thread the missing (nspin / formalism) index
through the real computation, and land a Tier-0/Tier-1 self-oracle test **in the
same commit** (the recipe from PR #45 nspin=2 post-SCF and PR #58 nspin=2 stress).

Regenerate the raw list with:

```bash
grep -rn "raise NotImplementedError" src/gradwave --include=*.py
```

Current count: **66** `raise NotImplementedError` sites, of which **4 are
abstract-method stubs** (interface contracts on base classes, not capability
gaps) — see the last section. So **62 real capability gates** remain.

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
| ~~inputs.py~~ (`hubbard.enabled and noncollinear` `InputError`) | SOC/NC-spinor + U | DFT+U on the noncollinear/spin-orbit spinor path | **removed, this PR** — not in the grep'd `NotImplementedError` list above (it was an `inputs.py`-level `InputError` combination guard, not a `NotImplementedError`, so the regen script would not have caught it). Generalized the occupation matrix to a 2×2 spin block per orbital pair, `N^{Iσσ'}_{mm'}` (`core.hubbard.occupation_matrices_noncollinear`/`hubbard_dmatrix_noncollinear`); `hubbard_energy`'s Dudarev trace is unmodified — it operates on the bigger composite matrix as-is and reduces exactly to the collinear sum in the z-polarized (no-canting) limit. Wired into `scf.noncollinear.scf_noncollinear`'s `SpinorHamiltonian` (the +U term is orthogonal to the SOC nonlocal term, so fully-relativistic pseudos get +U through the same apply — no separate SOC gate needed). SCF/energy path only; noncollinear +U forces/stress remain a follow-up (see the new Open row below). The noncollinear **USPP/PAW** SCF (`scf.uspp_noncollinear.scf_uspp_noncollinear`) does NOT get +U in this PR — it explicitly raises `NotImplementedError` on a `hubbard=` argument (see the Open table). Oracle: `test_occupation_matrix_noncollinear_reduces_to_collinear_limit` (exact algebraic reduction, 1e-12) + U=0 bit-for-bit SCF oracles (both a nonmagnetic diamond-C run through the input surface and a real ferromagnetic-Ni SCF through the driver directly) |
| ~~postscf/volumetric.py:266~~ | NC-only (spinor) | ELF for a noncollinear/SOC spinor result | **removed, this PR** — the ELF is the closed-shell form of the CHARGE density (`res.rho`, the trace of the spin-density matrix) and the total kinetic-energy density τ_0 = τ_↑↑ + τ_↓↓ (the trace of the 2×2 KE-density matrix `core.metagga.spinor_tau_matrix_b` already assembles). τ_0 ≥ \|∇ρ\|²/(8ρ) (von Weizsäcker bound on the total density), so D ≥ 0 and ELF ∈ [0, 1] just as collinear; no spin split, so no 2^{2/3} TF factor. Works with or without SOC. USPP/PAW still raises (no `batch`; the soft KE density needs the augmentation). Oracle: `test_elf_noncollinear_nonmagnetic_limit_matches_collinear` (nonmagnetic scalar-relativistic NC ELF == nspin=1 collinear ELF, ~2e-3) + the boundedness check folded into `test_noncollinear_spinor_export` (SOC GaAs, ELF ∈ [0, 1]) |
| ~~postscf/hubbard_u.py:227~~ | nspin2 | Sternheimer linear-response U was nspin=2 only (reverse gap — the simpler nspin=1 case was missing) | **removed, this PR** — `_response_columns` now loops `range(nspin)` with the spin degeneracy g=2/nspin folded into the occupation response (dN_I ×g) and the total Δρ that drives the Hxc kernel; nspin=1 solves the single spin-restricted Sternheimer channel (u↑=u↓ for the spin-symmetric probe) and screens with the non-spin `fxc_hvp` at ρ+ρ_core (matching the nspin=1 dielectric kernel), while nspin=2 keeps the collinear `_k_hxc_spin`. Oracle: `test_diamond_c_linear_response_u_nspin1_matches_nspin2` (nspin=1 PBE == nspin=2 SpinPBE nonmagnetic limit: U to <1e-4, χ/χ0 to <1e-6, using the working nspin=2 path as ground truth) |
| ~~postscf/hubbard_u.py:183~~ | +U | linear-response U for two Hubbard sites of different species or l | **removed, this PR** — the drivers now detect inequivalent sites (`_use_full_matrix`/`_all_sites_equivalent`, the existing species/l check) and build the full response matrix χ_IJ by perturbing each site independently (one Sternheimer solve / FD probe per site), inverting (χ0⁻¹ − χ⁻¹) as a general matrix (`_assemble_u_matrix`, Cococcioni–de Gironcoli general case); the cheap symmetric [[a,b],[b,a]] single-column shortcut is kept as a fast path for a lone site or two equivalent sites. The single-column `_assemble_u` keeps its guard (a single column can't reconstruct an asymmetric χ_IJ) as an internal precondition. Oracle: `test_diamond_c_linear_response_u_full_matrix_matches_shortcut` (full χ_IJ path == shortcut on the two equivalent C sites, U to <1e-4) + pure-logic `test_assemble_u_matrix_reduces_to_equivalent_shortcut` / `test_assemble_u_matrix_inequivalent_asymmetric` |
| ~~postscf/discretization_error.py:847~~ | NLCC | NLCC force term in the error estimate — blocked (stale) on the g.s. NLCC force in postscf.forces | **removed (nspin=1), this PR** — the blocker was stale: postscf.forces has carried the ground-state NLCC force term since PR #64 (`_core_correction_energy`, gated on `has_core`). `estimate_force_error` now rebuilds E_xc[ρ_val(ε)+ρ_core(τ)] on the SAME (ε, τ) leaves the local/nonlocal channels already differentiate (`scf.setup_common.assemble_core_density`, mirroring `_core_correction_energy`'s core-density build exactly, including the shared species/\|G\|-shell tables), so its mixed second derivative ∂²E_xc/∂ε∂τ — the XC-kernel cross term a fixed core misses (Hartree/XC otherwise have zero EXPLICIT τ-dependence at fixed ρ, which is why the no-core path never needed this term) — now falls out of the same eps-then-pos double-backward the estimator already ran. nspin=2 + NLCC stays gated (only the spin-summed `drho_first_order` is available, not the per-spin split the spin-resolved XC kernel needs — a new, narrower raise). Oracle: `test_nc_nlcc_force_error_vs_high_cutoff` (δF correlates with (0.97) and reduces the true low→high-cutoff force change on the low-symmetry NLCC carbon cell of `test_forces_nlcc.py`, mirroring the established `test_uspp_force_error_vs_high_cutoff` pattern — Gamma-only and a 35→70 Ry span, since carbon's hard ONCV core needs a wider cutoff span than the non-NLCC Si estimator elsewhere in this file to sit in the annulus correction's linear regime) + `test_nc_nlcc_force_error_requires_xc` / `test_nc_nlcc_force_error_nspin2_not_implemented` guard tests |
| ~~postscf/stress.py:146~~ | SOC + +U | DFT+U stress on the spin-orbit path — filed as a permanent feature boundary (#142), turned out stale | **removed, this PR** — +U and the SOC nonlocal term are orthogonal (the same statement PR #159 made for the SCF/energy path), so the collinear +U stress term generalizes directly onto `_energy_strained_fr`. `hubbard_energy_strained_nc` (postscf/_strain.py) builds the 2×2 spin-block composite occupation matrix N^I_{(σm),(σ'm')} (`core.hubbard.occupation_matrices_noncollinear`/`hubbard_dmatrix_noncollinear`, PR #159) from the SAME strained atomic-orbital projectors the collinear path already builds (`_hubbard_strain_setup`/`_hubbard_strain_q`, factored out of `hubbard_energy_strained` so the two paths cannot drift), contracted against BOTH spinor components; `hubbard_energy` (UNCHANGED) sums Tr[N(1−N)] over the bigger composite matrix, reducing exactly to the collinear per-spin sum in the z-polarized limit and to an exact zero at U=0 (D = (U−J)(½−N) vanishes identically). Oracle: `test_stress_soc_hubbard_autograd_vs_fd` (ε=0 strained expression reproduces the NC-SCF +U total to ~5e-8 eV; analytic stress == central FD of that energy to ~1e-10 eV/Å³ on the tested components) + `test_stress_soc_hubbard_u0_matches_plain_soc_stress` (U=0 == the pre-existing plain SOC stress of the SAME converged state, <1e-10, plus the no-manifolds ValueError guard) |
| ~~scf/noncollinear.py:639~~ | SOC/NC-spinor + metaGGA | noncollinear meta-GGA band structure (τ operator) — blocked (stale) on `NCResult` not carrying occupations | **removed, this PR** — the blocker was already half-resolved: `NCResult` has carried `occupations` since PR #103 (added for the SOC stress), so "NCResult carries coeffs but not occupations" no longer held. `band_structure_nc` now rebuilds the converged KE-density matrix (τ_0, τ⃗) from `res.coeffs`/`res.occupations` at the SCF's own k-mesh (`system.batch`, `core.metagga.spinor_tau_matrix_b`), projects to the local-frame per-spin τ_± (`core.xc.noncollinear.local_frame_tau`), evaluates v_xc at the converged τ (mirroring the SCF's `_nc_effective_potential`), and forms the fixed v_τ operator fields (v_τ0, v_τ⃗) (`vtau_up_dn` + `tau_operator_fields`/the nonmagnetic branch, mirroring `_nc_metagga_step`) — then applies `spinor_metagga_tau_operator` per k-path chunk through `SpinorHamiltonian`'s existing `metagga_op` slot (already used by the SCF, so no new Hamiltonian machinery). Oracle: `test_bands_nc_metagga_reproduces_scf_spectrum_on_mesh` (r2SCAN on a nonmagnetic fully-relativistic Si run; band structure recomputed at the SCF's own k-mesh reproduces the SCF eigenvalues to 1e-3 eV, mirroring `test_bands_nc.py`'s SOC reproduction pattern — an FR pseudo is used because `band_structure_nc`'s projector rebuild is unconditionally j-resolved, independent of this gate) |
| ~~scf/implicit.py:53~~ | nspin2 | implicit (adjoint) SCF backward for nspin=2 | **removed, this PR** — the norm-conserving analogue of the USPP twin (`uspp_implicit`, PR #58): the response vector doubles to the per-channel pair (δρ↑, δρ↓). χ₀ is block-diagonal over spin — each channel gets its own occupied bands, `v_eff^σ` and conduction-projected Sternheimer solves (`_hamiltonians`/`_occupied`/`_chi0_channel` now spin-indexed, degeneracy weight g=1 not 2) — while K_Hxc keeps its cross-spin blocks (Hartree on the total δρ + `fxc_hvp_spin`, NLCC core split half/half). The loss stays a functional of ρ_tot, so v̄=∂L/∂ρ seeds both channels equally. `solve_adjoint` switched from plain damping to `AndersonMixer` (the same NiO/spin-instability lesson the USPP/dielectric/Hubbard adjoints hit; nonmagnetic insulators still land the identical fixed point). Still insulators + `use_symmetry=False` (those gates below unchanged). Oracle: `test_nspin2_nonmagnetic_limit_matches_spin_restricted` (nspin=2 gradient == the FD-validated nspin=1 gradient, ~1e-4) + `test_nspin2_magnetic_gradient_vs_scf_finite_differences` (genuinely spin-split M=2 Si, analytic == central FD in the smooth window, few-% given the small-gap χ₀ conditioning) |
| ~~postscf/newton.py:60~~ | nspin2 | `newton_polish` raw-map plumbing for nspin=2 | **removed, this PR** — the finisher's packed residual/state vector doubles to the per-spin (δρ↑, δρ↓, δbec↑, δbec↓); the exact-Jacobian inner solve reuses the spin-resolved `_ConvergedUSPP.{apply_chi0, k_hxc_grid, hvp_onecenter}` (already nspin=2) and drives `_scf_iteration` with `nspin=res.nspin`. Independent of `scf/implicit.py` (newton consumes the USPP raw map, not the NC one); nspin=2 follows USPP's own smeared-occupation coverage (a shared Fermi level, magnetic insulators via a small width). +U raw-map still gated (below). Oracle: `test_newton_polish_nspin2_nonmagnetic_limit` (nonmag Al metal polishes to the floor, F == deep-converged ref; the genuinely-magnetic per-spin operators the finisher calls are the same `_ConvergedUSPP` ones validated magnetically by `test_uspp_implicit` — a dedicated magnetic newton oracle is impractical orthogonally to spin: magnetic metals floor above the finisher tol, molecular magnetic insulators stall the inner solve's vacuum 4π/G² mode) |

## Open — nspin=2 (next tranches)

| file:line | axis | operation blocked |
|---|---|---|
| postscf/dielectric.py:121 | nspin2 + symmetry | dielectric response with IBZ symmetry (nspin=2 magnetic-group vector fold) |

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
| postscf/volumetric.py:266 | USPP/PAW | ELF for USPP/PAW (no `batch`; the soft KE density needs the augmentation) — the NC-spinor part of this site was ungated this PR, USPP remains |

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
| postscf/stress_error.py:92 | SOC | pressure error for fully-relativistic pseudos |
| postscf/stress_error.py:95 | +U | pressure error with DFT+U |

## Open — NC-only / formalism routing

| file:line | axis | operation blocked |
|---|---|---|
| api.py:236 | NC-only | hybrid functionals need norm-conserving pseudos |
| api.py:872 | SOC | elastic constants for noncollinear/spin-orbit runs |
| api.py:970 | nspin2 / SOC | supercell phonons (forces path is NC, nspin=1) |
| api.py:975 | NC-only | supercell phonons need norm-conserving pseudos |
| calculator.py:575 | USPP/PAW nspin2 | nspin=2 through the ASE calculator is NC-only (USPP/PAW collinear spin not wired) |
| postscf/cohp.py:456 | NC-only | COHP `basis='iao'` needs the NC operator route (USPP/spinor absent) |
| postscf/cohp.py:550 | NC-only | `projection_rmsp`: NC SCFResult only (USPP/spinor absent) |
| postscf/forces.py:42 | metaGGA / NC | meta-GGA NLCC force needs the batched-k geometry (collinear NC only) |
| opt/joint.py:278 | metaGGA | meta-GGA joint (strain+orbital) minimization (τ rebuild) |

**COHP `basis='iao'` / `projection_rmsp` (cohp.py:456, :550) — genuine gaps,
deferred (audited this PR).** Both are real: `basis='iao'` and `projection_rmsp`
work on the norm-conserving collinear `SCFResult` (tested: `test_cohp_resolve_images_and_iao_o2`)
and raise for USPP/PAW and spinor results. Unlike the ELF wire-through, neither is
a mechanical extension — the IAO construction and the RMSp spilling both need the
S-metric (`⟨φ|S|φ⟩`, `⟨φ|S|ψ⟩`) for USPP/PAW and, for the operator route,
`_htilde_operator` has no USPP/spinor Hamiltonian analogue yet (the spinor COHP
uses the band-limited eigenvalue route, `cohp.py` docstring). `projection_rmsp` is
additionally a *differentiable* objective (no `torch.no_grad`), so its S-dressing
must stay on the autograd path. Left as a focused follow-up rather than forced into
this slice.

## Open — noncollinear / SOC (mostly feature boundaries, #142)

| file:line | axis | operation blocked |
|---|---|---|
| scf/paw_noncollinear.py:49 | SOC/NC-spinor | noncollinear one-center XC is LDA-only (GGA rejected) |
| scf/uspp_noncollinear.py:193 | SOC/NC-spinor | noncollinear USPP/PAW is LDA-only |
| scf/uspp_noncollinear.py:200 | SOC/NC-spinor + U | DFT+U on the noncollinear USPP/PAW path (the norm-conserving spinor path has it — see Done above) |
| postscf/forces.py | SOC/NC-spinor + U | +U forces on the (now +U-capable) noncollinear spinor SCF — SCF/energy path and now the fully-relativistic STRESS path have it (see Done above); forces are the remaining piece |
| checkpoint.py:76 | SOC/NC-spinor | checkpointing a `scf_uspp_noncollinear` result (no restart consumer) |
| postscf/dielectric.py:115 | SOC | dielectric response scalar-relativistic only |
| postscf/discretization_error.py:841 | SOC/NC-spinor | force-error estimate NC-collinear only (spinor force terms unassembled) |
| postscf/discretization_error.py:1010 | SOC/NC-spinor + symmetry | noncollinear disc. error requires `use_symmetry=False` |

**Noncollinear / SOC COHP and PDOS — NOT gaps, confirmed validation/routing
(audited this PR).** The spinor post-SCF analysis for COHP and PDOS is fully
implemented and tested; the `raise NotImplementedError` sites below are argument
guards or entry-point routing, not capability gaps:

| file:line | classification | what it actually is |
|---|---|---|
| postscf/cohp.py:623 | validation | `cohp_noncollinear` rejects a non-`NCResult`; the noncollinear (charge, spin-summed) COHP is fully implemented (`test_cohp_soc_bi2`) |
| postscf/cohp.py:662,665 | validation | `cohp_soc` rejects a non-`NCResult` / non-FR pseudo; the j-resolved SOC COHP is fully implemented (`test_cohp_soc_bi2`) |
| postscf/pdos.py:312 | routing | inside the COLLINEAR unpacker `_unpack_result`; a spinor result is served by the dedicated `projected_dos_noncollinear` / `projected_dos_soc` (both fully implemented and tested), so this raise only routes a spinor result away from the collinear entry point |
| postscf/pdos.py:387 | validation | `projected_dos_noncollinear` rejects a non-`NCResult`; the charge + spin-texture (m_x/m_y/m_z) projected DOS is fully implemented (`test_pdos_noncollinear_spin_texture`) |
| postscf/pdos.py:543,547 | validation | `projected_dos_soc` rejects a non-`NCResult` / non-FR pseudo; the j-resolved projected DOS is fully implemented (`test_pdos_soc_j_resolved`) |

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
