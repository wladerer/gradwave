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
Current count: **64** `raise NotImplementedError` sites, of which **4 are
abstract-method stubs** (interface contracts on base classes, not capability
gaps) — see the last section. So **62 real capability gates** remain. (The
noncollinear/SOC forces PR added a brand-new code path, `postscf/forces.py`,
previously absent entirely, gaining one narrower NLCC gate of its own; this PR
then ungated the fully-relativistic dielectric/Born-charge path but added its
own narrower magnetic-SOC and SOC+symmetry gates in its place — see Done/Open
below for both.)
gaps) — see the last section. So **60 real capability gates** remain. (This PR
removed two: `checkpoint.py`'s `scf_uspp_noncollinear` restart gate and
`postscf/uspp_bands.py`'s USPP+U bands gate; see Done below. The pressure-error
SOC/+U axes in `postscf/stress_error.py` were investigated but NOT closed this
PR — see the note under Open — PAW/NC stress & +U.)

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
| ~~postscf/forces.py~~ (no `is_fr`/noncollinear branch existed at all) | SOC/NC-spinor | Hellmann–Feynman forces on the noncollinear/SOC spinor SCF path — previously entirely missing (unlike `postscf.stress`/`dielectric`/`mae`/`pdos`/`cohp`/`stress_error`, which already had an `is_fr` branch) | **added, this PR** — `forces()` now dispatches a spinor `NCResult` (`res.formalism == "noncollinear"`) to a new `_forces_noncollinear`, generalizing the collinear path's three position-dependent terms (Ewald / E_loc structure factor / E_NL projector phase — kinetic/Hartree/XC carry no τ-dependence at fixed, detached ψ/ρ/m⃗) onto the spinor Hamiltonian, the way `postscf.stress`'s `is_fr` branch (`_energy_strained_fr`) already generalized stress. The nonlocal term splits exactly like `scf.noncollinear`'s `SpinorHamiltonian`: fully-relativistic (`is_fr`) pseudos use the j-resolved spinor projectors, now position-differentiable (`core.spinor_proj.build_so_projectors` gained an optional `positions=` override — phase-only, backward-compatible with its two existing callers); scalar-relativistic pseudos run through the noncollinear driver use plain KB projectors on the up/down spinor halves separately (`core.batch.projectors_b`/`becp_b`), mirroring `scf.spinor_common.spinor_scalar_nonlocal_energy`. NLCC on this path is a new, narrower, explicitly-gated follow-up (`_forces_noncollinear` raises `NotImplementedError` when `system.rho_core is not None` — the (ρ, m⃗) generalization of `_core_correction_energy` was out of scope here; see the new Open row below). Oracle: `test_forces_noncollinear_reduces_to_collinear_limit` (scalar-relativistic Si run through `scf_noncollinear` with m⃗ pinned to 0 == the plain collinear nspin=2 force on the same displaced geometry, ~5e-10) + `test_forces_soc_matches_finite_difference` (analytic SOC-GaAs spinor force == a central finite difference of the noncollinear free energy, ~3.5e-4 on a ~1.7 eV/Å force — genuinely exercises the j-resolved spinor nonlocal projector-phase term) |
| ~~postscf/forces.py~~ (no noncollinear +U force existed — the "forces are the remaining piece" row PR #165 left just above, after adding +U SOC stress) | SOC/NC-spinor + U | +U (Dudarev) Hellmann–Feynman force on the noncollinear/SOC spinor path | **added, this PR** — `hubbard_force_noncollinear`, the spinor generalization of the collinear `hubbard_force`: the same atomic-orbital projector phases (`core.hubbard.hubbard_projectors`) now feed the 2×2 spin-block occupation matrix (`occupation_matrices_noncollinear`, PR #159) the noncollinear +U SCF/energy path already self-consistifies, so the frozen-orbital autograd gradient of that same E_U expression is the force. A finite-displacement cross-check of the ISOLATED E_U term against independently re-converged SCFs is **not** a valid oracle here — the Hellmann–Feynman theorem only makes the TOTAL energy stationary in the orbitals at self-consistency, not E_U alone, so FD-of-E_U-alone picks up an orbital-relaxation contribution the frozen-orbital force correctly excludes (confirmed numerically while building the test: FD of E_U alone from re-converged SCFs disagreed with the analytic force by an O(1) factor — a real physics distinction, not a bug; the collinear `hubbard_force` has no full-SCF-FD test in this repo either, for the same reason). The oracle is instead a `torch.autograd.gradcheck` of the exact closed-form E_U(pos) expression at frozen orbitals (`tests/unit/test_hubbard.py::test_hubbard_force_noncollinear_gradcheck`, mirroring the existing collinear oracle `test_hubbard_force_gradcheck`) + an exact U=0 zero (`test_hubbard_force_noncollinear_u_zero_is_exact_zero` — the Dudarev trace is identically zero at U=J=0 for any occupation matrix, so the noncollinear +U force is bit-exact zero and the total force reduces to the plain spinor force above) + a real (non-synthetic) converged +U spinor SCF end-to-end sanity check (`test_hubbard_force_noncollinear_real_scf_sanity`: two-atom fully-relativistic Bi with U=3 eV on the 5d manifold — U=0 gives an exact-zero force, U>0 gives finite forces satisfying the net-force sum rule; every Hubbard-eligible (PP_PSWFC-carrying) fully-relativistic pseudo in the fixture set also carries an NLCC core charge, so this test exercises `hubbard_force_noncollinear` alone, not the still-NLCC-gated `forces()`). Between this PR and #165 (+U SOC stress, landed just before this one rebased), NC/SOC DFT+U now has both forces and stress. |
| ~~api.py:872~~ | SOC / NC | elastic constants (`run_elastic`) for noncollinear/spin-orbit runs | **removed, this PR** — a stale driver gate (the #152 pattern): `postscf.stress.stress` already differentiates the spinor strained energy (`_energy_strained_fr`, ungated in #103), the driver just never routed the SOC result to it. `run_elastic` now builds `NoncollinearXC`, sets the full-BZ / time-reversal per the magnetism rule, and folds C over the six strains. USPP/PAW spinor stress and DFT+U on the SOC path (#142) stay rejected. Oracle: `test_run_elastic_soc_matches_scalar` (fully-relativistic Si C == scalar-relativistic Si C to <3 GPa — the zero-SOC Si-valence limit) |
| ~~calculator.py:575~~ | USPP/PAW nspin2 | nspin=2 through the ASE calculator on the USPP/PAW route | **removed, this PR** — a stale driver gate: `scf_uspp`, `paw_forces.forces_uspp`, and `paw_stress.stress_uspp` all thread nspin=2 (benchmarked in #150). `_calculate_uspp` now resolves the spin channel via `_resolve_spin` (exactly as the NC path does), passes nspin/start_mag through, uses the spin-matched `_make_xc(nspin)`, and reports `magmom`. Oracle: `test_calculator_uspp_nspin2_matches_direct_scf` (FM O2 PAW: calculator == direct `scf_uspp`, energy + moment bit-for-bit) |
| ~~api.py:970~~ (nspin=2) | nspin2 | supercell phonons (`run_phonons`) for collinear nspin=2 | **removed, this PR** — a stale driver gate: the FD force fold calls `postscf.forces.forces`, which already sums per spin channel (#45); the driver hardcoded the nspin=1 SCF. `run_phonons` now threads nspin/start_mag/tot_magnetization into the per-displacement SCF and the NLCC-aware `xc` into `force_constants_home`. Noncollinear/spinor supercell phonons stay rejected; the USPP/PAW fold (api.py:975) is still open. Oracle: `test_run_phonons_nspin2_matches_nspin1` (nonmagnetic Si: nspin=2 frequencies == nspin=1, <1 cm⁻¹) |
| ~~postscf/dielectric.py:115~~ | SOC/NC-spinor | dielectric response / Born effective charges were scalar-relativistic only | **removed, this PR** — `_dielectric_born_soc` covers the fully-relativistic (spin-orbit, `system.is_fr`) spinor path in the NONMAGNETIC manifold (m⃗ ≡ 0). Structurally it is ONE channel of the existing `_dielectric_born_spin` (spinor bands hold f=1 each, and a single Sternheimer loop already sums the full electron count, so the same f=1 prefactors — 8π not 16π on ε, factor 2 not 4 on the density/Born nonlocal terms — apply directly with no extra channel sum). Two genuinely new pieces: (1) ∂H/∂k on a spinor (`_dhdk_psi_soc`) — the kinetic term acts on each doubled-axis component independently, and the nonlocal FD-in-k rebuilds the j-resolved SOC projectors at k±dk via `_so_projectors_at`, which reuses `core.spinor_proj.strained_so_projector_cols` (the strain-graph primitive `postscf.stress` already has) as a plain per-k/position evaluator — for both the FD-in-k RHS (dkvec≠0, pos fixed) and the position-differentiable Born-charge nonlocal term (dkvec=0, pos requiring grad), avoiding any new projector-assembly code; (2) the screening kernel `_k_hxc_soc`, built from a new shared primitive `postscf._response.fxc_hvp_noncollinear_nonmagnetic` — the noncollinear (locally-collinear) XC's f_xc Hessian-vector product pinned at m⃗ ≡ 0, which reduces exactly to the spin-restricted `fxc_hvp` there. `cg_sternheimer` needed NO changes — it only reads `bk.t`, so a `SimpleNamespace(t=torch.cat([bk.t, bk.t]))` shim is the only "generalization" required, confirming the task's premise that the Sternheimer machinery is already reusable. Magnetic SOC (m⃗ ≠ 0, needing the full coupled (ρ, m⃗) K_Hxc HVP) and IBZ symmetry (the magnetic-group polar-vector fold) remain gated with their own explicit raises inside `_dielectric_born_soc`. Oracle: `test_dielectric_soc_zero_splitting_matches_scalar` — a SYNTHETIC fully-relativistic pseudopotential built by duplicating each l≥1 scalar-relativistic beta into degenerate j=l±1/2 channels with identical radial data/D (a standard spin-angular completeness identity then makes the FR nonlocal operator EXACTLY the doubled scalar operator, zero SOC splitting by construction, verified independently to ~1e-13 on the SCF total energy) reproduces the scalar-relativistic ε∞/Z* to ~1e-9 (solver precision, not merely "small SOC") + `test_dielectric_soc_rejects_magnetic` |
| ~~checkpoint.py:76~~ | SOC/NC-spinor | checkpointing a `scf_uspp_noncollinear` result (no restart consumer) | **removed, this PR** — `save_checkpoint`/`as_start_from` gained a `uspp_noncollinear` branch archiving both halves of the spinor USPP/PAW state at once: the magnetization field m⃗ (the same field the `noncollinear` branch already archives) AND the 4-channel (n, mx, my, mz) one-center becsum (`rho_ij_chan`, the spinor analogue of `uspp`'s `rho_ij_atoms`). Unlike the norm-conserving spinor path (whose only warm-start hook is the atomic `mag_vec_init` seed via `nc_mag_seed`, since it has no density-level restart at all), `scf_uspp_noncollinear` gained a genuine `start_from=` parameter: m⃗ is already a full-grid field internal to the loop (unlike the atomic seed the collinear spinor path takes), so it restarts directly — rescaled by the cell-volume ratio, mirroring ρ — and the becsum restarts the same way `uspp_loop._seed_becsum`'s warm-start branch does. `USPPNCResult` also gained a `system` field (it carried none before, unlike every other result dataclass, which is why the checkpoint's grid/geometry metadata had nowhere to come from). Oracle: `test_uspp_noncollinear_checkpoint_roundtrip_and_restart` (bit-for-bit round trip of ρ/m⃗/becsum through save+load; a restart from the checkpoint reproduces the same converged free energy in far fewer SCF iterations) |
| ~~postscf/uspp_bands.py:28~~ | +U | USPP bands with DFT+U (V_U missing from frozen band H) | **removed, this PR** — `scf.uspp._HkS` (the per-k Hamiltonian `bands_uspp` diagonalizes) already applies a Hubbard term V_U = Σ\|Sφ_m⟩D_mm'⟨Sφ_m'\| whenever handed `hub_sphi`/`hub_d` — the exact pair the SCF loop's own `_solve_bands_uspp` already builds once per SCF k. `bands_uspp` now builds that same pair at each REQUESTED band k (which need not lie on the SCF mesh) from the converged `hub_occ`/`hub_sites` the result carries, reusing `scf.uspp_hubbard.build_uspp_hubbard`'s per-k body — factored out into `phi_free_at_sphere` (the position-independent radial/Y_lm setup computed once, phased and S-dressed per k) — instead of re-deriving the S-dressed atomic-orbital projector from scratch. Oracle: `test_uspp_bands_hubbard_reproduces_scf_spectrum` (bands at the SCF's own Γ point == the SCF eigenvalues, <5e-3 eV) + `test_uspp_bands_hubbard_u0_is_inert` (a U=0 manifold on the SAME converged state runs the whole S-dressed Hubbard band path but D_U vanishes identically, so the bands are bit-for-bit the pre-existing no-+U bands) |

## Open — nspin=2 (next tranches)

| file:line | axis | operation blocked |
|---|---|---|
| postscf/dielectric.py:169 | nspin2 + symmetry | dielectric response with IBZ symmetry (nspin=2 magnetic-group vector fold) |
| postscf/dielectric.py:627 | SOC/NC-spinor + symmetry | dielectric response with IBZ symmetry (spin-orbit magnetic-group polar-vector fold) |

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

**Pressure-error SOC/+U axes — investigated, NOT closed this PR.**
`estimate_pressure_error`'s `_denergy_at` measures `err.denergy`, the
Cances-Dusson-Maday-Stamm-Vohralik ANNULUS-COMPLEMENT correction
`estimate_density_error` already computes at each probe cutoff — a
second-order energy-lowering estimate from the missing high-G plane waves,
not the SCF total energy itself. Naively adding the REAL analytic +U/SOC
strain energy (`postscf._strain.hubbard_energy_strained`/`_nc`,
`postscf.stress._energy_strained_fr`, all landed earlier in this same
session) into that number would conflate an exact physical energy with a
basis-incompleteness error estimate; it would pass the required "U=0 / no-SOC
reduces to the existing result" reduction test trivially (both terms vanish
at U=0 by construction) without actually representing a discretization
*error*, which risks a silently-plausible-looking but physically meaningless
number in a scientific-computing codebase. The two axes need genuinely
different, larger pieces of new machinery:
  - **SOC**: `estimate_pressure_error` only ever dispatches through the plain
    collinear `effective_potentials`/`scf.loop.setup_system` path — it has no
    "noncollinear" formalism branch at all (unlike `discretization_error.
    estimate_density_error`, which already dispatches a `formalism ==
    "noncollinear"` result to `_estimate_density_error_noncollinear`). Closing
    this axis means building a parallel frozen-rebuild `_denergy_at` variant
    that reconstructs the strained spinor system, rebuilds the 2×2 (v, B⃗)
    potential from the frozen (ρ, m⃗) the way `_spinor_complement` already
    does, and calls the existing noncollinear density-error dispatch — real
    work, not a guard removal.
  - **+U**: `discretization_error._enlarged_hamiltonian` (the complement
    Hamiltonian `estimate_density_error`'s plain path builds) has ZERO
    Hubbard awareness — no mention of `hub_occ` anywhere in that module. A
    physically-correct +U pressure error needs the Hubbard nonlocal term
    threaded into THAT Hamiltonian (its own atomic-orbital projector set, built
    on the enlarged complement sphere — the analogue of this PR's
    `postscf/uspp_bands.py` fix, but for a shared helper several other
    formalisms' estimators also call, raising the regression surface well
    beyond `stress_error.py` alone).
  Both are real, tractable follow-ups, just bigger than "thread an existing
  building block" — flagged here rather than forced into this PR.

## Open — NC-only / formalism routing

| file:line | axis | operation blocked |
|---|---|---|
| api.py:236 | NC-only | hybrid functionals need norm-conserving pseudos |
| api.py (phonons) | SOC | supercell phonons for noncollinear/spinor runs (collinear nspin=2 now supported) |
| api.py:975 | NC-only | supercell phonons need norm-conserving pseudos (the FD fold calls the NC `postscf.forces` path; the USPP/PAW `paw_forces` route is not wired into `force_constants_home` yet) |
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
| postscf/forces.py:229 (`_forces_noncollinear`) | NLCC | NLCC core-correction force on the noncollinear/SOC path (new gate — forces themselves are now supported, see Done above, and PR #165 already closed the analogous +U SOC stress gate; no NLCC-free Hubbard-eligible fully-relativistic pseudo exists in the fixture set either, which is why the Stage-2 +U force test also stays clear of this) |
| checkpoint.py:76 | SOC/NC-spinor | checkpointing a `scf_uspp_noncollinear` result (no restart consumer) |
| postscf/dielectric.py:631 | SOC/NC-spinor | magnetic (m⃗ ≠ 0) fully-relativistic dielectric response — needs the coupled (ρ, m⃗) K_Hxc Hessian-vector product; the nonmagnetic case was ungated this PR (see Done above) |
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
| postscf/dielectric.py:163 | validation | dielectric response: nspin must be 1 or 2 |

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
