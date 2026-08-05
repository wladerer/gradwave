# Output schema

Every run writes two files into `output.dir`, a machine-readable `<task>.json`
and a human-readable `<task>.out`. The JSON is the parsing target and the `.out`
file is a rendering of the same dictionary for eyes. This page enumerates the
JSON keys per task. The summary is assembled by `api.build_summary` (SCF-derived
tasks) and `api._base_summary` (the post-SCF tasks), then `api.run` appends the
`provenance` and `outputs` blocks. `output.format_output` renders the `.out`
file from the same dictionary.

## Unit convention

A numeric key names its unit as a suffix, so the schema stays self-describing
without a separate legend. `energies_eV` is in electronvolts, `b0_GPa` in
gigapascals, `fmax_eV_ang` in eV/Å, `denergy_meV_per_atom` in meV per atom,
`volume_ang3` in Å³, `frequencies_cm1` in cm⁻¹, moments in `_muB`. A bare key is
dimensionless or non-numeric (`converged`, `n_iter`, `labels`, `scales`). Two
exceptions carry a non-suffix unit, the Becke-Johnson damping `a2_bohr` (in
Bohr) and `stress_eV_ang3` (in eV/Å³).

## Unconverged runs and exit codes

A run always writes its artifacts, and convergence is a flag inside the block
rather than a missing file. The `.out` header reads `NOT CONVERGED` in place of
`converged` when the flag is false. The CLI exit code follows the task.

| task | convergence flag | non-zero exit when |
|---|---|---|
| scf | `scf.converged` | not converged |
| relax | `relax.converged` | never (a step-limited relax still yields a usable geometry, exit 0) |
| eos | `eos.all_converged` | any volume's SCF did not converge |
| elastic | `elastic.all_converged` (and `relax_all_converged`) | any strained SCF did not converge |
| phonons | `min_frequency_cm1 > -1.0` | imaginary modes present |

Under `distributed: true` (a torchrun launch, wired for `scf`, `bands`, `relax`,
and `eos`), every rank computes the identical reduced summary and only rank 0
writes files.

## Blocks present on every task

| key | type | meaning |
|---|---|---|
| `code` | object | `{name, version, created}`, the writer identity and ISO timestamp |
| `task` | str | the task that produced this file |
| `structure` | object | cell, positions, species, and derived facts (below) |
| `parameters` | object | the resolved run settings (below) |
| `provenance` | object | machine and process context (below) |
| `outputs` | object | `{label: filename}` of the files written (below) |
| `runtime_s` | float | wall time of the task |

### structure

| key | type | meaning |
|---|---|---|
| `cell_ang` | 3×3 | lattice vectors, rows |
| `positions_ang` | N×3 | Cartesian positions |
| `species` | list[str] | chemical symbols |
| `n_atoms` | int | atom count |
| `volume_ang3` | float | cell volume |
| `density_g_cm3` | float | mass density |
| `spacegroup` | str | international symbol and number, present when spglib resolves the cell |
| `pointgroup` | str | conditional, alongside `spacegroup` |
| `n_symops` | int | conditional, symmetry-operation count |

### parameters

`formalism` (`nc`, `uspp/paw`, or `noncollinear`), `xc`, `ecut_eV`, `ecutrho_eV`
(USPP/PAW dense-grid cutoff, else null), `kmesh`, `nk_total`, `nspin`, `smearing`,
`width_eV`, `symmetry`, `mixing` (`{scheme, alpha, history, kerker, kerker_used,
precond}`), and `pseudos` (`{species: filename}`). SCF-derived tasks additionally
carry the materialized-system fields `nk`, `kweights`, `n_electrons`, `nbands`,
`fft_grid`, and `npw`. Tasks assembled without a System (relax, magnetism) leave
`nk`, `kweights`, and `kerker_used` null and omit the system-size fields. `hubbard`
(a list of `{species, l, U_eV, J_eV}`) is present only for a DFT+U run.

## scf

The `scf` task (and the SCF stage of `bands`) fills `summary["scf"]`.

| key | type | meaning |
|---|---|---|
| `converged` | bool | SCF reached the tolerances |
| `n_iter` | int | iteration count |
| `fermi_eV` | float or null | Fermi level |
| `gap_eV` | float or null | HOMO-LUMO gap, null for a metal or a fractionally occupied system |
| `energies_eV` | object | the energy breakdown (below) |
| `free_energy_per_atom_eV` | float | free energy divided by atom count |
| `trace` | list | per-iteration convergence history (below) |
| `convergence` | object | final residuals and diagnostics (below) |
| `total_magnetization_muB` | float | present for nspin=2 and noncollinear |
| `absolute_magnetization_muB` | float | present for nspin=2 and noncollinear |
| `magnetization_vector_muB` | list[3] | present only for the noncollinear formalism |

`energies_eV` holds the eleven-term breakdown from `checkpoint.energies_eV_dict`,
namely `kinetic`, `hartree`, `xc`, `local`, `nonlocal`, `ewald`, `smearing`,
`hubbard`, `onecenter` (PAW one-center), `dispersion`, `total`, and `free_energy`,
plus `fock` (exact exchange) and the derived `e0` (the total extrapolated to zero
smearing).

Each `trace` entry carries `iter`, `free_energy_eV`, `dE_eV` (null on the first
iteration), and `drho`. `t_s` is present when the driver timed its iterations. The
`energy_metric_eV` group (`energy_metric_charge_eV`, `energy_metric_longitudinal_eV`,
`energy_metric_transverse_eV`) is present only on the noncollinear energy-gate path.

`convergence` holds `criterion`, `final_dE_eV`, `final_drho`, `etol_eV`, `rhotol`,
`ratio_q` (the geometric decay rate of the energy residual, small is fast), and
`warm_started`. `entol_eV` and the `final_energy_metric_*` fields appear only when
`scf.convergence: energy`.

Two keys sit at the top level next to `scf`. `eigenvalues_eV` is the per-k, per-band
(and per-spin for nspin=2) eigenvalue array, and `occupations` its matching
occupations (empty for a noncollinear run). `scf_diagnostics` is present when the
SCF flight recorder ran (`scf.recorder`), holding `n_iter`, `final_residual`,
`diagnosis` (a list of `{tag, reason}`), `sloshing_fraction_final`,
`reordering_total`, and `wall_time_mean_s`, plus `final_magnetization_muB`
(nspin=2) and the `energy_metric_eV` group where the energy gate is active.

## relax

The `relax` task fills `summary["relax"]`. The nested engine is the default, and
the `joint`/`newton` engines replace some fields with their own H-apply provenance.

| key | type | meaning |
|---|---|---|
| `converged` | bool | fmax (and cell stress) reached the target |
| `method` | str | `nested`, `joint`, or `newton` |
| `n_steps` | int | optimizer steps (nested) or basis-rebuild cycles (joint/newton) |
| `optimizer` | str | `bfgs`, `fire`, `lbfgs`, or `steihaug-newton-cg` |
| `cell_relaxed` | bool | whether the cell degrees of freedom were relaxed |
| `fmax_target_eV_ang` | float | the force-convergence gate |
| `energy_eV` | float | final ASE-consistent potential energy |
| `fmax_eV_ang` | float | final maximum force |
| `max_displacement_ang` | float | largest atom displacement from the start |
| `species` | list[str] | final chemical symbols |
| `positions_ang` | N×3 | final Cartesian positions |
| `cell_ang` | 3×3 | final cell |
| `volume_ang3` | float | final volume |
| `trajectory` | list | per-step frames (below) |

Conditional relax keys. `scf_iter_per_step`, `scf_total_iter`, and
`scf_all_converged` record the inner SCF cost of the nested engine.
`extrapolation` names the density extrapolation, and `extrapolation_density_clamped`
appears only when a step's extrapolated density was clamped. `energy_change_eV` and
`volume_change_ang3` report the change from the first frame, and `nk_ibz` the
reduced k-count. A variable-cell relax adds `max_stress_eV_ang3`, `pressure_GPa`,
`pulay_correction`, and `pulay_pressure_GPa_final`. Selective dynamics adds `fixed`
(the per-atom axis mask) and `n_fixed_atoms`. The joint and newton engines add
`h_applies`, `h_seed`, `scf_iter_final`, and (newton) `n_grad`, `n_hvp`, `n_newton`.
A `joint`/`newton` request that falls back to nested records `requested_method` and
`fallback_reason`.

A `trajectory` entry holds `step`, `energy_eV`, `fmax_eV_ang`, `positions_ang`, and
`cell_ang`. `scf_iter` and `scf_converged` are present when the calculator cached an
inner SCF, and `pulay_pressure_GPa` on a variable-cell step.

`error_estimate` (below) is present at the final geometry when `error_estimate: true`.

## eos

The `eos` task fills `summary["eos"]` with an isotropic volume scan and its
third-order Birch-Murnaghan fit.

| key | type | meaning |
|---|---|---|
| `scales` | list[float] | isotropic volume-scale factors |
| `energy_kind` | str | which energy was fit (`free_energy`, `total`, ...) |
| `n_atoms` | int | atoms per cell |
| `volumes_ang3_per_atom` | list[float] | volume per atom at each scale |
| `energies_eV_per_atom` | list[float] | energy per atom at each scale |
| `fft_grid` | list[3] | the single grid shared across the scan |
| `v0_ang3_per_atom` | float | equilibrium volume |
| `b0_GPa` | float | bulk modulus |
| `b0_prime` | float | pressure derivative of the bulk modulus |
| `e0_eV_per_atom` | float | equilibrium energy |
| `rms_residual_eV_per_atom` | float | RMS fit residual |
| `b0_eV_ang3` | float | bulk modulus in eV/Å³ |
| `ev_a3_to_gpa` | float | the eV/Å³ → GPa conversion used |
| `all_converged` | bool | every volume's SCF converged |

## elastic

The `elastic` task fills `summary["elastic"]` with the 6×6 stiffness and the
Voigt-Reuss-Hill moduli.

| key | type | meaning |
|---|---|---|
| `strain` | float | the Voigt strain magnitude |
| `mode` | str | `clamped` or `relaxed` (internal coordinates re-relaxed) |
| `n_atoms` | int | atom count |
| `formalism` | str | `nc` or `uspp/paw` |
| `c_GPa` | 6×6 | the stiffness matrix in Voigt order (xx, yy, zz, yz, xz, xy) |
| `bulk_modulus_GPa` | object | `{voigt, reuss, hill}` |
| `shear_modulus_GPa` | object | `{voigt, reuss, hill}` |
| `young_modulus_GPa` | float | Young modulus (Hill) |
| `poisson_ratio` | float | Poisson ratio (Hill) |
| `mechanically_stable` | bool | the Born stability criteria |
| `residual_stress_GPa` | float | stress at the reference cell, nonzero means off-equilibrium |
| `all_converged` | bool | every strained SCF converged |

`relaxed` mode adds `relax_fmax`, `ref_fmax_eV_ang` (the reference-geometry force,
above the gate means a non-equilibrium input), `relax_steps` (twelve entries, one
per strain), and `relax_all_converged`.

## bands

The `bands` task runs an SCF, then fills `summary["bands"]` with the dispersion
along a k-path. The SCF block above rides in the same file.

| key | type | meaning |
|---|---|---|
| `kpts_frac` | Nk×3 | fractional k-points along the path |
| `x` | list[float] | the linear path coordinate |
| `labels` | list | `[[x, label], ...]` special-point ticks |
| `eigenvalues_eV` | array | eigenvalues along the path |
| `reference_eV` | float | the energy zero (Fermi level for a metal, else the valence-band maximum) |
| `irreps` | list | per-tick point-group irrep annotations, present when `bands.irreps: true` (norm-conserving only) |

## magnetism

The `magnetism` task fills `summary["magnetism"]` from `postscf.magnetism`.

| key | type | meaning |
|---|---|---|
| `ordering` | str | the detected magnetic ordering |
| `total_moment_muB` | float | total moment magnitude |
| `atomic_moments_muB` | list[float] | per-atom moment magnitudes |
| `moment_vectors_muB` | N×3 | per-atom moment vectors |
| `exchange_J_meV` | object or null | Heisenberg exchange constants by shell |
| `dmi_meV` | object or null | Dzyaloshinskii-Moriya constants by shell |
| `curie_temperature_mfa_K` | float or null | mean-field Curie temperature |

## phonons

The `phonons` task fills `summary["phonons"]` from the supercell
finite-displacement route.

| key | type | meaning |
|---|---|---|
| `supercell` | list[3] | the diagonal supercell |
| `n_atoms_supercell` | int | atoms in the supercell |
| `displacement_ang` | float | the finite-displacement amplitude |
| `kmesh_supercell` | list[3] | the folded k-mesh used |
| `qpts_frac` | Nq×3 | fractional q-points along the path |
| `x` | list[float] | the linear path coordinate |
| `labels` | list | `[[x, label], ...]` special-point ticks |
| `frequencies_cm1` | array | branch frequencies along the path |
| `min_frequency_cm1` | float | the minimum frequency, negative signals imaginary modes |
| `dos` | object | `{frequency_cm1, dos, mesh}`, present when `phonons.dos_mesh` is set |

## Optional blocks

These attach to an SCF-derived task under the named condition.

### error_estimate

Present on `scf`, `bands`, and `relax` when `error_estimate: true`, built by
`api._error_estimate_block`. When the run is outside coverage the block degrades to
`{available: false, reason}`. When available it carries `method`, `ecut_eV`,
`ecut_large_eV`, `denergy_eV`, `denergy_meV_per_atom`, `free_energy_extrapolated_eV`,
`drho_L1_per_electron`, `int_drho`, and `note`. Several sub-blocks are conditional
on the formalism. `force_error_max_eV_ang` and `force_error_rms_eV_ang` are present
for norm-conserving collinear (no NLCC) and USPP/PAW runs, or a `force_error`
`{available, reason}` when skipped. `gap_eV`, `gap_extrapolated_eV`, and `dgap_eV`
cover insulators, or a `gap_error` block for metals. `scf_convergence` and `smearing`
are self-consistency and smearing-extrapolation sub-blocks (or `smearing_error` for a
fixed-occupation run). `numerical_energy_error` is the leading-order sum of the
reachable terms (`total_eV`, `total_meV_per_atom`, `terms_eV`, `note`), with k-point
sampling excluded because it is not reachable from a single run.

### dispersion

Present on `scf` when `dispersion.enabled`, built by `api._apply_dispersion`, which
also folds the energy into `scf.energies_eV.dispersion`. Degrades to
`{available: false, reason}` when the element set is uncovered. When available it
carries `method` (`d3-bj` or `d4-bj`), `functional`, `damping` (`{s6, s8, a1,
a2_bohr}`), `energy_eV`, `energy_per_atom_eV`, `forces_eV_ang`, and
`stress_eV_ang3`.

### pdos and cohp

Present on `scf` when `projections.enabled` (and `projections.cohp.enabled`). Each
degrades to `{available: false, reason}` when the pseudopotentials omit `PP_PSWFC`
or the formalism is out of coverage. The energy-resolved curves are large, so the
full schema lives with the producers, `postscf.pdos.projected_dos` and
`postscf.cohp.cohp`. The `pdos` block carries `group_by`, `spilling`, `nspin`,
`energy_eV`, and `groups`. The `cohp` block carries `basis`, `method`, `spilling`,
`pairs`, `pair_icohp`, and `total_icohp`.

## provenance

Written by `runinfo.provenance_block`, a start-of-run machine snapshot plus an
end-of-run resample of the volatile parts. Optional fields are omitted when the
recording platform could not read them. The snapshot carries `timestamp`, `host`
(`{hostname, os, arch}`), `code` (`{gradwave, git, python, torch}`), `cpu`
(`{model, logical_cores, torch_threads}`), `memory` (`{total_gb, available_gb}`),
`gpu` (null on a CPU box), `load`, and `thermal`. The resample adds `load_end` and
`thermal_end`. `process` holds the accounting for this process, `wall_s`, `cpu_s`,
`effective_threads` (the ratio, a contested-box fingerprint), `peak_rss_gb`, and,
on CUDA, `cuda_peak_alloc_gb` and `cuda_peak_reserved_gb`.

## outputs

A `{label: filename}` map of the files written next to the JSON. `json` and
`report` are always present. `checkpoint` (an SCF-state `checkpoint.pt`),
`trajectory` (`relax.xyz`), `scf_trace` (the per-iteration `scf_trace.json` from
`scf.trace`), and the volumetric labels (`density`, `elf`, `magnetization`, and
`parchg_*`) are present when the corresponding output was requested.
