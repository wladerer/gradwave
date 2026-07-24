# Inputs and outputs

This page covers the input schema, the files a calculation writes, how to restart from a
checkpoint, and how to load results for analysis. The [Reference](reference.md) page
has the terse CLI and entry-point tables.

## Running

    gradwave init relax -o input.yaml   # write a starter input for a task
    gradwave input.yaml                 # outputs to the YAML's output.dir
    gradwave input.yaml -o results/     # override the output directory
    gradwave validate input.yaml        # parse and check, run nothing
    gradwave plot results/scf.json      # figure from a result file

`gradwave init` writes a commented starter input for a kind of calculation.
`gradwave init` with no name lists the templates: `scf`, `metal`, `relax`,
`relax-cell`, `bands`, `bands-soc`, `pdos`, `magnetism`, and `noncollinear`.
Each emits a complete, schema-valid file with an inline example structure and
placeholder pseudopotential paths; edit those two, then `gradwave validate` it.
Without `-o` the template goes to stdout (`gradwave init bands > bands.yaml`).

`gradwave validate` parses the input, resolves the structure and
pseudopotentials, and prints the calculation it would run without starting an
SCF. It is the fast way to catch a typo: an unknown key is rejected by name
(with a did-you-mean suggestion) rather than silently ignored, so a misspelled
`kpoint:` fails loudly instead of quietly running a Γ-only calculation.

`examples/input_si.yaml` documents every key with its default. The formalism is
detected from the UPF files, so the same input schema drives norm-conserving and
USPP/PAW calculations. `ecutrho` and the mixing scheme apply to the USPP/PAW path. The
explicit `gradwave run input.yaml` form remains valid.

## Input keywords

The YAML file parses into the frozen `Input` schema
([API reference](api/highlevel.md#gradwave.inputs.Input)). Every key except
`structure`, `pseudopotentials`, and `ecut` has a default, so a minimal input is
short. Energies are eV and lengths are Å throughout. A dash in the unit column
means the quantity is dimensionless or a plain count.

### Top level

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `structure` | *required* | Å | mapping or string | Inline `cell`/`positions`/`species` block, or a filename in any format ASE reads (cif, POSCAR, xyz). |
| `pseudopotentials` | *required* | — | mapping | `dir` and `map`; see below. |
| `ecut` | *required* | eV | float | Plane-wave kinetic-energy cutoff for the wavefunctions. |
| `ecutrho` | `4 × ecut` | eV | float | Density/augmentation cutoff. USPP/PAW only. Ignored for norm-conserving. |
| `xc` | `pbe` | — | string | Functional: `lda`, `pbe`, or `r2scan`. |
| `nbands` | `auto` | — | int or `auto` | Number of Kohn-Sham bands. `auto` picks from the electron count. |
| `symmetry` | `true` | — | bool | Reduce k to the IBZ and symmetrize the density each step. Forced off for a magnetic `noncollinear` run and the `magnetism` task (symmetry acts on the moment vector); setting it `true` there is an error. A spin-orbit-only run (`nonmagnetic: true`) keeps symmetry. |
| `nspin` | `1` | — | int | `1` unpolarized, `2` collinear spin. |
| `noncollinear` | `false` | — | bool | Spinor (non-collinear) SCF for `task: scf`, needed for spin-orbit coupling. Requires a fully-relativistic (FR) pseudopotential. |
| `nonmagnetic` | `false` | — | bool | With `noncollinear`: pin the moment to zero for a spin-orbit-only run (e.g. a nonmagnetic heavy metal). Keeps the full crystal symmetry via Kramers, so it is the efficient path when there is no magnetism. Requires `noncollinear: true`. |
| `start_mag` | `null` | — | mapping | Element → initial moment fraction in [-1, 1] (nspin=2 or a magnetic noncollinear seed). |
| `task` | `scf` | — | string | `scf`, `relax`, `bands`, or `magnetism`. |
| `device` | `cpu` | — | string | Torch device, e.g. `cpu` or `cuda`. |
| `verbose` | `true` | — | bool | Per-iteration SCF chatter on stdout. `gradwave run --quiet` silences a run regardless of this key. |
| `restart` | `null` | — | path | Checkpoint file to warm-start the density from. |

### `structure`

Three spellings reach the same geometry:

```yaml
structure: geometry.cif                              # bare filename, any ASE format
structure: {file: run.traj, format: traj, index: -1} # file with read controls
structure:                                           # inline block
  cell: [[...], [...], [...]]
  positions: {frac: [[...], ...]}
  species: [Si, Si]
```

Geometry goes through `ase.io.read`, so any format ASE reads works (cif, POSCAR,
xyz, extended-xyz, and so on). A structure with no periodic cell (a bare-molecule
xyz, for instance) is rejected at load time, because a plane-wave calculation
needs a cell; put the atoms in a box first. The lengths ASE returns are in Å, the
package convention.

The filename form accepts either a bare string or a mapping with read controls:

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `file` | *required* | — | string | Path to a geometry file, relative to the input. |
| `format` | *auto* | — | string | ASE format name, overriding the extension guess when it misfires. |
| `index` | `-1` | — | int | Frame to read from a multi-image file. `-1` is the last frame; a slice like `":"` is an error (pick one frame). |

The inline block:

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `cell` | *required* | Å | list[list[float]] | 3×3 lattice vectors as rows. |
| `positions.cart` | *required* | Å | list[list[float]] | Cartesian coordinates. Use this **or** `frac`. |
| `positions.frac` | *required* | — | list[list[float]] | Fractional coordinates. Use this **or** `cart`. |
| `species` | *required* | — | list[string] | Chemical symbols, one per atom. |

Reading from an external file means the geometry is no longer self-contained in
the YAML, so for an archived input either use the inline block or keep the
geometry file alongside it.

### `pseudopotentials`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `dir` | *required* | — | path | Directory of UPF files, relative to the input file. |
| `map` | *required* | — | mapping | Element → UPF filename. NC or USPP/PAW, auto-detected. |

### `kpoints`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `mesh` | `[1, 1, 1]` | — | list[int] | Monkhorst-Pack grid dimensions. |
| `shift` | `[0, 0, 0]` | — | list[int] | Grid offset. `[0,0,0]` is Γ-centered. |

### `smearing`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `type` | `none` | — | string | `none`, `fermi-dirac`, `gaussian`, `mp1`, or `cold`. |
| `width` | `0.1` | eV | float | Smearing width (electronic temperature scale). |

### `scf`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `max_iter` | `100` | — | int | Maximum self-consistency iterations. |
| `etol` | `1.0e-8` | eV | float | Total-energy convergence threshold. |
| `rhotol` | `1.0e-7` | — | float | Density-residual convergence threshold. |
| `diago.tol` | `1.0e-9` | — | float | Davidson eigensolver residual tolerance. |

### `scf.mixing`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `scheme` | `pulay` | — | string | `pulay`, `broyden`, `johnson`, or `linear`. |
| `alpha` | `0.7` | — | float | Linear mixing fraction. |
| `history` | `null` | — | int | Mixing history depth. `null` uses the per-scheme default (johnson 12, else 8). |
| `kerker` | `auto` | — | string or bool | Kerker preconditioner: `auto` (on when smearing is enabled), `true`, or `false`. |

### `relax`

Used when `task: relax`.

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `optimizer` | `bfgs` | — | string | `bfgs` or `fire`. |
| `fmax` | `0.01` | eV/Å | float | Convergence criterion. Gates the stress too under `cell`. |
| `max_steps` | `200` | — | int | Maximum ionic steps. |
| `cell` | `false` | — | bool | Variable-cell relaxation: relax the lattice with the atoms via `FrechetCellFilter` (stress). |
| `pressure` | `0.0` | GPa | float | External hydrostatic pressure, applied during cell relaxation. |

With `cell: true` the `relax.json` also reports `volume_ang3` and `max_stress_eV_ang3`. Relaxing a cell at fixed `ecut` carries Pulay (basis-incompleteness) stress, so converge `ecut` first or re-relax at the new cell. The plain filter does not constrain symmetry.

A relax task also writes `relax.xyz`, an extended-xyz trajectory with one frame
per ionic step carrying the energy and forces, readable by ovito, the ASE gui,
or `ase.io.read(..., index=":")`. Reaching `max_steps` still writes the
trajectory and exits 0. Convergence is carried by the `relax.converged` flag in
the JSON.

### `bands`

Used when `task: bands`.

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `path` | `""` | — | string | ASE bandpath string, e.g. `LGXUG`. Empty uses the lattice default. |
| `npoints` | `120` | — | int | Number of k-points along the path. |
| `nbands` | `null` | — | int | Bands to solve. `null` reuses the SCF count. |
| `irreps` | `false` | — | bool | Label bands at special points with Mulliken symbols. |

### `projections`

Adds a projected density of states to an `scf` calculation, written to the `pdos` block
of the JSON. Set `projections: true` for defaults, or a mapping for control.
Requires a pseudopotential with atomic orbitals (`PP_PSWFC`).

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `enabled` | `false` | — | bool | Compute the projected DOS; `projections: true` is shorthand. |
| `group_by` | `l` | — | string | Aggregate by `atom`, `l`, `lm`, or `total` (`j`, `jmj` for fully-relativistic). |
| `width` | `0.1` | eV | float | Gaussian broadening. |
| `npoints` | `800` | — | int | Energy grid points. |

Plot it with `gradwave plot <scf.json> --kind pdos`.

### `output`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `dir` | `./out` | — | path | Output directory, relative to the input file. |
| `checkpoint` | `true` | — | bool | Write `checkpoint.pt` after SCF tasks. |
| `wavefunctions` | `false` | — | bool | Include the wavefunction coefficients in the checkpoint (large). |
| `volumetric` | `false` | — | bool or table | Export real-space fields as `.cube`/`.xsf`. See [Volumetric export](#volumetric-export). |

## Output files

Each calculation writes three files into the output directory.

- `<task>.json` is the machine-readable summary and the parsing target. The
  top-level keys are stable, `code`, `task`, `structure`, `parameters`, `scf` (with
  `energies_eV` and the per-iteration `trace`, each entry carrying its wall time
  `t_s`), `eigenvalues_eV`, `occupations`, a `relax` or `bands` block for those
  tasks, `runtime_s`, `provenance`, and (on by default for `scf`/`bands`/
  `relax` — see [error estimation](error-estimation.md)) `error_estimate`.
- `<task>.out` is the human-readable report. Structure, parameters, the SCF
  iteration table (with per-iteration times), the energy breakdown, the gap or
  Fermi level and magnetization, eigenvalues for the first eight k-points, and
  the machine section.
- `checkpoint.pt` is restartable state, written for SCF tasks under the
  `output.checkpoint` key. Wavefunctions are excluded by default because they
  dominate the file size and a restart consumes only the density and becsum. Set
  `output.wavefunctions: true` to archive them.

`<task>` is `scf`, `relax`, or `bands`. A relax task additionally writes the
`relax.xyz` extended-xyz trajectory described above, referenced from the JSON
as `outputs.trajectory`. Setting `output.volumetric` adds `.cube`/`.xsf` field
files, described under [Volumetric export](#volumetric-export) and referenced
from the JSON under `outputs`.

Every output carries a `provenance` block (`gradwave/runinfo.py`) recording the
context a timing needs to be trusted months later: timestamp with timezone,
host and code versions (gradwave + git commit, torch, python), CPU model and
core/thread counts, RAM total and available, GPU name/VRAM/utilization/
temperature when CUDA is present, load averages and the busiest competing
processes sampled at run start AND end (the contested-machine indicator),
thermal-zone temperatures at both points (throttling shows up as the drift),
and process accounting: wall time, CPU time, effective threads (their ratio —
far below `torch_threads` fingerprints a contested box), peak RSS, and peak
CUDA memory. Collection is best-effort and dependency-free (`/proc`, `/sys`,
`nvidia-smi`); unreadable fields are absent rather than errors. The `.out`
report renders it as the closing `machine` section.

## Basis-set error estimate

Setting `output.error_estimate: true` (or `error_estimate: true` at the top level)
runs a post-SCF plane-wave (Ecut) discretization-error estimate and adds it to both
output files. This is the cheap complement correction of Cancès et al., a
post-processing pass that needs no larger SCF. The estimate answers "is my cutoff
converged" without a cutoff sweep.

The `<task>.json` gains an `error_estimate` block and the `<task>.out` a matching
section. The reported fields are the estimated energy error `denergy_eV` (a definite
lowering) and the extrapolated `free_energy_extrapolated_eV`, the density-error L1
norm per electron, and, for norm-conserving calculations (nspin=1 or 2), the Hellmann-Feynman
force error (`force_error_max_eV_ang` and the rms). It is a first-order indicator, not a
rigorous bound, so use it to gate convergence, not to quote an uncertainty. When the
calculation is outside coverage, USPP or PAW with symmetry on for example, the report notes
that the estimate was skipped and why, rather than failing the calculation. Coverage is
norm-conserving and USPP/PAW for the energy and density error, including the
non-collinear/spinor path (`noncollinear: true`), and norm-conserving
(nspin=1 or 2) for the force error. The `magnetism` task carries no estimate; run
`task: scf` with `noncollinear: true` to get one on a spinor SCF. The
[Basis-set error estimation](error-estimation.md#coverage) page has the full
coverage table.

## Volumetric export

Setting `output.volumetric` writes real-space fields to `.cube`, `.xsf`, or VASP
CHGCAR files that VESTA, Ovito, and tinykit read. These are the CHGCAR, PARCHG
and ELF analogs. The density and
the plane-wave coefficients are already in memory at the end of an SCF, so no rerun is
needed. The file encoding (units, voxel order, the periodic wrap plane) is handled by
ASE.

`output.volumetric: true` is shorthand for the density alone. The mapping form selects
fields and the file format:

```yaml
output:
  volumetric:
    density: true          # ρ(r), the CHGCAR analog
    elf: true              # electron localization function ELF(r)
    magnetization: true    # |m(r)|, noncollinear runs only
    bands: [[3, 0], [4, 0]]  # PARCHG |ψ_nk(r)|² for [band, kpoint] pairs
    format: cube           # cube (default), xsf, or chgcar
```

The `chgcar` format writes a VASP CHGCAR (ρ·Ω, so ASE's `VaspChargeDensity`
reader recovers ρ in e/Å³), which the POV-Ray front end
[tinykit](https://github.com/wladerer/tinykit) renders as an isosurface:
`tk viz density.chgcar --supercell 2 2 2 --isovalue 0.5`. The
[Post-SCF analysis](postscf-analysis.md) tutorial shows the density and ELF of
diamond rendered this way.

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `density` | `false` | — | bool | Total density ρ(r) [e/Å³]. |
| `elf` | `false` | — | bool | Electron localization function, a value in [0,1]. |
| `magnetization` | `false` | — | bool | Magnetization density \|m(r)\| [μ_B/Å³] for noncollinear runs. |
| `bands` | `[]` | — | list | `[band, kpoint]` pairs; each writes one PARCHG file, `parchg_b{band}_k{kpoint}`. |
| `format` | `cube` | — | string | `cube`, `xsf`, or `chgcar`. |

Files land in the output directory named by field (`density.cube`, `elf.cube`,
`magnetization.cube`, `parchg_b3_k0.cube`) and are listed under `outputs` in the JSON
summary. A single-state PARCHG density integrates to 1 over the cell; the total
density integrates to the electron count; the occupation-weighted sum of the PARCHG
densities reproduces the total density.

Coverage follows the result type. The total density is available for every SCF. PARCHG
covers collinear and noncollinear runs (the two spinor components are summed for the
latter). For USPP/PAW it is the soft pseudo-density, without the augmentation charge,
as in VASP. ELF[[38]](bibliography.md#elf) is available for collinear norm-conserving
results. Magnetization needs
a noncollinear run (`noncollinear: true`). A requested field that the result type does
not support is skipped with a note, and the rest of the run still writes.

The same fields are reachable from Python through
[`gradwave.postscf.volumetric`](api/properties.md#volumetric-export):

```python
from gradwave.postscf import volumetric as vol

vol.write_density(res, "density.cube")            # ρ(r)
vol.write_band_density(res, "homo.cube", band=3)  # |ψ_nk(r)|²
vol.write_elf(res, "elf.xsf")                     # ELF(r)
```

### Charge-response field

`gradwave.postscf.response` writes ∂n(r)/∂R_I, the change in the density when an atom
moves. Rendered as an isosurface it shows where charge flows as the atom is nudged, a
positive lobe ahead of the motion and a negative one behind, which is the
force-constant physics made spatial.

```python
from gradwave.postscf import response

# ∂n(r)/∂R for atom 0 along x, by central finite difference (two SCFs)
path, drift = response.write_density_response(inp, "dn_dx.cube", atom=0, direction=0)
```

The reference implementation is a central finite difference of two displaced SCFs, so
it covers every formalism. It returns the ∫ ∂n/∂R dr residual alongside the field.
Charge conservation puts it at zero, and a nonzero value flags an unconverged SCF or
too large a step. The 3N displacements are independent, so a full response set
parallelizes across the [remote workers](performance.md). For USPP insulators the
analytic `postscf.uspp_position.position_density_response` gives the same field from
the converged state in a single response solve, no re-runs.

## Checkpoints

```python
from gradwave.checkpoint import save_checkpoint, load_checkpoint, as_start_from

save_checkpoint(res, "checkpoint.pt")        # res: scf_uspp dict or NC SCFResult
payload = load_checkpoint("checkpoint.pt")   # plain dict of CPU tensors + metadata
res2 = scf_uspp(system, xc, start_from=as_start_from(payload))
```

The `restart:` YAML key does the same from the command line. A restart requires the
same FFT grid and spin count. The solver validates both and rescales the density by
the volume ratio, so small cell changes in EOS-style scans restart cleanly. Both
formalisms restart, and the ASE calculator applies the same density reuse
automatically between the ionic steps of a relaxation or MD run.

## Analysis

The analysis helpers return tidy pandas frames and matplotlib figures.

```python
from gradwave import analysis
r = analysis.load("out/scf.json")

analysis.scf_frame(r)             # iter, free_energy_eV, dE_eV, drho, dF_from_final_eV
analysis.eigenvalues_frame(r)     # spin, k, kweight, band, energy_eV, occupation
analysis.bands_frame(r)           # k, x, band, energy_eV; labels in df.attrs
analysis.dos_frame(r, width=0.1)  # gaussian DOS from eigenvalues and k-weights

analysis.plot_scf(r, path="scf.png")
analysis.plot_bands(r, path="bands.png")
analysis.plot_dos(r, path="dos.png")
```

`gradwave plot` wraps the same functions. `gradwave plot out/scf.json --kind dos
--width 0.2` selects the DOS view of an SCF result. The plot command dispatches over
scf, bands, and dos results, so a `relax.json` has no plot view. Read its trajectory
into pandas directly, as shown in the
[geometry optimization tutorial](geometry-optimization.md#plot-the-trajectory).
