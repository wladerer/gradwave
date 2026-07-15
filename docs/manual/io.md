# Inputs and outputs

This page covers the input schema, the files a run writes, how to restart from a
checkpoint, and how to load results for analysis. The [Reference](reference.md) page
has the terse CLI and entry-point tables.

## Running

    gradwave input.yaml                 # outputs to the YAML's output.dir
    gradwave input.yaml -o results/     # override the output directory
    gradwave plot results/scf.json      # figure from a result file

`examples/input_si.yaml` documents every key with its default. The formalism is
detected from the UPF files, so the same input schema drives norm-conserving and
USPP/PAW runs. `ecutrho` and the mixing scheme apply to the USPP/PAW path. The
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
| `ecutrho` | `4 × ecut` | eV | float | Density/augmentation cutoff. USPP/PAW only; ignored for norm-conserving. |
| `xc` | `pbe` | — | string | Functional: `lda` or `pbe`. |
| `nbands` | `auto` | — | int or `auto` | Number of Kohn-Sham bands. `auto` picks from the electron count. |
| `symmetry` | `true` | — | bool | Reduce k to the IBZ and symmetrize the density each step. |
| `nspin` | `1` | — | int | `1` unpolarized, `2` collinear spin. |
| `start_mag` | `null` | — | mapping | Element → initial moment fraction in [-1, 1] (nspin=2). |
| `task` | `scf` | — | string | `scf`, `relax`, or `bands`. |
| `device` | `cpu` | — | string | Torch device, e.g. `cpu` or `cuda`. |
| `restart` | `null` | — | path | Checkpoint file to warm-start the density from. |

### `structure`

Provide either a filename string or the inline block below.

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `cell` | *required* | Å | list[list[float]] | 3×3 lattice vectors as rows. |
| `positions.cart` | *required* | Å | list[list[float]] | Cartesian coordinates. Use this **or** `frac`. |
| `positions.frac` | *required* | — | list[list[float]] | Fractional coordinates. Use this **or** `cart`. |
| `species` | *required* | — | list[string] | Chemical symbols, one per atom. |

### `pseudopotentials`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `dir` | *required* | — | path | Directory of UPF files, relative to the input file. |
| `map` | *required* | — | mapping | Element → UPF filename. NC or USPP/PAW, auto-detected. |

### `kpoints`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `mesh` | `[1, 1, 1]` | — | list[int] | Monkhorst-Pack grid dimensions. |
| `shift` | `[0, 0, 0]` | — | list[int] | Grid offset; `[0,0,0]` is Γ-centered. |

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
| `fmax` | `0.01` | eV/Å | float | Force convergence criterion. |
| `max_steps` | `200` | — | int | Maximum ionic steps. |

### `bands`

Used when `task: bands`.

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `path` | `""` | — | string | ASE bandpath string, e.g. `LGXUG`. Empty uses the lattice default. |
| `npoints` | `120` | — | int | Number of k-points along the path. |
| `nbands` | `null` | — | int | Bands to solve. `null` reuses the SCF count. |
| `irreps` | `false` | — | bool | Label bands at special points with Mulliken symbols. |

### `output`

| keyword | default | unit | type | description |
|---|---|---|---|---|
| `dir` | `./out` | — | path | Output directory, relative to the input file. |
| `checkpoint` | `true` | — | bool | Write `checkpoint.pt` after SCF tasks. |
| `wavefunctions` | `false` | — | bool | Include the wavefunction coefficients in the checkpoint (large). |

## Output files

Each run writes three files into the output directory.

- `<task>.json` is the machine-readable summary and the parsing target. The
  top-level keys are stable, `code`, `task`, `structure`, `parameters`, `scf` (with
  `energies_eV` and the per-iteration `trace`), `eigenvalues_eV`, `occupations`, a
  `relax` or `bands` block for those tasks, and `runtime_s`.
- `<task>.out` is the human-readable report. Structure, parameters, the SCF
  iteration table, the energy breakdown, the gap or Fermi level and magnetization,
  and eigenvalues for the first eight k-points.
- `checkpoint.pt` is restartable state, written for SCF tasks under the
  `output.checkpoint` key. Wavefunctions are excluded by default because they
  dominate the file size and a restart consumes only the density and becsum. Set
  `output.wavefunctions: true` to archive them.

`<task>` is `scf`, `relax`, or `bands`.

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
