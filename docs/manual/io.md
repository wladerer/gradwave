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

## Basis-set error estimate

Setting `output.error_estimate: true` (or `error_estimate: true` at the top level)
runs a post-SCF plane-wave (Ecut) discretization-error estimate and adds it to both
output files. This is the cheap complement correction of Cancès et al., a
post-processing pass that needs no larger SCF, so it answers "is my cutoff
converged" without a cutoff sweep.

The `<task>.json` gains an `error_estimate` block and the `<task>.out` a matching
section. The reported fields are the estimated energy error `denergy_eV` (a definite
lowering) and the extrapolated `free_energy_extrapolated_eV`, the density-error L1
norm per electron, and, for norm-conserving nspin=1 runs, the Hellmann-Feynman force
error (`force_error_max_eV_ang` and the rms). It is a first-order indicator, not a
rigorous bound, so use it to gate convergence, not to quote an uncertainty. When the
run is outside coverage, USPP or PAW with symmetry on for example, the block records
`available: false` with the reason rather than failing the run. Coverage is
norm-conserving and USPP/PAW for the energy and density error, norm-conserving
nspin=1 for the force error.

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
