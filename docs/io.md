# Inputs, outputs, checkpoints, analysis (2026-07-15)

## Running

    gradwave input.yaml                 # outputs to the YAML's output.dir
    gradwave input.yaml -o results/     # override the output directory
    gradwave plot results/scf.json      # figure from a result file

`examples/input_si.yaml` documents every key. The formalism is detected
from the UPF files, so the same input schema drives norm-conserving and
USPP/PAW runs; `ecutrho` and the mixing scheme apply to the USPP/PAW
path. `gradwave run input.yaml` remains valid.

## Output files

Each run writes three files into the output directory.

- `<task>.json` — the machine-readable summary and the parsing target.
  Stable top-level keys: `code`, `task`, `structure`, `parameters`,
  `scf` (with `energies_eV` and the per-iteration `trace`),
  `eigenvalues_eV`, `occupations`, plus `relax` or `bands` blocks for
  those tasks, and `runtime_s`.
- `<task>.out` — the human-readable report: structure, parameters, the
  SCF iteration table, the energy breakdown, gap or Fermi level and
  magnetization, and eigenvalues for the first eight k-points.
- `checkpoint.pt` — restartable state (SCF tasks, `output.checkpoint`
  key). Wavefunctions are excluded by default because they dominate the
  file size and a restart only consumes the density and becsum; set
  `output.wavefunctions: true` to archive them.

## Checkpoints

    from gradwave.checkpoint import save_checkpoint, load_checkpoint, as_start_from

    save_checkpoint(res, "checkpoint.pt")            # res: scf_uspp dict or NC SCFResult
    payload = load_checkpoint("checkpoint.pt")       # plain dict of CPU tensors + metadata
    res2 = scf_uspp(system, xc, start_from=as_start_from(payload))

In YAML the `restart:` key does the same. Restarting requires the same
FFT grid and spin count (the solver validates both and rescales ρ by
the volume ratio, so small cell changes in EOS-style scans work).
Restart is a USPP/PAW feature; NC results save for archival and
analysis but the NC loop has no `start_from` yet.

## Analysis (pandas + matplotlib)

    from gradwave import analysis
    r = analysis.load("out/scf.json")

    analysis.scf_frame(r)          # iter, free_energy_eV, dE_eV, drho, dF_from_final_eV
    analysis.eigenvalues_frame(r)  # tidy (spin, k, kweight, band, energy_eV, occupation)
    analysis.bands_frame(r)        # (k, x, band, energy_eV); labels in df.attrs
    analysis.dos_frame(r, width=0.1)  # gaussian DOS from eigenvalues + k-weights

    analysis.plot_scf(r, path="scf.png")
    analysis.plot_bands(r, path="bands.png")
    analysis.plot_dos(r, path="dos.png")

`gradwave plot` wraps the same functions; `--kind dos --width 0.2`
selects the DOS view of an SCF result.
