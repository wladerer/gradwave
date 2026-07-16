# Installation and first run

The project is uv-managed. On NixOS the managed CPython loads wheel C libraries
through nix-ld, so no system pip is involved.

    cd QSuite
    uv venv && uv sync
    uv run gradwave --version

Run the shipped Si PAW example to confirm the SCF works end to end.

    uv run gradwave examples/input_si.yaml -o out/

The last line prints the converged free energy and the iteration count. The run
writes `out/scf.out`, `out/scf.json`, and `out/checkpoint.pt`.

## What you need

- A norm-conserving or USPP/PAW UPF file for every species. The examples ship
  ONCV[[6]](bibliography.md#oncv) and PAW[[7]](bibliography.md#paw) pseudos under
  `tests/fixtures/qe/pseudos`.
- A structure, either inline in the YAML or in any format ASE
  reads.[[4]](bibliography.md#ase)
- A plane-wave cutoff in eV. The density grid is set to four times the cutoff by
  default.

## Next

Continue to [Geometry optimization](geometry-optimization.md) for the first
tutorial, or read the [Reference](reference.md) page for the full CLI and output
schema.
