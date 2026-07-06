# gradwave

Differentiable plane-wave density functional theory for periodic solids, in PyTorch.

- Norm-conserving pseudopotentials (Quantum ESPRESSO UPF v2: PseudoDojo / SG15 ONCV), Kleinman–Bylander form
- Base units: **eV** and **Ångström**; float64/complex128 throughout
- SCF total energies, Hellmann–Feynman forces, geometry optimization (via ASE), band structures
- Autograd infrastructure: implicit differentiation through the SCF fixed point for
  learnable XC functionals and automatic Hessians/phonons

## Usage

```bash
gradwave run input.yaml
```

See `examples/` for input files. Any geometry format ASE can read is accepted.

## Scope fences (current)

- Fixed cell only: no stress tensor or variable-cell relaxation yet
  (do not wrap the calculator in `ExpCellFilter`).
- Norm-conserving pseudopotentials only — PAW/ultrasoft UPF files are rejected at parse time.

## Development

```bash
uv sync            # managed venv with all dev deps
uv run pytest -n auto
uv run ruff check
```

Reference data is generated against Quantum ESPRESSO `pw.x` with the *same* UPF files
(`tests/fixtures/qe/regenerate.py`; QE via `nix shell nixpkgs#quantum-espresso`).
CI never runs QE — fixtures are committed.
