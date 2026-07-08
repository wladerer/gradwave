# gradwave

Differentiable plane-wave density functional theory for periodic solids, in PyTorch.

## Validation vs Quantum ESPRESSO 7.5 (identical UPF / ecut / k-mesh)

| quantity | agreement |
|---|---|
| Si total energy (LDA & PBE, 30 Ry, 4×4×4) | ≤ 0.001 meV/atom |
| Al free energy (PBE, semicore, Gaussian smearing, 40 Ry) | < 2 meV/atom |
| Al Fermi level | < 10 meV |
| Si forces (displaced, vs `tprnfor`) | < 5 meV/Å |
| Si band structure L–Γ–X–U–Γ (occupied) | < 10 meV |
| dE/dθ, dL(ρ)/dθ (learnable XC) vs SCF finite differences | 1e-5 / 2e-4 rel |
| bcc Fe ferromagnet (spin-PBE, 60 Ry): free energy | < 0.1 meV/atom |
| bcc Fe magnetic moment | 2.2244 vs QE 2.22 μB (exp. 2.22) |

## Performance (Si LDA, 30 Ry, 4×4×4 — `benchmarks/bench_scf.py`)

| configuration | SCF wall time |
|---|---|
| per-k Python loop (v0), 8-core CPU | 218 s |
| k-batched + adaptive diago tolerance, 8-core CPU | 33 s |
| + spglib symmetry (36 → 8 k in the IBZ), 8-core CPU | 7.1 s |
| + symmetry, 22-thread CPU | 4.6 s |
| + symmetry, RTX 3050 laptop GPU (complex128) | **1.4 s** |

The SCF runs fully k-batched: padded `(nk, nb, npw_max)` layout, batched
FFT Hamiltonian applies, batched QR/eigh Davidson, band-chunked dense-grid
ops to bound GPU memory. `System.to("cuda")` moves a prepared calculation
to GPU. On NixOS, expose the driver to the managed-Python torch via
`/run/opengl-driver/lib` (see docs in repo).

### Cross-system matrix (`benchmarks/bench_matrix.py`, symmetry on)

| case | atoms | e⁻ | ecut | k (IBZ) | 8-core CPU | RTX 3050 |
|---|---|---|---|---|---|---|
| Si (diamond) | 2 | 8 | 30 Ry | 8 | 6.1 s | 1.4 s |
| C (diamond) | 2 | 8 | 40 Ry | 8 | 3.8 s | 0.9 s |
| GaAs (zincblende, Ga-3d, l=2) | 2 | 18 | 40 Ry | 8 | 16.5 s | 2.8 s |
| Al (fcc metal, smeared) | 1 | 11 | 40 Ry | 29 | 13.1 s | 3.9 s |
| Cu (fcc d-band metal, 3s3p semicore) | 1 | 19 | 45 Ry | 29 | — | 5.1 s |
| Cu₃Al (L1₂ intermetallic) | 4 | 68 | 45 Ry | 10 | — | 23.5 s |
| MgO (rocksalt) | 2 | 16 | 50 Ry | 8 | 7.0 s | 1.5 s |
| Si₈ (conventional cell) | 8 | 32 | 30 Ry | 4 | 22.9 s | 5.1 s |
| Si₆₄ (2×2×2 supercell, Γ) | 64 | 256 | 30 Ry | 1 | — | 231 s |

Large cells auto-enable Kerker mixing (charge-sloshing control: Si₆₄
converges in 25 iterations instead of 54).

All validated against QE at matched settings (≤ 1 meV/atom; GaAs at
0.003 meV/atom exercises the d-channel projectors). Si₆₄(Γ) reproduces
Si₈(2×2×2) to 3 µeV/atom — exact supercell folding equivalence.

- Norm-conserving pseudopotentials (Quantum ESPRESSO UPF v2: PseudoDojo / SG15 ONCV), Kleinman–Bylander form
- Collinear spin polarization (LSDA / spin-PBE) — `nspin: 2` + `start_mag` in YAML
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

- Stress is fixed-basis (Nielsen–Martin, same convention as QE): variable-cell
  relaxation via `FrechetCellFilter` works, but carries the usual Pulay
  pressure at low ecut — converge ecut or re-relax at the final cell.
- Stress with nspin=2, spin-orbit, or DFT+U is not implemented yet.
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
