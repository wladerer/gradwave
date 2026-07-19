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
| NiO Hubbard U vs `hp.x` DFPT | 6.449 vs 6.431 eV (0.3%) |
| Si Γ phonon (PAW) vs `ph.x` | 0.003% |
| GaAs spin-orbit split-off Δ₀ vs fully-relativistic QE | 0.336 eV, 2e-3 eV |

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
converges in 25 iterations instead of 54); slabs and molecules can opt into
the local-TF preconditioner (`precond="local_tf"`) for fewer iterations in
inhomogeneous cells.

All validated against QE at matched settings (≤ 1 meV/atom; GaAs at
0.003 meV/atom exercises the d-channel projectors). Si₆₄(Γ) reproduces
Si₈(2×2×2) to 3 µeV/atom — exact supercell folding equivalence.

- Norm-conserving (PseudoDojo / SG15 ONCV, Kleinman–Bylander) and ultrasoft/PAW
  pseudopotentials (psl `q_with_l` UPFs), detected from the UPF file
- SCF total and free energies, Hellmann–Feynman forces, stress, geometry and
  variable-cell relaxation (via ASE), band structures with irrep labels, total
  and projected (l, m, j) DOS, Γ-point phonons
- Collinear spin (`nspin: 2`), non-collinear magnetism, and spin-orbit coupling
  from fully-relativistic pseudopotentials
- Constrained non-collinear moments with autograd-exact torques, spin spirals,
  magnetocrystalline anisotropy, and Heisenberg/DMI exchange constants
- DFT+U with the Hubbard U from linear response and an exact dE/dU
- IBZ symmetry reduction with density/becsum symmetrization, including magnetic
  (Shubnikov) groups for non-collinear cells (`magmoms=`)
- Single-run error estimates: plane-wave (Ecut) discretization, SCF
  convergence, smearing, and k-point extrapolation
- Autograd infrastructure: implicit differentiation through the SCF fixed point
  for learnable XC functionals and automatic Hessians/phonons
- Base units: **eV** and **Ångström**; float64/complex128 throughout, CPU and GPU

## Usage

```bash
gradwave run input.yaml
```

See `examples/` for input files. Any geometry format ASE can read is accepted.

## Scope fences (current)

- Stress is fixed-basis (Nielsen–Martin, same convention as QE): variable-cell
  relaxation via `FrechetCellFilter` works, but carries the usual Pulay
  pressure at low ecut — converge ecut or re-relax at the final cell.
- On the norm-conserving path, forces and stress are nspin=1 only (no NLCC
  force term), and stress excludes fully-relativistic pseudos and DFT+U. The
  USPP/PAW path carries forces and stress at nspin=1 and 2. Its stress
  excludes DFT+U (the strained S-dressed projections are missing).
- USPP/PAW otherwise runs the full stack: k-space symmetry with becsum
  symmetrization, collinear spin, non-collinear/SOC spinors, +U, the
  implicit-differentiation (Sternheimer) machinery, and GPU batching — all
  validated vs QE (USPP energy 0.1 µeV; PAW energy 0.3 meV/atom, forces
  1e-4 eV/Å, stress 0.13 kbar, ferromagnetic Ni 1.6 meV/atom; Γ phonons via
  FD of the analytic forces, 0.003% vs ph.x — see examples/si_paw_phonon.py).
  Old-style USPPs with polynomial augmentation refits (nqf > 0, e.g. GBRV)
  are rejected at parse time.
- `dielectric_born` (ε∞, Born charges, IR) is nspin=1 scalar-relativistic
  insulators only. Hybrids/exact exchange and meta-GGA functionals are not
  implemented (semilocal LDA/GGA only).

## Development

```bash
uv sync            # managed venv with all dev deps
uv run pytest -m "not standard and not slow and not torture and not gpu"   # fast gate, ~80 s
uv run ruff check
```

The suite is tiered by marker (unmarked tests are the fast tier):

| tier | select | wall time | when |
|---|---|---|---|
| fast | `-m "not standard and not slow and not torture and not gpu"` | ~80 s | every commit |
| standard | `-m "not slow and not torture and not gpu"` | ~10 min | CI |
| nightly | `-m "not torture and not gpu"` | hours | nightly / pre-release |
| torture | `-m torture` | 10 min – hours each | manually, when their subsystem changes |

Reference data is generated against Quantum ESPRESSO `pw.x` with the *same* UPF files
(`tests/fixtures/qe/regenerate.py`; QE via `nix shell nixpkgs#quantum-espresso`).
CI never runs QE — fixtures are committed.
