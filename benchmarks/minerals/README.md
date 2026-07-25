# Magnetic-mineral SCF: gradwave vs Quantum ESPRESSO

A matched forward-SCF benchmark on real antiferromagnetic minerals. For each
mineral we run the **same** cell, pseudopotential, plane-wave cutoff, k-grid,
Gaussian smearing, PBE XC and magnetic ordering in both codes and compare
(a) total-energy agreement and (b) wall time.

Honest framing: QE (optimised Fortran + MPI) is expected to be much faster than
gradwave (differentiable PyTorch) for a forward SCF. The deliverable is the
*quantified gap* and *correctness*, not gradwave winning.

## Method

- **Structures** (`structures.py`): built with ASE and reduced to the
  **magnetic primitive cell** — the smallest cell that hosts the AFM order. The
  identical lattice + Cartesian positions + UPF are fed to both codes (QE does
  not auto-primitivise, so this is the fairness anchor). NiO type-II needs the
  doubled 4-atom rhombohedral cell; the corundum AFMs (hematite, eskolaite) are
  k=0 orders whose 10-atom rhombohedral primitive already hosts them.
- **gradwave** (`run_bench.py`): `setup_system(..., use_symmetry=True,
  magmoms=[[0,0,±m]...], collinear_magnetic=True)` on an **unshifted**
  Γ-centred mesh — this is the collinear-magnetic (Shubnikov) k-fold, which
  folds k into the *magnetic* IBZ (the spin-sublattice-swap enters as an
  anti-unitary op). `scf(SpinPBE(), nspin=2, start_mag=±m, smearing="gaussian",
  mixing_scheme="johnson")`, Kerker charge preconditioning on, Stoner
  spin-preconditioner off. 8 torch threads.
- **QE** (`qe.py`): `pw.x` scf, `ibrav=0`, matched `ecutwfc`/`ecutrho`
  (NC: ecutrho=4·ecutwfc), same k-grid, `occupations='smearing'`,
  `smearing='gaussian'`, matched `degauss`, `nspin=2`. Each magnetic sublattice
  is a distinct QE species (Fe1/Fe2…, same UPF) with opposite
  `starting_magnetization`, so QE detects the magnetic space group and reduces k
  comparably. `mixing_mode='plain'`, `mixing_beta=0.3`. 8 MPI ranks × 1 thread.
- The FFT grid is pinned to QE's dense-grid dimensions in gradwave
  (`fft_shape=`) so the XC grid integration matches at the meV level.
- **Energy validation.** We report `dE_electronic` = the total-energy
  difference with the ion-ion **Ewald** constant removed from both codes
  (`(E_tot − E_ewald)_gw − (E_tot − E_ewald)_QE`). This isolates the
  Kohn-Sham/electronic energy, which is the physics being validated. See the
  Ewald note below.

## Results

Runs on **asus** (Intel + 22 core), QE 7.5, PBE ONCV (SG15) pseudos.
gradwave = 8 torch threads; QE = 8 MPI ranks. Wall = full program (gradwave:
setup + SCF; QE: PWSCF WALL).

| mineral | atoms | k-grid | pseudo | IBZ k (gw mag / QE / gw TR-only) | gradwave wall (iters) | QE wall (ranks) | ratio QE:gw | ΔE_elec (meV/atom) | ΔE_total | status |
|---------|------:|:------:|:------:|:--------------------------------:|:---------------------:|:---------------:|:-----------:|:------------------:|:--------:|:------:|
| NiO (rocksalt, type-II AFM) | 4 | 6×6×6 | Ni+O ONCV (NC) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| α-Fe₂O₃ hematite (corundum, ++−−) | 10 | 4×4×4 | Fe+O ONCV (NC) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Cr₂O₃ eskolaite (corundum, +−+−) | 10 | 4×4×4 | Cr+O ONCV (NC) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Attempted (USPP/PAW path): _TBD_.

## Findings

_TBD after runs._

### Ewald note (a real gradwave bug this benchmark surfaced)

For the low-symmetry magnetic primitive cells here, gradwave's ion-ion Ewald
energy (`core/energies/ewald.py`) is **η-dependent** — it should be invariant
(and the module's own docstring claims "η-independence is a unit test"). At the
default η it under-converges the lattice sum, e.g. NiO Ewald = −479.53 Ry vs the
correct −478.65 Ry (an independent numpy Ewald and QE both give −478.6535). The
error is a **constant** offset (position/ecut-independent), so it does not
affect forces, k-reduction, magnetisation or the electronic energy — only the
absolute total. The high-symmetry fcc cells in the existing `test_scf_vs_qe`
suite have symmetric inverse cells that mask it. That is why we validate on
`dE_electronic` (Ewald removed) and separately flag `dE_ewald`.
