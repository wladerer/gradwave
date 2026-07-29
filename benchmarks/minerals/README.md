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

IBZ k is reported as `no-sym (TR-only) → Shubnikov-folded (gradwave) / QE`, so
the magnetic reduction is visible and the two codes' irreducible counts sit side
by side. ΔE_elec is the Ewald-removed electronic agreement (the validated
number); ΔE_total is the raw total-energy difference, both in meV/atom. The wall
ratio is `gradwave wall ÷ QE wall`, so 12.3× means QE finished 12.3 times faster.

| mineral | atoms | k-grid | pseudo | IBZ k: no-sym → gw / QE | gradwave wall (iters) | QE wall (ranks) | wall ratio (gw÷QE) | ΔE_elec (meV/atom) | ΔE_total (meV/atom) | status |
|---------|------:|:------:|:------:|:-----------------------:|:---------------------:|:---------------:|:------------------:|:------------------:|:-------------------:|:------:|
| NiO (rocksalt, type-II AFM) | 4 | 6×6×6 | Ni+O ONCV (NC) | 112 → 32 / 32 | 559.1 s (18) | 45.5 s (8) | 12.3× | −0.016 | −2969 † | converged ✓ |
| α-Fe₂O₃ hematite (corundum, ++−−) | 10 | 4×4×4 | Fe+O ONCV (NC) | 36 → 20 / 13 | 2274.9 s (17) | 87.1 s (8) | 26.1× | −0.013 | −13.2 | converged ✓ |
| Cr₂O₃ eskolaite (corundum, +−+−) | 10 | 4×4×4 | Cr+O ONCV (NC) | 36 → 16 / 13 | 1128.5 s (16) | 52.8 s (8) | 21.4× | −0.011 | −4.0 | converged ✓ |
| FeS₂ pyrite (Pa-3, diamagnetic) | 12 | 4×4×4 | Fe PAW + S USPP | — | — | — | — | — | — | unsupported ✗ |

† NiO's raw total carries the pre-#93 low-symmetry Ewald offset (dE_ewald =
−11.877 eV for this cell); the Ewald-removed ΔE_elec of −0.016 meV/atom is the
physics. The corundum cells trigger the same bug only weakly (dE_ewald = −0.132
eV hematite, −0.040 eV eskolaite), so their raw totals already sit at −13.2 and
−4.0 meV/atom. These SCF runs predate the #93 Ewald fix, so ΔE_elec is the
number to read across all three.

Attempted (USPP/PAW path): **FeS₂ pyrite** did not run. gradwave's UPF v2 reader
(`pseudo/upf.py::_read_root`, reused by `parse_upf_paw`) raises
`xml.etree.ElementTree.ParseError: junk after document element` on the Fe PAW
pseudopotential `Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`, and the PP_INFO-stripping
fallback does not recover it. The PAW free-text header is not parseable by the
current strict-XML path, so the run aborts at pseudopotential load before any QE
or gradwave SCF. This is a pseudopotential-parsing gap, not an SCF or physics
failure. Unsupported until the PAW UPF reader handles this header.

### GPU (asus, RTX 3050 6GB, fp64)

The wall times above are gradwave on CPU (8 torch threads). The same three SCFs
also run on the GPU with `--device cuda`, which moves the assembled system to the
card before the SCF (`system.to("cuda")`, setup stays on CPU). QE stays on CPU
MPI, so the GPU wall ratio compares CPU QE against GPU gradwave.

| mineral | GPU SCF (iters) | CPU SCF (iters) | GPU IBZ k | CPU IBZ k | ΔE_elec (meV/atom) |
|---------|:---------------:|:---------------:|:---------:|:---------:|:------------------:|
| NiO | 283.0 s (18) | 559.1 s (18) | 32 | 32 | −0.0155 |
| α-Fe₂O₃ hematite | 593.9 s (17) | 2274.9 s (17) | 13 | 20 | −0.0129 |
| Cr₂O₃ eskolaite | 670.2 s (16) | 1128.5 s (16) | 13 | 16 | −0.0110 |

NiO is the clean comparison, because its irreducible k-count is 32 in both runs.
The GPU SCF is 1.97 times faster than the 22-core CPU at the same 18 iterations,
and the total energy is unchanged to sub-0.001 meV/atom. The corundum runs fold
to 13 irreducible k-points, fewer than the 20 and 16 in the CPU table above,
because they postdate the #105 magnetic-fold fix. Their raw GPU walls are not a
like-for-like speedup, and normalizing by k-count the GPU gain is roughly 2.5×
(hematite) and 1.4× (eskolaite). A fair GPU column for those two needs the CPU
baseline re-run on current main.

fp64 runs at about 1/64 of fp32 on this consumer card, and the plane-wave SCF is
FFT-bound, so the GPU still beats 22 CPU cores by 1.4 to 2.5 times. Memory peaked
at 5.0 of 6.1 GB on the 10-atom cells with no out-of-memory, and the energies are
device-invariant, so the GPU path is a drop-in speedup for the forward SCF.

**Re-run after the Davidson batched-QR CPU-offload fix** (`_qr_offload` /
`_QR_CPU_MAX_COLS`, `solvers/davidson.py`, docs/manual/performance.md): hematite
GPU, same k-grid/cutoff/pseudo, `--device cuda --ranks 8 --threads 8`, converged
17 iterations, `ΔE_elec` −0.0129 meV/atom (unchanged):

| run | wall (iters) | QE wall (ranks) | wall ratio (gw÷QE) |
|---|:---:|:---:|:---:|
| pre-QR-fix (table above) | 593.9 s (17) | — | — |
| post-QR-fix | 586.8 s (17) | 76.3 s (8) | 7.69× |

Only a ~1% end-to-end gain, far short of the 1.5–1.7× the same fix gave the
diamond-C benchmark it was found on. A follow-up profiling pass
(`docs/manual/performance.md`, "Large-nb magnetic mineral: where GPU time
actually goes") explains why: hematite's Davidson rounds run at `nb=60`,
`npw≈6746`, well outside the `cols≤16`, `npw≤2500` regime the fix's own sweep
covered, and most of its QR calls land at `cols≈60` — above the safe
CPU-offload threshold, correctly, since a fresh sweep at hematite's actual
`npw` shows the offload is neutral-to-harmful there. The eigensolver's
per-round linear algebra (Rayleigh–Ritz subspace build and combination, plain
batched GEMM rather than a `cuSOLVER` factorization) is itself measurably
*slower* on this GPU than on the 22-core CPU at hematite's `nb`-driven subspace
width — a new, larger-scale expression of the same fp64-throughput limit the
QR/eigh overhead was one instance of, not a bug in the fix. The end-to-end GPU
run is still 3.9× faster than the unaffected CPU baseline (586.8 s vs
2274.9 s), because the FFT-heavy Hamiltonian apply and the moderate-subspace
`eigh` calls still favor the GPU; the win is just concentrated elsewhere than
this particular QR call for a system this size.

## Findings

The three antiferromagnetic oxides reproduce Quantum ESPRESSO's electronic total
energy to better than 0.02 meV/atom on the identical magnetic-primitive cell,
plane-wave cutoff, k-grid, smearing and PBE functional (NiO −0.016, hematite
−0.013, eskolaite −0.011 meV/atom). The agreement holds on the same FFT grid QE
reports, so the Kohn-Sham energy gradwave computes matches the mature Fortran
reference at the level the benchmark was built to test.

Quantum ESPRESSO runs the forward SCF 12.3× to 26.1× faster than gradwave on
these cells, which is the expected outcome for optimised Fortran plus MPI against
differentiable PyTorch. gradwave's value is exact gradients and inverse design on
top of a correct SCF, not raw forward-SCF speed, and the correctness is what this
benchmark establishes.

The collinear-magnetic (Shubnikov) fold cuts the irreducible k-count on every
real ordering here. NiO folds 112 unshifted mesh points to 32, matching QE's
magnetic-symmetry count of 32 exactly. On the corundum antiferromagnets gradwave
folds 36 points to 20 (hematite) and 16 (eskolaite) where QE reaches 13 in both,
so gradwave's magnetic IBZ is larger than QE's and gradwave integrates a few more
irreducible k-points. The total energies still agree to 0.013 meV/atom, so both
codes converge to the same physics on the same cell and gradwave's fold is
correct but less aggressive than QE's on the R-3c magnetic space group.

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
