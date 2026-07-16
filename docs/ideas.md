# Ideas and future work

A running backlog for gradwave, with enough reasoning attached that each item
can be picked up cold. Not commitments, just directions worth taking, ordered
roughly by how ready they are to build.

## RAIRS and a slab dipole moment

We can do vibrational frequencies now (`postscf/phonons.py`, validated against QE
`ph.x` to 0.003% on Si) and IR intensities for insulators and molecules (Born
charges and epsilon-infinity from `postscf/dielectric.py`). Metals are the gap.

The current `dielectric_born` refuses anything but an nspin=1 insulator, because it
splits valence from conduction with a conduction projector `P_c` and a `(H - eps_v)`
solve, and that construction goes singular at a metal Fermi level. A bulk metal also
has no IR-active optical phonon in the insulator sense and no static Born charge, so
"bulk metal IR" is not a real target.

The real target is RAIRS, the reflection-absorption IR of an adsorbate on a metal
surface (CO on Pt is the textbook case). The metal surface selection rule says only
the dynamic dipole perpendicular to the surface couples, and the slab surface-normal
dipole `mu_z` is well defined despite the metal because the vacuum gap gives a clean
reference. So we sidestep the singular DFPT entirely and finite-difference.

- New piece, a slab dipole-moment function `mu_z = integral z rho_tot(r) + ionic`,
  with the standard slab caveat that it needs a vacuum gap and a dipole correction
  so the two surfaces do not talk through the cell. This is the only genuinely new
  physics. gradwave already has rho on the grid and the ion charges, so it is modest.
- Reuse, finite-difference the dynamic dipole. Displace each adsorbate atom by
  plus/minus delta, compute `mu_z`, difference to get `Ztilde_{s,zbeta} = d mu_z / d tau`.
  For CO that is 2 atoms x 3 directions x 2 = 12 SCFs. Contract with the CO-projected
  Hessian modes from `gamma_hessian` and keep the z component.

Cost is finite-difference, roughly 12 extra slab SCFs on top of the Hessian, no DFPT.
It works on the metal because it uses `mu_z`, not `Z*`. Estimate about 1 to 2 days,
almost all of it in the slab dipole routine and its validation.

Raman on a metal stays hard. The Raman tensor is `d alpha / d Q` and alpha itself is
ill-defined for a metal, and surface-enhanced Raman is dominated by electromagnetic
field enhancement rather than a clean DFT observable. Leave it.

If someone wants the metal's own far-IR response, that is a different deliverable,
the optical conductivity `sigma(omega)` (Drude plus interband). It extends the
existing E-field Sternheimer to finite frequency and adds the intraband
Fermi-surface term `dielectric.py` omits. Roughly a week, and it produces optical
conductivity, not a vibrational spectrum.

## Little and orbit groups for DFPT under symmetry-breaking perturbations

The response calculations either use the full crystal symmetry or drop to
time-reversal only. A perturbation lowers the symmetry to its little group, and we
should reduce k-sampling and irreducible displacements by that residual group
instead of discarding symmetry outright.

- For a Gamma-phonon column, the displaced-atom pattern has a little group, the site
  symmetry intersected with the displacement direction. Only symmetry-inequivalent
  columns need computing, and the rest are reconstructed by the group action (the
  `HessianSymmetry` reconstruction already does the reconstruction half for the full
  group, this generalizes it to the perturbation little group).
- For the E-field response, the little group is the subgroup that leaves the field
  vector invariant, and k reduces to that subgroup's IBZ rather than to
  time-reversal only.

Payoff is direct k-point and displacement-column savings on exactly the expensive
response runs, which is what QE `ph.x` does with its `modes_of_q` and small-group
machinery. The building blocks (`find_spacegroup`, `reduce_mesh`) exist, the work is
computing the little group of a given perturbation and threading it into the DFPT
drivers.

## Phonon band structures

`gamma_hessian` is Gamma-only. Extend to finite q to get dispersions.

- Real-space force constants from finite displacements in a supercell, or an
  analytic force response at q with a q-dependent perturbation. The supercell route
  is simpler to land first.
- Fourier interpolate `D(q)` onto a band path. Reuse the electronic bands path
  builder for the q-path and labels.
- Acoustic sum rule, and for polar insulators the nonanalytic LO-TO term at q to
  Gamma, which needs Born charges and epsilon-infinity. Both already exist in
  `dielectric.py`, so the polar correction is reachable.

With a q-mesh this also gives the phonon DOS and the harmonic thermodynamics (free
energy, entropy, heat capacity), which pairs well with the EOS work for a full
thermal equation of state.

## Full nspin=2 and PAW coverage for every feature

Coverage is uneven across the postscf features. Several are nspin=1 or NC only.
`dielectric_born` is nspin=1 insulators, parts of the discretization-error force path
are NC nspin=1, and the noncollinear and SOC PDOS paths have their own constraints.

Make an explicit matrix of feature x {NC, USPP/PAW} x {nspin=1, 2} and close the
gaps. Most of the per-channel machinery exists, so the work is threading the spin
index and the S-metric or augmentation consistently, plus tests at each new cell of
the matrix. Unglamorous, but it is what makes the code trustworthy on real systems
like magnetic surfaces and spin-polarized adsorbates.

## Trajectory and extended-xyz output for optimizations

Relaxations and MD should write an extxyz trajectory next to the JSON, one frame per
step with positions and the energy and forces in the comment line. ASE writes extxyz
natively, so this is small. It makes trajectories viewable in ovito or the ASE gui
and re-loadable for restart or analysis. Add it to the relax and MD output path in
`api.py` / `output.py`.

## Batched multi-structure SCF, and the EOS-on-GPU question

Question, would an EOS go faster by batching several volumes on the GPU at once?

Measured on the asus RTX 3050 for the 1-atom fcc Pt EOS (40/400 Ry, 12x12x12), a
single point sits at 100% nvidia-smi util but only 24.7 W of draw (the card's TGP is
35 to 80 W) and 2.6 of 6 GB. The 100% util flag only means a kernel was in flight
during the sample. The low power and low memory say the GPU is not compute-saturated.
For a system this small it is launch and latency bound on many tiny kernels (small
matmuls, a 35^3 FFT, per-k Davidson steps), so there is real headroom. So yes,
concurrency would help here. Three ways, cheapest first.

- Run several volumes as concurrent processes sharing the GPU, either plain
  backgrounding or CUDA MPS. Zero code. Two points fit in 6 GB (2 x 2.6). Likely
  1.5 to 1.8x on a launch-bound system. The catch is that the current EOS chains the
  volumes with `start_from` warm starts, so they are serial by construction. Dropping
  the chain trades the warm-start iteration savings for the concurrency, which is
  close to a wash at N=2 but wins as the GPU empties.
- Batched multi-structure SCF, the real structural win. Stack the volumes as
  independent k-blocks in one padded generalized Davidson, the same way the batched
  Davidson already stacks k-points, so the small per-volume GEMMs become one big GEMM
  and the kernel-launch overhead amortizes. The SCF loop has to carry per-volume
  densities and potentials and mix them independently while sharing the linear
  algebra, which is a genuine feature, not a tweak. This is the version that would
  actually fill the card. It generalizes past EOS to any embarrassingly-parallel set
  of small structures (displacement stencils for phonons, rattled configs for
  training data, a k-convergence sweep).
- CUDA streams to overlap independent kernels. Hard to orchestrate from PyTorch
  eager, low priority.

Note that this only pays for small systems where a single SCF underfills the GPU. The
slab already uses more of the card, so batch structures for the cheap cases (bulk
EOS, phonon stencils) and run the heavy cases one at a time.

## torch.compile for the exchange-correlation layer

Landed as the opt-in `compile_xc` flag (`GradWave(compile_xc=True)` or
`xc.enable_compile()`). Measured 19x forward and 16x forward-plus-`v_xc` at 64³,
`v_xc` bit-accurate to 3e-16, with an eager fallback and automatic double-backward
routing so response and HVP code stays correct. Details in `docs/torch-compile.md`
and `docs/manual/performance.md`. The remaining backlog item below is the original
analysis, kept for the reasoning.

Full analysis in `docs/torch-compile.md`. The one-line version: the compiler is dead
on the complex, FFT-bound Hamiltonian apply, which two earlier attempts already
confirmed, but the real-valued XC functional was never isolated and compiles to 8x
forward and about 30x forward-plus-`v_xc` on a 64^3 grid, bit-accurate to 5e-15. The
end-to-end effect on a plain SCF is only a few percent because XC is a minority of
runtime and its FFT-based gradient assembly does not compile, but learned-XC training,
the PAW one-center angular loop, and the `f_xc` response HVPs call the XC
transcendental chain far more than once per iteration and are CPU-bound, so those are
the real targets. Insertion point is the single `XCFunctional.energy_density` choke
point, opt-in with an eager fallback for the NixOS toolchain gap.
