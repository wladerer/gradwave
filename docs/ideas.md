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

The best fit is GGA insulators. They are fixed-occupation, converge in few
iterations, and hold a small grid, so a single one badly underfills the card, which is
exactly the regime where stacking several into one padded solve wins. A batch of GGA
insulator structures is also the shape of a learned-XC training set and an EOS or
convergence sweep, so this feature and the meta-GGA training work reinforce each other.

## Optimization audit (dual grid, CheFSI, and whether the architecture holds)

Full writeup in `docs/optimization-audit.md`. Status of the ordered conclusions: (1)
the dual grid is DONE, landed as commit `71a5265`, about 2x on the USPP/PAW H-apply
FFT, matching the spec; (2) CheFSI is DONE and BENCHMARKED, and the go/no-go came
back no-go on the RTX 3050. The solver is in `solvers/chebyshev.py`, unit-tested and
wired opt-in as `scf(..., eigensolver="chebyshev")` for the NC and noncollinear
paths, bit-identical to Davidson on the real NC SCF regression. But the RTX 3050
fp32-deep benchmark
found it 2.5 to 5x SLOWER than Davidson at every grid size that fits in 6 GB, up to a
35^3 grid. The fp32 FFT advantage there is only about 3.4x, not the 12x the larger
systems would need, and CheFSI does 2 to 3x more H-applies, so the filter loses. It
stays opt-in and off by default. Revisit only on a bigger card where the grid can
grow into the regime where the fp32 FFT gain dominates; (3) two profile-visible
micro-costs remain, the per-round
`linalg.cond` SVD in the batched Davidson guard (the per-k path already catches
Cholesky failures instead) and the autograd one-center ddd; (4) a Kerker-plus-local-TF
metal preconditioner, the untried half of the 16-vs-7 iteration gap and now the
highest-value remaining perf item, since it is a 2.3x on every metal and the dual grid
has already taken the FFT win. A real-space rewrite is not warranted, with an explicit
go/no-go test in the doc.

## Atomic-orbital seeding for the initial wavefunctions

The SCF starts from a superposition-of-atomic-densities guess for the density, which is
fine and is what QE does too, but the initial *wavefunctions* handed to the first
Davidson solve are bare lowest-kinetic plane waves. `scf/loop.py` builds `c0` as an
identity block on the first `nb` sphere entries, which are ordered by `|k+G|²`, so the
starting subspace is the `nb` smoothest plane waves and nothing about the atoms. That
guess is poor enough that the loop deliberately runs the first diagonalization at a
loose `1e-3` tolerance (`loop.py:533`) before tightening. QE's default instead projects
the atomic pseudo-wavefunctions onto the plane-wave basis (`startingwfc='atomic'`),
which starts the eigensolver much closer to the occupied manifold.

We already have every piece to do the same. The atomic orbitals are parsed and in-tree,
`upf.pswfc` for norm-conserving and `paw.chi` for PAW, both `AtomicOrbital(l, label,
rchi=r·R_nl)`, with the `_species_orbitals` helper in `postscf/pdos.py` that already
pulls them per species. The projector that maps a radial orbital onto the plane-wave
basis is the same structure used everywhere else, `(4π/√Ω)(−i)^l Y_lm(k+G) F(|k+G|)
e^{−i(k+G)τ}` with `F` the spherical Bessel transform of `rχ·r`, identical to the KB,
Hubbard, and PDOS projector builds. So this is assembly of existing parts, not new
physics.

- New piece, small. Build the initial `(nk, nb, npw)` block from the atomic orbitals:
  stack the per-atom `|l, m⟩` projectors up to `nb` columns, pad with the current
  lowest-plane-wave columns when the atomic set is smaller than `nb` (the QE
  `atomic+random` fallback), and orthonormalize. One function next to `sad_density` in
  `scf/guess.py`, wired at the `c0` construction site in `loop.py`.
- Reuse, everything else. The SBT and Ylm projector machinery, the species-orbital
  helper, and the truncation conventions (the msh-at-10-bohr atomic-wfc cutoff that the
  +U path already honors) all carry over unchanged.

The honest magnitude. This mostly helps the first SCF iteration, since every later
iteration already warm-starts its orbitals from the previous one, so expect roughly one
to three iterations saved on a cold solve rather than a large fraction of the run. It
does not close the metal iteration gap against QE, which is a mixer and preconditioner
question, not a starting-orbital one. The reason to do it anyway is that it composes
with CheFSI, whose convergence rate depends directly on how much of the wanted subspace
is already present in the start. A Chebyshev filter fed atomic orbitals needs fewer
rounds than one fed smooth plane waves, so atomic seeding and CheFSI compound. Build it
alongside the CheFSI benchmark and measure the pair. Estimate about a day, almost all of it validation that the seeded
solve reaches the same converged energy as the plane-wave-seeded one, which it must to
machine precision since the guess only sets the starting point.

## Learned meta-GGA and the kinetic energy density

The learnable functional spans GGA form only, the two PBE parameters kappa and mu. Every
modern accurate semilocal functional (SCAN, r2SCAN) is meta-GGA, which means it depends
on the kinetic energy density `tau(r) = (1/2) Σ_i f_i |∇ψ_i(r)|²` on top of rho and
`|∇rho|²`. Without tau the learnable-XC path cannot fit, learn against, or even compare
with the functionals people actually use, so it cannot go past GGA form. This is the
natural next rung for the differentiable-XC work and the one that lets `train_xc_paw`
learn a real functional rather than only recover PBE.

- New piece, tau on the grid. Each occupied orbital's gradient is `i(k+G)` in reciprocal
  space, so `∇ψ_i` is one FFT per band per Cartesian direction, squared and accumulated
  with the occupations. This reuses the density-build FFT machinery with an extra factor
  of `i(k+G)`; the batched g-to-r path already carries the orbitals, so it is an added
  contraction, not a new solver.
- New piece, the meta-GGA potential. `v_tau = ∂e_xc/∂tau` does not act multiplicatively
  on rho. It enters the Hamiltonian as a tau-dependent modification of the kinetic term,
  `-∇·(v_tau ∇ψ)`, which makes this a generalized Kohn-Sham scheme and touches the
  H-apply, not just the functional. Autograd gives `∂e/∂tau` exactly the way it already
  gives `v_xc`, so no hand-derived kernel is needed, but the extra operator has to be
  wired into `BatchedHamiltonian.apply` and into the force and stress terms.
- Reuse, the functional interface. `XCFunctional.energy_density` gains a third argument
  `tau` beside rho and sigma; the autograd `v_xc`/`f_xc` machinery, the spin channels,
  and the learnable-parameter graph all extend without new derivations.

Validate against QE `input_dft='scan'` (or r2SCAN) at pinned settings to the usual
milli-eV, then expose a learnable meta-GGA (an r2SCAN-form functional with learnable
parameters) and repeat the `train_xc_paw` recovery test at the meta-GGA level. Estimate
about a week, most of it the generalized-KS potential in the H-apply and the matching
force and stress terms, which are the parts that are genuinely new rather than a threaded
argument. This is the item that most directly serves what makes gradwave distinct from a
very well-validated second copy of QE.

## Gamma-only real wavefunctions for slabs and molecules

At the Gamma point the orbitals can be taken real, because time reversal makes
`ψ(-G) = ψ*(G)`, so only half the plane-wave sphere is independent. A Gamma-specialized
path stores that half sphere and runs the H-apply on a real-to-complex FFT, which is
about 2x on the single hottest kernel, with the subspace algebra real rather than
complex on top of that. The performance notes deferred this because for a general
many-k run it caps the end-to-end gain at roughly 1.3 to 1.5x for the most invasive
change in the stack. The reason to build it now is the workload: slabs and molecules are
sampled at Gamma alone by construction, so for exactly those systems the Gamma path is
not a special case, it is the whole calculation, and the invasive change touches the one
k-point that matters.

- New piece, a real-wavefunction representation at Gamma. Impose `ψ(G) = ψ*(-G)`, store
  the independent half sphere with the `G=0` component real, and run the H-apply with
  `rfftn`/`irfftn`. The local potential multiply and the projector contractions carry
  over with the reality constraint; the subspace eigensolve becomes a real symmetric one.
- Reuse, the solver structure. Davidson and CheFSI both work unchanged in real
  arithmetic; CheFSI in particular gets cheaper, since a real filter halves the FFT and
  the arithmetic together, so Gamma-only slabs are where CheFSI and this specialization
  compound.

Scope it to a single Gamma k-point first, insulators and molecules, then metals at Gamma
with smeared occupations. The gain is 1.3 to 1.5x on the H-apply-dominated molecular
and slab SCF, which is the regime the RAIRS and surface-chemistry work lives in, so it
pairs with those. It is a real project, not a tweak, because it changes the core
wavefunction representation, so land it behind a flag and gate it against the complex
Gamma path to machine precision.

## Raising the system-size ceiling past the dense-allocation cliff

The GPU probe found peak memory scaling roughly linearly to about 96 atoms on the 6 GB
RTX 3050, then a hard cliff at 128 atoms from a single roughly 37 GB allocation, an
O(npw²) dense step (complex128 around 7.7 GB times the eigh workspace copies) that spikes
at `npw` near 22k. So the practical ceiling is about 96 to 110 atoms at that cutoff,
and the cliff is a specific dense allocation, not gradual fill, which means it is
tileable rather than fundamental.

- Identify the O(npw²) step. It is the dense object that scales with the square of the
  plane-wave count, most likely a subspace-related workspace or the eigensolve's internal
  copies, and the first task is to confirm which allocation trips at 128 atoms with a
  memory profile.
- Tile or avoid forming it. Block the offending contraction so the peak is bounded the
  way `BatchedHamiltonian.apply` and `density_b` already band-chunk their dense-grid
  temporaries, or restructure the step to never materialize the full O(npw²) array.

This only matters if larger cells become a goal, defects, bigger slabs, or supercells for
finite-q phonons, so it is a when-you-need-it item rather than a now item. But it is the
one thing standing between the current sub-100-atom validation regime and running the
kind of system where the code would do new science, so it is worth knowing the fix is a
tiling change and not an architecture change.

## torch.compile for the exchange-correlation layer

Landed as the opt-in `compile_xc` flag (`GradWave(compile_xc=True)` or
`xc.enable_compile()`). Measured 19x forward and 16x forward-plus-`v_xc` at 64³,
`v_xc` bit-accurate to 3e-16, with an eager fallback for the missing NixOS
toolchain. Compiled aot_autograd cannot double-backward, so the `f_xc` response
and HVP sites wrap their `xc.energy()` in `xc_eager()` to stay eager, which means
only the forward and first-order `v_xc` legs accelerate. Details in
`docs/torch-compile.md` and `docs/manual/performance.md`. The backlog item below
is the original analysis, kept for the reasoning.

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
