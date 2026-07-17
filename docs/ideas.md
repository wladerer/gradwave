# Ideas and future work

A running backlog for gradwave, with enough reasoning attached that each item
can be picked up cold. Not commitments, just directions worth taking. The open
backlog comes first, then a "Done and resolved" section that keeps the reasoning
for items already built or settled.

# Open backlog

## Scaling up: RI and tensor hypercontraction

Framing first, because it reframes the GPU results in the done section. The
CheFSI no-go, the EOS-batching analysis, and the 128-atom memory cliff were all
measured on a consumer RTX 3050, whose fp64 throughput is a small fraction of its
fp32 and whose 6 GB caps the grid. Those numbers bound that card, not GPUs in
general. A datacenter card with real fp64 changes the CheFSI arithmetic story on
its own, before any code change. But the more durable lever is to cut the
operation count itself, which helps on the CPU path and on any GPU, and that is
what resolution of identity and tensor hypercontraction do. They are also the
enabling substrate for exact exchange, the biggest single physics gap in the
code, so the scaling work and the accuracy work are the same work.

### Resolution of identity (RI, density fitting)

RI expands products of orbitals in an auxiliary basis so a four-center
electron-repulsion object factorizes through two- and three-center intermediates.
In a plane-wave code the Hartree term is already O(N log N) through the FFT, so
RI is not a Hartree win. Where it pays is exact exchange. The Fock term is the
O(N^4) bottleneck that keeps hybrids out of the code, and RI on the orbital pair
densities `rho_ij(r) = psi_i*(r) psi_j(r)` is the standard route to make it
affordable. So RI is not a standalone feature, it is the substrate that makes the
biggest missing physics piece, hybrid functionals, tractable.

- The auxiliary representation. A plane-wave code already carries a complete
  auxiliary basis in the dense-grid plane waves, so a pair density is exact on the
  grid. The cost problem is the number of pairs, O(N^2) of them, each needing an
  FFT to Coulomb-couple. RI proper compresses that, and its plane-wave-native form
  is the ISDF factorization below.
- The differentiable angle. The fit is a linear solve against a metric, which is
  differentiable end to end, so a learnable hybrid could carry the exchange-mixing
  fraction and the range-separation length as trained parameters on top of an
  RI-compressed Fock build.

### Tensor hypercontraction and ISDF

Tensor hypercontraction factorizes the pair-product tensor into a small set of
interpolation points and interpolation vectors, so an object that is O(N^2) in
orbital pairs and O(N_grid) in real space collapses to O(N) points times a
compact factor. The plane-wave-native form is interpolative separable density
fitting (ISDF, Lu and Ying), which writes `psi_i(r) psi_j(r)` approximately as
`sum_mu zeta_mu(r) psi_i(r_mu) psi_j(r_mu)` over a small chosen point set
`{r_mu}`. A QR-pivoted or centroidal-Voronoi point selection makes the rank grow
like N rather than N^2.

- Why it is the right scaling lever. It cuts the FLOP count of the exchange and
  correlation builds directly, so unlike the CheFSI fp32 story it does not depend
  on a particular card's fp64 throughput. It helps on the CPU path and on any GPU.
  That is the durable answer to "scale up", attack the operation count, not only
  the hardware.
- What it unlocks. ISDF is the standard enabling technique for affordable exact
  exchange and RPA correlation in plane-wave codes (Qbox, PWDFT, and the ISDF-K
  line of work). With ISDF in place a hybrid functional and an RPA correlation
  energy both become reachable, which is the jump from a very well validated GGA
  code to one that does electronic structure GGA cannot.
- Build order. Land ISDF first as a compression of the pair densities with a
  QR-pivoted point selection, validate the compressed Fock exchange energy against
  a direct plane-wave Fock build on a small molecule to milli-eV, then layer the
  learnable-hybrid parameters and, separately, the RPA correlation contraction.
  Each stage is a set of tensor contractions, so it stays inside the
  differentiable-by-construction design.
- The honest caveat. ISDF has its own accuracy knob, the interpolation-point
  count, and the point selection is the subtle part. Budget the validation against
  direct Fock, not against another approximate method, and treat the rank as a
  convergence parameter reported alongside the result.

## Exact exchange and hybrid functionals

The biggest single physics gap, and the reason the two scaling items above are
worth building. Every energy, gap, force, and adsorbate level gradwave produces
sits on a GGA electronic structure with self-interaction error, so band gaps come
out too small and defect and adsorbate levels land in the wrong place. There is no
exact exchange anywhere in the SCF Hamiltonian today. A hybrid needs a Fock exchange operator applied
each SCF step, which is the O(N^4) object RI and ISDF exist to tame. The payoff
that no mainstream code has is a learnable hybrid, the mixing fraction and
range separation as trained parameters through the existing `learnable.py` slot,
which only makes sense once the Fock build is affordable. Sequence it after ISDF.

## Learned meta-GGA and the kinetic energy density

The learnable functional spans GGA form only, the two PBE parameters kappa and mu.
Every modern accurate semilocal functional (SCAN, r2SCAN) is meta-GGA, which means
it depends on the kinetic energy density `tau(r) = (1/2) Σ_i f_i |∇ψ_i(r)|²` on
top of rho and `|∇rho|²`. Without tau the learnable-XC path cannot fit, learn
against, or even compare with the functionals people actually use, so it cannot go
past GGA form. This is the natural next rung for the differentiable-XC work and the
one that lets `train_xc_paw` learn a real functional rather than only recover PBE.
It is also the cheaper stepping stone before hybrids, roughly a week against a
much larger EXX build.

- New piece, tau on the grid. Each occupied orbital's gradient is `i(k+G)` in
  reciprocal space, so `∇ψ_i` is one FFT per band per Cartesian direction, squared
  and accumulated with the occupations. This reuses the density-build FFT machinery
  with an extra factor of `i(k+G)`; the batched g-to-r path already carries the
  orbitals, so it is an added contraction, not a new solver.
- New piece, the meta-GGA potential. `v_tau = ∂e_xc/∂tau` does not act
  multiplicatively on rho. It enters the Hamiltonian as a tau-dependent
  modification of the kinetic term, `-∇·(v_tau ∇ψ)`, which makes this a generalized
  Kohn-Sham scheme and touches the H-apply, not just the functional. Autograd gives
  `∂e/∂tau` exactly the way it already gives `v_xc`, so no hand-derived kernel is
  needed, but the extra operator has to be wired into `BatchedHamiltonian.apply`
  and into the force and stress terms.
- Reuse, the functional interface. `XCFunctional.energy_density` gains a third
  argument `tau` beside rho and sigma, and the autograd `v_xc`/`f_xc` machinery, the
  spin channels, and the learnable-parameter graph all extend without new
  derivations.

Validate against QE `input_dft='scan'` (or r2SCAN) at pinned settings to the usual
milli-eV, then expose a learnable meta-GGA (an r2SCAN-form functional with
learnable parameters) and repeat the `train_xc_paw` recovery test at the meta-GGA
level. This is the item that most directly serves what makes gradwave distinct from
a very well-validated second copy of QE.

## Magnetocrystalline anisotropy (MAE maps) and per-atom spin torques

The constrained-moment work (`postscf/moment_config.py`) already produces one half
of this for free. `constrained_moment_scf` returns a per-atom transverse torque
`-dW/de_I`, validated to a finite difference at ratio 1.000 — that *is* the
magnetic force-theorem spin torque on each atom. Without spin-orbit coupling it is
the inter-atomic exchange torque (what drives the config search and sets a spin
spiral's stiffness); with a fully-relativistic pseudo the same per-atom torque
picks up the on-site anisotropy term. So "individual spin torques" is not a future
capability, it is what the module returns today. The missing half is the *global*
anisotropy: MAE maps `E(theta, phi)` over the magnetization sphere.

The ingredients are in the tree. The SOC path exists — `core/spinor_proj.py` builds
the j = l ± ½ resolved projectors and `SpinorHamiltonian` accepts them (`q`,
`dij_so`). `NCResult.energies.free_energy` gives a total energy per direction, so
`MAE = E(n1) - E(n2)` and a full surface are directly a direction sweep. The
efficient route is the torque method: one SOC evaluation per direction yields the
anisotropy torque `-dE/dn`, and integrating it over the sphere reconstructs the
surface — and that torque is the machinery we already have, applied to the total
moment instead of a local one.

Three things are genuinely in the way, and the third is the only real code.

- **Precision floor.** `test_noncollinear.py` pins rotation-invariance (MAE ≡ 0
  without SOC) to ~0.2 µeV, the numerical noise floor. Cubic Fe's MAE is ~1 µeV/atom,
  sitting right on it — reproducible only with great care. Start instead on a
  high-anisotropy case that clears the floor by orders of magnitude: L1_0 FePt
  (~1 meV/atom), hcp Co (~65 µeV), or a uniaxial 2D magnet.
- **k-convergence.** Metal MAE converges painfully slowly in k (thousands of points,
  or fine-smearing / Fermi-surface-aware tricks). A cost problem, not a capability
  gap, but it is the reason the force theorem matters.
- **No force-theorem path for SOC yet.** The standard cheap recipe — converge the
  density scalar-relativistically once, add SOC *non-self-consistently*, and take
  occupied-band-energy differences per direction — is not wired. The frozen-potential
  band-solve infrastructure already exists (`postscf/uspp_bands.py`, `core/gamma.py`,
  the one-shot solve in `postscf/hubbard_u.py`); it just is not connected to the SOC
  Hamiltonian plus a directional band sum. Without it, MAE falls back to a full
  self-consistent SOC SCF per direction: affordable for FePt-class anisotropy, too
  expensive and too noisy for Fe.

What to build, all reusing what is here: (1) a global spin-axis control (rotate all
local `e_I` together — a one-line special case of the per-atom constraint, or seed
and let SOC pin it); (2) a force-theorem evaluator that freezes the converged
density, adds the SOC block, and does one non-SCF diagonalization per direction into
`dE(n)`, reusing the frozen-potential solve and the spinor projector block; (3) a
thin sweep/integrate layer over `(theta, phi)` taking energy differences or
integrating the torque into the anisotropy surface. The blocker is not the math —
the torque is already exact and autograd-derived — it is getting a fully-relativistic
pseudopotential into the fixtures and writing that force-theorem loop so the map is
affordable.

## One-center ddd analytic derivative

The one-center ddd is a named micro-cost from the performance audit, 5% of the PAW
profile through an autograd backward per iteration. It is already compiled when
`compile_xc=True` (the `energy_and_ddd` path is a single backward), so the remaining
question is only whether an analytic quadrature derivative beats the compiled
autograd, which is a small isolated experiment, not a feature. (The local-TF metal
preconditioner that used to head this section is now built, see the done section.)

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
like magnetic surfaces and spin-polarized adsorbates. The SCF core itself is already
even here, the batched USPP/PAW eigensolve is validated at nspin=2 (O2 triplet,
batched vs per-k to 7e-12 eV, and 21 iterations to QE's 20), so the gaps are in the
postscf property layer, not the solver.

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

The cleanest first target is a spin-spiral / magnetic-dispersion sweep (see
`examples/fe_spin_spiral.py`). Every angle theta is the *identical* cell, k-mesh, and
band count -- same FFT dims, same tensor shapes -- so the batch has zero raggedness in
the data layout; only the per-point convergence count differs. That is a strictly
cleaner batching case than the EOS, where the cells (and their FFT boxes) vary slightly
with volume. The one wrinkle is the same one everywhere: the frustrated large-angle
points need many more iterations than the collinear ones, so a lockstep batched solve
either over-iterates the easy members or needs per-member convergence masking. The real
blocker is the hardware, not the workload -- on the RTX 3050 the sweep is fp64-bound and
7.8x slower than the CPU (it runs as concurrent CPU processes today, see the done
section on the measured 3050 profile). On a card with real fp64 (A100/H100, fp64 = 1/2
fp32) and tens of GB, stacking these identical independent SCFs to fill the device is
exactly where the batched-multi-structure path first pays off.

The best fit is GGA insulators. They are fixed-occupation, converge in few
iterations, and hold a small grid, so a single one badly underfills the card, which is
exactly the regime where stacking several into one padded solve wins. A batch of GGA
insulator structures is also the shape of a learned-XC training set and an EOS or
convergence sweep, so this feature and the meta-GGA training work reinforce each other.

## Gamma-only real wavefunctions for slabs and molecules

At the Gamma point the orbitals can be taken real, because time reversal makes
`ψ(-G) = ψ*(G)`, so only half the plane-wave sphere is independent. The foundation for
this is built and validated in `core/gamma.py`, gated to machine precision against the
complex path (apply 1e-13, frozen-potential eigenvalues 5e-14). It stores the half
sphere, runs the local term on `irfftn`/`rfftn`, and solves the eigenproblem as a real
symmetric one in a feature embedding where the half-sphere metric is the plain dot
product, so the standard Davidson applies unchanged.

The premise was a roughly 2x real-FFT win on the hottest kernel. That did not appear on
the available CPU. The forward-plus-inverse real transform measured 0.75x to 1.25x the
complex pair on non-power-of-two boxes at 63^3 and 72^3, so the H-apply came out 0.97x
in isolation, and the full solver ran slower still (directionally 0.6x to 0.8x) once the
per-apply overhead compounds over the Davidson iterations. The real-transform advantage
is grid-size and library dependent, and MKL did not deliver it here. The correctness is
solid, so the work that remains is measurement and integration rather than the core
representation.

- Re-measure on a GPU. cuFFT's real transform behaves differently from MKL's, and the
  memory story is also better on the GPU, so the win may exist there even though it does
  not on this CPU. This is the first thing to check before investing more.
- Wire it into the SCF loop behind a flag, for a single Gamma k-point, insulators and
  molecules first, then metals at Gamma with smeared occupations. The density build,
  mixing, and energy assembly are unchanged, only the diagonalize call swaps.
- The memory angle stands on its own. The real-space fields are half the size, so this
  pairs with the size-ceiling item below independent of any speedup.

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
tiling change and not an architecture change. The ISDF work above is the complementary
lever, it lowers the operation count where this item lowers the peak memory.

## Acceleration frontier, 2024-2026 literature sweep

A focused survey of the recent literature (done after the local-TF preconditioner
landed) for levers that pass the filter "single GPU or CPU, small FFT-bound cell,
fp64". Two of the sweep's headline ideas turned out to be already implemented: the
Gong and Dal Corso trick of batching the H-apply FFTs across all bands and k-points
into one call (arXiv:2412.01695, worth 6x on their small-cell many-k H-apply) is
exactly what `core/batch.py` already does over `(nk, nb, grid)`, and the CPU FFT is
already on MKL rather than pocketfft, so the "free 1.5-2x pocketfft to MKL" swap is
not available here. What remains, ranked by how well it fits this code:

MEASURED on the RTX 3050 (2026-07-16, torch.profiler on 8 NC SCF iterations, aten-op
device time, no kernel double-count). This revises the "FFT-bound" framing for the GPU
small-cell regime, which came from CPU profiles and the molecule-in-large-box / USPP-Pt
cases. For an ordinary small crystal on the GPU the FFT is only about 12 percent:

    Si8 2x2x2 (nband 20, m~40, box 27^3): GPU-busy 2111 ms, launch/sync gap 996 ms
      = 32% of wall.  GEMM(bmm) 43%, eigh 21%, QR/ortho 14%, FFT 12%, other 10%.
    Si2 4x4x4 (nband  8, m~16, box 20^3): GPU-busy  652 ms, launch/sync gap 559 ms
      = 46% of wall.  QR/ortho 44%, GEMM 23%, FFT 12%, eigh 11%, other 10%.

Two things fall out. First, a small-cell GPU SCF is dense-linear-algebra-bound, not
FFT-bound: GEMM + eigh + QR are about 78 percent of GPU-busy time (small boxes make the
FFT cheap, and fp64 GEMM/eigh/QR pay the same 1/64 fp64 tax). Second, the launch/sync
gap is 32-46 percent of wall (profiler-inflated but consistent with the earlier finding
that eager dispatch of dozens of tiny kernels per Davidson round is the binding GPU
constraint) - that gap is exactly what a whole-step CUDA graph reclaims. The eigh cliff
is visible: eigh 11 percent at m~16 vs 21 percent at m~40 (the n>32 cusolver-batched
fallback, measured 2.5-4.5x on its own). Reprioritized by this data: (1) whole-step CUDA
graph to close the 32-46 percent launch gap, (2) cut the dense subspace LA - RMM-DIIS is
now attractive because it removes the Rayleigh-Ritz (eigh) AND the subspace
orthonormalization (QR), together 35 percent (Si8) to 54 percent (Si2) of GPU-busy - and
a c64 subspace reduction on the NC standard problem would dodge both the fp64 tax and the
eigh cliff, (3) the FFT is no longer the thing to chase on GPU small cells.



- Whole-SCF-step CUDA-graph capture of the dispatch-bound glue. The measured GPU
  negatives so far were an apply-only CUDA graph (1.0-1.1x, the back-to-back FFT
  kernels have no launch gap) and torch.compile on the XC functional in isolation.
  Neither touched the 55-65 percent of a step that is many-tiny-kernel real-valued
  glue between the FFTs (XC assembly, mixing, occupations, PAW one-center, density
  build). Capturing the whole step as one CUDA graph (the PyGraph line,
  arXiv:2503.19779, averages 1.18x and never regresses where naive reduce-overhead
  degrades up to 32 percent) removes the per-kernel launch overhead across that glue,
  which is exactly where an 8-core host plus a consumer GPU hurt most. CUDA-graph
  capture, unlike torch.compile fullgraph, tolerates the complex FFTs (the earlier
  apply probe captured them fine), so the whole step is capturable. It cannot speed
  the FFTs themselves. Estimate 1.2-1.5x on the non-FFT fraction, GPU only, needs
  measuring on the RTX 3050. Highest-value new software lever.
- The batched `eigh` size cliff (diagnostic, cheap). `davidson_batched` calls
  `torch.linalg.eigh` on the `(nk, m, m)` subspace matrix with `m` about `2*nband`.
  On CUDA the fast `cusolverXsyevBatched` path is used only for `n <= 32`; above that
  PyTorch loops per-matrix (measured about 83x slower at the boundary, pytorch#175585).
  Every real system has `m > 32`, so the subspace diagonalization is probably on the
  slow per-k loop on the 3050. It is only about 5 percent of the CPU profile, but the
  cliff can inflate it on GPU. A ten-minute microbenchmark on asus settles whether it
  matters; if it does, cap or tile the subspace or split the batched solve.
- ML density initializer, plane-wave-native. "Global Plane Waves From Local
  Gaussians" (arXiv:2601.19966) and a transferability study (arXiv:2509.25724) report
  25-33 percent fewer SCF iterations, and show a density init transfers out of
  distribution where an ML-Hamiltonian init collapses. It only cuts iteration count,
  not per-iteration FFTs, so about a 1.3x ceiling on a single point, but it stacks
  with everything and its training set is the same shape as the learned-XC data. For
  MD and relaxation the cheaper analog is wavefunction/Grassmann extrapolation across
  geometries (about 3 iterations per step, JCTC 2022 1c00751), which QE and VASP
  already do and gradwave's warm-start approximates.

Skip, from the same sweep, because they do not transfer: distributed GPU eigensolvers
(ELPA, ChASE, SIRIUS all lose on small subspace matrices), ML Hamiltonian predictors
and learned preconditioners (they need a localized basis; our kinetic preconditioner
is already analytic), tensor-core FP16 FFT (accuracy-fatal against QE-grade fp64),
FP8-emulated fp64 FFT (Blackwell-only, no FP8 on Ampere), NUFFT (our grid is uniform),
and VkFFT (wins only at large-prime grids; `good_fft_size` restricts to 2*3*5*7
radices cuFFT already handles). RMM-DIIS is the one prototype-worthy eigensolver, it
removes the Rayleigh-Ritz that CheFSI could not, but the RR is cheap at small cell
size so the win is uncertain. The through-line matches the earlier audit: on a single
small SCF the consumer-GPU fp64 tax is the wall, and the durable levers are throughput
(batch many small structures), fewer iterations (learned or extrapolated start), and a
datacenter fp64 GPU.

# Done and resolved

Kept for the reasoning. Each of these is either landed in the code or settled as a
measured negative.

## RMM-DIIS solver and whole-step CUDA graph (both TRIED, measured negatives)

Prompted by the GPU profile above (dense-LA-bound, 32-46 percent launch gap), two of
the three levers it suggested were built and measured, and neither pays for small
cells.

RMM-DIIS (a `solvers/rmm_diis.py` prototype, since removed) replaces the block
Davidson's growing Rayleigh-Ritz subspace with per-band residual minimization, so it
has no per-round eigh and no m x m subspace GEMM - the 64 percent the profile flagged.
It needed two fixes to converge at all: a units-correct preconditioner (teter_b is a
dimensionless filter, right for Davidson subspace expansion but not for a direct Jacobi
step) and an exact line search (Teter-Payne preconditioned CG - a fixed step does not
converge). After both it converges on a FIXED operator (synthetic batched Hermitian, err
2e-11) but in about 100 iterations to the block Davidson's 22, at two H-applies per
iteration. In the real SCF it is worse than slow: on smeared fcc Al it hit the iteration
cap without converging and returned the wrong energy (-368 vs -1828 eV), at 1,548,800
band-applies against Davidson's 10,512 (147x). The reasons are exactly the textbook
ones: subspace methods converge in far fewer iterations, the SCF drives the solver with
a loose-early tolerance schedule that a residual method handles poorly, and a metal's
near-degenerate bands break the per-band tracking. RMM-DIIS is a large-system solver
(where the O(N^3) subspace eigh/GEMM finally dominates) and an MD warm-start refiner, not
a small-cell Davidson replacement - and the dense LA it removes, while 64 percent of GPU
time, is cheap in absolute terms at small cell size. Removed the prototype.

The whole-step CUDA graph is blocked upstream: `torch.linalg.eigh` is not CUDA-graph
capturable (it does a host-side info check), and it sits in the Davidson inner loop every
expansion round, so a whole-step capture fragments into tiny pieces around each eigh
rather than removing the launch gap. It genuinely needs the eigh out of the hot loop,
which was RMM-DIIS's job, and RMM-DIIS is not viable here. So the 32-46 percent launch
gap is real but not reclaimable by either lever without a solver that avoids eigh
altogether. Net: the measured GPU bottleneck resists these fixes because the eigh is both
cliff-hit and non-capturable and removing the subspace method costs convergence speed;
the durable levers stay throughput batching and a datacenter fp64 GPU.

## Local Thomas–Fermi metal preconditioner (DONE)

Landed as opt-in `precond="local_tf"` on both `scf` and `scf_uspp`
(`scf/local_tf.py`, default `"kerker"`). The bare Kerker filter screens charge
sloshing with a single length `1/q0`, right for a bulk metal but wrong for an
inhomogeneous cell, where a fixed `q0` over-screens the vacuum. Following QE's
`mixing_mode='local-TF'`, `LocalTFPrecond` lets the screening wavevector track the
local density, `q²(r)=min(q²_TF(r), q0_max²)` with `q²_TF=(4/π)k_F(r)`, capped at
the bare `q0` so a bulk metal is unchanged. It is applied by a short
preconditioned-CG solve of the screened-Poisson operator (a few box FFTs per
mixing step, warm-started across iterations), acting on the ρ-total block only.

Measured (NC, fcc Al, PBE, gaussian 0.1 eV): energies bit-identical to bare Kerker
(same fixed point). Bulk 8×8×8 neutral (9→9), Al(100) slab 21→17 (4 layers) and
27→21 (6 layers) iterations, the margin growing with cell inhomogeneity, exactly the
inhomogeneous regime the operator targets. So the original framing was right that a
fixed Kerker is the wrong operator away from a uniform bulk, but the win is on slabs
and molecules, not on a homogeneous bulk metal, where Kerker at a sensible `q0` is
already near-optimal. The bulk-Pt 16-vs-7 iteration gap is therefore a
starting-density and Broyden-history question more than a screening-length one, and
this preconditioner does not by itself close it. Unit tests pin the three operator
limits (`tests/unit/test_local_tf.py`), integration tests gate the fixed-point
invariant on NC and USPP (`tests/integration/test_local_tf_scf.py`), and the
slab iteration-count win lives in `benchmarks/bench_precond.py`.

Two follow-ups worth noting. First, building this surfaced and fixed a separate
bug: `setup_uspp` sized the FFT box as a blanket cube for any symmetric cell, so an
anisotropic slab got a 105³ box instead of 20×20×105, a 27.6× over-allocation that
OOMs during setup, now fixed by porting the NC path's symmetry-coupled axis grouping
(`symmetry.coupled_axis_groups`). Second, the modern parameter-free successor to
local-TF is the LDOS preconditioner of Herbst and Levitt (arXiv:2009.01665, DFTK's
default), which adapts the screening to whether each region is metallic or
insulating from the local density of states rather than a Thomas–Fermi model. If
local-TF ever underdelivers on a strongly mixed metal-vacuum-insulator cell, that is
the next rung, and it reuses the same reciprocal-space mixing hook.

## torch.compile for the exchange-correlation layer (DONE)

Landed as the opt-in `compile_xc` flag (`GradWave(compile_xc=True)` or
`xc.enable_compile()`). Measured 19x forward and 16x forward-plus-`v_xc` at 64³,
`v_xc` bit-accurate to 3e-16, with an eager fallback for the missing NixOS
toolchain. Compiled aot_autograd cannot double-backward, so the `f_xc` response
and HVP sites wrap their `xc.energy()` in `xc_eager()` to stay eager, which means
only the forward and first-order `v_xc` legs accelerate. Details in
`docs/manual/performance.md`.

The original analysis, kept for the reasoning. The compiler is dead on the complex,
FFT-bound Hamiltonian apply, which two earlier attempts already confirmed, but the
real-valued XC functional was never isolated and compiles well on a 64^3 grid. The
end-to-end effect on a plain SCF is only a few percent because XC is a minority of
runtime and its FFT-based gradient assembly does not compile, but learned-XC
training, the PAW one-center angular loop, and the `f_xc` response HVPs call the XC
transcendental chain far more than once per iteration and are CPU-bound, so those are
the real targets. Insertion point is the single `XCFunctional.energy_density` choke
point, opt-in with an eager fallback for the NixOS toolchain gap.

## Dual FFT grid (DONE)

Landed as commit `71a5265`, about 2x on the USPP/PAW H-apply FFT by running the
smooth wavefunctions on a coarse grid and the augmentation on the dense grid,
matching the audit spec.

## CheFSI, benchmarked no-go on the RTX 3050 (DONE)

Chebyshev-filtered subspace iteration is in `solvers/chebyshev.py`, unit-tested and
wired opt-in as `scf(..., eigensolver="chebyshev")` on the NC collinear path,
bit-identical to Davidson on the real NC SCF regression. The noncollinear spinor
twin was tried but left unwired, CheFSI converges too slowly on the dense metal
spinor spectrum (100-iteration cap vs Davidson's 18). The RTX 3050 fp32-deep
benchmark found it 2.5 to 5x slower than Davidson at every grid size that fits in 6
GB, up to 35^3. The fp32 FFT advantage there is only about 3.4x, not the 12x the
larger systems would need, and CheFSI does 2 to 3x more H-applies, so the filter
loses. It stays opt-in and off by default. Revisit on a bigger card where the grid
can grow into the regime where the fp32 FFT gain dominates, which is the same
hardware caveat the scaling section above opens with.

## Batched Davidson conditioning guard, cond-SVD removed (DONE)

The k-batched USPP/PAW generalized Davidson computed a full `linalg.cond` of the
subspace overlap every round on top of the `cholesky_ex` it already ran. Probing a
low-ecut Si PAW SCF (8, 10, 12 Ry) showed the overlap tips into non-PD, which
`cholesky_ex` flags with info>0, long before its condition number nears the 1e14
trip (max observed ~9e7), so the SVD never fired independently and was pure cost.
Removed it. Batched-vs-per-k equality (identical eigenpairs) and USPP/PAW-vs-QE
regression still pass, including nspin=2 PAW (O2 triplet, 7e-12 eV). Recorded in
`docs/manual/wisdom.md` under Eigensolvers.

## Extended-xyz trajectory output for relax (DONE)

`run_relax` accumulates an ASE frame per optimizer step with energy and forces
frozen on a `SinglePointCalculator`, and `run` writes them to `relax.xyz` (extxyz)
next to the JSON, re-readable in ovito or the ASE gui. The relax CLI now returns
exit 0 on normal completion, since reaching the ionic-step limit still yields a
valid trajectory, with convergence carried by the JSON `relax.converged` flag.
Regression in `tests/integration/test_io.py::test_relax_writes_extxyz_trajectory`.
MD does not have an output path yet, so the same frame accumulation extends there
once it lands.

## Atomic-orbital seeding for the initial wavefunctions (TRIED, no net gain)

The idea was to hand the first Davidson solve a superposition of pseudo-atomic
orbitals instead of bare lowest-kinetic plane waves. `scf/loop.py` builds `c0` as an
identity block on the first `nb` sphere entries, the smoothest plane waves and
nothing about the atoms, poor enough that the loop runs the first diagonalization at
a loose `1e-3` tolerance before tightening. QE's default instead projects the atomic
pseudo-wavefunctions onto the plane-wave basis (`startingwfc='atomic'`). All the
pieces existed in-tree, the `upf.pswfc`/`paw.chi` orbitals and the SBT-and-Ylm
projector build shared with the KB, Hubbard, and PDOS paths.

Built `lcao_seed` (per-k atomic-orbital block, QR-orthonormalized to 8e-15, padded
with plane waves past the orbital count) and wired it at the `c0` site. It reaches
the plane-wave-seeded energy to machine precision, as it must (NC O2 gives dF = 5e-12
eV, fcc Ni gives dF = 3e-11 eV). The predicted one-to-three iteration saving is real
but small (O2 goes 28 to 26 iterations, fcc Ni 6x6x6 goes 12 to 12), and the per-k
seed build costs enough that wall time came out neutral to slightly worse (Ni, 108 s
to 122 s). The reason is the one the prediction named. The loop already runs the
first diagonalization at a loose 1e-3 tolerance, so a crude plane-wave start
converges the cheap early eigensolves fine, and the total SCF count is set by density
mixing, not by initial-orbital quality. Reverted the wiring rather than add per-k
overhead to the default path for no measured gain. Recorded in
`docs/manual/wisdom.md` under SCF and mixing.

The remaining reason to revisit is that it composes with CheFSI, whose convergence
rate depends directly on how much of the wanted subspace is already in the start. A
Chebyshev filter fed atomic orbitals needs fewer rounds than one fed smooth plane
waves, so the pair should be measured together. That is the only configuration where
the seed cost might be repaid, and it is worth building `lcao_seed` back only
alongside a CheFSI-default benchmark that shows the compound win.
