# Wisdom

Things that cost real time to learn and are not obvious from the code. Each entry is
a rule with the symptom that motivates it. If what you are seeing matches a symptom
below, trust the entry before re-deriving it. The clearest example is that the
small-system speed gap is fp64 precision and kernel maturity, not an architectural
defect, which the [Performance](performance.md) page works through in full.

## Conventions and units

- Keep one FFT convention, decided once in `core/fftbox.py`. Wavefunctions are
  $\psi(r) = \Omega^{-1/2} \sum_G c(G)\, e^{i(k+G)r}$ with unit coefficient norm.
  Fields are Fourier series, $\tilde f(G) = \mathrm{fftn}(f)/N$. Real-space integrals
  are $\int fg = \Omega \sum_G \tilde f^* \tilde g$. Most normalization bugs come from
  mixing these two families.
- Grid gradients from autograd carry a volume element. A derivative with respect to
  grid density values equals the functional derivative times $\Omega/N$, and a second
  backward carries the factor once more. Apply that conversion once, at the final
  pairing, never per term. The hardest USPP adjoint bug was converting one pairing but
  not its composite partner, which silently broke the becsum block.
- torch returns conjugate Wirtinger gradients for a real scalar of complex inputs.
  Every explicitly implemented backward must match that convention. It is documented in
  `fftbox.py` and nowhere else on purpose.
- Everything is float64 or complex128, always explicit. `torch.tensor(x)` defaults to
  float32 and once broke Fermi-level bisection.

## Pseudopotentials and UPF parsing

- Respect the UPF unit traps. PP_DIJ and PP_LOCAL are in Ry, the mesh is in Bohr,
  PP_BETA stores $r\beta(r)$, and PP_RHOATOM stores $4\pi r^2 \rho(r)$. Respect the
  per-projector kkbeta cutoffs, since SG15 hard-truncates.
- Truncate local-channel radial integrals exactly where the reference code does. QE
  truncates all of them at 10 bohr. psl meshes run far past that with tails that
  deviate from $-Z/r$ at the 2e-5 eV level, which becomes a rigid eigenvalue offset
  times the electron count (132 meV/atom for Ni) unless you truncate identically. The
  same truncation is load-bearing in the local potential, the atomic densities, and
  the atomic wavefunctions for +U.
- For PAW +U, use the raw PP_PSWFC amplitudes. PAW pseudo-orbitals are not norm-one by
  design, since the overlap supplies the rest, and renormalizing them is a 100 meV
  class error.
- Take the S-operator weights from the same radial integrals as the augmentation
  tables. PP_Q and $\int q^0 dr$ disagree at file precision (5e-8), and if the weights
  come from a different integral than the tables, charge conservation breaks by exactly
  that mismatch, which the DIIS metric then amplifies through its $1/q_0^2$ weight at
  $G=0$.
- Know your pseudo's contents. SG15 carries no PP_PSWFC, PseudoDojo tarballs do.
  PseudoDojo fully-relativistic pseudos have NLCC and are unsupported on the spin-orbit
  path, while SG15 fully-relativistic works.

## Grids

- Use one fixed FFT grid per material for an EOS scan, the elementwise max over the
  volumes. A volume-varying minimal box steps E(V) inconsistently in any code that does
  it.
- Pin the reference grid dimensions in QE comparisons. Two valid boxes that both hold
  the density sphere give XC energies differing at the meV level for sharp semicore
  densities.
- Make non-symmorphic translations commensurate with the grid. The diamond glide
  $(\tfrac14,\tfrac14,\tfrac14)$ needs dimensions divisible by 4. On an incommensurate
  box the un-projected full-mesh fixed point genuinely carries a 2e-4 asymmetric density
  component, so an IBZ-versus-full comparison fails at 1e-4 with no bug present. QE's
  ph.x enforces the same constraint harder.
- Mask the symmetrizer to the density sphere. Box-Nyquist folding is ill-defined for
  glide phases.
- A hard USPP/PAW pseudo wastes the wavefunction FFT on the dense grid. gradwave sizes
  one FFT box from `ecutrho` and runs the Davidson H-apply (`g_to_r`/`r_to_g` per band,
  per k, per subspace vector, the hottest kernel) on it. For Pt at 40/400 Ry that box is
  35^3 where the wavefunctions only need the smooth box 21^3 that holds their products,
  2*Gmax(ecutwfc). That is 4.6x too many points on the single most-called transform.
  Norm-conserving does not have the problem, its `ecutrho = 4*ecutwfc` box already is the
  smooth box, so this is USPP/PAW only.
- The dual grid is exact, not an approximation. For the local term
  `<psi_i|V|psi_j>` only the smooth part of V contributes, because any two wavefunction
  G-vectors differ by at most `2*Gmax(ecutwfc)`, so V truncated to the smooth sphere
  reproduces `H|psi>` to round-off (verified, relative error 6e-16). Implemented for the
  batched USPP H-apply local term (`setup_uspp` builds the smooth box and a per-sphere
  `flat_idx` into it, aligned to the dense spheres because `build_gsphere` gives the same
  Miller ordering, and the loop filters `v_eff` to the smooth box each iteration through a
  precomputed smooth-to-dense Miller index map). The augmentation, the density build, the
  one-center work, and Hartree/XC stay on the dense grid, so the density-build FFT is not
  yet dual-gridded (a follow-up). Validated two ways, the batched path matches the dense
  per-k reference to 2e-13 eV on Al, and the fcc Pt free energy is bit-identical before
  and after. Measured on Pt at 40/400 Ry it halves the FFT time (34 percent of the SCF)
  for about 1.2x, more on harder pseudos and less on softer ones because the gain scales
  with `ecutrho/ecutwfc`.
- A real FFT is not automatically faster than a complex one. The Gamma-real path
  (`core/gamma.py`) stores the half sphere and runs the H-apply on `irfftn`/`rfftn`,
  which in theory halves the hottest kernel. Measured on this CPU (MKL, non-power-of-two
  boxes) the forward-plus-inverse real transform ran 0.75x to 1.25x the complex pair
  across 63^3 and 72^3 at 8 and 24 bands, so the H-apply was 0.97x in isolation,
  neutral to slightly slower. The full Davidson solver ran slower still (0.6x to 0.8x
  under CPU contention, so directional rather than precise) because the per-apply embed
  and full-sphere reconstruction overhead compounds over the iterations. The
  real-transform advantage is grid-size and library dependent and did not appear here.
  The path is correct to machine precision and stands as the substrate for a GPU (cuFFT)
  re-measure and for the memory-ceiling angle, where the real-space fields are half the
  size, but it is not a CPU speedup as built.

## Eigensolvers

- Re-orthonormalize the reused Ritz block on a Davidson restart. Drift compounds at
  tight tolerance into eV-scale energy jumps, and it manifests on CUDA first while staying
  latent on CPU.
- For a complex generalized problem, form $L^{-1} H L^{-\dagger}$, not $L^{-1} H
  L^{-1}$. The error is invisible on real matrices and produces Ritz values below the
  true minimum. That fingerprint means a broken reduction, not a Davidson problem.
- Guard the indefinite overlap. The USPP S is indefinite on truncated spheres at low
  cutoff, and QE errors identically. The batched solver must mirror the per-k path's
  drop-oldest conditioning guard, whose Cholesky failures are silently caught, and never
  contract to the contaminated Ritz block.
- Do not add a condition-number check to that guard. The `cholesky_ex` info flag already
  catches the non-PD overlap that produces the below-minimum Ritz values, and it fires
  before a still-PD overlap ever gets near-singular. Probing low-cutoff Si PAW (8 to 12
  Ry), the overlap condition number stayed below 1e8 while the factorization failure
  triggered, so a per-round `linalg.cond` SVD never fired independently and was pure cost.
- Build the preconditioner from positive band kinetic expectations. Eigenvalues go
  negative below the Fermi level, the clamp makes near-zero rows, and rank-safety jitter
  then replaces physical rows with noise.
- Put rank-safety jitter only on near-zero-after-projection rows, and normalize
  expansion rows before orthonormalization. Otherwise near-converged residuals floor
  around 1e-8, whose signature is exact energies with `converged=False` for 60
  iterations.
- Do not feed a walked-off warm start to the mixer. A warm-started Davidson can
  deterministically diverge from a converged-quality density, with 100 eV
  energy jumps and a frozen residual that mixer resets do not touch. The rescue is
  discarding the warm start and re-solving from salted seeds without feeding the garbage
  to the mixer.

## SCF and mixing

- Mix the composite (density, becsum) pair for USPP/PAW. Mixing becsum outside the
  Pulay vector produces a gain-per-iteration charge oscillation on semicore-metal PAW.
- Kerker damps the density-total block only. A per-channel Kerker with a $G=0$ zero
  freezes interspin charge transfer and makes spin SCFs diverge, so spin mixing runs in the
  (total, magnetization) basis.
- Solve DIIS in the diagonally-normalized basis with scale-invariant Tikhonov.
  Regularization scaled to the raw matrix swamps the newest small-residual entries and
  stalls the tail. On ill-conditioning drop the oldest entry, since a full reset
  discards curvature the next steps need.
- Give trust-region resets a windowed baseline and a cooldown. An all-time best residual
  locks at the eigensolver noise floor, after which every iteration resets the mixer,
  DIIS never re-learns, and pure damped iteration diverges geometrically for the
  gain-above-one modes.
- Gate convergence on the energy tail. QE's criterion is an energy criterion, where the
  error scales as the residual squared, so a density threshold 100 to 1000 times tighter
  explains most of an apparent iteration gap before any mixer difference. For smeared
  metals the density residual floors at occupation noise while the free energy is long
  settled, and the flag should say so honestly.
- Treat ferromagnetic metals near the Stoner instability as the adversarial case. The
  map has a measured gain near $-6$ on the spin mode, and its consequences were each
  learned separately. Default damping collapses the moment to the nonmagnetic branch
  silently. Manual damping converges slowly. Scalar adaptive step controllers either
  over-damp permanently or ride the stability boundary. Plain unweighted Broyden diverges
  because early garbage secant pairs poison the inverse Jacobian. Johnson's weights, the
  normalization plus the w0 regularization, are the load-bearing part of QE's mixer, not
  a refinement.
- Do not expect the mixer to select the physical branch. The nonmagnetic state is a
  genuine stationary point tens of meV away, and every code can land on it. Warm-start
  chains across scan points are the practical defense, plus an explicit moment gate as a
  detector.
- Prefer preconditioning to step-size control. The Stoner-expansive mode needs an
  operator, the $\chi_0$-diagonal preconditioner whose ingredients are the Fermi-surface
  adjoint terms, not a schedule. Adaptive damping line searches match good fixed damping
  at best and fail on transition metals.
- Re-audit every crutch when you change mixers. A stabilizer tuned for one mixer slows convergence under a better one. The 0.4 becsum step scale that suppressed the on-site
  becsum-ddd mode for Pulay cost Johnson eleven iterations on ferromagnetic Ni. QE mixes
  becsum unscaled, and matching that closed the bulk of the remaining iteration gap.
- A better initial wavefunction does not cut the SCF iteration count. Seeding the first
  Davidson from atomic orbitals instead of plane waves reaches the same energy to machine
  precision but saves at most a couple of iterations (O2 28 to 26, fcc Ni 12 to 12), and
  the per-k build cost makes wall time neutral to worse. The count is set by density
  mixing, and the first diagonalization already runs at a loose 1e-3 tolerance, so the
  better start is masked. Atomic seeding is worth revisiting only paired with CheFSI,
  whose rate depends on how much of the wanted subspace is already in the start.

## Metals and smearing

- Compare the free energy. QE's printed smeared "total energy" is the free energy F.
- Keep occupation and entropy as scheme-paired objects. Mixing one scheme's occupation
  with another's entropy is a classic silent error.
- Assert that fractional occupations exist before testing metallic physics. A coarse
  k-mesh can have none at small smearing. Al on a 2×2×2 mesh needs 0.5 eV of smearing
  before anything is fractional.
- Match the reference's occupation scheme before suspecting forces. A smeared calculation against a fixed-occupation reference legitimately differs, since displaced Si at 0.05
  eV smearing has 0.9 percent fractional occupations.
- The mixing scheme sets the iteration count on a metal, the smearing kernel does not.
  On 1-atom fcc Pt (PAW, 40/400 Ry, 6x6x6, 0.2 eV) `johnson` converges in 13 iterations
  where `pulay` takes 17 and `broyden` 20, and gaussian/cold/mp1 are all within one
  iteration of each other at fixed scheme. Johnson (Kerker-preconditioned Broyden with
  the metric on the total density) is the right default for a smeared metal. The
  converged free energy is bit-identical across schemes, so this is pure iteration
  count. It does not close the gap to QE's 7 iterations, which is a starting-density and
  preconditioner-quality difference, not a scheme choice.
- The QE iteration gap is metal-specific, not a PAW or spin problem. The O2 triplet
  nspin=2 PAW converges in 21 iterations to QE's 20 on the identical input, so a gapped
  magnetic system shows no gap. Only the metal opens the 16-to-7 distance, which points
  at the density preconditioner and leaves spin orthogonal to it. Do not chase the metal
  gap through the spin or PAW machinery.

## Spin

- Test spin-GGA at the potential level, not with energy-only $\zeta=0$ tests. Those
  cannot see a GGA vector-field bug. The spin-GGA field needs $2\, e_{tt}\, \nabla
  \rho_\text{tot}$, and the factor of 2 was wrong while every energy gate passed.
- Generate the one-center potential by autograd through the quadrature. The conventional
  divergence-form PAW one-center potential is not the exact derivative of the evaluated
  quadrature, 0.05 to 1 percent off on the lm-truncated expansion. Codes agree with each
  other on forces because they share the inexactness. When your code matches QE but
  disagrees with a finite difference of its own energy, suspect a shared convention, not
  the finite difference.
- Gate the moment sign. The $\pm m$ branches are exactly degenerate without spin-orbit
  and the trajectory selects one.
- Some references are not worth generating. The displaced-Ni₂ torture geometry made QE
  itself oscillate for 54 to 100 Broyden iterations. Validate on degenerate limits and clean
  integer-occupation systems like the O₂ triplet instead.
- The direction-only moment constraint has a magnitude loophole. The Ma-Dudarev
  $|M^\perp|^2$ penalty is minimized at $M=0$, so a frustrated non-collinear target
  demagnetizes to satisfy it at no penalty. It is fine at collinear endpoints (a forced
  antiparallel target relaxes to the true smaller AFM moment), but at oblique angles use
  the magnitude-robust $|M - m_0\hat e|^2$ ("vector") penalty, which charges $\lambda
  m_0^2$ for demagnetizing. bcc Fe forced to a 135° spiral, perp collapses to 1.7 μB and vector holds 2.2.
- Spin Hamiltonians (J, D, K) come from the torque, not energy mapping. The exchange
  tensor is the site-to-site derivative of the autograd torque, $\mathcal{J}_{IJ} =
  \partial T_I/\partial\hat e_J$. Tilt one moment, read the induced torque on the
  others. Because the torque is already an analytic gradient this is *one* finite-
  difference order, not the two of energy mapping, so it is lower-noise. bcc Fe
  validates: $J_1 \approx 22$ meV (LKAG ~15-19), DMI zero by centrosymmetry, mean-field
  $T_c$ 1388 K matching Pajda's MFA. A small cell folds periodic images, so it yields the
  shell-summed $J(q=0)$ (hence $J_1 \approx J_{01}/8$, a slight overcount). Per-shell
  $J_n$ needs a supercell or the reciprocal-space $J(q)$ route.
- Seed non-collinear references high-spin. The bare unconstrained non-collinear SCF is
  multi-stable. A weak moment seed collapses O₂ or Fe to a low-spin or nonmagnetic
  solution. Seed above saturation (`mag_init_scale` ~1.5) and let it relax down, or pass
  the target magnitude explicitly.
- Constrained non-collinear at strongly-frustrated oblique angles can limit-cycle at a
  small residual rather than converge tightly, though the moment values stay stable. Raise
  the penalty $\lambda$. A collinear axis (parallel or antiparallel) converges far more
  easily than an oblique one.

## Response, adjoints, and autograd

- Never trace the SCF loop or the eigensolver with autograd. Derivatives are taken at
  the converged point, by stationarity for energy gradients and by Sternheimer plus
  autograd HVP kernels for everything else. No explicitly derived $f_{xc}$ exists anywhere in
  the code, which is why every solver works for learnable functionals unchanged.
- Use Anderson mixing on response fixed points, not plain damped iteration, which
  diverges near spin instabilities. NiO's antisymmetric mode has a $K\chi_0$ eigenvalue
  near $-6$.
- Project the full degenerate occupied subspace together in the Sternheimer solve.
- For metals, project out the entire computed band window and sum the window-pair
  contributions with divided-difference occupation weights. This keeps the shifted
  operator positive definite at the Fermi level, and the empty buffer bands make the de
  Gironcoli partition unnecessary.
- Keep the insulator path as the same code, not a special case. The occupation response
  carries a rank-one $\delta\mu$ coupling across the whole Brillouin zone from particle
  conservation, and it vanishes smoothly in the insulating limit.
- Floor densities before fractional powers. `torch.where` evaluates both branches, and a
  NaN in the dead branch poisons the backward.
- torch.compile cannot double-backward. The `f_xc` kernel is a double backward through
  `E_xc`, so a compiled XC functional would crash the second grad, and the failure lands
  in the caller's grad call where no try/except reaches it. The opt-in `compile_xc` path
  handles this by having the `f_xc` response and HVP call sites wrap their
  `xc.energy()` in the `xc_eager()` context manager (`core/xc/base.py`), a thread-local
  switch that forces `eval_energy_density` back to eager. First backward (`v_xc`) stays
  compiled, second backward (`f_xc`) falls back, and no response caller changes.
- Validate finite differences on a ladder. Analytic against a finite difference of
  complete SCF re-runs at 1e-5 to 1e-7 relative is the floor for first derivatives. When
  two analytic routes disagree at 3e-7, energy-space Richardson finite differencing
  arbitrates.

## Discretization error estimation

The `postscf/discretization_error.py` estimator perturbs the converged solution
into the high-G complement to estimate the plane-wave (Ecut) basis error, then
propagates it to quantities of interest. It follows Cancès et al. and the
differentiable-DFT coupling of arXiv:2509.07785. A few things are not obvious.

- The complement correction is cheap and needs no larger SCF. The enlarged sphere
  is a subset of the density FFT box, and the first-order correction is a diagonal
  divide δφ = −R/(T_G − ε) on the annulus, since the Laplacian dominates at high G.
  Keep it a post-processing step, never in the SCF hot path.
- The density change integrates to zero. The correction is orthogonal to the
  occupied space, so ∫δρ = 0 to first order. If it does not, the residual or the
  padding is wrong.
- The forward-mode force propagation is δF ≈ (∂F/∂P)δP with a FIXED δP, one AD
  pass. This works because the force's δP-response is dominated by ⟨δφ|∂R/∂τ⟩, the
  ion moving. It reproduces the reference error to a 0.99 correlation.
- The energy error is second order, not first. The naive first-order term
  Σ f 2Re⟨δφ|R⟩ is halved by the second-order term at the variational optimum, so
  the correct estimate is δE = Σ f ⟨δφ|R⟩ with factor 1, not 2. A factor-2 form
  overshoots by 2x. It is a definite energy lowering.
- The naive force recipe does NOT extend to stress. Propagating a fixed δP through
  the fixed-basis stress gives a cleanly anti-correlated estimate (correlation near
  −1, about −0.5x the true error). The reason is that σ = ∂E/∂ε and its δP-response
  is dominated by the strain-response of the orbital correction, the ⟨∂δφ/∂ε|R⟩
  term, which a fixed-δφ forward pass omits. Forces avoid this because the ion-motion
  term dominates there. A correct stress estimate needs a strain-parameterized
  residual, δφ differentiated through strain, not one AD pass.
- The coarse-space Dyson refinement is opt-in and not yet validated. Dressing the
  first-order δρ by (1 − χ₀K)⁻¹ is structurally derived but was neutral to slightly
  negative on the one insulator tested, so the exact Schur coupling still needs
  pinning against the reference. Leave `dyson=False` for now.
- The USPP/PAW density error has two channels, not one. The complement residual uses
  the generalized metric, R = P_annulus(H − εS)ψ, and the density change is the smooth
  part from δψ PLUS an augmentation part from the on-site occupation (becsum) change,
  δbecsum fed through the Q functions. On Si2 the augmentation channel lifts the density
  correlation from 0.51 (smooth only) to 0.74. ∫δρ is no longer exactly zero: the smooth
  part cancels but the augmentation carries a small S-orthogonality residual (~3e-5 per
  electron on Si), which is negligible. The USPP energy ratio (0.99 on Si) was tighter than the NC diamond case (0.73), but that is the system, not the method, so do
  not read it as a general improvement.
- The nspin=2 path runs the correction per spin channel, each with its own v_eff and
  eigenvalues, and sums δρ and δE over the two channels. The nonmagnetic limit
  (start_mag=0) reproduces the nspin=1 estimate to machine precision, which is the
  cheapest regression test. On ferromagnetic bcc Fe the density correlation is 0.96 and
  the energy ratio 0.97. Because nspin=2 forces smearing, partially occupied bands carry
  only the leading first-order term, so the estimate is best on gapped or well-separated
  bands.
- Symmetry does not have to be off, for the right reason. The complement correction is a
  fully symmetric perturbation (the high-G complement of psi_Sk is the complement of psi_k
  rotated), so for NC nspin=1 the estimate runs on the IBZ k-points and folds the density
  error over the star with the same operator the SCF applies to rho, and the force error
  symmetrizes the output dF vector exactly as ground-state forces do. On [111]-displaced
  diamond Si (n_ops=12, 36 to 13 k-points) the symmetric dF matches the full-BZ dF to
  0.1 percent and is symmetry-invariant to 1e-19. What genuinely needs the full k-mesh is
  a real response to a symmetry-breaking perturbation (USPP augmentation, nspin=2 spin
  channels, the Dyson dressing, and stress), where folding the IBZ output is not valid.
- Coverage. Density and energy error: NC and USPP/PAW, nspin=1 and 2. Force error: NC
  nspin=1 (the USPP force needs the augmentation and one-center force terms, and the nspin=2
  force needs the per-spin channels threaded through the propagation). Symmetry is
  supported for the NC nspin=1 density, energy, and force error. USPP/PAW, nspin=2, and
  Dyson still require use_symmetry=False.

## Process and validation

- Check what a reference structure actually is. The Lejaeghere reference for carbon is
  graphite, not diamond.
- Pin everything in a cross-code comparison. Pseudo, cutoffs, k-mesh, FFT dimensions,
  occupation scheme, and compare the quantity the other code prints.
- Use two comparison axes. The Δ-factor against all-electron mixes pseudization with
  implementation, while Δ against QE at pinned settings isolates the implementation. Cu
  and Ni look bad on the first axis and sit exactly at the psl pseudization limit on the
  second.
- Write long remote calculations to a file on the remote host. A killed ssh pipe takes the output
  of an hour-long calculation with it, and `timeout` kills the wrapper while the python orphans.
- Do not match a process by its own command line. `pgrep -f` matches the shell running it
  when the pattern appears in the command line. Use `ps aux | grep "[p]attern"` or exact
  PID kills. The same failure occurs remotely too: `ssh host 'pkill -f pat'` matches the ssh command's
  own remote shell and kills the job you just launched.
- Detach remote background jobs with `setsid`, not bare `nohup`. `nohup cmd &` launched
  inside `ssh host '...'` still takes SIGHUP when the ssh session closes and dies. Wrap
  the work in a launcher that spawns the parallel jobs and `wait`s, and run it with
  `setsid nohup bash launcher.sh >log 2>&1 </dev/null &`. That survives the disconnect.
- Non-interactive ssh has no login environment. `ssh host 'python ...'` fails with
  `python: not found` and `module: command not found` because the profile is never
  sourced. Call the interpreter by full path (`~/.venvs/base/bin/python`) and `source`
  the module init explicitly.
- A shared HPC cluster's Python environment works against you. Build it from conda-forge. On
  Hoffman2 the login profile forced `pip --user`, put a broken py3.9 `~/.local` on
  `PYTHONPATH`, shipped GCC < 9.3 (so any source-built wheel dies), and threw an OpenBLAS
  memory error on the login node. The robust fix is conda-forge for the whole scientific
  stack (self-contained binaries, no compiler, no user-site) with pip only for the CUDA
  torch wheel, then run with `PYTHONPATH=<your src>`, `PYTHONNOUSERSITE=1`, and an
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` cap so imports work on the login node.
- Independent SCF points parallelize. Do not run them serially. A spin-spiral or
  dispersion sweep is embarrassingly parallel. Running the angles as concurrent
  processes (a few threads each) cut a ~2-hour serial sweep to ~30 minutes, and it is the
  same shape the batched-multi-structure GPU work would eventually fill.
