# Wisdom

Things that cost real time to learn and are not obvious from the code. Each entry is
a rule with the symptom that motivates it. If what you are seeing matches a symptom
below, trust the entry before re-deriving it. The marquee example is that the
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
  Every hand-written backward must match that convention. It is documented in
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

## Eigensolvers

- Re-orthonormalize the reused Ritz block on a Davidson restart. Drift compounds at
  tight tolerance into eV-scale energy jumps, and it bites CUDA first while staying
  latent on CPU.
- For a complex generalized problem, form $L^{-1} H L^{-\dagger}$, not $L^{-1} H
  L^{-1}$. The error is invisible on real matrices and produces Ritz values below the
  true minimum. That fingerprint means a broken reduction, not a Davidson problem.
- Guard the indefinite overlap. The USPP S is indefinite on truncated spheres at low
  cutoff, and QE errors identically. The batched solver must mirror the per-k path's
  drop-oldest conditioning guard, whose Cholesky failures are silently caught, and never
  contract to the contaminated Ritz block.
- Build the preconditioner from positive band kinetic expectations. Eigenvalues go
  negative below the Fermi level, the clamp makes near-zero rows, and rank-safety jitter
  then replaces physical rows with noise.
- Put rank-safety jitter only on near-zero-after-projection rows, and normalize
  expansion rows before orthonormalization. Otherwise near-converged residuals floor
  around 1e-8, whose signature is exact energies with `converged=False` for 60
  iterations.
- Do not feed a walked-off warm start to the mixer. A warm-started Davidson can
  deterministically walk off a cliff from a converged-quality density, with 100 eV
  energy jumps and a frozen residual that mixer resets do not touch. The rescue is
  discarding the warm start and re-solving from salted seeds without feeding the garbage
  to the mixer.

## SCF and mixing

- Mix the composite (density, becsum) pair for USPP/PAW. Mixing becsum outside the
  Pulay vector produces a gain-per-iteration charge oscillation on semicore-metal PAW.
- Kerker damps the density-total block only. A per-channel Kerker with a $G=0$ zero
  freezes interspin charge transfer and blows up spin SCFs, so spin mixing runs in the
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
  silently. Hand damping converges slowly. Scalar adaptive step controllers either
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
- Re-audit every crutch when you change mixers. A stabilizer tuned for one mixer becomes
  a brake under a better one. The 0.4 becsum step scale that tamed the on-site
  becsum-ddd mode for Pulay cost Johnson eleven iterations on ferromagnetic Ni. QE mixes
  becsum unscaled, and matching that closed the bulk of the remaining iteration gap.

## Metals and smearing

- Compare the free energy. QE's printed smeared "total energy" is the free energy F.
- Keep occupation and entropy as scheme-paired objects. Mixing one scheme's occupation
  with another's entropy is a classic silent error.
- Assert that fractional occupations exist before testing metallic physics. A coarse
  k-mesh can have none at small smearing. Al on a 2×2×2 mesh needs 0.5 eV of smearing
  before anything is fractional.
- Match the reference's occupation scheme before suspecting forces. A smeared run
  against a fixed-occupation reference legitimately differs, since displaced Si at 0.05
  eV smearing has 0.9 percent fractional occupations.

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
  itself bounce for 54 to 100 Broyden iterations. Validate on degenerate limits and clean
  integer-occupation systems like the O₂ triplet instead.

## Response, adjoints, and autograd

- Never trace the SCF loop or the eigensolver with autograd. Derivatives are taken at
  the converged point, by stationarity for energy gradients and by Sternheimer plus
  autograd HVP kernels for everything else. No hand-derived $f_{xc}$ exists anywhere in
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
- Validate finite differences on a ladder. Analytic against a finite difference of
  complete SCF re-runs at 1e-5 to 1e-7 relative is the floor for first derivatives. When
  two analytic routes disagree at 3e-7, energy-space Richardson finite differencing
  arbitrates.

## Process and validation

- Check what a reference structure actually is. The Lejaeghere reference for carbon is
  graphite, not diamond.
- Pin everything in a cross-code comparison. Pseudo, cutoffs, k-mesh, FFT dimensions,
  occupation scheme, and compare the quantity the other code prints.
- Use two comparison axes. The Δ-factor against all-electron mixes pseudization with
  implementation, while Δ against QE at pinned settings isolates the implementation. Cu
  and Ni look bad on the first axis and sit exactly at the psl pseudization limit on the
  second.
- Write long remote runs to a file on the remote host. A killed ssh pipe takes the output
  of an hour-long run with it, and `timeout` kills the wrapper while the python orphans.
- Do not match a process by its own command line. `pgrep -f` matches the shell running it
  when the pattern appears in the command line. Use `ps aux | grep "[p]attern"` or exact
  PID kills.
