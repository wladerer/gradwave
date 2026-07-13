# Lessons

Institutional knowledge that is otherwise scattered across commit messages,
docstrings, and session notes. Each entry is something that cost real time
to learn. If a symptom below matches what you are seeing, trust the entry
before re-deriving it.

## Conventions and units

- One FFT convention, decided once in `core/fftbox.py`. Wavefunctions are
  `ψ(r) = (1/√Ω) Σ c(G) e^{i(k+G)r}` with unit coefficient norm. Fields are
  Fourier series, `f̃(G) = fftn(f)/N`. Real-space integrals are
  `∫fg = Ω Σ_G f̃*g̃`. Most normalization bugs come from mixing these two
  families.
- Grid gradients from autograd carry a volume element. `d E/d ρ_j` on grid
  values equals the functional derivative times `Ω/N`. A second backward
  carries the factor once more. The USPP adjoint's hardest bug was applying
  the `N/Ω` conversion to one pairing but not the composite partner, so the
  becsum block silently broke. One shared conversion at the final pairing,
  never per-term.
- torch returns conjugate Wirtinger gradients for real scalars of complex
  inputs. Every hand-written backward must match that convention. It is
  documented in `fftbox.py` and nowhere else on purpose.
- `torch.tensor(x)` defaults to float32 and once broke Fermi-level
  bisection. Everything is float64/complex128, always explicit.

## UPF and pseudopotential parsing

- PP_DIJ and PP_LOCAL are in Ry, the mesh is in Bohr, PP_BETA stores r·β,
  PP_RHOATOM stores 4πr²ρ. Respect per-projector kkbeta cutoffs, SG15
  hard-truncates.
- QE truncates ALL local-channel radial integrals at 10 bohr (readpp msh,
  forced odd). psl meshes run far past that with tails that deviate from
  −Z/r at the 2e-5 eV level, which becomes a rigid eigenvalue offset times
  the electron count (132 meV/atom for Ni) unless you truncate identically.
  The same msh truncation is load-bearing in three places, local potential,
  atomic densities, and atomic wavefunctions for +U.
- For PAW +U, use the RAW PP_PSWFC amplitudes. PAW pseudo-orbitals are not
  norm-one by design (S supplies the rest), and renormalizing them is a
  100 meV class error.
- PP_Q and ∫q⁰dr disagree at file precision (5e-8). The S-operator weights
  MUST come from the same radial integrals as the augmentation tables or
  charge conservation breaks by exactly that mismatch, which the DIIS
  metric then amplifies through its 1/q0² weight at G=0.
- SG15 carries no PP_PSWFC, PseudoDojo tarballs do. PseudoDojo FR pseudos
  have NLCC and are unsupported on the SOC path, SG15 FR works.

## Grids

- EOS scans need ONE fixed FFT grid per material (elementwise max over
  volumes). A volume-varying minimal box steps E(V) identically in any
  code that does it.
- Two valid boxes that both hold the density sphere give XC energies
  differing at the meV level for sharp semicore densities. QE comparisons
  must pin the reference grid dims.
- Non-symmorphic translations must be commensurate with the grid. The
  diamond glide (¼,¼,¼) needs dims divisible by 4. On an incommensurate
  box the un-projected full-mesh fixed point genuinely carries a 2e-4
  asymmetric density component, so IBZ-vs-full comparisons "fail" at 1e-4
  without any bug present. QE's ph.x enforces the same constraint harder.
- The symmetrizer must be masked to the density sphere. Box-Nyquist
  folding is ill-defined for glide phases.

## Eigensolvers

- Davidson restarts MUST re-orthonormalize the reused Ritz block. Drift
  compounds at tight tolerance into eV-scale energy jumps, and it bit CUDA
  first while staying latent on CPU.
- For complex generalized problems, forming L⁻¹HL⁻¹ instead of L⁻¹HL⁻† is
  invisible on real matrices and produces Ritz values BELOW the true
  minimum. That fingerprint means a broken reduction, not a Davidson
  problem.
- The USPP S is INDEFINITE on truncated spheres at low cutoff (QE errors
  identically). The batched solver must mirror the per-k path's
  drop-oldest conditioning guard, whose cholesky failures are silently
  caught, and never contract to the contaminated Ritz block.
- Preconditioner scales must be positive band kinetic expectations.
  Eigenvalues go negative below E_F, the clamp makes near-zero rows, and
  rank-safety jitter then replaces physical rows with noise.
- Rank-safety jitter belongs only on near-zero-after-projection rows, and
  expansion rows need unit normalization before orthonormalization, or
  near-converged residuals get floored around 1e-8. The signature is exact
  energies with converged=False for 60 iterations.
- A warm-started Davidson can deterministically walk off a cliff from a
  converged-quality density (energy jumps 100 eV, residual freezes, mixer
  resets change nothing). The rescue is discarding warm starts and
  re-solving from salted seeds WITHOUT feeding the garbage to the mixer.

## SCF and mixing

- The mixed variable for USPP/PAW is the composite (ρ, becsum) pair.
  Mixing becsum outside the Pulay vector produces a gain-per-iteration
  charge oscillation on semicore-metal PAW.
- Kerker damps the ρ-total block only. Per-channel Kerker's G=0 zero
  freezes interspin charge transfer and blows up spin SCFs. Spin mixing
  runs in the (total, magnetization) basis.
- Solve DIIS in the diagonally-normalized basis with scale-invariant
  Tikhonov. Regularization scaled to the raw matrix swamps the newest
  small-residual entries and stalls the tail. On ill-conditioning drop the
  OLDEST entry, a full reset discards curvature the next steps need.
- Trust-region resets need a WINDOWED baseline and a cooldown. An all-time
  best residual locks at the eigensolver noise floor, after which every
  iteration resets the mixer, DIIS never re-learns, and pure damped
  iteration diverges geometrically for gain>1 modes.
- QE's convergence criterion is an energy criterion (error ~ residual²).
  Demanding |Δρ| thresholds 100 to 1000 times tighter explains most of the
  apparent iteration-count gap before any mixer quality difference. For
  smeared metals the residual floors at occupation noise while F is long
  settled, so gate on the energy tail and report the flag honestly.
- FM metals near the Stoner instability are the adversarial case. The map
  has a measured gain near −6 on the spin mode. Consequences, each learned
  separately. Default damping collapses the moment to the NM branch
  silently. Hand damping (α 0.3) converges but slowly. Scalar adaptive
  step controllers either over-damp permanently (monotone rules react to
  startup transients) or ride the stability boundary (any recovery rule),
  measured across three controller designs. Plain unweighted Broyden
  DIVERGES catastrophically, early garbage secant pairs poison the inverse
  Jacobian. Johnson's weights (normalization plus w0 regularization) are
  the load-bearing part of QE's mixer, not a refinement.
- Branch selection is not a mixer property. The NM state is a genuine
  stationary point tens of meV away and every code can land on it. Warm
  start chains across scan points are the practical defense, plus an
  explicit moment gate as a detector. One EOS volume flipped branches at
  seed 0.6 and needed 0.8, at the volume where the moment is largest.
- Preconditioning beats step-size control. The Stoner-expansive mode needs
  an operator (the χ₀-diagonal preconditioner of arXiv:2606.26693, whose
  ingredients are the Fermi-surface adjoint terms), not a schedule.
  Adaptive damping line searches match good fixed damping at best and fail
  on transition metals per their own authors.

## Metals and smearing

- QE's printed smeared "total energy" is the free energy F. Compare F.
- Keep occupation and entropy functions as scheme-paired objects. Mixing
  one scheme's occupation with another's entropy is a classic silent error.
- Coarse k-meshes can have NO fractional occupations at small smearing (Al
  2×2×2 needs σ 0.5 eV before anything is fractional). Test premises
  should assert fractional occupations exist before testing metallic
  physics.
- Smeared runs against fixed-occupation references legitimately differ
  (displaced Si at 0.05 eV smearing has 0.9% fractional occupations).
  Match the reference's occupation scheme before suspecting forces.

## Spin

- Energy-only ζ=0 unit tests CANNOT see GGA vector-field bugs. The
  spin-GGA field needs 2·e_tt·∇ρ_tot and the factor 2 was wrong while
  every energy gate passed. Test at the ddd/potential level.
- The conventional divergence-form PAW one-center potential is NOT the
  exact derivative of the evaluated quadrature (0.05 to 1% off on the
  lm-truncated expansion). Codes agree with each other on forces because
  they share the inexactness. Generating the potential by autograd through
  the quadrature itself makes the SCF variational and becsum-level FD
  exact. When code matches QE but disagrees with FD of its own energy,
  suspect a shared convention, not the FD.
- ±m branches are exactly degenerate without SOC and trajectory selects
  one. Gate |m|.
- Some references are not worth generating. The displaced-Ni₂ torture
  geometry made QE itself bounce for 54 to 100 Broyden iterations.
  Validate on degenerate limits and clean integer-occupation systems
  (O₂ triplet) instead.

## Response, adjoints, autograd

- Autograd never traces the SCF loop or eigensolver. Derivatives are taken
  at the converged point, stationarity for energy gradients, Sternheimer
  plus autograd HVP kernels for everything else. No hand-derived f_xc
  exists anywhere in the code, and that is why every solver works for
  learnable functionals unchanged.
- Plain damped iteration on response fixed points DIVERGES near spin
  instabilities (NiO's antisymmetric mode has Kχ₀ eigenvalue near −6).
  Anderson mixing is mandatory, not optional.
- Sternheimer must project the full degenerate occupied subspace together.
- For metals, project out the ENTIRE computed band window and sum
  window-pair contributions explicitly with divided-difference occupation
  weights. This keeps the shifted operator positive definite for states at
  the Fermi level, and the empty buffer bands make the de Gironcoli θ̃
  partition unnecessary.
- The occupation response carries a rank-one δμ coupling across the whole
  BZ from particle conservation. It vanishes smoothly in the insulating
  limit, so the insulator path should be the same code, not a special case.
- FD validation has a ladder. Analytic vs FD of complete SCF re-runs at
  1e-5 to 1e-7 relative is the floor for first derivatives. When two
  analytic routes disagree at 3e-7, energy-space Richardson FD arbitrates.
- `torch.where` evaluates both branches, NaN in the dead branch poisons
  the backward. Floor densities before fractional powers.

## Performance

- IBZ symmetry is the big lever (5 to 14×), then GPU. Small systems on GPU
  are launch-latency bound and the PAW one-center chain is CPU-resident,
  so GPU wins grow with system size and are modest for one-atom cells.
- Batched generalized Davidson pays 1.44× on GPU at 36 k-points and near
  parity on CPU at 8.
- Mixed-precision drafting is NOT a general win, measured across six
  systems. It helps moderate-grid many-k smeared/SOC cases (1.45×) and
  REGRESSES insulators (fp32 drafts inflate SCF iterations for fixed
  occupations). fp32 leaves norms good to only 1e-6, renormalize in fp64
  or the G=0 charge assert trips.
- torch.compile was tried and removed. Inductor does not codegen complex
  ops and the compilable real-decomposed slice is too small next to the
  FFTs.
- Warm-starting band-path chunks from a single previous point made things
  2.5× SLOWER, near-degenerate seeded subspaces stall adaptive Davidson.
  Warm-starting SCF density/becsum across EOS volumes is the opposite
  story, same fixed point, fewer iterations, and branch stability.
- Testing mixers on real SCFs costs 15 to 50 minutes per data point. The
  linearized rig (Arnoldi on FD applies of the true one-iteration map)
  reduces mixer experiments to milliseconds and measures the actual gain
  spectrum. Screen there, confirm winners once on the real SCF, and
  remember the rig only sees local convergence, never basin selection.

## Process

- The Lejaeghere reference for carbon is GRAPHITE, not diamond. Check what
  a reference structure actually is before fitting an EOS to it.
- Pin everything in cross-code comparisons, pseudo, cutoffs, k-mesh, FFT
  dims, occupation scheme, and compare the quantity the other code
  actually prints.
- Two comparison axes beat one. Δ vs all-electron mixes pseudization with
  implementation, Δ vs QE at pinned settings isolates the implementation.
  Cu and Ni look bad on the first axis and are exactly at the psl
  pseudization limit on the second.
- Long remote runs write to a file on the remote host. A killed ssh pipe
  takes the output of an hour-long run with it, and `timeout` kills the
  wrapper while the python orphans.
- `pgrep -f` matches the shell that is running it when the pattern appears
  in the command line. Use `ps aux | grep "[p]attern"` or exact-PID kills.
