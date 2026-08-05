# Improving convergence

Three things decide whether a hard calculation finishes and how much it costs.
The first is electronic convergence, the SCF
loop that will not settle on a metal, a magnet, or a slab. The second is dual
descent, the joint geometry-plus-electronic optimizer that avoids nesting a full
SCF inside every ionic step. The third is distributed k-point parallelism, which
makes a dense k-mesh affordable across more than one machine.

The [Wisdom](wisdom.md) page records the hard-won rules behind the SCF advice
here, and [Performance](performance.md) has the measured mixing and
preconditioner numbers. This page covers what you see go wrong and which input
option changes it.

## Improving SCF convergence

Three symptoms cover almost every stuck SCF. The density residual oscillates
without falling (a metal with a partially filled band at the Fermi level), the
magnetic moment flips or collapses between iterations (spin metastability), or a
low-frequency residual grows with cell size (charge sloshing in a metal or slab).
Each has its own set of knobs, and reaching for them in the wrong order wastes
time.

### Degeneracies at the Fermi level

A partially filled band at the Fermi level with no smearing gives an integer
occupation that jumps as eigenvalues cross, so the density never settles and the
residual oscillates at a fixed amplitude. The fix is a smearing scheme, set with
`smearing.type` (`none`, `fermi-dirac`, `gaussian`, `mp1`, or `cold`) and
`smearing.width` (default 0.1 eV). A metal needs one of the fractional-occupation
schemes, and `cold` or `mp1` recover the zero-temperature energy more accurately
than plain `gaussian` at the same width.

Assert that fractional occupations exist before deciding smearing is the problem.
A coarse k-mesh can carry none at a small width. Al on a 2×2×2 mesh needs 0.5 eV
of smearing before any occupation is fractional, so a too-fine width on a too-coarse
mesh looks metallic and behaves like a fixed-occupation insulator.

The band count also matters at the Fermi level. `nbands` defaults to 20 percent
headroom over the occupied count, at least four extra bands. The buffer bands hold
the states that fractional occupation spills into, and if a near-degenerate cluster
sits right at the top of the window the smearing tail gets truncated. Raise
`nbands` when the highest computed band carries non-negligible occupation.

The smearing kernel does not set the iteration count, the mixing scheme does. On
one-atom fcc Pt (PAW, 40/400 Ry, 6×6×6, 0.2 eV) `gaussian`, `cold`, and `mp1` land
within one iteration of each other at a fixed scheme. Pick the smearing for the
physics, not for convergence speed.

For a smeared metal the density residual floors at occupation noise while the free
energy is long settled. Gate on the energy tail rather than the density residual,
which in practice means a looser `scf.rhotol` around 1e-5 for a ferromagnetic
metal instead of the 1e-7 default.

### The energy-metric gate

The residual floor is not noise on a magnetic metal, it is physics. On a
ferromagnet near the Stoner boundary the magnetization-channel residual settles
around 2e-3 and does not fall further, mixer-independently, while the charge
channel sits five to seven times lower. A `scf.rhotol` gate then either never
fires or has to be loosened by hand to a value chosen per system.

The energy metric replaces that hand-tuning with the residual's exact energy
error. Because the energy is stationary at the self-consistent fixed point, the
error left by stopping at a finite residual r is second order, and its leading
term is the kernel contraction $\tfrac{1}{2}\langle r | K_\text{Hxc} | r\rangle$
with $K_\text{Hxc}$ the Hartree-plus-XC kernel. gradwave already carries those
operators for the Dyson dressing, so the estimate is computed exactly once per
iteration from the same $4\pi e^2/G^2$ Hartree kernel and an autograd
Hessian-vector product of $E_\text{xc}$. The Hartree kernel amplifies the charge
channel and leaves the magnetization channel alone, so the energy error is set by
the well-converged charge residual and settles far below any reachable `rhotol`
while the magnetization residual sits on its floor. This is the same quantity
VASP gates on through `EDIFF` and QuantumESPRESSO estimates through its
`conv_thr` accuracy, computed here rather than approximated. QE reports the
un-halved $\langle r | K_\text{Hxc} | r\rangle$, so its threshold is about twice
this one.

Select it with `scf.convergence: energy`, which converges when the estimated
energy error falls below `scf.entol` (default 1e-6 eV) with the energy tail
(`scf.etol`) and the stale-solve guard both still enforced. The default
`scf.convergence: density` leaves the residual gate unchanged. The per-iteration
estimate and its charge and magnetization decomposition are recorded in the SCF
diagnostics block and the `scf_trace.json` sidecar, so a trace shows which
channel carries the remaining error. On the norm-conserving path the
Harris-Foulkes and Kohn-Sham free energies at each iteration bracket the same
error with no extra machinery, and their gap is recorded alongside the estimate
as an independent cross-check. The estimate is the kernel-only term. It
omits the independent-particle $\chi_0$ response, whose one application needs a
Sternheimer solve per band restricted to insulators and is neither cheap per
iteration nor applicable to the metals this gate targets, and it omits the
Hubbard and Fock second-order kernels. A meta-GGA is rejected rather than
estimated without its kinetic-energy-density response.

### The spinor path

A magnetic spinor SCF (`task: scf` with `noncollinear: true` and a nonzero
moment) does not reach `rhotol` 1e-5 under any mixer. The magnetization-channel
residual floors near 2e-3 while the charge channel sits five to seven times
lower, and the floor is a transverse instability rather than a stuck fixed
point. Long-wavelength transverse magnetization is the magnon-soft direction of
a ferromagnet, whose linear response has near-unit gain, so the mixed iteration
amplifies it about threefold per step until it saturates near 1e-4. The fixed
point underneath is converged. Every fcc Ni + SOC arm that holds the
ferromagnetic branch agrees on the free energy to 4e-5 eV and on the moment to
four digits, so the residual gate reports an error the energy does not have.

`scf.convergence: energy` gates the spinor path on the same second-order energy
error as the collinear one, decomposed here into charge, longitudinal, and
transverse magnetization channels. The kernel is the exact coupled (ρ, m⃗) f_xc
Hessian-vector product of the noncollinear functional plus the charge-channel
Hartree kernel, evaluated at the iteration's input density. The transverse
magnon-soft modes carry the residual floor but almost none of the energy error
(3e-6 eV at a stop whose magnetization residual sits at 2e-2). The per-channel
decomposition is recorded in the `scf_trace.json` sidecar, so a trace shows which
channel carries the remaining error.

Set `scf.entol` to 1e-4 for a magnetic spinor run rather than the 1e-6 default,
which is calibrated to the collinear paths. On fcc Ni + SOC the johnson recipe
below stops at iteration 10 under `entol` 1e-4, at the cross-arm consensus energy
to 6e-5 eV with the moment intact, and its estimate dips to the 5e-6 eV scale a
few iterations later without reliably crossing 1e-6. The estimate is conservative
on the near-Stoner magnetization channels, whose soft modes make the kernel-only
contraction an overestimate of the energy they carry, so a fired gate is trusted
and a lower `entol` costs iterations rather than accuracy. Pair the energy gate
with the `quadratic` diagonalization schedule. Under the stock `linear` schedule
the eigensolves track the flooring residual loosely and inject 1e-4-scale energy
noise each iteration, which the estimator honestly reports, and the gate then
floors near 5e-4 instead.

The magnetization mixing has its own controls under `scf.magnetic`, because the
spinor driver resolves the charge and moment mixing independently of
`scf.mixing`. `mixer` selects the (ρ, m⃗) mixer class (`pulay`, `johnson`, or
`broyden`), `spin_precond` turns on the Stoner preconditioner for the
longitudinal moment channel, `mixing_alpha` sets the moment step, and
`diago_schedule` picks the adaptive diagonalization-tolerance schedule. Under
`johnson` the pulay-tuned moment step collapses the near-Stoner moment onto the
nonmagnetic branch, measured on the SOC-free spinor run of the Ni cell, so an
unset `mixing_alpha` with `mixer: johnson` takes a lower default (0.3) instead of
the pulay guard. See [Non-collinear magnetism and spin-orbit
coupling](noncollinear-soc.md) for the measured arms.

### Magnetic systems

A spin-polarized run is set with `nspin: 2` and seeded with `start_mag`, a map
from element symbol to an initial moment fraction (the collinear seed defaults to
0.5 per atom). The nonmagnetic state is a genuine stationary point tens of meV
away, so an under-seeded or over-damped SCF collapses the moment to it silently
and validates against nothing.

Magnetic systems slosh because the magnetization channel, not the charge, sits
near the Stoner instability. The one-iteration map has a measured gain near −6 on
the spin mode for a ferromagnetic metal, so default damping collapses the moment
to the nonmagnetic branch and manual damping converges slowly. The robust operator
is Johnson mixing, whose normalized multisecant update with the w0 regularization
handles that expansive mode. On the norm-conserving path `scf.mixing.scheme`
resolves automatically to `johnson` for `nspin: 2` and `pulay` otherwise, so a
collinear magnet already runs on the right scheme with no setting. The USPP/PAW
path defaults to `johnson` for every `nspin`, including `nspin: 2`. An earlier
bcc Fe blowup kept it on `pulay`, on the theory that Johnson discarded a becsum
step-damping the ferromagnet needed, but that damping was a crutch tuned for
Pulay, and matching QuantumESPRESSO's unscaled becsum mix removed the penalty.
Johnson now converges bcc Fe in 16 iterations against Pulay's 30, fcc Ni near the
Stoner boundary in 18 against 27, and a two-sublattice AFM Fe cell in 31 against
58, each at the same moment and energy.

The mixer does not select the physical branch. The practical defense across a scan
is a warm-start chain, carrying the converged density from one point to the next so
branch selection stays stable, plus an explicit moment check as a detector. If you
see the moment drift toward zero over the first few iterations, raise the seed and
confirm the smearing width is not so wide it washes out the exchange splitting.

### Charge sloshing in metals and slabs

A large metal or a slab with vacuum sloshes charge at long wavelength, because the
Hartree response amplifies the smallest-|G| density modes like $4\pi e^2\chi/G^2$.
The residual oscillates at low frequency and the amplitude grows with the cell.
The Kerker filter screens those modes with a single screening length. It is
controlled by `scf.mixing.kerker` (`auto` by default), and the auto policy turns
it on for any smeared system and for an insulator once the cell grows past roughly
8 Å, where the smallest nonzero |G| drops below about 0.8 Å⁻¹.

A single Kerker screening length is the right operator for a bulk metal and the
wrong one for a cell with vacuum, where a fixed screening over-damps the modes
that must stay free in the vacuum region. The local Thomas-Fermi preconditioner
lets the screening wavevector track the local density, capped at the bare Kerker
value so a homogeneous bulk is unchanged. It is selected with `precond: local_tf`
under `scf.mixing`, or through the Python calculator as `GradWave(precond="local_tf")`.
The default is `kerker`, the constant filter described above, whose on and off state
`scf.mixing.kerker` sets. Choosing `local_tf` replaces that filter on the charge
channel, so the `kerker` setting no longer applies there. On fcc Al slabs it cut
the 4-layer Al(100) slab from 21 to 17 iterations and the 6-layer from 27 to 21,
with the converged energy bit-identical to bare Kerker, while bulk Al is unchanged
at 9 iterations either way. The gain grows with the vacuum fraction.

Two more mixing knobs sit under `scf.mixing`. `alpha` (default 0.7) is the linear
mixing fraction, and lowering it damps a persistent oscillation at the cost of
iterations. `history` (default 8, lifted to 12 for Johnson) is the number of
residual vectors the Pulay/Broyden/Johnson update keeps, and a longer history helps
a stiff response at the cost of memory. Reach for the preconditioner before either
of these, since a step-size or history change trades against convergence rate while
the right operator fixes the mode directly.

### Large U on a metallic adsorbate system

A large Hubbard U on a metallic slab carrying an adsorbate can diverge where the
bare slab converges. On Pt(111) with an adsorbed H, a bulk-derived U(Pt 5d) of
8.949 eV oscillates the energy by about 30 eV and never converges, while U of 2 and
4 eV converge in about 30 iterations, which places the threshold between 4 and 6 eV.
The bare slab at the same U is fine, so the adsorbate is required.

The mechanism is occupation flip-flop, not the charge sloshing of the section above
and not the adsorbate dipole. When U/2 exceeds the smearing width, the +U potential
shift moves levels through the Fermi window faster than the one-step-lagged
occupation update tracks, so occupations reorder every iteration and the density
limit-cycles. The reorder count scales with U, reaching 28597 at U = 8.95 eV with a
Fermi-level swing of 0.235 eV against a 0.14 eV smearing width. Charge sloshing is
present in the converged runs too, so it is not the driver, and a symmetric cell that
cancels the net adsorbate dipole still diverges, which rules the dipole out as well.

The structural fix is occupation-matrix damping, which removes the one-step lag.
Give the `hubbard` block as a mapping and set `occ_mix` (β in (0, 1]) below 1, with
`u_ramp_iters` to ramp U in linearly over the first iterations — see
[Differentiable Hubbard U](hubbard-u.md) for both knobs. Beyond that, converge at a
lower U and carry that density into a higher-U run, and distrust a bulk-derived U
near 9 eV transferred to a metallic adsorbate system where the level structure
differs. The full diagnosis is in the ideas.md record "Large-U divergence on
metallic adsorbate systems".

### Which knob, in which order

1. If occupations are integer and the system is a metal, add `smearing` first.
   Confirm fractional occupations actually appear at your mesh and width.
2. If a near-degenerate cluster sits at the top of the band window, raise `nbands`.
3. If the residual sloshes at long wavelength, confirm `scf.mixing.kerker` is on
   (it is by default for a smeared cell), then set `scf.mixing.precond: local_tf`
   for a slab or a molecule in a box.
4. For a magnet, seed with `start_mag` and rely on the automatic Johnson scheme on
   the norm-conserving path. Chain warm starts across a scan to hold the branch.
5. For a metallic +U system that limit-cycles, damp the occupation matrix
   (`hubbard: {manifolds: [...], occ_mix: 0.3, u_ramp_iters: 15}`) before touching
   the charge-channel knobs — the instability is in the occupations, not the density.
6. Only then tune `scf.mixing.alpha` down or `scf.mixing.history` up, and gate a
   magnetic metal with `scf.convergence: energy` rather than loosening
   `scf.rhotol` to ~1e-5 by hand.

## Dual descent for geometry optimization

The standard relaxation nests a full SCF inside every geometry step, which is the
robust route and the default. See [Geometry optimization](geometry-optimization.md)
for that workflow. gradwave's total energy is an explicit differentiable function
of the cell, the positions, and the orbital coefficients, so one optimizer can
descend on all three at once and let the electronic state converge alongside the
geometry instead of inside it. This is dual descent, and it trades far fewer
Hamiltonian applications for a narrower scope.

### When it wins

The saving is measured in Hamiltonian applications, one $\hat H$ acting on one band
vector, about two FFTs plus a projector contraction. The nested engine pays one full
SCF per ionic step, and every SCF is many Davidson solves. Dual descent charges a
loose SCF seed once plus one apply per band and k-point per line-search closure.

On a fixed-cell Si2 relaxation (LDA, 12 Ry, 2×2×2, one atom displaced 0.1 Å) the
joint engine converged to `fmax` < 0.005 eV/Å in 80 closures, reaching the ideal
bond to 2e-5 Å at an equivalent cost of 3992 applies, less than a single cold SCF
at the same settings. On a variable-cell Si primitive (LDA, 15 Ry, 2×2×2) it and
the nested BFGS-plus-`FrechetCellFilter` reference relaxed to the same `fmax` =
0.005 eV/Å, with final energies agreeing to 1.2e-7 Ha and the bond length to 8e-5
Å, at 9632 applies against the nested engine's 74920, 7.8 times fewer.

### How to invoke it

Set `relax.method: joint` (the default is `nested`). The joint engine routes a
norm-conserving insulator through the joint driver and transparently falls back to
nested on any unsupported system or on non-convergence, so it is always safe to
request. `relax.fmax` and `relax.cell` carry over from the nested path unchanged.
A second-order variant, `relax.method: newton`, runs an exact Hessian-vector
Newton-CG step under the same contract and fallback.

### Scope and failure modes

Dual descent is insulator-only by construction. The occupied subspace energy is
invariant under band rotations, so occupied near-degeneracies are harmless, but a
smeared metal reintroduces level-crossing discontinuities that break the smoothness
L-BFGS assumes. The applicability guard falls back to nested for a USPP/PAW
pseudopotential, a spin-polarized or noncollinear run, any smearing other than
`none`, an odd electron count, or an external pressure (the joint functional
minimizes the energy, not the enthalpy). The USPP/PAW gap is the largest one. The
generalized S-overlap orthonormalization is done, but the augmented energy assembly
(the augmentation charge, the screened D-matrix, and the PAW one-center term on the
strain graph) is a deferred follow-up, so a PAW relax still nests.

The fallback is logged and printed under `verbose`, and the relax block reports
`method`, `h_applies`, and `n_closures` so you can confirm which engine ran. When
the descent does not converge within `40·max_steps` closures the engine returns to
nested rather than shipping a half-relaxed geometry. The final energy, forces, and
stress are recomputed with one calculator SCF at the relaxed geometry, so the
reported numbers are consistent with the nested path.

## Distributed k-point parallelism

A dense k-mesh makes a metal or a small cell affordable to converge, and its
diagonalization time scales with the k-point count. gradwave already batches every
k-point into one set of tensor ops on a single box. Distributed mode splits that
batch of k-points across `torchrun` ranks with `torch.distributed` (Gloo), each rank
diagonalizing a disjoint k-shard and stitching the density, occupations, and energy
back together with a small collective call every SCF iteration.

Enable it with `distributed: true` in the input, which does nothing outside a
multi-rank launch. Launch through the wrapper, `scripts/gradwave_distributed.sh
input.yaml --nproc-per-node 2` for N ranks on one box, or the two-machine form with
`--nnodes 2 --node-rank {0,1} --master-addr <tailscale-ip>` and
`GLOO_SOCKET_IFNAME=tailscale0` set on every rank.

The `distributed: true` input path routes the norm-conserving and USPP/PAW
collinear SCF, and DFT+U (Dudarev) rides along on either, reduced the same way as
the density. #196 and #197 extended the sharding machinery to the USPP/PAW
collinear SCF and to DFT+U on that path, and `api.run_scf` now shards either
formalism straight from an input file. IBZ symmetry reduction composes with
distribution: the ranks shard the reduced k-list, so the 5 to 14 times k-point
reduction from symmetry and the rank count multiply (the world size just cannot
exceed the IBZ k-count). See [Distributed k-point parallelism](distributed.md)
for the full scope, the reduction detail, and the launch reference.
