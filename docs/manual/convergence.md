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
path mirror-inverts this, keeping `pulay` for `nspin: 2` because Johnson discards
the becsum step-damping that a robust ferromagnet leans on and blows up (bcc Fe 29
to 93 iterations at the same moment and energy), while Johnson wins near the Stoner
boundary (fcc Ni 27 to 18).

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

### Which knob, in which order

1. If occupations are integer and the system is a metal, add `smearing` first.
   Confirm fractional occupations actually appear at your mesh and width.
2. If a near-degenerate cluster sits at the top of the band window, raise `nbands`.
3. If the residual sloshes at long wavelength, confirm `scf.mixing.kerker` is on
   (it is by default for a smeared cell), then set `scf.mixing.precond: local_tf`
   for a slab or a molecule in a box.
4. For a magnet, seed with `start_mag` and rely on the automatic Johnson scheme on
   the norm-conserving path. Chain warm starts across a scan to hold the branch.
5. Only then tune `scf.mixing.alpha` down or `scf.mixing.history` up, and gate a
   smeared metal on the energy tail (`scf.rhotol` ~1e-5) rather than the density
   residual.

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
formalism straight from an input file. IBZ
symmetry reduction is mutually exclusive with distribution for now, so build with
`symmetry: false` for a distributed run and reach for symmetry first on a single box,
where it already gives a 5 to 14 times k-point reduction. See [Distributed k-point
parallelism](distributed.md) for the full scope, the reduction detail, and the
launch reference.
