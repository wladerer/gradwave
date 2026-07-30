# Adversarial testing probe: gradient-ascent bug hunting vs random sampling

Research-notes register. Feasibility probe, not a finished harness. No PR.

## Question

gradwave is differentiable end to end. If two things that should agree are wired
into a disagreement functional D(x), can gradient ASCENT maximize D to surface
worst-case configurations that equal-budget random sampling misses? Every found
maximum would become a regression fixture.

Two attacks: Attack 1 is the calibration target (a real, known artifact);
Attack 2 is the implementation cross-check, gated on Attack 1 working.

## Verdict up front

NO-GO on the ascent harness as designed. The cheap (one-SCF-per-evaluation)
gradient of D is not followable: it is the frozen-density partial derivative,
which for this D is a second derivative of the energy, and the density-response
term it omits dominates. Measured against the true re-converged gradient it is
wrong by 3 to 4 orders of magnitude and gets the sign wrong at 1 of 3 probe
points. In the equal-budget head-to-head the ascent runs collapsed into a
2-cycle around a D minimum; every good point ascent "found" came from its
random starting points, not from following the gradient. Attack 2's
implementation cross-check is a clean null (disagreement ceiling 5.1e-5
eV/Ang, the FD floor), and its one apparent hit was a false positive from a
non-converged SCF, which is itself a design lesson for any future harness.
Details below, alternatives in the go/no-go section.

## Attack 1: the FFT eggbox

The FFT grid is fixed in the cell while the atoms move, so a rigid translation
of all atoms by tau within one grid voxel is not exactly invariant: the net
force Sum_a F_a, which must be zero under rigid translation, deviates from
zero. That deviation is the eggbox artifact. We take

    D(tau) = | Sum_a F_a |    (net force magnitude, eV/Ang)

System: 2-atom Si FCC primitive cell (a = 5.43 Ang), `Si_ONCV_PBE-1.2.upf`,
PBE, kmesh 2x2x2, smearing none, nspin 1, symmetry OFF. Cutoff DELIBERATELY
loose at 12 Ry so the eggbox is visible. FFT grid 18x18x18, grid spacing
0.2133 Ang along each axis. `forces()` semantics matter here: its default
`remove_net=True` subtracts exactly the mean force we are measuring, so
everything below uses `remove_net=False`.

### How forces are computed (postscf/forces.py)

`forces()` returns F = -dE/dtau at the converged SCF point, differentiating
only the three position-dependent energy terms (Ewald, local-potential
structure factor, nonlocal projector phases) with density and orbitals
DETACHED. This is the Hellmann-Feynman regime: at convergence the energy is
stationary in the density, so the frozen-density partial equals the total
dE/dtau. First derivatives are exact in this regime; that exactness does NOT
extend to second derivatives (see the reliability check below).

### Differentiability regime used

RE-CONVERGE the SCF at every evaluation, treat that evaluation's converged
density/orbitals/occupations as FIXED, and take the explicit partial
derivative of D through the same HF energy assembly forces() differentiates.
For a rigid shift s every position is pos0_a + s, so

    Sum_a F_a = - Sum_a dE/dpos_a = - dE/ds,   D(s) = | dE/ds |.

Ascent needs dD/ds, one extra autograd pass through the frozen-density energy.
Cost per evaluation is ONE SCF, so equal budget = equal SCF count. tau is
parametrized as fractional voxel coordinates u in [0,1)^3 with Cartesian shift
s = u @ V, V rows = cell_i / N_i. One voxel is one eggbox period, so [0,1)^3
is the fundamental domain.

Value-level cross-check: the differentiable net force here reproduces
`postscf.forces(res, remove_net=False).sum(0)` to 0.0 (bitwise identical,
abs diff 0.00e+00 at u = (0.37, 0.21, 0.63)). The functional is the right
object; only its derivative is in question.

### Gradient reliability (the central finding)

Directional derivative of D along the analytic gradient direction, frozen
density autograd vs central finite difference of the RE-CONVERGED D (h = 1e-3
in u, extra SCFs outside the 30/30 budget):

| u                  | analytic dD | re-converged dD | ratio   | same sign |
|--------------------|-------------|-----------------|---------|-----------|
| (0.30, 0.40, 0.50) | 59.810      | 0.0216          | 2771    | yes       |
| (0.65, 0.15, 0.85) | 54.229      | -0.0029         | 18935   | NO        |
| (0.10, 0.90, 0.45) | 32.918      | 0.0027          | 11985   | yes       |

The frozen-density gradient overestimates the true gradient by 3 to 4 orders
of magnitude and can point the wrong way. This is expected in hindsight: dD/du
is a second derivative of E, and Hellmann-Feynman stationarity only kills the
response term in the first derivative. The omitted d(rho)/du response term
dominates the second derivative. Consequence: raw lr * grad steps overshoot
the voxel by orders of magnitude, so the ascent below uses the NORMALIZED
gradient direction with a fixed step of 0.08 voxel. That is the most favorable
honest use of this gradient.

### Equal-budget head-to-head (30 SCFs each, seed-fixed)

| method                          | SCFs | max \|Sum F\| (eV/Ang) | at u (voxel coords)   |
|---------------------------------|------|------------------------|-----------------------|
| random uniform over the voxel   | 30   | 3.2326e-03             | (0.580, 0.299, 0.672) |
| gradient ascent, 3 starts x 10  | 30   | 3.7850e-03             | (0.149, 0.973, 0.890) |

Ascent/random ratio 1.171, but the step log shows the ratio is not credit to
the gradient: ascent's best point IS its third random start (start 2, step 0,
D = 3.785e-3). From every start, within 2-3 steps the iterate collapsed into a
2-cycle bouncing across the high-symmetry corner (u = 0.018 <-> u = 0.972 on
all axes), where D is near ZERO (D ~ 1.4e-4). The frozen-density gradient
consistently points toward the grid-aligned symmetric point, a MINIMUM of the
re-converged D. Following it is descent, not ascent. At equal budget the
method found nothing random did not hand it.

### Eggbox magnitude and landscape (15-point diagonal scan, 15 SCFs)

- Energy spread over one voxel: 0.1753 meV (rigid-translation variance of the
  converged free energy at 12 Ry).
- Max net force on the diagonal: 3.822e-3 eV/Ang, slightly above both search
  methods' 30-SCF maxima, at t = 0.571 (equivalently 0.071 by the half-voxel
  symmetry below).
- Structure: D = 0 to 1e-9 at t = 0, 0.5, 1.0 (grid-aligned AND half-voxel
  shifts are exact zeros: at those shifts the sampling error is symmetric
  around the atoms and the net force cancels). Maxima sit at t ~ 0.07 and
  0.43, i.e. NOT midway between grid planes: the midpoint t = 0.25 is a local
  minimum (D ~ 7e-5). Physical intuition ("maximum between grid points") is
  wrong for the NET force; the landscape oscillates at roughly twice the grid
  frequency with zeros at all the symmetric shifts.

Total Attack 1 cost: 87 SCFs, 241 s wall (2 torch threads). Budget accounting:
30 random + 30 ascent + 15 scan + 12 diagnostic SCFs (setup probe, value
cross-check, and 3 gradient-reliability points at 1 + 2 SCFs each plus their
grad evals). Every SCF counted; the 30/30 comparison used exactly 30 each.

## Attack 2: analytic vs finite-difference force, disagreement ceiling

Gate: the brief runs Attack 2 only if Attack 1 works. Attack 1's ascent
mechanism failed, and Attack 2's D cannot be ascended at all in the cheap
regime, because F_fd is itself a finite difference of re-converged SCF runs
(not cheaply differentiable). So there is no ascent variant of Attack 2 to
test. We still ran the scan version to deliver the disagreement ceiling the
brief asks for, since it is the null-result measurement that bounds what an
adversarial search could ever find on this pair.

D(x) = | F_analytic(x) - F_fd(x) |_max over displacement x of atom 2 in
[0, 0.3] Ang along a fixed random direction, same Si cell at 24 Ry (2x2x2,
no symmetry, smearing none, etol 1e-10):

- F_analytic = `postscf.forces` (Hellmann-Feynman, frozen density,
  remove_net=False)
- F_fd = -(E(tau+h) - E(tau-h)) / 2h per component, h = 5e-3 Ang, each E a
  full SCF re-convergence (7 SCFs per scan point: 1 analytic + 6 FD)

Result over 6 displacements (42 SCFs, all counted; see attack2_results.json):

| disp (Ang) | \|F_an\|_max (eV/Ang) | \|F_an - F_fd\|_max (eV/Ang) | all SCFs converged |
|------------|-----------------------|-------------------------------|--------------------|
| 0.050      | 0.7551                | 1.02e-05                      | yes                |
| 0.100      | 1.4846                | 1.96e-05                      | yes                |
| 0.150      | 2.1866                | 2.97e-05                      | yes                |
| 0.200      | 2.8591                | 4.00e-05                      | yes                |
| 0.250      | 3.4995                | 5.06e-05                      | yes                |
| 0.300      | 5.5662                | 3.99e+00                      | NO                 |

Two findings:

1. Disagreement ceiling in the valid regime: 5.06e-05 eV/Ang (at d = 0.25 Ang,
   relative 1.4e-5 of the force), growing linearly with force magnitude and
   consistent with the h^2 FD truncation floor at h = 5e-3 Ang, not with an
   implementation defect. This confirms the null result the brief anticipated:
   both force paths track everywhere the comparison is defined, matching the
   validated FD gate quoted in benchmarks/derivatives/README.md.
2. The d = 0.30 point is a false positive of exactly the kind an adversarial
   harness would mass-produce. Diagnosis (extra SCFs, logged): at d >= 0.25 the
   displaced cell has a NEGATIVE indirect gap (band overlap: homo 8.09 eV above
   lumo 6.48 eV at d = 0.30, metallic) while smearing="none" keeps integer
   per-k occupations; at d = 0.30 the SCF oscillates and stops non-converged
   (converged=False at max_iter=120). HF stationarity only holds at
   convergence, so the 3.99 eV/Ang "disagreement" measures the SCF failure,
   not the forces. gradwave reports converged=False honestly; the first
   version of this script simply did not check it. The script now records
   every SCF's converged flag and excludes non-converged points from the
   ceiling. Lesson for any future harness: a maximizer of D over inputs
   preferentially drives into non-converged or unphysical regions (metallic
   configurations under smearing="none", near-degenerate crossings), so SCF
   convergence must be a domain constraint on D, not an afterthought.

## Go / no-go on building the attack harness properly

NO-GO for the cheap-gradient ascent harness. The design premise (one SCF per
evaluation, frozen-density autograd for dD/dx) fails structurally, not
incidentally, whenever D is built from forces: the gradient of a force-based
D is a second derivative of E, HF stationarity does not protect it, and the
missing density-response term dominates (measured 3 to 4 orders of magnitude,
sign errors). Worse, on the eggbox the frozen gradient systematically points
toward symmetric configurations, which are minima of D, so ascent actively
descends. At equal SCF budget random sampling is at parity, and a 15-point
1-D scan beat both.

What could still work, in decreasing order of promise:

1. Exact dD/dx via the implicit-function machinery (scf/implicit.py computes
   response-aware parameter gradients today; a position analogue would make
   dD/dtau exact at roughly the cost of one adjoint solve per step). The
   ascent question then becomes worth re-asking, but each evaluation costs
   several SCF-equivalents, so the equal-budget bar rises accordingly.
2. Energy-based D (e.g. translation variance of E itself, or E_low_ecut vs
   E_high_ecut on a shared geometry). First derivatives of E ARE HF-exact,
   so dD/dtau is cheap and correct for those functionals.
3. Drop gradients: the landscape found here is smooth, low-dimensional, and
   oscillatory at grid frequency. Bayesian optimization or a coarse scan plus
   local refinement is a better fit per SCF than ascent, and the 15-point
   scan in fact found the largest D of this whole study.

Whatever the search method, Attack 2's false positive fixes one requirement:
SCF convergence (and physical validity, e.g. an actual gap under
smearing="none") must be a hard domain constraint on D. An unconstrained
maximizer will report convergence failures as "bugs" and bury the real signal.

The differentiable plumbing itself was flawless (bitwise agreement between the
hand-assembled HF energy gradient and postscf.forces), and the probe did
produce a usable regression artifact: the eggbox maximum at u ~ (0.07, 0.07,
0.07) voxel with D = 3.8e-3 eV/Ang at 12 Ry, and the exact zeros at grid- and
half-grid-aligned shifts, are pinnable invariants for a future eggbox test.

## Reproduce

    uv run python experiments/adversarial_probe/attack1_eggbox.py            # -> attack1_results.json
    uv run python experiments/adversarial_probe/attack2_force_crosscheck.py  # -> attack2_results.json

Both set torch.set_num_threads(2) (shared laptop) and use fixed RNG seeds.
Attack 1: 87 SCFs, ~4 min. Attack 2: 42 SCFs at 24 Ry.
