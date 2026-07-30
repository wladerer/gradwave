# Differentiable pseudopotential correction (feasibility probe)

Research-notes register. This is a feasibility probe, not a merge candidate. The
goal is to learn a small correction to an element's local pseudopotential by
descending an equation-of-state loss through the self-consistent solution, using
the exact `dLoss/dtheta` that gradwave's differentiability provides.

Target: Si (diamond), PseudoDojo NC-SR-v0.4 PBE standard UPF
(`benchmarks/delta_gauge/pseudos/Si.upf`).

## Parameterization

`correction.py`. A smooth few-parameter local-potential correction in reciprocal
space, added to the species local form-factor table `[eV.Ang^3]`:

    dv_loc(G; theta) = sum_k theta_k * (G/mu_k)^2 * exp(-(G/mu_k)^2)

- the `(G/mu_k)^2` prefactor forces `dv(G=0) = 0`, so the alpha-Z / charge
  normalization the base `vloc_tables` carries is untouched (the G=0 entry is
  handled exactly as `setup_common.build_vloc_tables` handles it).
- the Gaussian forces `dv -> 0` as `G -> inf`, so it is a genuine short-range
  correction rather than a shifted potential.
- centers `mu_k = [1, 2, 3, 4] Ang^-1` (K = 4), so each `theta_k` is a
  near-orthogonal knob on the low-|G| shells where the local form factor lives.

Plumbing: `dv` is gathered onto the dense box through the same unique-|G|-shell
inverse index as the base table, so a corrected per-atom table drops straight
into `System.vloc_atom`, which `local_potential_g` already consumes. `theta` is a
`requires_grad` tensor; the correction threads into the SCF exactly the way the
alchemical local-table blend does (`scf/alchemical.py`).

## Gradient

Only the local term carries `theta`. At the self-consistent density the total
energy is stationary in the density, so by the envelope theorem
`dE_total/dtheta = dE_local/dtheta` at the frozen density (Hellmann-Feynman, the
same argument `alchemical_energy_gradient` uses for `dE/dlambda`).
`probe.energy_of_theta` builds a differentiable scalar equal to the converged
total at the training `theta` whose autograd derivative is exactly this HF
gradient, so `loss.backward()` gives `dLoss/dtheta` without differentiating
through the SCF fixed point.

## Steps

1. `step1_endpoint.py` -- endpoint exactness: `theta=0` reproduces the
   uncorrected SCF energy bit-for-bit.
2. `step2_gradient.py` -- HF `dE/dtheta` vs central finite difference with full
   SCF re-convergence at each displaced `theta`.
3. `step3_synthetic.py` -- synthetic recovery oracle (the go/no-go): perturb Si
   by a known `theta*`, generate a synthetic reference E(V), train `theta` from
   zero to match, report parameter-space distance and EOS residual.
4. `step4_real.py` -- one real step against the WIEN2k all-electron Si EOS:
   Delta(Si) before/after, an off-training force check on a displaced cell, and
   the basis-error estimate (probe ecut vs converged 48 Ry / 8^3).

## Settings

Probe: ecut 28 Ry, 6^3 Gamma mesh, no smearing (Si is an insulator). Converged
reference for the basis-error estimate: 48 Ry / 8^3 (the `delta_gauge` Si
setting). All EOS chains warm-start along volume and across training steps.

Compute discipline: `torch.set_num_threads(2)` locally; the training loops were
run on asus (22 cores) because the laptop was saturated by sibling agents.

## Results

All runs on asus (22-core, idle; laptop was at load 56/8 from sibling agents).
Raw numbers in `results_step{1,2,3,3b,4}.json`.

### Step 1: endpoint exactness (PASS)

theta=0 through the vloc_atom path reproduces the plain SCF bit-for-bit:
E_base = E_theta0 = -230.25119705855093 eV, |dE| = 0.0 exactly (28 Ry, 6^3,
grid 25^3).

### Step 2: gradient credential (PASS)

At a nonzero theta0 = [0.15, -0.10, 0.05, -0.02], Hellmann-Feynman dE/dtheta by
autograd vs central finite difference (eps 1e-3, full SCF re-convergence per
displaced theta):

    grad_hf = [ 0.0375487064,  0.1436920889, -0.0365698429, -0.1883876964]
    grad_fd = [ 0.0375487064,  0.1436920889, -0.0365698429, -0.1883876964]
    max relative error 3.2e-10

Four decades below the ~1e-6 target and the repo's ~1e-5 FD floor, because the
energy is exactly linear in theta at frozen density, so HF here is not an
approximation and the only noise is SCF convergence residual.

### Step 3: synthetic recovery oracle (MIXED -- the key finding)

Known perturbation theta* = [0.30, -0.20, 0.10, -0.04], 5 volumes (94-106%),
40 Adam steps (lr 0.05) from theta = 0, warm-started everywhere. 205 SCFs,
395 s wall.

    EOS shape residual:   0.935 -> 0.077 meV/atom   (12x, observable recovered)
    |theta - theta*|:     0.367 -> 0.364            (parameters NOT recovered)
    theta_fit = [0.0716, 0.0592, 0.0714, 0.0721]    (collapsed to a common value)

The EOS observable trains to near-zero while theta goes somewhere else entirely.
This is not an optimizer failure; it is rank deficiency of the loss, quantified
in step 3b.

### Step 3b: Jacobian diagnosis of the recovery failure

At frozen density the energy is exactly linear in theta, so the mean-subtracted
5-volume EOS shape is a near-linear map r = A theta with A the (5 x 4)
Hellmann-Feynman Jacobian (autograd, 5 SCFs). SVD of A
(`results_step3b.json`):

    singular values  [1.67e-2, 2.84e-4, 4.18e-6, 1.55e-7]   (condition 1.1e5)

Decomposing the recovery error theta_fit - theta* (norm 0.364) over the right
singular directions:

    dir 0  sigma 1.67e-2   err component -0.0053   residual contribution 8.8e-5 eV
    dir 1  sigma 2.84e-4   err component +0.0388   residual contribution 1.1e-5 eV
    dir 2  sigma 4.18e-6   err component +0.2960   residual contribution 1.2e-6 eV
    dir 3  sigma 1.55e-7   err component -0.2088   residual contribution 3.2e-8 eV

The one well-determined direction is recovered to 0.005; the error sits almost
entirely in the two weakest directions, whose contribution to the EOS residual
(1e-6 eV and 3e-8 eV) is invisible below the achieved 7.7e-5 eV shape residual.
The EOS window determines one, at most two, directions of a four-parameter
correction. Physically the four Gaussian bumps (mu = 1..4 1/Ang) act on E(V)
almost interchangeably, because the volume window samples dv only through
slowly varying combinations of the few populated low-|G| shells (|G_min| ~ 2.0
1/Ang for Si diamond), and the mu = 1 bump is sampled only through its tail.

Conclusion for the go/no-go: the *machinery* (differentiable correction, exact
gradient, trainable loop) is validated end to end, but an EOS-only loss cannot
identify a multi-parameter correction. Any real training needs either (a) a
parameterization with as many effective degrees of freedom as the loss actually
constrains (1-2 for an EOS window), or (b) a richer loss (eigenvalues, log
derivatives, forces, a second structure) that fills the null space.

### Step 4: one real step against WIEN2k Si

7 volumes (94-106%), 10 Adam steps (lr 0.05) on the WIEN2k BM3 shape.
93 SCFs, 283 s wall (plus one earlier attempt of ~92 SCFs discarded to an
NLCC/forces API crash at the force check).

    Delta(Si) converged basis (48 Ry, 8^3):        0.1090 meV/atom
    Delta(Si) probe basis (28 Ry, 6^3), theta=0:   0.2578 meV/atom
    Delta(Si) probe basis, after training:         0.0570 meV/atom
    basis-error estimate |probe - converged|:      0.1488 meV/atom

    V0 (Ang^3/atom): before 20.4669, after 20.4564, converged 20.4478, WIEN2k 20.4530
    B0 (GPa):        before  88.47,  after  88.61,  converged  88.37,  WIEN2k  88.545

Off-training force check (atom displaced 0.10 Ang, no symmetry, probe basis):
max|F| uncorrected 1.3042 eV/Ang, max component change from the trained
correction 0.0002 eV/Ang (0.015 percent). The EOS fix does not degrade forces
at any meaningful level.

Honesty caveats, in order of importance:

- The basis error at the probe settings (0.149 meV/atom) is comparable to the
  Delta improvement (0.258 -> 0.057, a 0.20 meV/atom change). Part of what the
  correction learned is basis truncation, not pseudization. A real training run
  must sit at the converged basis (48 Ry / 8^3 for Si, roughly 4x the probe
  cost per SCF).
- Si is a nearly-perfect pseudopotential to begin with (PseudoDojo dfact 0.146
  meV/atom; our converged Delta 0.109). It was chosen because it is the
  cheapest clean insulator, not because it needs fixing. The interesting
  target (Cu, Delta ~7.9) is a metal with smearing and a 16^3 mesh, roughly
  20x the SCF cost per volume.
- Adam at lr 0.05 oscillates around the optimum (Delta dips to 0.013 at step 1,
  ends at 0.057); the trajectory minimum and the final point differ. A real run
  wants a smaller step or a line search; with the loss quadratic and the map
  linear at frozen density, one Gauss-Newton step on the (already computed)
  Jacobian would land at the optimum directly.

## Budget

All heavy runs on asus (22 cores, idle). Totals: 314 kept SCFs (+~92
discarded in the crashed step-4 attempt), about 16 minutes of asus wall time
across step1 60 s, step2 90 s, step3 395 s, step3b ~60 s, step4 283 s, plus
setup. Local laptop use was limited to plumbing checks after an initial timing
probe found the box at load 56 on 8 cores from sibling agents.

## Go/no-go and recommended next scope

GO on the machinery, NO-GO on EOS-only training. Everything the idea needs from
the code side exists and is exact: endpoint bit-for-bit, HF gradient to 3e-10,
stable warm-started training at ~2 s/SCF-step-volume on asus. The blocker is
identifiability, not differentiability. Recommended next scope, in order:

1. Multi-observable loss before any new element: add valence eigenvalues at a
   few k-points (already differentiable via the same HF argument) and forces on
   a displaced cell to the EOS loss, then rerun the synthetic recovery oracle.
   Success there is the real go signal.
2. Then Cu at converged basis on asus (the actual 7.9 meV/atom target), with a
   1-2 parameter correction chosen along the leading singular directions of the
   multi-observable Jacobian, or Gauss-Newton on the linear map instead of Adam.
3. The D_ij (KB coefficient) rung only after the local channel shows a real
   Delta gain on Cu: the nonlocal channel has the same linear-in-parameter HF
   structure through `dij_full` (see `blend_projector_data`), so the plumbing
   cost is small, but it multiplies the null-space question, so it needs the
   richer loss first.
