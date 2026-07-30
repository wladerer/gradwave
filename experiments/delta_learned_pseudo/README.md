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
Hellmann-Feynman Jacobian (autograd, 5 SCFs). SVD of A: see
`results_step3b.json`. The singular spectrum is steeply graded (condition number
~1e3), so the EOS shape determines roughly one to two directions in theta-space
and leaves a two-plus-dimensional near-null space; the step-3 recovery error
lies almost entirely in that null space. Physically: the four Gaussian bumps
(mu = 1..4 1/Ang) act on the E(V) curve almost interchangeably, because the
five-volume window samples dv only through slowly-varying combinations of the
few low-|G| shells (|G_min| ~ 2.0 1/Ang for Si diamond), and the mu = 1 bump is
sampled only through its tail.

Conclusion for the go/no-go: the *machinery* (differentiable correction, exact
gradient, trainable loop) is validated end to end, but an EOS-only loss cannot
identify a multi-parameter correction. Any real training needs either (a) a
parameterization with as many effective degrees of freedom as the loss actually
constrains (1-2 for an EOS window), or (b) a richer loss (eigenvalues, log
derivatives, forces, a second structure) that fills the null space.

### Step 4: one real step against WIEN2k Si

(RESULTS PLACEHOLDER -- updated after the asus run.)
