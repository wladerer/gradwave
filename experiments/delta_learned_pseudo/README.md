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

See `results_step{1,2,3,4}.json`. Filled in by the run; summary below.

(RESULTS PLACEHOLDER -- updated after the asus run.)
