# Basis-set error estimation

A plane-wave calculation has one systematic convergence knob, the kinetic-energy
cutoff `ecut`. The usual way to know whether it is converged is a cutoff sweep,
several full runs at rising `ecut`. gradwave instead estimates the remaining
plane-wave (Ecut) error from a **single** converged run, as a cheap post-SCF
pass that needs no larger SCF, following the perturbation post-processing of
Cancès et al.[[18]](bibliography.md#cances)

Turn it on and the run reports how far the energy still has to fall, the
extrapolated energy, the density error, the band-gap error, and — for a
norm-conserving run (spin-unpolarized or spin-polarized) — the force error.

## Theory

The occupied orbitals are converged inside the sphere $T_G \le E_\text{cut}$ but
truncated at its edge. Enlarge the sphere to $E_\text{cut} < T_G \le
E_\text{cut}^\text{large}$ and estimate the piece of each orbital that lives on
that annulus. At high kinetic energy the Hamiltonian is dominated by the diagonal
Laplacian, so the first-order correction is a diagonal divide,

$$ \delta\psi_i(G) = -\frac{R_i(G)}{T_G - \varepsilon_i}, \qquad R_i = P_\text{annulus}\, \hat{H}\, \psi_i, $$

no larger SCF required. Three facts make the estimate trustworthy and cheap.

- **The density change integrates to zero.** $\delta\psi_i$ is orthogonal to the
  occupied space, so $\int \delta\rho = 0$ to first order. A nonzero integral
  means the annulus or the padding is wrong.
- **The energy error is second order, a definite lowering.** The correct estimate
  is $\delta E = \sum_i f_i \langle \delta\psi_i | R_i \rangle$ with a factor of
  one, not two: at the variational optimum the naive first-order term is halved by
  the second-order term. $\delta E < 0$ always — the exact energy is below the
  computed one.
- **The force error is one extra pass.** Propagating the *fixed* orbital
  correction through the force, $\delta F \approx (\partial F / \partial P)\,
  \delta P$, needs a single automatic-differentiation pass, no response solve. It
  works because the force's sensitivity is dominated by the ion-motion term
  $\langle \delta\psi | \partial R / \partial \tau\rangle$; the estimate tracks
  the true error closely (correlation ~0.99 on displaced diamond).

For ultrasoft and PAW the density error has **two channels**: the smooth part from
$\delta\psi$ and an augmentation part from the change in the on-site occupations
(becsum), fed through the $Q$ functions, using the generalized residual $R =
P_\text{annulus}(\hat{H} - \varepsilon \hat{S})\psi$.

!!! note "Not stress, and not a rigorous bound"
    The same fixed-$\delta P$ recipe does **not** extend to stress: it comes out
    cleanly *anti*-correlated with the true error, because the stress error is
    dominated by the strain-response of the orbital correction, which a fixed-$\delta\psi$
    pass omits. Stress error is deferred. And the estimate is a first-order
    indicator for gating convergence, not a certified error bar.

## Run it from an input file

Set `error_estimate: true` (either at the top level or under `output:`) on an
`scf` task.

```yaml
task: scf
structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions: {frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]}
  species: [Si, Si]
pseudopotentials:
  dir: ../tests/fixtures/qe/pseudos
  map: {Si: Si_ONCV_PBE-1.2.upf}
ecut: 204.0                 # eV (15 Ry) — deliberately loose, to have an error to see
xc: pbe
kpoints: {mesh: [4, 4, 4]}
output:
  dir: ./out_si
  error_estimate: true
```

The `scf.json` gains an `error_estimate` block and `scf.out` a matching section.

## Read the output

The human report prints a `basis-set error estimate` section; the JSON block
carries the same fields.

| field | meaning |
|---|---|
| `denergy_eV` | the estimated energy error $\delta E$ (negative, a definite lowering) |
| `denergy_meV_per_atom` | the same per atom |
| `free_energy_extrapolated_eV` | $F + \delta E$, the energy the cutoff is converging to |
| `drho_L1_per_electron` | $\int |\delta\rho|$ per electron, the density-error norm |
| `int_drho` | $\int \delta\rho$; near zero confirms the correction is charge-conserving |
| `force_error_max_eV_ang`, `force_error_rms_eV_ang` | the force error (NC nspin=1 or 2) |
| `gap_eV`, `gap_extrapolated_eV`, `dgap_eV` | the band gap, its extrapolation, and the error (NC insulators) |
| `ecut_eV`, `ecut_large_eV` | the base and enlarged cutoffs |

When the run is outside the supported coverage the block is
`{"available": false, "reason": ...}` and the report prints `unavailable —
<reason>` rather than failing the run.

## Drive it from Python

The estimator is a small set of functions. `estimate_density_error` takes a
converged `scf` result (norm-conserving) or a `scf_uspp` dict (USPP/PAW);
`estimate_force_error` turns that into a per-atom force error, and
`estimate_eigenvalue_error` / `estimate_gap_error` (below) give band and gap
errors.

```python
from gradwave.postscf.discretization_error import (
    estimate_density_error, estimate_force_error)

# res from scf(...) at a loose ecut
err = estimate_density_error(res, ecut_large=35 * RY)
print(err.denergy)                 # eV, < 0
print(err.drho.abs().sum())        # density-error magnitude
dF = estimate_force_error(res, err)   # (na, 3) eV/Å; add to F to approach the limit
```

`ecut_large` defaults to `2.5 * ecut` and must satisfy `ecut_large <= 4 * ecut`
so the enlarged sphere fits inside the density FFT box. For USPP/PAW pass the
functional too, `estimate_density_error(res_uspp, ecut_large=30*RY, xc=PBE())`.

## Eigenvalue and gap error

The per-band term the energy error sums over occupations, $\delta\varepsilon_i =
\langle \delta\psi_i | R_i \rangle$, is exactly the second-order shift of the
$i$-th Kohn-Sham eigenvalue toward the infinite-basis limit ($\delta\varepsilon
\le 0$, a definite lowering). Running it on the empty bands as well turns the
estimator into a band-structure and band-gap error tool at no extra SCF cost.

```python
from gradwave.postscf.discretization_error import (
    estimate_eigenvalue_error, estimate_gap_error)

eige = estimate_eigenvalue_error(res, ecut_large=35 * RY)  # per-band δε [eV]
gap = estimate_gap_error(res, eige)     # dict: gap, extrapolated gap, δgap, VBM/CBM
print(gap["gap_eV"], gap["gap_extrapolated_eV"], gap["dgap_eV"])
```

Because the occupied shifts are the same quantity the energy error integrates,
their occupation-weighted BZ sum reproduces `denergy` exactly. `estimate_gap_error`
locates the VBM and CBM over the BZ (and both spin channels) and reports the raw
gap, the extrapolated gap $\varepsilon + \delta\varepsilon$ at each edge, and
their difference; it raises for a metal/semimetal. On loosely converged silicon
the extrapolated gap recovers roughly half of the remaining basis-set gap error.

## Coverage

| quantity | norm-conserving | USPP/PAW |
|---|---|---|
| density error | nspin=1, nspin=2 | nspin=1, nspin=2 |
| energy error | nspin=1, nspin=2 | nspin=1, nspin=2 |
| force error | nspin=1, nspin=2 (no NLCC) | not available |
| eigenvalue / gap error | nspin=1, nspin=2 | not available |
| stress error | not available (deferred) | not available |

Symmetry is supported for the norm-conserving nspin=1 density, energy, and force
error: the estimate runs on the IBZ and folds the density error over the star
with the same symmetrizer the SCF applies to the density, matching the full-BZ
result to ~1e-4. USPP/PAW, nspin=2, and the opt-in Dyson dressing require
`use_symmetry=False`.

## Gotchas

- It is a first-order *indicator*, meant to gate convergence ("is my cutoff good
  enough"), not to quote an uncertainty.
- `int_drho` should be ~0. A nonzero value flags an under-converged density or a
  padding problem, not a small real error.
- The coarse-space **Dyson refinement** (`dyson=True`) is opt-in, Python-only, and
  not yet validated — leave it off.
- Choose `ecut` genuinely loose to see a signal; at a well-converged cutoff
  `denergy_eV` is already tiny, which is the answer you want.

## Next

Continue to [Differentiable Hubbard U](hubbard-u.md). The symmetric error estimate
reuses the IBZ machinery of [Symmetry reduction](symmetry.md).
