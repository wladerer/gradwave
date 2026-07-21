# Basis-set error estimation

A plane-wave calculation has one systematic convergence knob, the kinetic-energy
cutoff `ecut`. The usual way to know whether it is converged is a cutoff sweep,
several full calculations at rising `ecut`. gradwave instead estimates the remaining
plane-wave (Ecut) error from a **single** converged calculation, as a cheap post-SCF
pass, following the perturbation post-processing of
Cancès et al.[[18]](bibliography.md#cances)

Turn it on and the calculation reports how far the energy still has to fall, the
extrapolated energy, the density error, the band-gap error, and (for a
norm-conserving calculation, spin-unpolarized or spin-polarized) the force error.

## Theory

The occupied orbitals are converged inside the sphere $T_G \le E_\text{cut}$ but
truncated at its edge. Enlarge the sphere to $E_\text{cut} < T_G \le
E_\text{cut}^\text{large}$ and estimate the part of each orbital on
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
  the second-order term. $\delta E < 0$ always, so the exact energy is below the
  computed one.
- **The force error is one extra pass.** Propagating the *fixed* orbital
  correction through the force, $\delta F \approx (\partial F / \partial P)\,
  \delta P$, needs a single automatic-differentiation pass, no response solve. It
  works because the force's sensitivity is dominated by the ion-motion term
  $\langle \delta\psi | \partial R / \partial \tau\rangle$. The estimate tracks
  the true error closely (correlation ~0.99 on displaced diamond).

For ultrasoft and PAW the density error has **two channels**: the smooth part from
$\delta\psi$ and an augmentation part from the change in the on-site occupations
(becsum), fed through the $Q$ functions, using the generalized residual $R =
P_\text{annulus}(\hat{H} - \varepsilon \hat{S})\psi$.

!!! note "Stress: hydrostatic part only, and not a rigorous bound"
    The same fixed-$\delta P$ recipe does **not** extend to the full stress
    tensor: it is *anti*-correlated with the true error, because the stress error
    is dominated by the strain-response of the orbital correction, which a
    fixed-$\delta\psi$ pass omits. The **hydrostatic (pressure) component** -- the
    ~95% of the stress error that is the spurious Pulay pressure of a too-small
    basis -- is available through `estimate_pressure_error` (below); the full
    anisotropic tensor is still deferred. As everywhere here, the estimate is a
    first-order indicator for gating convergence, not a certified error bar.

## Run it from an input file

The estimate is on by default for `scf`, `bands`, and `relax` tasks (for a
relax, it describes the final geometry — the calculator's last converged SCF).
Set `error_estimate: false` (either at the top level or under `output:`) to
skip it. The `magnetism` task carries no automatic block, but the estimator does
cover the spinor path: run `task: scf` with `noncollinear: true` and the block
attaches to the non-collinear/SOC SCF (density, energy, and eigenvalue error;
`use_symmetry=False`).

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
```

The `scf.json` gains an `error_estimate` block and `scf.out` a matching section.

## Read the output

The human report prints a `basis-set error estimate` section. The JSON block
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
| `scf_convergence` | sub-block: the SCF self-consistency energy error (see below) |
| `smearing` | sub-block: the finite-temperature (smearing) energy error (see below) |

When the calculation is outside the supported coverage the report prints
`unavailable — <reason>` rather than failing the calculation.

## Drive it from Python

The estimator is a small set of functions. `estimate_density_error` takes a
converged `scf` result (norm-conserving) or a `scf_uspp` dict (USPP/PAW).
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

## Beyond the basis: SCF, smearing, and k-point error

The plane-wave cutoff is one axis of the numerical error budget. Three more sit
in `postscf.convergence_error`, and each has a different structure. None of them
touch the exchange-correlation model error, which no internal estimate reaches
(the reasoning is in `docs/ideas.md`).

**SCF convergence error.** Stopping the iteration at a finite tolerance leaves
the reported free energy a little above the fully self-consistent value
$E_\infty$. The headline estimate reads this distance off the recorded energy
trajectory: in the convergence basin the tail is geometric,
$E_i - E_\infty \sim q^{\,i}$, so the unobserved remainder sums to

$$ E_\infty - E_\text{last} \approx \delta E_\text{last}\,\frac{q}{1 - q}, $$

with $q$ the ratio of the last energy steps. This needs one run and no response
solve, and because it reads only the recorded energies it works for every system
(metal, spin, symmetry, USPP, noncollinear), reporting a non-negative `denergy`
and the extrapolated `energy_converged_estimate`. `reliable` is `False` when the
tail is too short or not clearly contracting, where `denergy` falls back to the
last $|\delta F|$ as an order-of-magnitude proxy.

```python
from gradwave.postscf.convergence_error import estimate_scf_error

scfe = estimate_scf_error(res)          # res from a (possibly loose) scf(...)
print(scfe.denergy, scfe.reliable, scfe.energy_converged_estimate)
```

A second-order *response* form is available as a diagnostic when the functional
and the collinear response primitives are supplied. Because the energy is
stationary at the fixed point the error is second order in the residual
$r = \rho_\text{out} - \rho_\text{in}$, and the exact form is
$\tfrac{1}{2}\langle x | (K_\text{Hxc} - \chi_0^{-1}) | x\rangle$ with
$x = (1 - \chi_0 K_\text{Hxc})^{-1} r$. The code can only form
$\tfrac{1}{2}\langle r | K_\text{Hxc}(1 - \chi_0 K_\text{Hxc})^{-1} | r\rangle$,
which omits the $\chi_0^{-1}$ kinetic-response term (that term needs a
near-singular $\chi_0^{-1}$ solve; see `docs/ideas.md`) and is therefore not
sign-definite. It is reported as `denergy_response`/`denergy_unscreened` for
analysis and never drives the headline.

```python
scfe = estimate_scf_error(res, PBE())   # adds the response diagnostic (nspin=1 insulator)
print(scfe.denergy_response, scfe.screened)   # NOT sign-definite; diagnostic only
```

For a ground-truth number, run the SCF loose and tight and compare with
`estimate_scf_error_bracket(res_loose, res_tight)`, which returns the measured
$F_\text{loose} - F_\text{tight}$ next to the loose run's extrapolated estimate.

**Smearing error.** A finite electronic temperature $\sigma$ reports the free
energy $F = E - \sigma S$, not the $\sigma\to 0$ energy. The scheme-matched
extrapolation $E_0 = (E + F)/2$ cancels the leading entropy-order term for every
smearing this code carries, because each is a matched occupation/entropy pair.
The reported free energy differs from $E_0$ by $-\sigma S/2$, and that difference
is `dsmearing`. Be careful with the technique. The extrapolation is a variational
one, valid only because the pair is built for it; a fixed-occupation run has no
smearing error at all (the estimator raises), and a deliberately
physical-temperature Fermi-Dirac run wants the finite-$T$ free energy kept rather
than removed. The `note` field states the per-scheme caveat.

```python
from gradwave.postscf.convergence_error import estimate_smearing_error

sme = estimate_smearing_error(res, scheme="mp1", width=0.2)
print(sme.energy_extrapolated, sme.dsmearing)   # E0 and the F -> E0 correction
```

**k-point sampling error.** Brillouin-zone integration is a quadrature, not a
truncated variational space, so the complement/second-order structure does not
transfer. It is reached instead by mesh extrapolation: run the same cell at a few
rising meshes and fit $E(N_k) = E_\infty + c\,N_k^{-p}$, reporting the dense-k
limit and the residual of the finest mesh. This one needs more than a single run,
so it is a Python helper rather than part of the automatic block. Extrapolate at
a fixed smearing width, since a metal's k-convergence rate is set by the
Fermi-surface discontinuity and changes with the width.

```python
from gradwave.postscf.convergence_error import estimate_kpoint_error

# free energies from scf(...) at 4x4x4, 6x6x6, 8x8x8
kp = estimate_kpoint_error([4**3, 6**3, 8**3], [E4, E6, E8])
print(kp["e_infinity_eV"], kp["error_eV"], kp["exponent"])
```

`examples/kmesh_error.py` runs the whole sweep on silicon end to end. Keep the
meshes in the asymptotic regime: a too-coarse mesh sits off the power law and
drags the fit (dropping a 2×2×2 Si point, ~2.5 eV off, is what turns the
extrapolation monotone). At 4/6/8³ it reports $E_\infty \approx -214.483$ eV with
the 8×8×8 residual near 7 meV.

**Pressure (hydrostatic stress) error.** The plane-wave error in the stress is
dominated by its trace, the spurious Pulay pressure of a too-small basis (on
sheared silicon the shear part of the basis-set stress error is ~1% of the
hydrostatic part). Estimate it by differentiating the frozen-state energy error
along a homogeneous strain at a *fixed* Miller set,
$P_\text{error} = -\,\mathrm{d}(\delta E_\text{error})/\mathrm{d}V$. Holding the
integer $G$-labels while the metric strains (the $e_\text{cut}\to e_\text{cut}/s^2$
map for a scale $s$) is what makes this work: differentiating through a basis
whose plane-wave count *jumps* as $G$-vectors cross $e_\text{cut}$ -- the naive
volume derivative at fixed cutoff -- is anti-correlated, like the fixed-$\delta P$
tensor form. This one needs `use_symmetry=False` (the frozen strained rebuild
reproduces the run's full $k$-point set).

```python
from gradwave.postscf.stress_error import estimate_pressure_error

pe = estimate_pressure_error(res, PBE())   # res from a (loose) scf(...), no symmetry
print(pe["pressure_error_kbar"])           # add to the reported pressure toward the large-basis limit
```

It is correctly signed (the load-bearing property the naive forms get wrong) and
captures ~0.5–0.75× of the true pressure error over $e_\text{cut}\sim$ 10–18 Ry on
silicon, the ratio rising toward 1 as the cutoff converges -- a consistent
under-estimate, so it does not give false confidence. The full anisotropic
stress-error tensor is deferred (it needs the strain-differentiated orbital
residual, not just its trace).

## Coverage

| quantity | norm-conserving | USPP/PAW | non-collinear / SOC |
|---|---|---|---|
| density error | nspin=1, nspin=2 | nspin=1, nspin=2 | spinor |
| energy error | nspin=1, nspin=2 | nspin=1, nspin=2 | spinor |
| force error | nspin=1, nspin=2 (no NLCC) | not available | not available |
| eigenvalue / gap error | nspin=1, nspin=2 | not available | eigenvalue (spinor) |
| stress error (pressure) | nspin=1, no symmetry | not available | not available |
| stress error (full tensor) | not available (deferred) | not available | not available |
| SCF error (trajectory) | any run | any run | any run |
| SCF error (response diagnostic) | nspin=1 insulator, no symmetry | not available | not available |
| smearing error | any smeared run (all schemes) | any smeared run | any smeared run |
| k-point error | mesh sweep (any run) | mesh sweep (any run) | mesh sweep (any run) |

Symmetry is supported for the norm-conserving nspin=1 density, energy, and force
error: the estimate runs on the IBZ and folds the density error over the star
with the same symmetrizer the SCF applies to the density, matching the full-BZ
result to ~1e-4. USPP/PAW, nspin=2, the non-collinear/SOC spinor path, and the
opt-in Dyson dressing require `use_symmetry=False`.

## Gotchas

- It is a first-order *indicator*, meant to gate convergence ("is my cutoff good
  enough"), not to quote an uncertainty.
- `int_drho` should be ~0. A nonzero value flags an under-converged density or a
  padding problem, not a small real error.
- The coarse-space **Dyson refinement** (`dyson=True`) is opt-in, Python-only, and
  not yet validated. Leave it off.
- Choose `ecut` genuinely loose to see a signal. At a well-converged cutoff
  `denergy_eV` is already tiny, which is the answer you want.

## Next

Continue to [Differentiable Hubbard U](hubbard-u.md). The symmetric error estimate
reuses the IBZ machinery of [Symmetry reduction](symmetry.md).
