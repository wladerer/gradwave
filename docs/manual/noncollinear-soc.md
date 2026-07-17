# Non-collinear magnetism and spin-orbit coupling

gradwave solves the two-component (spinor) Kohn-Sham problem, so magnetic moments
can point in any direction and spin-orbit coupling (SOC) can mix them. The showcase
is a topological band inversion: in Bi₂Se₃ spin-orbit coupling swaps the parity of
the states across the gap at Γ, the fingerprint of a topological insulator.

## Theory

A non-collinear wavefunction is a two-component spinor, and gradwave stores it as a
doubled plane-wave vector, the spin-up block followed by spin-down. The density
becomes a scalar charge plus a vector magnetization through the Pauli decomposition,

$$ \rho = \psi^\dagger \psi, \qquad \mathbf{m} = \psi^\dagger \boldsymbol\sigma\, \psi, $$

and the potential gains a magnetic part, $\hat V = (v_H + v_\text{loc} + v_\text{xc})\,
\mathbb{1} + \mathbf{B}_\text{xc}\cdot\boldsymbol\sigma$, with $\mathbf{B}_\text{xc}$
from autograd of the non-collinear functional. The charge and the three
magnetization fields are mixed jointly, with Kerker preconditioning on the charge
channel only.

**Spin-orbit coupling** comes from a fully-relativistic pseudopotential, whose
projectors are resolved by total angular momentum $j = l \pm \tfrac12$. gradwave
builds these $j$-resolved spinor projectors from complex spherical harmonics and
Clebsch-Gordan coefficients[[21]](bibliography.md#dalcorso) and adds them as a genuine $2\times2$ block in
the non-local Hamiltonian. Because spin-orbit coupling breaks the separate spin
and spatial rotation symmetries, time-reversal k-reduction is kept only through
Kramers degeneracy for a nonmagnetic cell, and the mesh falls back to the full
Brillouin zone once a net moment breaks it.

## The non-collinear SCF

`scf_noncollinear` takes an initial per-atom moment direction and magnitude and
converges the spinor density.

```python
from gradwave.scf.noncollinear import scf_noncollinear
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92

res = scf_noncollinear(
    system, NoncollinearXC(LSDA_PW92()),   # wrap any collinear SpinXC
    mag_vec_init=[[0.0, 0.0, 0.4]],   # (na, 3): direction · fraction per atom
    smearing="gaussian", width=0.1,
)
res.mag_vec     # ∫ m⃗ dr — the net moment vector
res.mag_abs     # ∫ |m⃗| dr
res.m           # (3, grid) magnetization field
```

For a nonmagnetic system where only the spin-orbit splitting matters, pin the
moment to zero with `nonmagnetic=True` (QE's `domag=false`): the spinor structure
and SOC stay, the magnetization does not. The converged moment *direction* is
whatever the unconstrained SCF settles into.

## Spin-orbit band inversion in Bi₂Se₃

`examples/bi2se3_inversion.py` runs the calculation twice — a scalar-relativistic
`scf` and a fully-relativistic `scf_noncollinear(..., nonmagnetic=True)` — and
labels the parity of the Γ states around the gap. Without SOC the ordering is the
normal-insulator one; with SOC the conduction and valence parities **swap**, the
$Z_2$-nontrivial signature of the topological surface states.[[22]](bibliography.md#bi2se3) The band overlay
is committed alongside the script.

The SOC machinery is validated quantitatively on the GaAs valence split-off: the
$\Gamma_8$ (four-fold) sits above $\Gamma_7$ (two-fold) with a spin-orbit gap
$\Delta_0 = 0.336$ eV against QE's fully-relativistic reference (experiment 0.34 eV),
agreeing to $2\times10^{-3}$ eV. Spin-orbit character is also resolvable in the
projected density of states, separating a shell into its $j$ channels (a
$6P_{1/2}$ from a $6P_{3/2}$, for instance).

## Collinear spin, with numbers

When the moments are collinear a full spinor solve is unnecessary — set `nspin=2`
and an initial moment fraction, and read the converged moment off the result.

```python
res = scf(system, SpinPBE(), nspin=2, start_mag=[0.4],   # bcc Fe
          smearing="gaussian", width=0.1)
res.mag_total    # ∫ (ρ↑ − ρ↓) dr [μB]
```

This path is QE-validated: bcc Fe converges within 0.02 $\mu_B$ of QE's moment and
under 1 meV/atom, and the triplet O₂ molecule lands the $m = 2\,\mu_B$ moment to
$10^{-3}$. The USPP/PAW spin path (`scf_uspp`, `nspin=2`) carries the same
`start_mag` and `mag_total`.

## Optimizing the magnetic configuration

The moment *directions* are themselves degrees of freedom, and gradwave finds the
ground-state configuration by constrained density-functional theory: each atomic
moment is held toward a target direction $\hat{\mathbf{e}}_I$ by a penalty field,
and the torque that would rotate the *unconstrained* moment is read off and
descended.

Following Ma and Dudarev,[[23]](bibliography.md#madudarev) an atomic moment $\mathbf{M}_I = \int
w_I(\mathbf{r})\,\mathbf{m}(\mathbf{r})\,\mathrm{d}^3r$ — with a Hirshfeld weight
$w_I$ localizing on atom $I$ — is pinned to $\hat{\mathbf{e}}_I$ by adding a
penalty $E_p = \sum_I \lambda\,|\mathbf{M}_I^\perp|^2$ to the energy, which
contributes a constraining field $\mathbf{B}_c = 2\lambda \sum_I w_I\,
\mathbf{M}_I^\perp$ to the spinor Hamiltonian. The gradient of the constrained
functional $W = E_\text{KS} + E_p$ with respect to a target direction is

$$ \frac{\partial W}{\partial \hat{\mathbf{e}}_I} = -2\lambda\,(\mathbf{M}_I \cdot \hat{\mathbf{e}}_I)\,\mathbf{M}_I^\perp, $$

which gradwave validates against a finite difference of $W$ (they agree to a ratio
of 1.000). A configuration is a stationary point of the true energy when no
constraint is needed, $\mathbf{M}_I^\perp \to 0$.

```python
from gradwave.postscf.moment_config import relax_moment_directions

# two O moments started 45° apart, in the x–z plane
dirs0 = [[0.0, 0.0, 1.0], [0.707, 0.0, 0.707]]
final, history = relax_moment_directions(
    system, NoncollinearXC(LSDA_PW92()), dirs0,
    lam=2.0, step=0.5, smearing="gaussian", width=0.1)
```

Each sweep runs a constrained SCF, reads the torque, and rotates the targets
downhill. For triplet O₂, a ferromagnet, two moments started 45° apart collapse
to parallel in three sweeps and the energy falls to the unconstrained
ground-state value:

| sweep | relative angle | energy (eV) |
|---|---|---|
| start | 45.0° | −840.675 |
| 1 | 4.6° | −840.823 |
| 2 | 0.3° | −840.825 |

`constrained_moment_scf` runs a single constrained point (returning the atomic
moments, the constraining field, and the torque) and `relax_moment_directions`
wraps the descent. With a fully-relativistic pseudopotential the same machinery
gives the magnetocrystalline-anisotropy torque, since spin-orbit coupling ties
the moment to the lattice.

### The magnitude problem, and holding a moment at any angle

The $|\mathbf{M}^\perp|^2$ penalty constrains only the moment *direction*, and it
is minimized ($E_p \to 0$) at $\mathbf{M}=0$. So a strongly-coupled magnet forced
to a large relative angle has a cheap escape: it demagnetizes rather than holding
its moments apart. This is not a numerical artifact — reducing $|\mathbf{M}|$ is a
real way out of frustration — but it means `mode="perp"` cannot represent, say, a
metastable antiferromagnetic state.

The `mode="vector"` penalty pins the full moment vector,
$E_p = \sum_I \lambda\,|\mathbf{M}_I - m^0_I\,\hat{\mathbf{e}}_I|^2$, so
demagnetizing now costs $\lambda\,(m^0_I)^2$. The target magnitude $m^0_I$ defaults
to the unconstrained self-consistent $|\mathbf{M}_I|$ (measured once by
`reference_moment_magnitudes`). Forcing the two O moments of O₂ *antiparallel*
shows the difference sharply:

```python
m0 = reference_moment_magnitudes(system, xc, [[0, 0, 1], [0, 0, 1]], weights=w)
afm = [[0, 0, 1], [0, 0, -1]]                       # target: antiparallel
_, perp = constrained_moment_scf(system, xc, afm, lam=8.0, weights=w, mode="perp")
_, vec  = constrained_moment_scf(system, xc, afm, lam=8.0, weights=w,
                                 mode="vector", target_mag=m0)
```

| mode | $\lvert\mathbf{M}_I\rvert$ (μB) | $M_z$ (μB) | outcome |
|---|---|---|---|
| `perp`   | 0.00, 0.00 | 0.00, 0.00 | demagnetized — constraint met for free |
| `vector` | 0.81, 0.81 | +0.81, −0.81 | genuine antiferromagnet, held |

The held antiferromagnetic state sits ≈3.0 eV above the ferromagnetic ground
state — O₂'s exchange splitting, a number `perp` cannot produce because it never
holds the moments. The field, the torque, and both penalty forms are one
differentiable definition (`gradwave.scf.moment_penalty`), so the SCF field and
the config-search gradient stay consistent by construction, and the gradient
matches a finite difference of $W$ to a part in $10^3$.

!!! note "Penalty stiffness vs. convergence"
    `vector` holds *magnitude* robustly, but a finite $\lambda$ is a soft
    constraint: it trades the target *angle* against the exchange energy, and the
    constrained SCF for a strongly-frustrated forced angle can settle into a small
    residual plateau rather than converging tightly. Raise $\lambda$ to hold the
    angle stiffer; a natural collinear axis (parallel or antiparallel) converges
    far more easily than an oblique one.

## Gotchas

- A fully-relativistic pseudopotential is required for SOC; the collinear `scf`
  rejects one. PseudoDojo fully-relativistic sets work; note that their NLCC is
  unsupported on the spin-orbit path, while SG15 fully-relativistic works.
- Symmetry is off for a magnetic non-collinear run — a net moment breaks the
  crystal symmetry — so these use the full mesh. A nonmagnetic SOC run keeps
  time-reversal (Kramers) reduction.
- The $\pm\mathbf{m}$ branches are exactly degenerate without spin-orbit coupling;
  the branch the SCF lands on depends on the trajectory. Gate on the moment
  magnitude, not its sign.

## Next

See the [Reference](reference.md) page for the CLI, output files, and entry points.
