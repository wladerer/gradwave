# Non-collinear magnetism and spin-orbit coupling

gradwave solves the two-component (spinor) Kohn-Sham problem, so magnetic moments
can point in any direction and spin-orbit coupling (SOC) can mix them. A topological
band inversion demonstrates it. In Bi₂Se₃ spin-orbit coupling swaps the parity of
the states across the gap at Γ, the fingerprint of a topological insulator.

This page covers the spinor SCF, SOC from a fully-relativistic pseudopotential, and
collinear spin as the cheaper special case. To extract magnetic structure, the
ground-state moment configuration, exchange constants, and spin Hamiltonians, see
[Magnetic structure and spin Hamiltonians](magnetism.md).

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
res.mag_vec     # ∫ m⃗ dr, the net moment vector
res.mag_abs     # ∫ |m⃗| dr
res.m           # (3, grid) magnetization field
```

For a nonmagnetic system where only the spin-orbit splitting matters, pin the
moment to zero with `nonmagnetic=True` (QE's `domag=false`). The spinor structure
and SOC stay, the magnetization does not. The converged moment *direction* is
whatever the unconstrained SCF settles into.

## Spin-orbit band inversion in Bi₂Se₃

`examples/bi2se3_inversion.py` runs the calculation twice, a scalar-relativistic
`scf` and a fully-relativistic `scf_noncollinear(..., nonmagnetic=True)`, and
labels the parity of the Γ states around the gap. Without SOC the ordering is the
normal-insulator one. With SOC the conduction and valence parities **swap**, the
$Z_2$-nontrivial signature of the topological surface states.[[22]](bibliography.md#bi2se3) The band overlay
is committed alongside the script.

The SOC machinery is validated quantitatively on the GaAs valence split-off. The
$\Gamma_8$ (four-fold) sits above $\Gamma_7$ (two-fold) with a spin-orbit gap
$\Delta_0 = 0.336$ eV against QE's fully-relativistic reference (experiment 0.34 eV),
agreeing to $2\times10^{-3}$ eV. Spin-orbit character is also resolvable in the
projected density of states, separating a shell into its $j$ channels (a
$6P_{1/2}$ from a $6P_{3/2}$, for instance).

## Collinear spin, with numbers

When the moments are collinear a full spinor solve is unnecessary. Set `nspin=2`
and an initial moment fraction, and read the converged moment off the result.

```python
res = scf(system, SpinPBE(), nspin=2, start_mag=[0.4],   # bcc Fe
          smearing="gaussian", width=0.1)
res.mag_total    # ∫ (ρ↑ − ρ↓) dr [μB]
```

This path is QE-validated. bcc Fe converges within 0.02 $\mu_B$ of QE's moment and
under 1 meV/atom, and the triplet O₂ molecule lands the $m = 2\,\mu_B$ moment to
$10^{-3}$. The USPP/PAW spin path (`scf_uspp`, `nspin=2`) carries the same
`start_mag` and `mag_total`.

## Gotchas

- A fully-relativistic pseudopotential is required for SOC, and the collinear `scf`
  rejects one. PseudoDojo fully-relativistic sets work, though their NLCC is
  unsupported on the spin-orbit path. SG15 fully-relativistic works.
- Symmetry is off for a magnetic non-collinear run, because a net moment breaks the
  crystal symmetry, so these use the full mesh. A nonmagnetic SOC run keeps
  time-reversal (Kramers) reduction.
- The $\pm\mathbf{m}$ branches are exactly degenerate without spin-orbit coupling,
  and the branch the SCF lands on depends on the trajectory. Gate on the moment
  magnitude, not its sign.

## Next

Continue to [Magnetic structure and spin Hamiltonians](magnetism.md), which reads
the ground-state moment configuration and the exchange constants out of this spinor
SCF.
