# Symmetry reduction

Reducing the k-mesh to the irreducible Brillouin zone (IBZ) and symmetrizing the
density each step is the largest single source of speedup in gradwave. The reduction
gives 5 to 14 times, depending on the point group. It is on by default (`symmetry: true`) and
gated by tests that check the reduced and full-mesh energies agree.

## Theory

A crystal's space group is a set of operations $\{W \mid w\}$, a rotation $W$ and
a fractional translation $w$, that map the lattice onto itself. Two consequences
save work.

**k-point reduction.** Bloch states at $k$ and at $W^{-\top} k$ are related by
symmetry, so only one representative per orbit needs solving. gradwave finds the
space group with spglib,[[17]](bibliography.md#togo) builds each k-point's orbit under the inverse-transpose
rotations (plus $k \to -k$ when time reversal holds), and keeps one representative
with a weight equal to its orbit size. A larger point group means larger orbits
and a smaller IBZ. For example, diamond Si ($Fd\bar{3}m$, 48 operations) folds a $4\times4\times4$
mesh from 64 points to 8, matching QE.

**Density symmetrization.** The self-consistent density must carry the full point-group
symmetry. gradwave averages the reciprocal-space density over the star of each
$G$-vector every SCF iteration,

$$ \tilde\rho_\text{sym}(G) = \frac{1}{N_\text{ops}} \sum_{\{W|w\}} e^{-i G \cdot w}\, \tilde\rho(W^{-1} G), $$

masked to the density sphere where the Miller-index map is exact. This averaging projects
out the small asymmetric component that an IBZ sum would otherwise leave, so the
reduced calculation reproduces the full-mesh energy. PAW additionally symmetrizes the
on-site occupancies (becsum), the same job QE's `PAW_symmetrize` does.

## Inspect the symmetry

`find_spacegroup` and `reduce_mesh` work standalone. A pseudopotential is not
needed to count the IBZ.

```python
import numpy as np
from gradwave.symmetry import find_spacegroup, reduce_mesh

a = 5.43                                   # diamond Si
cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
cart = np.array([[0.0, 0, 0], [a / 4] * 3])
frac = cart @ np.linalg.inv(cell)

sg = find_spacegroup(cell, frac, species_of_atom=[0, 0])
print(sg.international, sg.n_ops)          # Fd-3m 48

k, w = reduce_mesh((4, 4, 4), (0, 0, 0), sg, time_reversal=True)
print("full", 4 ** 3, "-> IBZ", len(k))    # full 64 -> IBZ 8
assert abs(w.sum() - 1.0) < 1e-12          # weights sum to one
```

`find_spacegroup(cell, frac_positions, species_of_atom, symprec=1e-6)` returns a
`SpaceGroup` with `.rotations`, `.translations`, `.atom_map`, `.international`,
and `.n_ops`. `reduce_mesh(mesh, shift, sg, time_reversal=True)` returns the IBZ
k-points and their weights. Reduction is valid for unshifted Γ-centered
Monkhorst-Pack meshes. A shifted mesh may not be group-invariant, and the caller
falls back to time-reversal-only folding.

## Magnetic (Shubnikov) symmetry

A magnetic cell does not have to fall back to the full mesh. A finite moment
field changes the symmetry group rather than destroying it: the moment is an
axial vector locked to the lattice, so each paramagnetic operation either maps
the moment field onto itself (a unitary operation of the magnetic group), maps
it onto its reversal (surviving only combined with time reversal, the
anti-unitary half of the Shubnikov group), or relates two different magnetic
configurations and is dropped. `magnetic_spacegroup(sg, magmoms, cell)` performs
that filter, cross-checked against spglib's magnetic symmetry detection, and
`reduce_mesh_magnetic` folds the k-mesh with the unitary operations acting as
$W^{-\top}$ and the anti-unitary ones as $-W^{-\top}$.

To use it, pass the per-atom moment directions at setup:

```python
system = setup_system(cell, pos, species, pseudos, ecut=ecut, kmesh=(6, 6, 4),
                      use_symmetry=True, magmoms=[[0, 0, 3.0], [0, 0, 0.4]])
res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 3.0], [0, 0, 0.4]], ...)
```

`setup_uspp` takes the same argument for the spinor USPP/PAW path. The SCF loop
then re-symmetrizes the charge and the magnetization each iteration under the
full magnetic group, with the magnetization transformed as an axial vector and
reversed under the anti-unitary operations (PAW additionally symmetrizes the
four Pauli channels of the on-site occupancies). Zero moments reproduce the
paramagnetic time-reversal fold exactly.

The savings are largest on the systems where cost is highest. L1_0 FePt with the
moment along [001] folds a $6\times6\times4$ mesh from 144 to 30 k-points, the
in-plane orientation to 48, and bcc Fe folds $4\times4\times4$ from 64 to 13.
The fold introduces no approximation: the magnetic-IBZ SCF reproduces the
full-mesh free energy to $5\times10^{-11}$ eV on the FePt spin-orbit case, and
each orientation may be folded by its own magnetic group for an anisotropy
difference as long as both share the same underlying mesh, because the folded
sum is the full-mesh sum re-weighted. The force-theorem MAE evaluator applies
the same idea per evaluation: `force_theorem_mae(..., magmoms=...)` folds each
one-shot solve into its direction's own magnetic IBZ
([Magnetocrystalline anisotropy](magnetism.md#magnetocrystalline-anisotropy)).

### Collinear FM/AFM meshes

A collinear antiferromagnet forfeits the ordinary space-group reduction. The
operation that swaps the two spin sublattices is a symmetry of the crystal but
not of either spin channel, so a per-channel density symmetrizer would fold ρ↑
and ρ↓ together. Such a system drops to a near-full time-reversal-only mesh (bcc
Cr AFM, a 2-atom cell at $8\times8\times8$, runs 260 k-points). The magnetic
group keeps that swap as an anti-unitary operation $g\cdot T$, which folds k as
$-W^{-\top}$ and, in the density, folds the spatial operation together with a
spin-channel swap.

`collinear_magnetic=True` engages that fold for the collinear (ρ↑, ρ↓) `nspin=2`
path. Pass it with the collinear sublattice pattern in `magmoms=` on an unshifted
mesh, and the collinear `scf` consumes the resulting `CollinearMagneticSymmetrizer`.

```python
system = setup_system(cell, pos, species, pseudos, ecut=ecut, kmesh=(4, 4, 4),
                      use_symmetry=True, collinear_magnetic=True,
                      magmoms=[[0, 0, 1.0], [0, 0, -1.0]])   # 2-atom AFM
res = scf(system, SpinPBE(), nspin=2, start_mag=[0.6, -0.6],
          smearing="gaussian", width=0.1)
```

The moments must be collinear (all parallel to one axis) or the constructor
rejects them. The k-fold (`reduce_mesh_magnetic`) shares the magnetic group with
the spinor path, and the density symmetrizer differs, swapping the spin channels
under the anti-unitary operations. The collinear path adds one thing the spinor
path cannot: the plain $k \to -k$ time reversal. Each spin channel has a real
Hamiltonian, $H_\sigma(-k) = H_\sigma(k)^*$, so $n_\sigma(-k) = n_\sigma(k)$ with
no spin swap — folding $-k$ onto $k$ only doubles its weight and leaves the
moment untouched. The fold reproduces the unreduced collinear SCF exactly and
introduces no approximation. bcc Cr AFM at $4\times4\times4$ folds the
time-reversal-only 36 k-points to 18 and reproduces the full-mesh free energy per
atom to $10^{-6}$ eV and the moment to $10^{-4}$ μB. The $8\times8\times8$ mesh
reclaims more, 260 to 75.

The plain $k \to -k$ fold is what a magnetic group leaves out when its
sublattice-swap operation carries no $k$-inversion. bcc Cr gets $k \to -k$ for
free: its swap is a pure lattice translation ($W = I$, so the anti-unitary
$-W^{-\top} = -I$ already inverts $k$). The corundum antiferromagnets (hematite
$\alpha$-Fe$_2$O$_3$ and eskolaite Cr$_2$O$_3$, R-$\bar{3}$c, #167) do not: their
swap is inversion$\cdot T$ ($W = -I$, so $-W^{-\top} = +I$, a trivial
$k$-action). Without the added time reversal their $4\times4\times4$ mesh folds
only to 20 and 16; with it both reach 13, matching Quantum ESPRESSO, and the
folded free energy still equals the full-mesh value because the reference mesh
assumes the same per-channel $k \to -k$.

## Use it in a calculation

Symmetry is a single input flag, on by default:

```yaml
symmetry: true        # IBZ reduction + density symmetrization (default)
```

Set `symmetry: false` to run the full mesh. Under the hood `setup_system` /
`setup_uspp` detect the space group, reduce the mesh, and attach a density
symmetrizer that the SCF loop applies to the output density each iteration. A P1
cell (no symmetry) transparently falls back to the full mesh. From Python the
same flag is `use_symmetry=True`.

## Validation

The reduced and full-mesh calculations must give the same energy, and the suite gates on
it:

- norm-conserving Si and Al: IBZ vs full-mesh free energy agree to $5\times10^{-7}$ eV.
- PAW diamond Si: IBZ + ρ/becsum symmetrization vs the full time-reversal mesh
  agree to $10^{-7}$ eV, folding 36 k-points to 8 for ~5× less work.
- the symmetrizer is an exact idempotent projector on the density sphere
  (applying it twice matches to $10^{-14}$) and enforces $F_1 = -F_2$ on a
  two-atom cell to $10^{-10}$.

See [Performance](performance.md) for how the 5-to-14× speedup breaks down by system.

## Gotchas

- **Non-symmorphic operations must be commensurate with the grid.** A glide like
  the diamond $(\tfrac14, \tfrac14, \tfrac14)$ needs FFT dimensions divisible by 4.
  gradwave equalizes symmetry-coupled axes automatically. On an incommensurate box
  the full-mesh fixed point itself carries a ~$2\times10^{-4}$ asymmetric density
  component, so an IBZ-vs-full comparison fails at $10^{-4}$ with no bug present.
- **The symmetrizer is masked to the density sphere.** At the box Nyquist boundary
  the folded Miller map misidentifies $G$-vectors for glide phases. Physical
  densities are zero there, and masking makes the operator exactly idempotent.
- **Turn symmetry off for symmetry-breaking work.** A real response to a
  perturbation (the implicit-differentiation backward, USPP/PAW or nspin=2
  discretization-error estimates, the Dyson dressing) needs `use_symmetry=False`,
  because the perturbation lowers the crystal symmetry. For a magnetic ordering
  the right tool is the magnetic group above: pass `magmoms=` instead of turning
  symmetry off. The spinor paths take it directly, and the collinear `nspin=2`
  path takes it with `collinear_magnetic=True`. A collinear system built with
  `magmoms=` but without that flag is rejected, since the plain per-channel
  symmetrizer would mis-fold the spin channels.
- **Time reversal** ($k \to -k$) is on by default and shrinks the IBZ further.
  A net moment breaks it as a standalone symmetry, but operations that reverse
  the moment survive combined with it as the anti-unitary half of the magnetic
  group, so part of the reduction comes back through `magmoms=`. A nonmagnetic
  spin-orbit calculation keeps the full time-reversal fold through Kramers
  degeneracy.

## Next

Continue to [Basis-set error estimation](error-estimation.md).
