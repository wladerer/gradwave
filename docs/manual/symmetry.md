# Symmetry reduction

Reducing the k-mesh to the irreducible Brillouin zone (IBZ) and symmetrizing the
density each step is the largest single source of speedup in gradwave, giving 5 to 14
times depending on the point group. It is on by default (`symmetry: true`) and
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

See [Performance](performance.md) for where the 5-to-14× speedup lands.

## Gotchas

- **Non-symmorphic operations must be commensurate with the grid.** A glide like
  the diamond $(\tfrac14, \tfrac14, \tfrac14)$ needs FFT dimensions divisible by 4.
  gradwave equalizes symmetry-coupled axes automatically. On an incommensurate box
  the full-mesh fixed point itself carries a ~$2\times10^{-4}$ asymmetric density
  component, so an IBZ-vs-full comparison fails at $10^{-4}$ with no bug present.
- **The symmetrizer is masked to the density sphere.** At the box Nyquist boundary
  the folded Miller map misidentifies $G$-vectors for glide phases. Physical
  densities are zero there, and masking makes the operator exactly idempotent.
- **Turn symmetry off for symmetry-breaking work.** Antiferromagnetic and
  ferrimagnetic orderings, a real response to a perturbation (the implicit-differentiation
  backward, USPP/PAW or nspin=2 discretization-error estimates, the Dyson dressing)
  all need `use_symmetry=False`, because the perturbation lowers the crystal
  symmetry. gradwave raises a clear error when a magnetic ordering needs it.
- **Time reversal** ($k \to -k$) is on by default and shrinks the IBZ further. It
  is switched off for magnetic systems where $k \not\equiv -k$. A nonmagnetic
  spin-orbit calculation keeps it through Kramers degeneracy.

## Next

Continue to [Basis-set error estimation](error-estimation.md).
