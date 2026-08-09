# Simulated scanning tunneling microscopy

In the Tersoff-Hamann approximation, this chapter images monolayer graphene and a
spin-polarized Fe(100) surface from a converged SCF. The example is in
`experiments/stm/`.

## Theory

For an s-wave tip the tunneling current is proportional to the sample local density
of states at the tip position, at the Fermi level[[46]](bibliography.md#tersoff).
gradwave computes the LDOS from the converged Kohn-Sham orbitals,

$$
\rho(\mathbf{r}; E) = \sum_{n,k} w_k\, g_\sigma(E_{nk} - E)\, |\psi_{nk}(\mathbf{r})|^2,
$$

with the states windowed by a Gaussian $g_\sigma$ of width $\sigma$ about the
reference energy $E$. The reference is the Fermi level at low bias. The map is a
post-SCF quantity, so it needs no tip model and no transport calculation.

Two imaging modes follow. A constant-height map is the LDOS on a fixed plane a set
distance above the surface. A constant-current map is the height at which the LDOS
reaches a target isovalue, the quantity an experiment records while it holds the
current fixed.

A symmetry-reduced SCF sums $|\psi_k|^2$ over the irreducible zone, which omits the
star of each k-point and does not carry the crystal point-group symmetry of the
real-space map. `ldos_grid` symmetrizes the map over the space group afterwards, so
a reduced calculation gives the same image as a full-zone one.

## Graphene

Graphene is a semimetal, so the density of states vanishes at the Dirac point at the
Fermi level and imaging there samples that node. A small negative bias images the
occupied $\pi$ band instead. A monolayer calculation, two carbon atoms with PBE and
37 k-points in the irreducible zone, converges in 15 iterations. The constant-height
map 2 Å above the sheet resolves the honeycomb, with the carbon $\pi$ states forming
the bright network and the ring centers dark.

![Graphene, constant-height LDOS at the Fermi level, 2 Å above the sheet. The cyan
circles mark the carbon atoms.](img/graphene_stm.png)

## Spin-polarized Fe(100)

Spin resolution comes from imaging each spin channel on its own, set by the `spin`
argument. A three-layer bcc Fe(100) slab converges to a moment of +8.0 μB, or
2.7 μB per atom. At the Fermi level the minority spin-down $d$ states dominate the
local density of states, so the spin-down map carries the four-lobe $d$-orbital
pattern at roughly six times the majority amplitude. The spin asymmetry
$(\rho_\uparrow - \rho_\downarrow)/(\rho_\uparrow + \rho_\downarrow)$ is near −0.7
across the cell. This negative tunneling spin polarization is the signal that
spin-polarized STM measures on iron[[47]](bibliography.md#sptm).

![Fe(100) spin-polarized STM. The left and center panels show the spin-up and
spin-down LDOS at the Fermi level, and the right panel shows the spin
asymmetry.](img/fe_sp_stm.png)

A true graphene-on-iron interface is a moiré superstructure of a few hundred atoms,
so the tractable iron example here is the clean surface.

## Running it

The LDOS grid and the two map modes are in `postscf.stm`.

```python
from gradwave.postscf.stm import stm_constant_height

# constant-height LDOS at the Fermi level, tip 2 A above the topmost atom
image, z_tip = stm_constant_height(result, height=2.0, sigma=0.3)

# spin-down channel only (nspin=2)
image_dn, _ = stm_constant_height(result, height=2.0, spin=1)
```

The reference energy defaults to `result.fermi`. Passing `energy` images a
different bias, and `stm_constant_current` returns the isovalue height map.

## What is differentiable

The map is built from the orbitals by fixed arithmetic, so it is differentiable in
the tip energy and height. The gradient with respect to atomic position runs through
the orbitals and needs the eigenvector response, which is not part of this post-SCF
path. For the full tunneling current at finite bias or strong tip coupling, the NEGF
transport module (`postscf.transport`) takes a tip lead.
