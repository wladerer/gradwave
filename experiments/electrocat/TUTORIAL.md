# Adsorption energetics and constant-potential electrochemistry on Pt(111)

This tutorial computes hydrogen and CO adsorption free energies on Pt(111), then
the electrode potential from a constant-µ SCF, on one H100 GPU. The scripts are in
`experiments/electrocat/`.

Build the input structures once.

```bash
uv run python structures.py
```

## The surface

![Clean Pt(111) slab](renders/Pt_slab.png)

The slab is a 2×2×4 Pt(111) cell with 15 Å of vacuum, the bottom two layers held
at the PBE lattice constant a = 3.968 Å. PAW pseudopotentials (psl.1.0.0, PBE) run
at ecutwfc = 50 Ry and ecutrho = 400 Ry, on a 4×4×1 k-mesh with cold smearing at
0.15 eV. A vacuum slab converges in fewer iterations with the local Thomas-Fermi
preconditioner than with a constant Kerker screening, because a fixed screening
over-damps the vacuum region. The preconditioner and Johnson mixing are set in
`config.py`.

## Hydrogen adsorption

`run_pair.py` runs one adsorbate-surface pair from start to finish. It relaxes the
clean slab, relaxes the adsorbate at the ontop, bridge, fcc, and hcp sites, relaxes
the gas-phase reference, and forms the adsorption free energy.

```bash
uv run python run_pair.py Pt H
```

H binds most strongly in the threefold fcc hollow at ΔE = −0.48 eV, with the hcp
hollow weaker by 0.05 eV and the atop site weaker by 0.07 eV.

![H in the fcc hollow](renders/Pt_H_fcc.png)

The hydrogen evolution descriptor is the adsorption free energy ΔG_H*. It is formed
from ΔE plus the standard 298 K zero-point and entropy correction. On Pt(111) at
1/4 ML, ΔG_H* = −0.243 eV. A |ΔG_H*| near zero is the condition for fast hydrogen
evolution, and Pt is the reference catalyst against which the descriptor was
calibrated.

## The CO puzzle

CO on Pt(111) is a standard failure of semilocal DFT. PBE places CO in a threefold
hollow, whereas the experiment finds it atop. The workflow reproduces the site
error and the overbinding together.

```bash
uv run python run_pair.py Pt CO --skip bridge
```

The `--skip bridge` flag drops the bridge site, which relaxes into a hollow anyway.

CO binds most strongly in the hcp hollow at ΔE = −1.742 eV, with the fcc hollow
within 0.01 eV and the atop site weaker by 0.17 eV. The adsorption free energy at
the hollow is ΔG = −1.642 eV.

![CO in the hcp hollow](renders/Pt_CO_hcp.png)

PBE selects the hollow, and it overbinds relative to the experimental adsorption
energy near −1.4 to −1.5 eV. The atop geometry that the experiment prefers is the
weaker-binding site here.

![CO atop, the experimentally preferred site](renders/Pt_CO_ontop.png)

## Electrode potential from a constant-µ SCF

The adsorption energetics hold the electron count fixed. An electrode holds the
potential fixed, and the charge floats. Typically the fixed potential is reached
with a charged periodic cell and a compensating background. Instead gradwave uses
the open-boundary (ESM) electrostatics, where metal capacitor plates at the cell
edges carry the counter-charge, and floats the electron count to a target Fermi
level.

```bash
uv run python constant_potential.py
```

At the potential of zero charge the neutral slab gives a work function Φ = 5.53 eV,
and an absolute electrode potential of Φ − 4.44 = +1.09 V versus SHE. Literature
values are 5.9 eV from experiment and about 5.7 eV in PBE, so Φ here is comparable
to the PBE reference at this smearing. From the fixed-µ runs the electron
count floats away from neutrality, and its slope ∂N/∂µ is the differential
capacitance of the interface.

## Convergence on a transition metal

The constant-µ SCF converges at once on a free-electron metal like Na. On Pt it
does not, and the reason is the d-band. At a fixed Fermi level the electron count is
recomputed from the occupied states each iteration. Pt has a large density of states
at the Fermi level, so dN/dµ is large, and with a sharp 0.1 eV smearing the floating
electron count oscillates and the SCF never settles. The neutral fixed-count run
converges without trouble, so the µ constraint is the difficulty, not the metal
itself.

You have a few options. Broadening the smearing to 0.3 eV shrinks dN/dµ and is
enough here. Warm-starting each fixed-µ run from the converged neutral density means
only the electron count has to move, not the whole density. The local Thomas-Fermi
preconditioner damps the charge sloshing between the slab and the plates. The
principled fix is to deflate the soft mode of the charge response, the near-singular
mode with large dN/dµ that the instability follows.

## What each run writes

Every stage writes to `results/`. The energies land in `<pair>.json` as they
converge, so a re-run skips finished stages. Each relaxation writes the final
geometry to `<tag>_relaxed.xyz` with forces, the full trajectory to `<tag>.traj`,
and a diagnostics sidecar `<tag>.diag.json` recording the SCF iteration count, the
Fermi level, whether fractional occupations exist, and whether the optimizer reached
the force threshold. A gas molecule is flagged non-metallic in that sidecar, which
is correct.

## What is differentiable

Every number above is an energy from a converged SCF. Because the SCF is
differentiable, its derivatives follow from autograd without a finite-difference
sweep. `differentiable.py` reads the strain derivative of the adsorption energy from
the autograd stress, one calculation rather than a lattice scan. The same route
gives the derivative with respect to the electrode potential through the constant-µ
SCF, and with respect to composition through the alchemical path.
