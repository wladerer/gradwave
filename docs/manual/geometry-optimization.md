# Geometry optimization

This tutorial relaxes diamond carbon. The second atom starts about 0.08 Å off its
ideal (a/4, a/4, a/4) site and BFGS moves it back.

## Theory

The total energy is a function of the nuclear positions $\{\mathbf{R}_I\}$. The
force on atom $I$ is the negative gradient,

$$ \mathbf{F}_I = -\frac{\partial E}{\partial \mathbf{R}_I}. $$

At self-consistency the energy is stationary with respect to the density, so the
implicit dependence of the wavefunctions on $\mathbf{R}_I$ drops out of the
gradient. Only the explicit dependence of the Hamiltonian survives, which is the
Hellmann-Feynman theorem,[[3]](bibliography.md#feynman)

$$ \mathbf{F}_I = -\left\langle \psi \left| \frac{\partial \hat{H}}{\partial \mathbf{R}_I} \right| \psi \right\rangle. $$

gradwave does not code this by hand. It differentiates the detached converged
energy with one reverse-mode pass, so the positions enter through the Ewald sum,
the structure factors, and the projector phases, and the result matches the
Hellmann-Feynman force to autograd precision. A relaxation drives the largest
force below a threshold,

$$ \max_I \left| \mathbf{F}_I \right| < f_\text{max}, $$

by a quasi-Newton method. The default is BFGS,[[8]](bibliography.md#nw) which
builds an approximate inverse Hessian from the force history and is efficient near
a smooth minimum. FIRE[[9]](bibliography.md#fire) is the robust fallback for
starts far from the minimum. For a variable cell the stress tensor is the strain
derivative,

$$ \sigma_{\alpha\beta} = \frac{1}{\Omega} \frac{\partial E}{\partial \varepsilon_{\alpha\beta}}, $$

which gradwave also obtains by autograd through the differentiable radial
transforms.

## Write the input

`examples/input_diamond_relax.yaml`:

```yaml
structure:
  cell: [[0.0, 1.7835, 1.7835], [1.7835, 0.0, 1.7835], [1.7835, 1.7835, 0.0]]
  positions:
    cart: [[0.0, 0.0, 0.0], [0.9518, 0.8618, 0.9318]]   # ideal: 0.89175 each
  species: [C, C]

pseudopotentials:
  dir: ../tests/fixtures/qe/pseudos
  map: {C: C_ONCV_PBE-1.2.upf}

ecut: 680.28          # eV (50 Ry, hard C ONCV pseudo)
xc: pbe
kpoints: {mesh: [4, 4, 4]}

scf:
  etol: 1.0e-8
  rhotol: 1.0e-7

task: relax
relax:
  optimizer: bfgs
  fmax: 0.01          # eV/Å
  max_steps: 100

output:
  dir: ./out_diamond
```

`ecut` is in eV. `fmax` is the force threshold $f_\text{max}$ in eV/Å. The
optimizer default is `bfgs`, which is right for a smooth problem near the minimum.
Use `fire` when the start is far from the minimum or the surface is rough.

## Run it

    uv run gradwave examples/input_diamond_relax.yaml -o out_diamond/

ASE prints the BFGS log as it runs, one line per ionic step with the energy and
the maximum force. The final line reports whether the run converged, the energy,
the fmax reached, and the step count.

## Read the output

Three files land in `out_diamond/`.

- `relax.out` is the human report. Structure, parameters, the ionic step table,
  and the final geometry.
- `relax.json` is the machine-readable summary and the parsing target. The
  `relax` block holds `converged`, `n_steps`, `energy_eV`, `fmax_eV_ang`,
  `max_displacement_ang`, the final `positions_ang` and `cell_ang`, and a
  `trajectory` list of `{step, energy_eV, fmax_eV_ang, positions_ang}`.
- `checkpoint.pt` is restartable SCF state for the final geometry.

## Plot the trajectory

The `relax.json` trajectory reads straight into pandas.

```python
import json
import pandas as pd

data = json.load(open("out_diamond/relax.json"))
traj = pd.DataFrame(data["relax"]["trajectory"])
traj.plot(x="step", y="fmax_eV_ang", logy=True)
```

A log-scale $\max_I |\mathbf{F}_I|$ against step shows the approach to the
threshold. The energy column shows the monotone descent BFGS produces on a convex
basin.

## Drive it from Python

For programmatic control, attach the ASE calculator to an `Atoms` object and use
any ASE optimizer.

```python
from ase.build import bulk
from ase.optimize import BFGS
from gradwave.calculator import GradWave

atoms = bulk("C", "diamond", a=3.567)
atoms.rattle(0.05)
atoms.calc = GradWave(
    ecut=680.28,
    pseudopotentials={"C": "tests/fixtures/qe/pseudos/C_ONCV_PBE-1.2.upf"},
    xc="pbe",
    kpts=(4, 4, 4),
)
BFGS(atoms).run(fmax=0.01)
```

The calculator caches the grids and form-factor tables and reuses them when only
positions change, which is the common case during a relaxation. It also reuses the
previous step's density as the SCF start, so same-position restarts drop to a
couple of SCF iterations.

!!! note "Variable cell"
    Variable-cell relaxation works through `ase.filters.FrechetCellFilter` because
    the calculator returns the stress from the differentiable radial transforms.
    Relaxing the cell at fixed `ecut` carries a Pulay stress from basis
    incompleteness,[[2]](bibliography.md#payne) so converge `ecut` or re-relax at
    the new cell before trusting the volume.

## Gotchas

- Forces sum to zero to about 1e-6 eV/Å by construction, since $E$ is invariant
  under a rigid translation. A larger residual means the density or the grid is
  under-converged.
- BFGS needs a few steps for this diamond. FIRE needs roughly ten times as many on
  the same problem, so reserve it for hard starts.
- On small cells gradwave is slower per ionic step than `pw.x`, mostly from FFT
  and small batched linear-algebra kernel maturity against decades-tuned FFTW and
  LAPACK. The gap shrinks on GPU and on larger systems. See
  [Performance](performance.md) for the measured comparison, which works through why
  the gap is not architectural.

## Next

Continue to [Learning XC by AD](learning-xc.md), the second tutorial.
