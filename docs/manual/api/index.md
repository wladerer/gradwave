# API reference

This section documents the Python API. Every page renders directly from the
docstrings in `src/gradwave`, so the signatures here are the ones the code
exposes.

Most work goes through one of three entry points. Pick the one that matches how
much control you need.

| you want to | use | page |
|---|---|---|
| run a task from an input file or dict | `gradwave.api.run` | [High-level API](highlevel.md) |
| plug gradwave into ASE (energy, forces, stress) | `gradwave.calculator.GradWave` | [High-level API](highlevel.md) |
| drive the SCF yourself and keep the result object | `scf.loop.scf`, `scf.uspp.scf_uspp` | [SCF engine](scf.md) |
| compute a property from a converged result | `gradwave.postscf.*` | [Properties](properties.md) |
| differentiate the energy through the SCF | `core.xc`, `postscf.uspp_implicit` | [Exchange–correlation](xc.md) |
| read a pseudopotential | `pseudo.upf`, `pseudo.upf_paw` | [Pseudopotentials](pseudopotentials.md) |

## Layers

The code is organized in three layers, and the API mirrors them.

- **Layer A** is the pure, differentiable core. Every energy term is a tensor
  function of the positions, cell, density, and functional parameters. Autograd
  runs here. See [Exchange–correlation](xc.md) and the total-energy assembly.
- **Layer B** is the SCF driver and the eigensolver. They run under `no_grad`
  and stay invisible to autograd. See [SCF engine](scf.md).
- **Layer C** is the user-facing surface: the YAML/dict input schema, the task
  runner, the ASE calculator, and the analysis helpers. See
  [High-level API](highlevel.md).

Base units are eV and Å throughout. Every tensor is `float64` or `complex128`
unless a run opts into mixed precision. The formalism (norm-conserving or
ultrasoft/PAW) is detected from the UPF file, so the same input drives both.
