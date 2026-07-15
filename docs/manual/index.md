# gradwave

gradwave is a differentiable plane-wave density functional theory code written
in PyTorch. It solves the Kohn-Sham equations[[1]](bibliography.md#ks) in a
plane-wave basis with norm-conserving or ultrasoft/PAW pseudopotentials. Every
energy term is differentiable, so one reverse-mode pass through the total energy
gives forces, a second gives the Hessian, and an implicit-differentiation wrapper
gives the response of the self-consistent density to any parameter of the
functional.

This manual is organized as a wiki. Start with [Installation](installation.md),
then work through the two tutorials. The first relaxes a geometry with exact
autograd forces. The second trains an exchange-correlation functional back to
PBE[[10]](bibliography.md#pbe) through the self-consistent density. Each tutorial
opens with the theory it rests on, then runs a shipped example. The
[Reference](reference.md) page collects the CLI, the output files, and the entry
points, and the [Bibliography](bibliography.md) lists every citation.

## Design

Every energy term in gradwave is a pure tensor function of the atomic positions,
the cell, the density, and the exchange-correlation parameters. The
self-consistent field (SCF) loop and the eigensolver run under `no_grad` and stay
invisible to autograd, following the plane-wave pseudopotential formulation of
Payne et al.[[2]](bibliography.md#payne) The converged quantities are then detached
and fed back into the pure energy, so a single backward pass differentiates it.

Two capabilities follow from that design.

- Geometry optimization uses exact Hellmann-Feynman
  forces[[3]](bibliography.md#feynman) from autograd, driven by any ASE
  optimizer.[[4]](bibliography.md#ase)
- Functionals are trained by gradient descent through the SCF fixed point, where
  each gradient is one adjoint solve and no finite differences are taken.

The code is validated against Quantum ESPRESSO
`pw.x`[[5]](bibliography.md#qe) at identical cutoff, k-mesh, and pseudopotential.
The formalism is detected from the UPF file, so the same input schema drives the
norm-conserving (ONCV[[6]](bibliography.md#oncv)) and
USPP/PAW[[7]](bibliography.md#paw) paths. Base units are eV and Å, and every
tensor is float64 or complex128.

## Pages

| page | what it covers |
|---|---|
| [Installation](installation.md) | set up the environment and run the first SCF |
| [Geometry optimization](geometry-optimization.md) | relax a structure with autograd forces |
| [Learning XC by AD](learning-xc.md) | train a functional through the SCF density |
| [Inputs and outputs](io.md) | input schema, output files, checkpoints, analysis |
| [Reference](reference.md) | CLI and entry points |
| [Performance](performance.md) | where time goes, what helps, the GPU precision story |
| [Wisdom](wisdom.md) | non-obvious do and do-not rules learned the hard way |
| [Bibliography](bibliography.md) | numbered references |
