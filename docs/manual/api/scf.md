# SCF engine

Layer B. `setup_system` builds the frozen per-geometry object; `scf` runs the
self-consistent loop on it and returns an `SCFResult`. The USPP/PAW path has the
same shape through `setup_uspp` and `scf_uspp`. Both run under `no_grad`, so the
converged quantities they return are detached and ready to feed back into the
pure energy for autograd.

## Norm-conserving

::: gradwave.scf.loop.setup_system

::: gradwave.scf.loop.scf

::: gradwave.scf.loop.System

::: gradwave.scf.loop.SCFResult

::: gradwave.scf.loop.vxc_potential

::: gradwave.scf.loop.vxc_spin_potential

## Ultrasoft / PAW

The USPP/PAW driver takes `PAWData` pseudopotentials and, for `nspin=2`, a
`SpinXC` functional with per-species `start_mag`. The augmentation charge and
the one-center term are handled inside the loop.

::: gradwave.scf.uspp.setup_uspp

::: gradwave.scf.uspp.scf_uspp

::: gradwave.scf.uspp.USPPSystem

## Options

The runner accepts flat keyword arguments, but the loop internally normalizes
them into these frozen option objects. Construct them directly for finer
control over the mixer.

::: gradwave.scf.options.SCFOptions

::: gradwave.scf.options.MixerOptions

## Density mixing

The mixers implement the input-to-output density map that drives convergence.
The default is Pulay; Johnson's Broyden variant matches Quantum ESPRESSO's
scheme.

::: gradwave.scf.mixing.PulayMixer

::: gradwave.scf.mixing.JohnsonMixer

::: gradwave.scf.mixing.BroydenMixer
