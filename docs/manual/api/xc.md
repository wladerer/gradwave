# Exchange–correlation and differentiation

Layer A. An `XCFunctional` maps the density to the exchange-correlation energy
density as a pure tensor function, so autograd gives the potential and, through
the SCF fixed point, the response to any functional parameter. The learnable
functionals expose (κ, μ) as trainable parameters, and the learnable hybrid
exposes the exchange mixing α and screening ω; the implicit-differentiation
helpers turn a density-dependent loss into a gradient with one adjoint solve.
For the training workflows in prose, see [Learning XC by AD](../learning-xc.md)
and [Hybrid functionals](../hybrid-functionals.md).

## Functional interface

::: gradwave.core.xc.base.XCFunctional

::: gradwave.core.xc.base.to_au

::: gradwave.core.xc.base.eps_to_ev_density

## Standard functionals

::: gradwave.core.xc.lda_pw92.LDA_PW92

::: gradwave.core.xc.pbe.PBE

### Spin-polarized

::: gradwave.core.xc.spin.SpinXC

::: gradwave.core.xc.spin.LSDA_PW92

::: gradwave.core.xc.spin.SpinPBE

### Non-collinear

Wraps any collinear `SpinXC` into a functional of the density and the
magnetization vector, in the locally-collinear approximation.

::: gradwave.core.xc.noncollinear.NoncollinearXC

::: gradwave.core.xc.noncollinear.vxc_and_bxc

## Learnable functionals

::: gradwave.core.xc.learnable.LearnableX

::: gradwave.core.xc.learnable.LearnableSpinX

::: gradwave.core.xc.learnable.energy_param_grads

## Hybrid functionals

Exact (Fock) exchange acting in the SCF, with the mixing α and screening ω
trainable. `hybrid_scf` solves a global (PBE0-form) or screened hybrid;
`differentiable_hybrid_energy` turns the converged result into an objective
differentiable in (α, ω). For the workflow in prose, see
[Hybrid functionals](../hybrid-functionals.md).

::: gradwave.postscf.hybrid.hybrid_scf

::: gradwave.postscf.hybrid.differentiable_hybrid_energy

::: gradwave.postscf.hybrid.hybrid_energy_gradient

::: gradwave.postscf.exchange_multik.HybridExchangeParams

::: gradwave.postscf.hybrid.ScaledExchangePBE

::: gradwave.postscf.hybrid.MultiKFockExchange

## Differentiating through the SCF

`energy_param_grads` covers the norm-conserving path at convergence.
`uspp_energy_param_grads` and `uspp_density_loss_param_grads` extend
differentiation to the USPP/PAW fixed point, including the one-center term and a
density-dependent loss through the adjoint.

::: gradwave.postscf.uspp_implicit.uspp_energy_param_grads

::: gradwave.postscf.uspp_implicit.uspp_density_loss_param_grads

## Total energy

The pure function autograd differentiates. Everything above ultimately feeds
this assembly.

::: gradwave.core.energies.total.total_energy

::: gradwave.core.energies.total.EnergyBreakdown

## DFT+U

The rotationally-invariant Dudarev correction: projectors onto the localized
manifolds, the occupation matrix, and the corrective energy and potential.

::: gradwave.core.hubbard.HubbardManifold

::: gradwave.core.hubbard.build_hubbard_projectors

::: gradwave.core.hubbard.hubbard_energy
