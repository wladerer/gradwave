# Hybrid functionals

A hybrid functional replaces a fraction of the semilocal exchange with exact
(Fock) exchange evaluated on the Kohn-Sham orbitals. The global PBE0
form[[25]](bibliography.md#pbe0) mixes a constant fraction $\alpha$,

$$
E_\text{xc} = (1-\alpha)\,E_x^\text{PBE} + \alpha\,E_x^\text{Fock} + E_c^\text{PBE},
$$

with $\alpha = 0.25$ the usual choice. The exact-exchange term is orbital
dependent, so unlike a PBE run the exchange operator changes as the orbitals do.
gradwave *solves* that fixed point rather than evaluating exchange once on a fixed
density, and because the whole exchange energy is a differentiable tensor
function of the mixing, the fraction $\alpha$ and the screening length $\omega$
are themselves trainable.

Everything here runs through the Python API. The YAML `xc` key stays restricted
to `lda` and `pbe`; hybrids are reached with `hybrid_scf` and the classes in
`gradwave.postscf.hybrid`.

## Why exact exchange is expensive, and how it is made cheap

The Fock operator couples every occupied orbital to every other through the
Coulomb kernel, an $O(N^4)$ object naively. Two standard compressions bring it
into range.

- **ISDF** (interpolative separable density fitting[[27]](bibliography.md#isdf))
  factorizes the occupied orbital-pair products onto a small set of
  interpolation points, so the co-densities that feed the Coulomb solve are
  represented by $O(N)$ vectors instead of $O(N^2)$ pairs.
- **ACE** (adaptively compressed exchange[[28]](bibliography.md#ace)) freezes the
  action of the operator on the occupied space into a low-rank factor. Once built
  each SCF step, applying it to a trial orbital is a pair of small matrix
  multiplies.

The Fock operator is rebuilt from the current orbitals each SCF iteration and
lags one step, exactly the way the DFT+U occupation matrices do
([Differentiable Hubbard U](hubbard-u.md)). ACE makes that frozen operator cheap
to re-apply through the inner Davidson solves.

## Range separation

Splitting the Coulomb kernel at a length $1/\omega$ gives the screened hybrids.
The short-range part (erfc) is the operator a screened functional such as
HSE[[26]](bibliography.md#hse) keeps; the long-range part (erf) is the complement.
`mode` selects the kernel: `"full"` (bare $1/r$, the PBE0 operator),
`"short_range"`, or `"long_range"`, with `omega` in Å⁻¹ for the screened modes.

!!! warning "Screened modes are the operator half only"
    The screened Fock *operator* is exact and energy-consistent, but a complete
    HSE also range-separates the *semilocal* exchange it replaces, keeping the
    long-range PBE exchange and removing only the short-range fraction.
    `ScaledExchangePBE` scales the whole PBE exchange by $(1-\alpha)$, which is
    correct for full-range PBE0 but double-counts the long-range exchange for a
    screened hybrid. The matching range-separated (wPBE) enhancement on the
    semilocal side is not implemented. Use `mode="full"` for a physically
    complete self-consistent hybrid; treat the screened modes as the exact
    operator half while the semilocal side is finished.

## Run a hybrid SCF

`hybrid_scf` wraps the standard SCF: it scales the semilocal exchange by
$(1-\alpha)$ and adds $\alpha\,E_x^\text{Fock}$ through the `fock` hook. At
$\alpha = 0$ it is exactly a PBE SCF, a reduction gate worth checking. Extra
keyword arguments pass through to `scf`.

```python
from gradwave.postscf.hybrid import hybrid_scf
from gradwave.scf.loop import setup_system

# Si at Gamma, a loose 18 Ry cutoff
res = hybrid_scf(system, alpha=0.25, smearing="none",
                 etol=1e-9, rhotol=1e-8, max_iter=80, verbose=False)

res.energies.free_energy   # eV, total with the Fock term included
res.energies.fock          # eV, the alpha * E_x^Fock contribution (< 0)
```

On the shipped Si cell (`examples/hybrid_train.py` builds it) PBE0 lowers the
energy by the Fock term ($E_x^\text{Fock}$ contribution $-4.96$ eV at
$\alpha = 0.25$) and opens the $\Gamma$ gap from the PBE $2.32$ eV to $2.97$ eV.
The operator is acting on the eigenvalues, which is the point of solving exchange
self-consistently rather than evaluating it once.

### On a k-mesh

`hybrid_scf` uses the multi-k Fock build, which reduces to the $\Gamma$ build at a
single k-point. Each k's exchange sums over the whole Brillouin zone through the
co-density momentum $\mathbf q = \mathbf k - \mathbf k'$, so the mesh has to be
the full BZ:

```python
system = setup_system(cell, pos, species, upfs, ecut=..., kmesh=(2, 2, 2),
                      use_symmetry=False, time_reversal=False, nbands=8)
res = hybrid_scf(system, alpha=0.25, mode="full", smearing="none")
```

A symmetry-folded mesh is an invalid quadrature for the exchange sum, the same
constraint the [magnetocrystalline anisotropy](mae.md) reference SCF carries.

## A learnable hybrid

At self-consistency the density is stationary, so by the Hellmann-Feynman
argument the total energy's derivative with respect to the mixing is the
*explicit* one, evaluated on the frozen converged orbitals. This is the same
free-derivative-at-convergence property the [learnable XC](learning-xc.md) slot
uses for $(\kappa, \mu)$, now on the exchange-mixing parameters.

`HybridExchangeParams` holds $\alpha$ and $\omega$ as reparameterized
(sigmoid / softplus) trainable parameters, so training stays in the physical
range. `differentiable_hybrid_energy` returns a scalar equal to
`res.energies.total` whose gradient in $(\alpha, \omega)$ is exact.

```python
from gradwave.postscf.exchange_multik import HybridExchangeParams
from gradwave.postscf.hybrid import differentiable_hybrid_energy, hybrid_scf

params = HybridExchangeParams(alpha=0.10, mode="full")
res = hybrid_scf(system, alpha=float(params.alpha.detach()), smearing="none")
e = differentiable_hybrid_energy(res, params)   # equals res.energies.total
loss = (e - e_target) ** 2
loss.backward()                                 # dE/dalpha into params
```

`hybrid_energy_gradient(res, params)` is the convenience wrapper that runs the
backward pass and returns the physical $(\mathrm dE/\mathrm d\alpha,
\mathrm dE/\mathrm d\omega)$ directly ($\omega$ is `None` for `mode="full"`).

### The training example

`examples/hybrid_train.py` is the recovery check: it fixes a target
$\alpha^\star = 0.25$, records the converged PBE0 total energy, then starts from a
perturbed $\alpha = 0.10$ and descends $(E(\alpha) - E^\star)^2$ back onto the
target. Each step re-converges the hybrid SCF at the current $\alpha$ (the
stationary gradient is exact only at self-consistency) and takes one backward
pass. It recovers $\alpha^\star$ to within $10^{-3}$ in about 45 Adam steps and
writes `examples/hybrid_train.json` with the per-step history.

    uv run python examples/hybrid_train.py

Matching a single scalar energy determines one parameter, so the example trains
$\alpha$ alone in PBE0 mode. Training $\omega$ as well needs a loss that
constrains the range separation, for example a reference the screened operator
reproduces.

## Gotchas

- **Full-BZ mesh.** The multi-k exchange sum requires
  `use_symmetry=False, time_reversal=False`. A folded mesh raises or gives a
  wrong energy.
- **`mode="full"` for a complete SCF.** The screened modes supply the exact
  screened operator but not yet the matching range-separated semilocal exchange,
  so a self-consistent screened hybrid double-counts the long-range piece. Use
  `full` (PBE0) for a complete self-consistent hybrid.
- **`differentiable_hybrid_energy` is nspin=1.** The learnable path is
  spin-unpolarized for now; a spin-polarized hybrid SCF still runs, only its
  $(\alpha, \omega)$ gradient is unavailable.
- **The SCF runs under `no_grad`.** Pass a detached `alpha` (or `params`) to
  `hybrid_scf`; the gradient comes from `differentiable_hybrid_energy` on the
  converged result rather than from the SCF loop.
- **Reduction gate.** At $\alpha = 0$ the hybrid SCF must reproduce the PBE run to
  machine precision, and a single $\Gamma$ point with `mode="full"` must
  reproduce the $\Gamma$-only Fock build. Both are cheap sanity checks before a
  production run.

## Next

Continue to [Symmetry reduction](symmetry.md), which the full-BZ requirement here
motivates, or back to [Learning XC by AD](learning-xc.md) for the semilocal
analog of the trainable-parameter machinery.
