# Plan: a contracted / occupied-space local basis for COHP

## Motivation

`postscf/cohp.py` is candid (module docstring, "QUANTITATIVE STATUS") that its
absolute solid-state ICOHP is not calibrated to LOBSTER. On diamond (PBE) LOBSTER
reports IpCOHP ≈ −9.64 eV per C–C bond; gradwave's operator route overshoots ~2×
and the band-limited eigenvalue route undershoots ~2×, bracketing the true value.
Two independent causes:

1. **Bond resolution.** The projectors carry the Bloch phase `e^{-i(k+G)·τ_a}`
   (`pdos._ao_projectors_k`), so `H̃_pq(k)` is the interaction of orbital `p` on
   atom `i` with the *entire* atom-`j` sublattice (all periodic images), not one
   bond. For diamond a "pair" is ~4 nearest bonds. LOBSTER reports one bond.

2. **Basis diffuseness.** The projector basis is the pseudo-atomic `PP_PSWFC`
   orbital read straight from the UPF. After Löwdin orthonormalization it is more
   diffuse than LOBSTER's contracted local orbitals, so inter-atomic overlap is
   large and `O^{-1/2}` inflates the off-site `H̃_ij`. The COHP magnitude comes out
   too big (operator) or, band-limited, too small.

## The theory, and the one trap

The projected-COHP idea and its quality metric are due to
Sánchez-Portal, Artacho & Soler (*Solid State Commun.* 1995; the **spilling**
parameter and its variational minimisation) and Deringer, Tchougréeff &
Dronskowski (*J. Phys. Chem. A* 2011, projected COHP). LOBSTER (Maintz et al.,
*J. Comput. Chem.* 2016) projects PW/PAW states onto **minimal contracted
Slater-type orbitals** fitted to free-atom valence orbitals, reports **absolute
charge spilling** and **RMSp** (a G-space, model-independent residual), and
Löwdin-orthonormalises the local basis for COHP.

**Trap.** The Sánchez-Portal spilling objective is correct for reproducing band
energies, but *minimising spilling pushes the basis toward more diffuse /
multi-ζ*, which makes COHP **worse** — more inter-atomic overlap, larger off-site
`H̃`. LOBSTER's authors say their bonding methods are "bound to minimal basis sets
on purpose." The objective for a COHP basis is **localization** (a minimal,
well-shaped valence orbital), not spilling → 0. gradwave's spilling is already
small; completeness is not the problem, extent is.

## The occupied-space answer: Intrinsic Atomic Orbitals (IAO)

Knizia (*J. Chem. Theory Comput.* 2013) builds a minimal set of polarized atomic
orbitals that **exactly span the occupied manifold** — occupied-space spilling is
zero by construction — while staying minimal and localized. Given occupied KS
states `|ψ_n>` and a free-atom minimal basis `|φ_p>` (here the `PP_PSWFC` set),

    Õ = orthonormalize(P^{B2} |ψ>)              depolarised occupied space
    |A_p> = ( O Õ + (1−O)(1−Õ) ) |φ_p>          IAO, O = |ψ><ψ|, P^{B2}=Σ|φ̃><φ̃|

The IAOs live in the plane-wave basis (linear combinations of `ψ` and the
PW-represented `φ`), so they drop straight into the existing operator route
`H̃ = ⟨Ã|Ĥ|Ã⟩`. This is the smallest code change that fixes cause (2): no external
basis tables, no radial refit, and it is the natural fit for a differentiable
code (pure linear algebra on `becp`/overlap). Bloch/periodic IAO follows Lehtola
& Jónsson-style constructions (see arXiv:2407.00852).

## Metrics

`spilling` / `charge_spilling` are already reported. Add **RMSp**, the LOBSTER
G-space residual

    RMSp² = Σ_{k,n,G} w_k |ψ_n(k+G) − X_n(k+G)|² / Σ_{k,n,G} w_k,
    X_n = P^{B2} ψ_n = Σ_p <φ̃_p|ψ_n> φ̃_p,

which for a Löwdin reconstruction equals the k-weighted mean state spilling but is
computed directly in reciprocal space, is bounded, and — computed *without*
`torch.no_grad` — is a differentiable objective for variationally contracting a
basis. That is the long game the differentiable framework enables and LOBSTER
cannot.

## Work order (this branch)

1. **Per-image-R bond resolution.** For a pair `(i,j)` isolate the single bond at
   the min-image lattice vector `R*`. Real-space hopping `h_pq(R)=Σ_k w_k
   e^{-ik·R} H̃_pq(k)` and an `e^{ik·R}` phase on the density side; `Σ_R` over the
   image shell reconstructs the current sublattice COHP (validation). Requires the
   full (unreduced) k-mesh for `R≠0`; exact at Γ. Unblocks any comparison to a
   per-bond LOBSTER number.
2. **IAO projectors.** `basis="iao"` in `cohp()` (collinear, norm-conserving,
   operator route). Verify charge spilling ≈ 0, bonding sign, sum rule.
3. **RMSp.** Report on `COHP`; expose a differentiable `projection_rmsp` helper.
4. **Consider contracted STOs** (LOBSTER route) + an external LOBSTER/QE fixture —
   evaluated last; needs shippable basis tables and a real oracle, so likely
   staged as follow-up rather than landed here.

### Decision on step 4 (contracted STOs)

**Deferred to a follow-up.** Fitting per-element contracted Slater basis tables
(Bunge / pbeVaspFit2015 style) buys one thing IAO does not: a *literal* LOBSTER
number match, because the projection basis would be the same family. But it costs
a per-element data set to ship and maintain, a fitting pipeline, and — to be worth
anything — an external LOBSTER or QE oracle fixture, of which the tree has none for
COHP. IAO (step 2) already fixes the diffuseness cause with no external data and is
the natural differentiable-code choice. So the ordering is: land steps 1–3 (image
resolution, IAO, RMSp) now; open a follow-up that (a) adds a diamond LOBSTER
IpCOHP fixture and (b) only then decides whether IAO already matches −9.64 eV/bond
closely enough to make STO fitting unnecessary. Committing to STO tables before we
have the oracle would be building the expensive half first.

## Validation

No external COHP oracle exists in-tree yet (only the internal sum rule + sign).
Per-image `Σ_R` reconstruction and IAO zero-spilling are internal checks that can
land now. A LOBSTER cross-check (diamond −9.64 eV/bond) needs step 4's fixture.
Heavy runs go to `asus` (idle); local is reserved for the small O2/Bi2 gates.

## Follow-up: magnitude check against the diamond −9.64 eV/bond citation (post-landing)

Steps 1–3 above are landed (`resolve_images`, `basis="iao"`, `projection_rmsp`).
This follow-up ran the comparison step 4 deferred — not a real LOBSTER fixture
(none is available: no license/binary, `nix search lobster` has no COHP
package), but a direct magnitude check against the one published number, on
diamond (PD_C_PBE_std, ecut 45 Ry, 2×2×2 unreduced k-mesh, nbands=24 — the
`diamond_c` fixture in `tests/integration/test_cohp.py`).

**Bond resolution (cause 1): confirmed closed.** Scanning every integer lattice
shift `R` that places atom 1 at the nearest-neighbour distance from atom 0
finds exactly 4 (diamond's tetrahedral coordination), and `resolve_images`
gives the IDENTICAL −20.9589 eV for all 4 — not approximately equal, identical
to the digits measured. Summing the 4 reconstructs the −83.5 eV sublattice
number to <0.4%. `resolve_images` is a verified, exact per-bond decomposition,
not a partial fix.

**Basis diffuseness (cause 2): IAO does NOT close it — the plan's central
hypothesis was wrong.** Measured directly: `basis="iao"` gives −21.2 eV/bond
vs. `basis="pswfc"`'s −21.0 eV/bond — 1% LARGER, not smaller, despite
`charge_spilling` collapsing from 0.0038 to ~0. IAO fixes occupied-manifold
*completeness* (which is what `charge_spilling`/RMSp measure); it does nothing
to shrink the orbitals' real-space *extent*, which is what actually drives the
inter-atomic `H̃` overlap and hence the COHP magnitude. In hindsight this
should have been predictable from the construction: `A = [OÕ + (1−O)(1−Õ)]φ`
is built to reproduce the occupied KS states exactly, including whatever
bonding character extends onto neighbouring atoms — there is no term in it
that penalises spatial spread. The "resemblance" between "spans the occupied
space" and "is a contracted local basis" was a false equivalence.

A separate, unshipped diagnostic (Gaussian-damping the PP_PSWFC radial tail,
`exp(-(r/rc)^2)`, post-SCF — cheap because PP_PSWFC is a postscf-only
projection table, not a SCF input) confirms a naive truncation is not a
shortcut either: the per-bond magnitude barely moves for `rc` down to ~0.8×
the bond length, and by the `rc` (~0.7 Å) where it crosses −9.64 eV,
`charge_spilling` has grown past 20% (from 0.0038 undamped) — i.e. it only
"matches" the LOBSTER number by no longer representing the occupied states,
which is not a real fix.

**Conclusion:** cause 2 is confirmed real (~2.1–2.2× overshoot, both routes
agree at converged nbands, so this is not a route-choice artifact) and remains
open. The step 4 decision above holds: closing it needs an actual
contracted/fitted local orbital (LOBSTER-style minimal Slater-type basis
tables), which is new machinery — per-element basis data plus a fitting
pipeline, evaluated against a real external oracle — not a parameter tweak on
the existing PP_PSWFC radial or a different orthogonalization of it. Scoped,
not started.

Also worth correcting: the original "operator overshoots ~2×, eigenvalue
undershoots ~2×, true value bracketed between" framing does not survive
per-bond resolution at a well-converged band count. At nbands=24 (well past
the reference-leak knee `cohp.py`'s docstring already quantifies), the
eigenvalue route (−20.4 eV/bond) sits only ~2% below the operator route
(−21.0 eV/bond) — both on the SAME side of LOBSTER's −9.64 eV, not bracketing
it. What looked like a bracket was the eigenvalue route's partial convergence
toward the operator value at lower nbands, not two independent estimates of
the true bond strength.

## References

- Sánchez-Portal, Artacho, Soler, *Solid State Commun.* **95**, 685 (1995) — spilling, variational basis optimisation.
- Deringer, Tchougréeff, Dronskowski, *J. Phys. Chem. A* **115**, 5461 (2011) — projected COHP.
- Maintz, Deringer, Tchougréeff, Dronskowski, *J. Comput. Chem.* **37**, 1030 (2016) — LOBSTER: contracted STOs, absolute spilling, RMSp, Löwdin.
- Knizia, *J. Chem. Theory Comput.* **9**, 4834 (2013) — intrinsic atomic/bond orbitals.
- Bloch IAO for periodic systems — arXiv:2407.00852.
