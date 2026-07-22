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

## Validation

No external COHP oracle exists in-tree yet (only the internal sum rule + sign).
Per-image `Σ_R` reconstruction and IAO zero-spilling are internal checks that can
land now. A LOBSTER cross-check (diamond −9.64 eV/bond) needs step 4's fixture.
Heavy runs go to `asus` (idle); local is reserved for the small O2/Bi2 gates.

## References

- Sánchez-Portal, Artacho, Soler, *Solid State Commun.* **95**, 685 (1995) — spilling, variational basis optimisation.
- Deringer, Tchougréeff, Dronskowski, *J. Phys. Chem. A* **115**, 5461 (2011) — projected COHP.
- Maintz, Deringer, Tchougréeff, Dronskowski, *J. Comput. Chem.* **37**, 1030 (2016) — LOBSTER: contracted STOs, absolute spilling, RMSp, Löwdin.
- Knizia, *J. Chem. Theory Comput.* **9**, 4834 (2013) — intrinsic atomic/bond orbitals.
- Bloch IAO for periodic systems — arXiv:2407.00852.
