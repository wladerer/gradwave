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
**[SUPERSEDED — see the 2026-09-13 follow-up below: a free CC-BY LOBSTER oracle
now exists (Zenodo 10.5281/zenodo.8091844), and the ~2× is mostly a spin
convention, not basis diffuseness.]**
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
**[UPDATE 2026-09-13: the contracted-STO machinery WAS in fact built
(`pseudo/sto_basis.py`, `cohp(basis="contracted")`) and is now calibrated —
measured-negative for parity — and the ~2× turns out to be mostly a spin
reporting convention, not diffuseness. See the 2026-09-13 follow-up below.]**

Also worth correcting: the original "operator overshoots ~2×, eigenvalue
undershoots ~2×, true value bracketed between" framing does not survive
per-bond resolution at a well-converged band count. At nbands=24 (well past
the reference-leak knee `cohp.py`'s docstring already quantifies), the
eigenvalue route (−20.4 eV/bond) sits only ~2% below the operator route
(−21.0 eV/bond) — both on the SAME side of LOBSTER's −9.64 eV, not bracketing
it. What looked like a bracket was the eigenvalue route's partial convergence
toward the operator value at lower nbands, not two independent estimates of
the true bond strength.

## Follow-up 2026-09-13: oracle obtained, cross-material calibration, convention reconciliation

Three things resolved this session (all measured on asus): (1) a free external
LOBSTER oracle now exists, (2) the contracted-STO path was finally calibrated and
is measured-**negative** for parity, and (3) the ~2× "overshoot" this plan (and
`cohp.py`'s QUANTITATIVE STATUS) attributed to basis diffuseness is **mostly a
spin-degeneracy reporting-convention difference**, not basis physics.

### Oracle — supersedes "no external COHP oracle exists in-tree"
The George-group bonding database (Naik, Ertural, Dhamrait, Benner, George,
*Sci. Data* **10**, 610 (2023)) is freely downloadable, CC-BY, **no LOBSTER binary
and no Materials Project API key**: Zenodo DOI 10.5281/zenodo.8091844
(`Lightweight_lobster_jsons.tar.gz`, 737 MB, one `mp-<id>.json.gz` per material,
1520 compounds). The references are VASP-PBE-**PAW**-derived. Five per-bond ICOHP
(eV, as-reported, integrated to E_F): C −9.586, Si −4.495, GaAs −4.350,
MgO −0.920, NaCl −0.566.

### Contracted-STO path is measured-negative — updates the step-4 decision and "Scoped, not started"
`pseudo/sto_basis.py` (`ContractedSTO`, `fit_contracted_sto` with a `tail_reg`
localization penalty) and `cohp(basis="contracted")` are in fact **built** — the
"not started" note above was stale. Calibrated on the `diamond_c` fixture
(operator route, `resolve_images`, per-bond):

- `pswfc` −21.26, `iao` −21.58, `contracted` (`tail_reg` 1e-3…0.3) −20.3…−21.4 —
  all cluster at **2.1–2.2× LOBSTER**, charge_spilling ~0.004–0.006.
- Only `tail_reg=3` reaches −11.6 (1.2×), but at **27 % charge_spilling** — the
  basis has stopped representing the occupied manifold.
- A pure free-atom single-ζ Slater sweep is numerically **unstable / non-monotonic**
  (−13.8 → −9.1 → −25.0 over small ζ steps); at carbon's actual free-atom exponent
  it gives −25.0 (2.6×, *worse*). "Fit a more contracted orbital" is refuted: no
  low-spilling, principled projector in this family reaches the target.

Conclusion: shipping per-element contracted-STO tables on the **norm-conserving**
projection path does not close the gap. Do not build that pipeline.

### Cross-material calibration (operator route, per-bond) — the actual resolution

| material | bonding | gradwave | LOBSTER | ratio | ÷2 |
|---|---|---|---|---|---|
| C diamond | covalent | −21.26 | −9.586 | 2.22 | 1.11 |
| Si | covalent | −11.02 | −4.495 | 2.45 | 1.23 |
| GaAs | covalent/polar | −9.43 | −4.350 | 2.17 | 1.08 |
| MgO | ionic | −1.835 | −0.920 | 2.00 | 1.00 |
| NaCl | ionic | −1.003 | −0.566 | 1.77 | 0.89 |

Ratios cluster tightly (1.77–2.45, mean 2.12); sign and the full ordering
(C > Si > GaAs > MgO > NaCl) match LOBSTER exactly. A near-constant ~2× across
covalent/ionic and 2s2p/3s3p/4s4p+d character is a **convention** signature, not a
basis-overlap effect (which would scatter with orbital character).

**Reconciliation.** gradwave weights each off-site bond by `factor 2` (block +
Hermitian conjugate, in `_pair_block_weights`) × `g_spin=2` (spin degeneracy), and
its total is pinned to the physical band-structure energy by its own sum rule
(Σ_pairs ICOHP = Σ_n f_n ε_n, f_n=2) — so gradwave is on the physical **2-electron**
scale. The Hermitian `2` is *required by the sum rule* and both codes must include
it. LOBSTER's `ICOHPLIST` for a non-spin-polarized (ISPIN=1) run stores the value
under `Spin.up` only, and pymatgen/lobsterpy do not double it — so the reported /
database value is the **per-spin (1-electron)** number. Net: gradwave (2-electron)
vs LOBSTER (per-spin) → ~2× by convention; dividing gradwave by 2 collapses the
ratio to mean **1.06** (C 1.11, Si 1.23, GaAs 1.08, MgO 1.00, NaCl 0.89). (The
spin-convention direction is inferred from pymatgen's Spin.up-only, non-doubling
storage plus the empirical ÷2 collapse; not a direct LOBSTER-manual quote — worth a
one-line confirmation against a raw `ICOHPLIST.lobster` if one becomes available.)

So the dominant offset is the **spin reporting convention**, and the ~±10–20 %
residual after ÷2 is the real, *secondary* NC-pseudo-vs-PAW effect: gradwave
slightly overestimates covalent bonds (1.08–1.23 — a more diffuse projector gives
larger bond-region off-site H̃) and matches/underestimates the ionic pair (0.89–1.00,
little shared bond density → basis matters least).

### Revised path to quantitative ICOHP
1. **Convention fix (cheap, principled).** To compare against LOBSTER ISPIN=1
   values, multiply LOBSTER by 2 (or add a per-spin option to `cohp`). gradwave's
   2-electron convention is the sum-rule-correct one — this is a
   comparison/documentation fix, not a physics change. Brings all five to ~±10–20 %.
2. **Residual NC-vs-PAW (~10–20 %, covalent-biased).** Close to a few % only via
   all-electron / PAW-reconstructed wavefunctions (gradwave's PAW path, or FLAPW as
   the COHP source), or document as a known systematic. The contracted-STO basis
   swap does **not** address this.
3. **Supersede the "basis diffuseness is the dominant cause" framing** in this
   plan's Motivation and in `src/gradwave/postscf/cohp.py`'s QUANTITATIVE STATUS:
   diffuseness is real but secondary; the dominant 2× was spin convention. The
   cohp.py docstring should be updated to match (tracked separately).

Validation harness now exists: 5 oracle points (1520 available via the Zenodo
pull), the `diamond_c` fixture in `tests/integration/test_cohp.py`, and the
operator-route calibration pattern.

## References

- Sánchez-Portal, Artacho, Soler, *Solid State Commun.* **95**, 685 (1995) — spilling, variational basis optimisation.
- Deringer, Tchougréeff, Dronskowski, *J. Phys. Chem. A* **115**, 5461 (2011) — projected COHP.
- Maintz, Deringer, Tchougréeff, Dronskowski, *J. Comput. Chem.* **37**, 1030 (2016) — LOBSTER: contracted STOs, absolute spilling, RMSp, Löwdin.
- Knizia, *J. Chem. Theory Comput.* **9**, 4834 (2013) — intrinsic atomic/bond orbitals.
- Bloch IAO for periodic systems — arXiv:2407.00852.
