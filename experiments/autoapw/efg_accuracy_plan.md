# EFG accuracy-improvement plan (research-backed)

Status: **plan** (2026-08-23). Built from four parallel research agents (gradwave code
review, Elk-source + WIEN2k/QE cross-code, all-electron EFG literature, differentiable
basis-optimization feasibility), triangulated. Follows the converged-k validation
(`efg_converged_k_validation.md`, PR #362) which showed the EFG accuracy is site-specific:
anions good (corundum O 0.95, MgF₂ F 0.98 on-site gw/Elk, matching experiment), several
cations poor (rutile Ti 51%, rutile O full 0.63×, MgF₂ Mg²⁺ wrong sign, anatase non-gating).

## Unified diagnosis (triangulated)

**The frozen CORE is NOT the problem — the core-polarization / Sternheimer track is dropped.**
- Elk source (`writeefg.f90`, task 115): EFG = curvature of the total muffin-tin Coulomb
  potential with the l=0 component removed; the core is added *only* to l=0 (spherical,
  `rhocore.f90`) and is self-consistent only spherically → **Elk carries no aspherical core
  term either**. Our reference omits it too.
- Literature (Petrilli–Blöchl PRB 57, 14690 (1998); Sternheimer γ∞): core polarization is
  only **~10–20% for a 3d cation**; grows large only for heavy/high-γ∞ ions (rare-earth,
  5d, actinide). Cannot explain a factor-of-2 miss.
- gradwave code: frozen core is spherical by construction → contributes 0 to the l=2 EFG
  (confirmed absent) — but so is Elk's.

**The cation problem is the SEMICORE and the aspherical-density completeness, not the core.**
Anions work because their EFG-active shell (O 2p) is already valence; cations fail where an
EFG-active p-semicore is frozen or the l=2 *shape* is incomplete.

## Per-failure plan

### ① Rutile Ti (51%) — one empirical question, then apply
Conflicting hypotheses, resolved by a cheap ablation (step 1 below):
- **H1 (literature/Elk):** frozen **Ti 3p semicore**. Blaha, Singh, Sorantin, Schwarz,
  PRB 46, 1321 (1992): in rutile the Ti 3p V_zz contribution is *valence-sized*; a frozen
  spherical 3p is the missing ~half. Elk's `Ti.in` keeps 3s/3p in valence via semicore LOs.
- **H2 (gradwave code review):** Ti 3s/3p are *already* in the production valence config
  (`los={"Ti":[(0,"3s"),(1,"3p")]}`) → the residual is the **3d channel** (bare single-energy
  `{u(3d),u̇(3d)}`), needing a **d-HELO** (second-energy 3d radial, `confine=False` — the
  analog of the shipped anion p-HELO).
- Open fact: unclear which Ti basis the converged-k validation (`kconv_efg.py`) used. →
  the ablation resolves it. Target: |V_zz|≈2.0–2.1×10²¹ V/m², C_Q(⁴⁹Ti)≈13.4 MHz (η≈0.2).

### ② Mg²⁺ (wrong sign) — likely config-only, today
Mg is run frozen [Ne]; the 2p polarization sets the sign. Fix = Mg 2p → valence via LO
(`los`/`core`/`val_e`/`el_override` — all exist, no new code). Caveat (literature): Mg²⁺
V_zz is tiny and sign-set by near-cancelling lattice + completeness, so if 2p-in-valence
doesn't flip it, it is a boundary/completeness problem, not semicore.

### ③ Rutile O biaxiality (η 0.11 vs Elk 0.74) — aspherical completeness, not radial
Code review + literature agree: it is the l=2 *asymmetry* — the m-balance of the 2p hole
(pₓ,p_y vs p_z) set by the interstitial↔sphere matching + boundary assembly, not another LO.
Rutile O is planar-tricoordinate (η≈0.6 real, stringent). Fix = an m-projected O-2p
occupation diagnostic vs Elk, then matching/assembly work. No existing knob; harder.

### ④ Anatase (never gates) — SCF convergence, orthogonal to EFG
Marginal interstitial mode diverges the fullpot continuation. Fix = the Newton/Kerker/mixing
machinery already built for metals. Unlocks a currently-unmeasurable case.

### Deprioritized: deep-core Sternheimer
Real but ≤10–20%, neglected even by Elk. Only worth it if extending to heavy ions.

## The differentiable-basis multiplier (novel, gated)

Make-or-break honest fact: **the moat does not currently reach the basis parameters.** The
forward EFG path is numpy/`float()` end-to-end (E_l, LO energy, R_MT enter as plain floats
that sever the graph); the shipped `dV_zz/du` differentiates only w.r.t. *displacement*, not
basis knobs (a different perturbation class — both H and S move, a Pulay/overlap term the
current machinery does not carry).

So: **start with finite differences** (E_l is one scalar; dV_zz/dE_l = 2 solves — exactly the
Ti probe). If the lever moves V_zz to Elk within a physical window → escalate to a
**per-element, multi-material transferable fit** (shared E_l/LO-energy per element, jointly
over all 6 materials, regularized toward the band-center prior, validated leave-one-material-out)
— reaching Elk accuracy with *fewer* LOs than WIEN2k's brute-force recipe. Prior art:
**unclaimed** — nobody has autodiff-optimized LAPW/APW+lo basis parameters (nearest: 2025
DFTK.jl work on plane-wave pseudopotential params, arXiv:2509.07785). First-of-kind,
publishable; gradwave's differentiable all-electron FLAPW is the only substrate that hosts it.
Fragility favorable: no backprop-through-SCF; V_zz is in the code's "stable/splitting" class
(independent of the interstitial zero).

Build tiers: (1) FD (zero new code) → (2) frozen-coefficient torch re-plumb of the
augmentation→V_zz chain (~150 lines, explicit ∂V_zz/∂E_l) → (3) full self-consistent
basis-adjoint (large, the augmentation-Pulay analogue). Only escalate as parameter count grows.

## Recommended sequence

1. **Rutile-Ti ablation probe** (3p-in-valence vs d-HELO) — settles the Ti diagnosis, IS the
   FD sensitivity demo, near-zero code. **In flight (task-tracked).**
2. **Mg 2p → valence** (config-only) — likely fixes the sign; parallelizable.
3. **Apply the winning Ti lever** across TM cations; re-validate at converged k.
4. **Rutile-O m-population diagnostic** → the angular/assembly work.
5. **Anatase convergence** (Newton/Kerker) — enables measurement.
6. **Differentiable per-element basis fit** — after FD proves the lever: frozen-torch tier,
   then the transferable multi-material fit (the moonshot).

## Key references
- Petrilli, Blöchl, Blaha, Schwarz, PRB 57, 14690 (1998) — PAW EFG decomposition.
- Blaha, Singh, Sorantin, Schwarz, PRB 46, 1321 (1992) — Ti 3p semicore is valence-sized (rutile).
- Blaha, Schwarz, Herzig, PRL 54, 1192 (1985); Blaha, Schwarz, Dederichs, PRB 37, 2792 (1988).
- Singh, PRB 43, 6388 (1991) — LAPW local orbitals. Michalicek et al., CPC 184, 2670 (2013) — HDLOs.
- Nikolaev et al., PRB 102, 174305 (2020); Bastow & Stuart, Chem. Phys. 143, 459 (1990) — rutile Ti/O EFG.
- Yakovlev et al., Crystals 9, 507 (2019) — PAW-set-vs-WIEN2k EFG (semicore is the divider).
- Schmitz, Ploumhans, Herbst, arXiv:2509.07785 (2025) — AD in DFTK.jl (nearest differentiable-basis analogue).
