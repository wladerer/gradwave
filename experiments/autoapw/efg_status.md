# FLAPW EFG accuracy — current state (2026-08-23)

Consolidates the EFG accuracy picture after the convergence re-diagnosis. Supersedes the
"convergence-limited" framing of the first `fullpot_convergence.md` (corrected in #370).

## The corrected picture

**Convergence is solved, not the accuracy bottleneck.** The muffin-tin FLAPW SCF gates on the
documented recipes (Kerker + joint Anderson; `newton_polish` for the hard-config r_v plateau).
The earlier "fullpot diverges → EFG non-reproducible" verdict came from pathological basis
configs (frozen Ti-3p, `best_rv=37`) and under-converged/non-polished runs — not a methodology
gap. See `fullpot_convergence.md` (rewritten).

**The accuracy story is basis/angular, with one structural residual.**

## Validated results vs Elk 11 (ecut 300, fp-lmax 4, k 2×2×2, kerker 0.7, shift-invert)

| system | site | metric | gradwave | Elk | note |
|---|---|---|---|---|---|
| rutile TiO₂ | O | η | **0.654** | 0.740 | biaxiality fixed by the l=1 anion HELO (was 0.168, wrong [001] axis) |
| rutile TiO₂ | O | V_zz | −15.07 (79%) | −19.10 | correct [110] frame + sign; magnitude is the structural residual |
| rutile TiO₂ | Ti | V_zz | +17.10 (88%) | +19.34 | magnitude good; η still low |
| corundum Al₂O₃ | O | on-site V_zz | +25.56 (94%) | +27.08 | l=1 HELO (was 0.656) |
| corundum Al₂O₃ | O | C_Q(¹⁷O) | 2.11 MHz | 2.19 | 96% |
| corundum Al₂O₃ | Al | C_Q(²⁷Al) | 2.485 MHz | 2.19 | within 4% of the 2.38 MHz experiment |
| MgF₂ | F | V_zz | +46.70 (91%) | +51.55 | l=1 HELO (was 73%) |
| MgF₂ | Mg | V_zz | **−3.15** | −5.51 | **sign fixed** by 2p-in-valence (was +11.12) |
| MgF₂ | Mg | η | **0.546** | 0.553 | near-exact once 2p polarizes |

## What shipped

- **Anion basis recipe** (`flapw.basis.efg_anion_basis`, `FlapwParams.efg_anion_basis=[…]`, #371):
  the validated l=1 unconfined HELO (90 eV) + l=0 2s→2p semicore LO, opt-in per named anion
  species. The one-line fix for the rutile-O biaxiality (η 0.17→0.65) and the corundum/MgF₂
  anion undershoot (→0.91–0.96 of Elk).
- **LO overlap Schur-complement conditioning** (`flapw.lo_schur`, `GRADWAVE_FLAPW_LO_DIAG=1`, #372):
  in-context per-LO `resid_frac` — a redundant LO prints as a named, localized number instead of
  a spectrum-wide S⁻¹ᐟ² NaN.
- **Mg²⁺ 2p-in-valence** (config: freeze 1s,2s only; `val_e=8`; l=1 linearized at 2p): flips the
  Mg sign and fixes η. Not yet a shipped default cation recipe.

## The remaining frontier: the ~80% anion magnitude

After the on-site fix (HELO → 0.94 on-site), the *full-tensor* anion magnitude sits at ~79–91%,
geometry-dependent (corundum O 96% vs rutile O 79%: rutile's O–Ti spheres nearly touch, support
ratio 0.98). The forensics (`elk_efg_forensics.md`) attributes it to the boundary/lattice term;
D1 (Coulomb-only grid) and D2 (exact own-field subtraction) are fixed (#22); **D4** (neighbor
near-field matched to L≤4 vs Elk's L≤6) is the residual, biting through a delicate large-opposite
on-site/boundary cancellation.

**aug-lmax 6 is opt-in, not a default.** It lifts the rutile-O magnitude (79→89% at matched
convergence) but *drops* the biaxiality η (0.65→0.35 vs Elk 0.74), and lmax-6 is convergence-
fragile (Anderson stalls ~1e-2; `newton_polish` fell into a spurious sign-flipped basin). So it's
a magnitude lever with an η cost, not a clean win. Closing the anion magnitude without the η cost
is the open EFG problem — likely a dual/local near-field grid (D4) rather than a global lmax bump.
