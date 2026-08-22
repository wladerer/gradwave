# Oxygen EFG: the l=1 (O 2p) local orbital is a decisive NEGATIVE

Follow-up to `oxygen_efg_diagnosis.md` (the p×p/l=1 finding) and PR #344 (the conditioned l=0
O 2s LO, which took O full |V_zz| to 73% of Elk but left the on-site biaxiality at η ~0.91).
#344's residual diagnosis was: *the on-site O EFG is p×p (l=1) dominated; an l=0 LO enriches the
l=0 density but does not relax the O 2p (l=1) radial that sets the on-site biaxiality — the lever
is l=1.* This document tests that hypothesis directly by adding an l=1 O LO. **It fails.** The
l=1 radial freedom does not move the on-site η. The biaxiality is not a p-channel radial-freedom
problem.

All SCFs on asus (`~/gw-ol1`, worktree of `origin/main` incl. #344), rutile TiO₂ aug4/fp4/k222,
ecut 300, kerker as noted. Harness `ol1_accept.py` (env-driven l=0/l=1 O LOs + el_override +
kerker), conditioning pre-probe `ol1_probe.py`.

## VERDICT

Adding an l=1 (O 2p) local orbital on top of #344's l=0 O 2s LO:

- **does NOT move the on-site O η**: 0.911 (l=0 only) → **0.889** (l=0 + l=1). Elk is **0.099**.
  This is the decisive test the l=0 LO failed, and l=1 fails it too — by 0.02, not the ~0.8 needed.
- **flips the on-site O V_zz to the wrong sign** (+19.4; #344's l=0-only was −16.9, matching Elk's
  −10.27 sign) — the l=1 LO makes the on-site tensor *worse*, not better.
- raises full |V_zz| to ~90% of Elk but keeps the **wrong principal frame** ([001], not Elk's
  [110]) and **degrades Ti** (+17.1 → +14.3).
- is **numerically unstable**: it diverges the SCF at the standard kerker=0.7 (r_v → 1e1) from
  every starting basin and on both the old and the φ-direct density path; it only gates under
  heavy damping (kerker=0.3). The gated (k0.3) and marginal (k0.7) fixed points agree to η 0.001,
  so η 0.89 is the true self-consistent value, not a mixing/basin artifact.

The deficit is therefore **not** a p-channel radial-freedom problem. Inside the small O MT sphere
(R = 0.824 bohr) the valence span{u(2p), u̇(2p)} already captures the O p-radial space; a third
confined p-radial is near-redundant (adds no biaxiality-relaxing freedom) and merely destabilizes
the self-consistency. See the re-diagnosis below.

## The numbers (eV/Å²)

| config | Ti full V_zz | O full V_zz (frame) | O on-site V_zz / η |
|---|---|---|---|
| no LO (#344 baseline) | +18.07 | +10.63 [001] (56%) η 0.589 | −13.39 / **0.924** |
| + O 2s LO, l=0 (#344) | +17.10 | +13.89 [001] (73%) η 0.169 | −16.84 / **0.913** |
| + O 2s LO, l=0 (this work, φ-direct) | +17.11 | +13.89 [001] (73%) η 0.167 | −16.86 / **0.911** |
| + O 2s + O 2p LO (l=0+l=1), k0.3 GATED | +14.36 | +17.18 [001] (90%) η 0.147 | +19.41 / **0.890** |
| + O 2s + O 2p LO (l=0+l=1), k0.7 marginal | +14.33 | +17.20 [001] (90%) η 0.146 | +19.42 / **0.889** |
| Elk 11 | +19.34 | −19.10 [110] η 0.740 | −10.27 / **0.099** |

- l=0 (this work) reproduces #344 to within self-consistency noise — confirms the harness and the
  φ-direct density carry are faithful to the merged baseline.
- The l=1 LO's E₂ was swept -8…-28 eV in the conditioning pre-probe; **resid_frac stays 0.13–0.14
  across the whole ladder** (see below), so the confined l=1 LO adds nearly the same near-redundant
  direction regardless of E₂. The gated run above used E₂ = −20 eV; other E₂ give the same η within
  noise (E₂-independence is the point). The el_override "2p-semicore" variant (valence l=1 moved to
  a +5 eV scattering energy, LO anchored at the true 2p) is **even less stable** — it diverges at
  kerker=0.3 (r_v → 1.7e2 by n_it 50), the damping that gates the deep-E₂ LO. No l=1 O LO
  construction tried both converges and helps.

## Why the l=1 LO is near-redundant (the pre-probe)

`ol1_probe.py` resumes the #344 l=0 state, rebuilds the O sphere exactly as `_multi_iterate`, and
for each candidate E₂ builds the l=1 LO and reports resid_frac = the fraction of ‖φ‖ orthogonal to
the valence span{u(2p), u̇(2p)}:

| E₂ (eV) | −8 | −11.7 (2p) | −14 | −16 | −18 | −20 | −24 | −28 |
|---|---|---|---|---|---|---|---|---|
| resid_frac | 0.143 | 0.142 | 0.141 | 0.140 | 0.140 | 0.139 | 0.137 | 0.136 |

A confined l=1 LO is **~86% inside** the valence p-span at every energy. Contrast l=0, where the
deep O 2s LO sits distinctly below the valence 2s. There is essentially one p-radial degree of
freedom in the O sphere and {u, u̇} already hold it; the LO's 14% residual is a fixed high-curvature
remainder that carries no biaxiality lever. #344's overlap conditioning (`resid_frac < 0.18` →
Löwdin-orthogonalize) correctly fires on all of these, so the LO enters the pencil as an
orthonormal H-only coupling — but an orthonormal *near-null* direction is exactly what feeds the
instability below.

## Why it destabilizes the SCF (and why l=0 didn't)

The l=1 LO perturbs the O **p**-amplitude, which enters the on-site density through the **p×p**
cross term — i.e. it directly drives the **aspherical l=2** density/potential channel. That channel
is weakly damped in the fixed-point mixer relative to the spherical channel. The l=0 O 2s LO only
touches s×s → l=0 (spherical), which is why #344 gated cleanly at kerker=0.7 while the l=1 LO
diverges there (r_v → 38 by ~n_it 35–100) and needs kerker=0.3 to converge. This is a genuine
coupling instability of a near-redundant p-basis into the aspherical channel, not a density-
reconstruction precision problem: the φ-direct density carry (which removes the O(1e4)-coefficient
catastrophic cancellation of the fold path) improves accuracy but does not stop the divergence.

## Re-diagnosis — where the on-site biaxiality actually lives

With both the l=0 (#344) and l=1 (this work) radial levers exhausted and the angular cutoffs
already saturated (aug-lmax 4≈6, fullpot-lmax, per `oxygen_efg_diagnosis.md`), the on-site η ~0.9
vs Elk 0.10 is **not** a muffin-tin basis-completeness problem. What remains:

1. **The O 2p angular occupation (crystal-field / hybridization), not its radial shape.** The
   on-site η is set by the *m*-balance of the 2p hole (p_x,p_y vs p_z), which is a property of the
   occupied KS states' angular character — the crystal field and O–Ti hybridization — not of the
   radial function the p density is expanded in. gradwave's p_x/p_y/p_z occupation balance should be
   compared directly against Elk's on-site density matrix ρ_2M decomposition. A biaxial (η→1) vs
   axial (η→0.1) on-site tensor is an *angular* population difference. This is the prime suspect.
2. **The l=2 aspherical potential self-consistency**, which the l=1 LO destabilizes rather than
   improves — the near-redundant p-basis makes the p-amplitude (hence the l=2 density) *more*
   ill-determined, which is why the on-site V_zz even flips sign. A better-conditioned route to the
   aspherical channel (not a redundant LO) would be needed to test this.
3. **The O 2s/2p core-valence partition** (frozen 1s only): unlikely to set η but not excluded.

The actionable next step is a direct **O 2p m-projected occupation comparison vs Elk** (density-
matrix level), not another radial basis function. If the p_x,p_y,p_z populations already match Elk
and η still differs, the defect is in the aspherical-potential closure; if they differ, the lever
is the crystal-field/hybridization description (k-mesh, XC, or the interstitial–sphere matching),
none of which a local orbital touches.

## Bottom line

The l=1 O 2p LO is a decisive negative on all three acceptance gates: it does not move the on-site
η (0.911 → 0.889 vs target 0.099), it does not stabilize on Elk's [110] frame, and it is not even
self-consistently stable at production damping. No src change ships from this. The remaining O EFG
deficit is an **angular-occupation / aspherical-potential** problem, not a radial-basis one.

## Files

- `ol1_accept.py` — env-driven l=0/l=1 O LO acceptance SCF + axis-resolved & on-site EFG vs Elk.
- `ol1_probe.py` — seconds-fast conditioning pre-probe (resid_frac vs E₂ ladder), no SCF.
