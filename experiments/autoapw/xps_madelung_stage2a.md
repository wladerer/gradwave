# Stage 2a — where the FLAPW interstitial-Madelung (C0_ext) deficit comes from

This is the diagnosis stage for the within-cell O1s core-level shift on the synthetic
Ti + 2 O demo (one Ti, two O at 1.75 and 2.30 Angstrom in a 13-Bohr cubic box; LDA,
Gamma-only, `use_symmetry=False`, `fullpot=False`). Prior stages localized the wrong-signed
O1s shift to the l=0 external Madelung constant `C0_ext = v_bc(R) - E2(q_sph-Z)/R`
(`core_levels.onsite_madelung_potentials`): gradwave gives `dC0_ext(long-short)` too small
vs Elk 11.0.2 at matched basis. Stage 2a compares the interstitial Coulomb POTENTIAL
directly (not just the site constants) and attributes the deficit.

All numbers were measured on the asus peer against the committed matched-basis Elk decks
(`~/xps_elk_dq/o{100,140}_gwbasis`, rgkmax 3.834, LDA, Gamma, matched R_MT). gradwave was
run at ecut 200 through the internal `_multi_*` driver (`experiments/autoapw/xps_s2a_probe.py`).

## 1. Confirm the gap (both codes, same decomposition, matched basis)

`C0_ext` decomposed identically in each code (Elk from VCLMT l=0 at R minus the analytic
own monopole, `xps_elk_decomp.py`; gradwave from `st.v_hart` via `onsite_madelung_potentials`):

| O R_MT (Bohr) | gradwave dC0_ext | Elk dC0_ext | gap |
|---|---|---|---|
| 1.00 | +0.736 | +2.903 | 2.17 |
| 1.40 | +2.814 | +4.794 | 1.98 |

Elk's `V_ext(r)` is flat in r inside the sphere (e.g. +2.902 at every radius at R 1.00),
confirming the external field is a rigid l=0 constant, as it must be. The earlier
`xps_validation.md` claim that gradwave "matches Elk at R 1.40" conflated Elk's TOTAL
EVALCORE eigenvalue shift (+1.22) with its decomposed `C0_ext` (+4.79); the matched-basis
decomposition here is the apples-to-apples comparison and it shows a genuine ~2 eV deficit.

## 2. The gap is in the interstitial potential, not the own-monopole subtraction

`C0_ext = vbc0 - E2(q_sph-Z)/R`. The in-sphere charges `q_sph` match Elk (O sites agree to
~0.005 e; see below), so the analytic own-monopole term matches. The entire gap is therefore
in `vbc0` = the l=0 surface average of the interstitial Coulomb grid `v_hart` at R:

| R_MT | quantity | gradwave | Elk |
|---|---|---|---|
| 1.00 | d(vbc0) (long-short) | -0.739 | +1.62 |
| 1.40 | d(vbc0) (long-short) | +0.263 | +2.47 |

gradwave's `v_hart` surface-value difference between the two O is ~2.3 eV too small, even
wrong-signed at R 1.00. Direct sampling of both grids (`elk_vclir_compare.py`) shows why:
Elk's interstitial Coulomb potential (VCLIR) is nearly flat (~+1.2 to +1.7 eV across the whole
interstitial), while gradwave's `v_hart` swings to -3..-20 eV at genuinely-interstitial points
and to -170 eV inside the muffin tins. gradwave's interstitial potential is corrupted.

## 3. Two independent causes (attributed on the converged state)

### 3a. Dominant (~1.3-1.45 eV): the Ti in-sphere charge differs (a semicore choice, NOT Coulomb)

The Ti muffin-tin radii match exactly (1.70075 Bohr in both codes), but the Ti in-sphere
charge does not:

| R_MT | gradwave Ti q_sph | Elk Ti q_l0 | difference |
|---|---|---|---|
| 1.00 | 19.518 | 18.668 | +0.850 e |
| 1.40 | 19.520 | 18.751 | +0.769 e |

gradwave's Ti holds ~0.8 e more, so its Ti net charge is less negative (q-Z = -2.48 vs Elk
-3.33), and a weaker cation produces a weaker Madelung-field DIFFERENCE between the near
(O_short) and far (O_long) oxygens. The point-charge estimate
`E2 (q_Ti-Z)(1/d_long - 1/d_short)` gives +4.9 eV (gradwave) vs +6.6 eV (Elk) — a +1.7 eV
difference from the Ti charge alone, matching the residual almost exactly.

Cause: gradwave freezes the Ti **3s** as localized core (`scf._CORE["Ti"]` = 1s,2s,3s,2p),
while Elk treats 3s as valence (spcore=F, via a local orbital). The frozen atomic 3s sits
entirely inside the sphere; the crystal-valence 3s has a small interstitial tail and
hybridizes, so ~0.8 e less charge stays in Elk's Ti sphere. This is a core/valence
partition choice, not a bug in `_weinert_multi`, and it cannot be fixed there.

Direct confirmation (`ti_charge_test.py`): scaling gradwave's Ti in-sphere density to Elk's
charge, with the masked interstitial (3b), reproduces Elk:

| R_MT | masked, gw Ti charge | masked, Ti scaled to Elk | Elk |
|---|---|---|---|
| 1.00 | +1.153 | +2.603 | +2.903 |
| 1.40 | +3.512 | +4.823 | +4.794 |

### 3b. Secondary (~0.4-0.7 eV): gradwave uses the UNMASKED plane-wave interstitial density

Elk's interstitial density (VCLIR's source, rhoir) is exactly zero inside every muffin tin —
Elk masks the plane-wave density with the characteristic function Theta_I and gives its
Weinert pseudocharge the full true moments q^MT (`elk_rhoir_check.py`: Ti +0.0000 e,
O +0.0001 e inside). gradwave instead feeds the FULL unmasked plane-wave density to
`_weinert_multi` — its smooth continuation piles +25 to +32 e inside the small Ti sphere
(`q_rhoI_in`), forcing a huge deficit pseudocharge (q^MT - q_i ~ -27 e). This catastrophic
cancellation is what corrupts the interstitial potential.

The two variants are mathematically identical (both reproduce the true moments), but not
numerically. Masking gradwave's rho_I (variant B, matching Elk) on the converged state
(`mask_variant_test.py`) recovers part of the gap; the unmasked variant is grid-CONVERGED at
the wrong value (flat from nfft 36 to 108), so it is not a resolution knob:

| R_MT | unmasked dC0_ext | masked dC0_ext | Elk |
|---|---|---|---|
| 1.00 | +0.736 | +1.153 | +2.903 |
| 1.40 | +2.814 | ~+3.1-3.5 (grid-noisy) | +4.794 |

A naive real-space mask (`np.where`) adds a sharp step at R that aliases (the masked value
oscillates ~3.0-3.5 across grids); a clean fix would need a band-limited characteristic
function like Elk's `gencfun`/`cfunir`.

## 4. Verdict — report, do not force a `_weinert_multi` change

The task premise (the entire deficit is the external Madelung constant, localized to
`_weinert_multi`) is only partly right. The unmasked-rho_I cancellation IS a real formulation
difference in `_weinert_multi` vs Elk, but it is the SMALLER cause (~0.4-0.7 eV). The DOMINANT
cause (~1.3-1.45 eV) is the Ti 3s core/valence partition — a density/basis choice outside the
Coulomb layer. So:

* No single change to `_weinert_multi` reaches the target (Elk +2.90/+4.79, O1s sign flip).
  Masking alone leaves ~1.3-1.5 eV on the table and does not flip the O1s sign to match Elk.
* The masking prototype changes `v_hart` everywhere and thus the l=2 EFG lattice term; its EFG
  impact is unverified, and the naive mask is noisy. Per the stage-2 guardrail (do not risk the
  validated EFG for an incomplete fix), it is not merged here.

### Open questions for a follow-up (owner decision)

1. **Masking in the SCF (not post-hoc).** Whether masking rho_I inside `_weinert_multi` and
   re-converging also pulls the Ti charge toward Elk's (a feedback path the post-hoc test cannot
   see) — if so, masking could close more of the gap than the +0.4-0.7 eV measured on the frozen
   density. Needs a band-limited Theta_I and an EFG (corundum V_zz, torture) re-validation.
2. **Ti 3s as valence.** Matching Elk's core/valence partition (3s -> valence local orbital)
   is the dominant lever, but it is a semicore change with its own consequences (Ti EFG, Ti core
   levels) and belongs in its own investigation, not the Coulomb layer.

## Reproduce (asus)

```bash
# gradwave converged state + per-site table (slow, ~15-30 min/radius):
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 uv run python experiments/autoapw/xps_s2a_probe.py 1.40 200 60
# Elk C0_ext decomposition (fast):
uv run python experiments/autoapw/xps_elk_decomp.py ~/xps_elk_dq/o140_gwbasis/{STATE,EVALCORE}.OUT 8 1,2
# Elk masks rho_I (rhoir=0 in MTs):
uv run python experiments/autoapw/elk_rhoir_check.py ~/xps_elk_dq/o140_gwbasis/STATE.OUT
# variant + Ti-charge attribution on the pickled state:
uv run python experiments/autoapw/mask_variant_test.py /tmp/gw_state_r140.pkl 4.794
uv run python experiments/autoapw/ti_charge_test.py /tmp/gw_state_r140.pkl 18.751 4.794
```

Elk decks (`~/xps_elk_dq/o{100,140}_gwbasis`) and the pickled gradwave states
(`/tmp/gw_state_r{100,140}.pkl`) live on asus and are not committed.
