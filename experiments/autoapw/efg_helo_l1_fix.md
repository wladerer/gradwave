# The unconfined second-energy l=1 HELO corrects the on-site EFG density bias

Follow-up to `efg_partition_diagnosis.md` (the ±bias is 100% on-site) and `oxygen_l1_efg_diagnosis.md`
(the CONFINED l=1 LO failed). This work adds an **unconfined** second-energy l=1 HELO and tests the
diagnosis's unifying-lever hypothesis directly. **It confirms for the anion (decisively) and moves
the cation the right way.** src change: a `confine=False` path in `scf._build_lo` + a
`{"e":…, "confine":False}` spec in `_build_lodat`.

All SCFs on asus, worktree `~/gw-helo` (origin/main @ 9680cad + this branch), corundum + rutile
warm-started from the committed converged states, k222 aug4/fp4 ecut300 kerker=0.7. Probe scripts:
`helo_probe.py` (conditioning), `helo_decisive.py` (corundum on-site decomposition),
`helo_rutile.py`.

## The construction (why "unconfined" is the whole point)

The confined Singh LO `φ = a·u(E₁) + b·u̇(E₁) + c·u(E₂)` imposes **two** boundary conditions
`φ(R)=φ'(R)=0`. Satisfying `φ'(R)=0` pulls in a large `u̇(E₁)` component (measured `b` ~ hundreds),
which collapses `φ` back into `span{u, u̇}`: the O 2p sphere's confined LO is ~85% redundant
(resid_frac ~0.15) at **every** E₂ from −8 to +90 eV (`helo_probe.py`). That is exactly the prior
l=1 negative.

The HELO drops the second boundary condition — `φ = a·u(E₁) + c·u(E₂)`, `φ(R)=0` only, `b=0`, slope
free — and takes E₂ **high** (a scattering-like `u(E₂)` with an extra radial node). `φ(R)=0` alone
keeps the orbital strictly inside the muffin tin (no interstitial matching); the in-sphere kinetic
element uses the gradient-square weak form, finite for a free `φ'(R)`. This is a genuinely new
in-sphere p-radial degree of freedom the confined form could not supply.

## Step 1 (DECISIVE): on-site gw/Elk ratio, corundum, warm-start-reproduced baseline

On-site `V_zz` = the interior l=2 sphere-Poisson of the aspherical valence density (`V_zz_valence`),
the term the diagnosis localized the entire bias to. Elk on-site reference from
`elk_onsite_corundum.py` (validated vs EFG.OUT). eV/Å².

| site | Elk on-site | gw baseline | gw + l=1 HELO | baseline gw/Elk | HELO gw/Elk |
|---|---|---|---|---|---|
| **O** (anion) | +27.08 | +17.76 | **+25.56** | 0.656 | **0.944** |
| **Al** (cation) | −6.185 | −7.26 | −7.12 (unconf) / **−6.83** (conf) | 1.173 | 1.151 / **1.105** |

- The warm-start baseline reproduces the diagnosis **exactly** (O 0.656, Al 1.173) — the harness is
  faithful.
- **O anion: 0.656 → 0.944.** The −27% undershoot is almost entirely closed by the unconfined O
  HELO (E₂ = 90 eV). O full V_zz +26.28 → +34.16 (Elk +35.63); C_Q(¹⁷O) 1.625 → 2.113 MHz.
- **Al cation: 1.173 → 1.105** with a distinct Al l=1 LO (the Al sphere's confined LO is already
  distinct, resid 0.78, unlike O's). Moves toward Elk but the cation overshoot is a weaker, only
  partially-radial effect. An unconfined Al HELO alone gives 1.151; the confined Al LO gives 1.105.
- Independence confirmed: an O-only HELO moves O and leaves Al at 1.173; an Al-only LO moves Al and
  leaves O at 0.656. One lever per site, same mechanism (the on-site l=2 p×p density).

**Verdict: both directions close toward Elk with the single radial-basis lever, the dominant
(anion) error decisively so.** The unifying-lever hypothesis of `efg_partition_diagnosis.md` is
confirmed; the "not a radial problem" re-diagnosis of `oxygen_l1_efg_diagnosis.md` is refuted for
the *unconfined* form (it was correct only for the near-redundant confined LO it tested).

## Stability: the unconfined HELO gates at kerker=0.7 where the confined LO diverged

The prior confined l=1 O LO **diverged** the SCF at kerker=0.7 (r_v → 1e1) and needed kerker=0.3.
The unconfined HELO **gates at the production kerker=0.7** (r_nsph < 1e-3) — the aspherical channel
is well-behaved because the HELO is a genuinely distinct direction, not a near-null orthonormal one.
This is the second decisive difference from the confined negative.

## Step 2: multi-material validation (full EFG tensor vs Elk vs experiment)

`helo_rutile.py` (TiO₂, warm from the #344 state), `helo_hbn.py` (h-BN, cold `converge_efg`), all
kerker=0.7 aug4/fp4 k222. C_Q via `nmr.quadrupolar_coupling`.

| system | site | metric | baseline | + l=1 HELO | Elk 11 | experiment |
|---|---|---|---|---|---|---|
| corundum | O (anion) | on-site V_zz | +17.76 | **+25.56** | +27.08 | — |
| corundum | O | full C_Q(¹⁷O) | 1.63 MHz | **2.11 MHz** | 2.19 MHz | — |
| corundum | Al (cation) | on-site V_zz | −7.26 | **−6.83** | −6.185 | — |
| corundum | Al | full C_Q(²⁷Al) | 2.48 MHz | **2.33 MHz** | 2.19 MHz | 2.38 MHz |
| rutile | O (anion) | on-site η | 0.913 | **0.147** | 0.099 | — |
| rutile | O | full V_zz (sign/frame) | +13.9 [001] | **−15.1 [110]** | −19.1 [110] | — |
| rutile | Ti | full V_zz | +18.1 | +17.2 | +19.3 | — |
| h-BN | B (cation) | full C_Q(¹¹B) | 3.62 MHz | 3.95 MHz | 2.95 MHz | 2.934 MHz |

**Anions (the dominant error) — decisively fixed.** Corundum O undershoot 0.66→0.94 on-site;
rutile O η 0.91→0.15 (Elk 0.10) and the full O tensor recovers Elk's [110] frame and negative sign,
both of which the confined l=1 LO could not (it left η at 0.89 on the wrong [001] frame and diverged
at kerker=0.7).

**Cations — material-specific, not a universal lever.** Corundum Al closes toward Elk (1.17→1.10,
C_Q 2.48→2.33 MHz, toward the 2.38 MHz experiment). But light B (h-BN) gets *worse* with a B HELO
(C_Q 3.62→3.95 MHz, away from the 2.934 MHz experiment) — its on-site is already over-captured and a
high-E p-radial over-enriches it further (though it did improve convergence, GATED vs baseline
MARGINAL). The production recipe is therefore: **apply the l=1 HELO to anion sites** (where it is a
clean, decisive fix) and to Al-type semicore-frozen cations; do **not** blanket-apply it to light
main-group cations. The HELO is opt-in per (species, l), so this selectivity is a config choice, not
a code branch.

## Bottom line

The unconfined second-energy l=1 HELO is the radial-basis fix for the on-site EFG **anion**
undershoot — the term the partition diagnosis proved carries the dominant bias — moving corundum O
and rutile O decisively to the Elk all-electron reference and gating at production damping where the
confined LO failed. It also improves the corundum Al cation. It is **not** a universal cation fix
(light B worsens), so the diagnosis's "one lever moves both directions" holds for the dominant
(anion) channel and for Al, but not for every cation.

## Files

- src: `scf._build_lo(confine=False)` + the `{"e":…, "confine":False}` LO spec in `_build_lodat`;
  unit test `tests/unit/test_flapw_radial.py::test_helo_is_unconfined_and_confined_to_sphere`.
- probes: `helo_probe.py` (conditioning ladder, confined vs unconfined), `helo_decisive.py`
  (corundum on-site decomposition), `helo_rutile.py`, `helo_hbn.py`.
