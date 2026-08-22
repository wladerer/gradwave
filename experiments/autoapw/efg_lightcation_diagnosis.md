# Light-cation EFG "overshoot" is a k-mesh mismatch, not a radial/partition bias

Diagnostic-first follow-up to `main_group_quadrupolar_validation.md` (the ¹¹B/⁷Li "cation +20 %
overshoot") and `efg_helo_l1_fix.md` (the HELO worsens light cations). The campaign hypothesis was
that light main-group cations over-capture the on-site l=2 density (a radial/aspherical-shape bias)
and need the *opposite* of the anion HELO — a contracted/low-energy p-radial. **That hypothesis is
refuted.** The measured mechanism is a **k-point-sampling mismatch**: the published gradwave
main-group numbers were computed at **k222** while the Elk all-electron reference was at **k444**,
and the l=2 EFG (a p-population anisotropy) is strongly k-dependent in these small covalent/ionic
cells. At a **matched** k-mesh the "+20 % overshoot" vanishes (it slightly reverses to a small
undershoot). No radial-basis lever is warranted for light cations; the fix is k-convergence and a
matched-k comparison.

All work on asus, fresh worktree `.claude/worktrees/lightcation-efg-diag` off origin/main @ 6992795
(the #353 HELO fix). gradwave `converge_efg` (muffin-tin → warm fullpot → newton → one exact
`efg=True` pass), aug-lmax 4, fp-lmax 4, kerker 0.7, smearing 0, exact solves, anion 2s LO.
Elk 11.0.2 references built by `setup_elk_mgroup.py` (rmt **forced** to gradwave's spheres, rgkmax
matched, LDA, lmaxi 2). Term decomposition (on-site l=2 sphere-Poisson of rhomt vs boundary =
full − on-site) via the validated `elk_onsite.py` STATE.OUT reader (reproduces corundum Al −6.185 /
O +27.082 and every EFG.OUT full tensor to 3 digits; l=0 in-sphere charge matches INFO.OUT
muffin-tin charge to 4 digits).

## Step 1 — reproduce the overshoot, term-decomposed (gw k222 vs Elk k444)

| site | on-site gw / Elk | boundary gw / Elk | full gw / Elk | val in-sphere gw / Elk |
|---|---|---|---|---|
| ¹¹B (h-BN, R=0.70 Å) | −45.08 / −39.44 = **1.14** | +8.41 / +9.42 = 0.89 | −36.67 / −30.02 = **1.22** | 1.0715 / 1.0715 = **1.00** |
| ⁷Li1 (Li₃N 1b, R=0.90) | +5.79 / +5.33 = 1.09 | +2.71 / +1.98 = 1.37 | +8.51 / +7.31 = **1.16** | 0.211 / 0.182 = **1.16** |
| ⁷Li2 (Li₃N 2c) | −3.33 / −3.02 = 1.10 | −0.87 / −0.49 = 1.77 | −4.20 / −3.51 = **1.20** | 0.213 / 0.187 = **1.14** |

The full-tensor overshoot (B 1.22, Li 1.16–1.20) reproduces the campaign. But the term decomposition
already breaks the "100 % on-site, boundary exact" premise: for B the boundary is 0.89× (under), for
Li it is 1.37–1.77× (strongly over). eV/Å² throughout; Elk on-site from the STATE.OUT rhomt reader,
gw on-site from `V_zz_valence`, boundary = full − on-site (axial η=0 sites share the c principal axis).

## Step 2 — partition (RMT / l=0-charge) sanity check: SHAPE for B, mild partition for Li

The valence l=0 in-sphere charge at the **same forced RMT** (gw `sphere_charge` vs Elk INFO.OUT
muffin-tin charge minus in-sphere core):
- **B: gw 1.0715 e = Elk 1.0715 e — identical to 4 digits.** Same RMT, same in-sphere valence
  charge, yet the l=2 moment is 14 % larger → a genuine aspherical-*shape* effect, not a partition
  or normalization (RMT) effect.
- **Li: gw 0.211/0.213 vs Elk 0.182/0.187 — gw over-captures valence charge by +14–16 %.** For Li⁺
  (2s¹ only, no occupied p) the l=2 EFG is a small induced polarization, and the excess in-sphere
  charge tracks the full overshoot (1.16/1.14 ≈ full ratios) — a partition-flavoured component.

## Step 3 — radial localization for B: the l=2 shape is Elk's; only the amplitude differs

Cumulative on-site moment ∫₀^r ρ₂₀/r′ dr′ as a fraction of the total, at matched r/R (gw dump via a
warm efg pass; Elk from rhomt l=2 m=0):

| r/R | 0.25 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| gw B  | 23.1 % | 52.6 % | 72.7 % | 86.8 % | 95.9 % |
| Elk B | 21.6 % | 49.9 % | 71.4 % | 86.7 % | 95.3 % |

The **radial buildup is nearly identical** (gw only marginally front-loaded). With the l=0 charge and
the l=2 *radial shape* both matched, the 14 % on-site excess is not a contracted/expanded p-radial —
it is a larger **angular anisotropy of the p-population** (more l=2 per unit charge, uniformly in
radius). That pointed at the sampling of the band occupations, i.e. the k-mesh.

## Step 3′ — the mechanism: k-mesh convergence of the p-anisotropy (the decisive result)

gradwave ¹¹B on-site vs k-mesh (Elk reference is at k444):

| k-mesh | B on-site V_zz | B boundary | B full V_zz | C_Q(¹¹B) | val in-sphere |
|---|---|---|---|---|---|
| gw k222 | −45.08 | +8.41 | −36.67 | 3.599 | 1.0715 |
| gw k333 | −38.87 | +9.32 | −29.55 | 2.900 | 1.0680 |
| gw k444 | −36.68 | +9.55 | −27.13 | 2.663 | 1.0699 |
| **Elk k444** | **−39.44** | **+9.42** | **−30.02** | (2.95) | 1.0715 |

The on-site moves −45.1 → −36.7 (**−19 %**) from k222 → k444 — the entire "overshoot" and more. At the
**matched k444** the boundary is 1.01× (essentially exact, as the corundum forensics found) and the
on-site is **0.93×** (gw now slightly *under* Elk), full **0.90×**. The +22 % overshoot was an
artifact of gw@k222 vs Elk@k444. `sphere_charge` is k-flat (1.068–1.072) — the l=0 charge was never
the issue; the k-dependence lives entirely in the l=2 *anisotropy*.

Full convergence table (¹¹B, eV/Å²):

| | gw k222 | gw k333 | gw k444 | Elk k444 |
|---|---|---|---|---|
| B on-site | −45.08 | −38.87 | −36.68 | −39.44 |
| B boundary | +8.41 | +9.32 | +9.55 | +9.42 |
| B full | −36.67 | −29.55 | −27.13 | −30.02 |
| gw/Elk (on-site) | 1.14 | 0.99 | 0.93 | — |

gradwave's on-site is **not yet plateaued at k444** (k333→k444 still moves it −2.2 eV), so the true
converged gw value is *below* −36.7 in magnitude — i.e. the small matched-k undershoot vs Elk is real
and this cell class needs ≥k666 for a converged EFG. (A gw k555 and an Elk k666 were queued to pin the
plateau; asus was saturated at write time. They refine the plateau, not the conclusion — the crossover
through the Elk reference between k333 and k444 is already unambiguous.)

### Li₃N ⁷Li — the on-site collapse repeats (2nd independent light cation)

| site | on-site k222 → k444 | vs Elk k444 on-site | boundary k444 / Elk | full k444 / Elk | charge k222→k444 |
|---|---|---|---|---|---|
| ⁷Li1 (1b) | +5.79 → **+4.94** | +5.33 → **0.93×** (under) | +3.31 / +1.98 = **1.68** | +8.25 / +7.31 = 1.13 | 0.211 → 0.194 |
| ⁷Li2 (2c) | −3.33 → **−2.73** | −3.02 → **0.90×** (under) | −1.09 / −0.49 = 2.21 | −3.82 / −3.51 = 1.09 | 0.213 → 0.200 |

Li₃N confirms B: the **on-site "overshoot" collapses to a slight undershoot (0.90–0.93×) at matched
k444**, for both Li sites, exactly as B. The residual full overshoot for Li (1.09–1.13×) is now
carried entirely by the **boundary/Weinert lattice term** (1.7–2.2× Elk) — a *separate*, non-k,
non-radial effect: the D4 near-field / uncorrected-high-L neighbour-moment mechanism of
`elk_efg_forensics.md`, which bites for the tiny, close-packed Li spheres (R=0.90 Å in a dense
nitride) and is absent for B (B boundary 1.01× ≈ exact). This is a lattice-term, not on-site-density,
question and is out of scope for a radial-basis lever.

## Step 4 — no lever built (clean negative)

The matched-k data is unambiguous: the on-site l=2 "over-capture" is a k-undersampling artifact, not
a radial-basis or partition over-capture. A contracted/low-E "anti-HELO" p-radial would be treating a
non-existent radial disease and would *mis-fit* the (already Elk-matching) radial shape. **No lever is
warranted.** Since no src/radial change is proposed, the #353 anion/Al HELO gains cannot regress —
there is nothing to change.

## Implication for the #352 / #353 validation set (the important takeaway)

**The entire main-group + HELO validation was run at gradwave k222** (`main_group_quadrupolar_
validation.md`, `efg_helo_l1_fix.md`, and the corundum/rutile HELO probes all use k-mesh 2×2×2),
while every Elk reference is at k444. Because the on-site l=2 EFG is demonstrably ~19 % k-sensitive
(k222→k444 for B), **the k222-vs-k444 mismatch contaminates the whole comparison**, not just the
light cations:
- The "**cation +20 % overshoot**" (B, Li, corundum Al) is, at least for B and Li, entirely this
  artifact — gone at matched k.
- The "**anion −30 % under-capture**" (corundum O 0.66, rutile O η 0.91) and the #353 **HELO gains**
  (rutile O η 0.91→0.15, corundum Al C_Q 2.48→2.33) were **also measured at k222** and are therefore
  **not established at converged k**. The HELO may be partly compensating a k-undersampling error of
  the *opposite* sign on the anion. This is a flag, not a refutation — I did not re-run the anion set —
  but those numbers should be re-taken at matched, converged k before the HELO's anion benefit is
  considered bankable.

**Recommendation — standardize the EFG validation on a k-converged mesh, compared at matched k.**
For these 4-atom hexagonal/rhombohedral cells the EFG is not converged at k222 or even k444 (B on-site
is still drifting −2.2 eV k333→k444); the set should adopt at least **k666** (or a per-cell k-convergence
gate on V_zz, e.g. <2 % change on doubling the linear density) for *both* codes, and every gw/Elk
ratio in the campaign re-taken there. Until then, on-site EFG ratios from the k222 runs — cation *and*
anion — should be treated as k-unconverged.

## Verdict

- **Confirmed mechanism: k-point sampling of the p-population anisotropy, NOT a radial-basis or
  partition over-capture.** Proven for B by (i) the l=0 in-sphere charge matching Elk exactly
  (1.0715 = 1.0715) while the l=2 moment is 14 % high; (ii) the l=2 radial shape matching Elk;
  (iii) the on-site collapsing −19 % from k222→k444 to reach (and slightly cross) the Elk reference;
  (iv) `sphere_charge` being k-invariant. **Confirmed independently for Li₃N ⁷Li** — both Li sites'
  on-site over-capture collapses to a 0.90–0.93× undershoot at matched k444.
- **Matched-k444 on-site ratios (the answer):** ¹¹B **0.93×**, ⁷Li1 **0.93×**, ⁷Li2 **0.90×** — the
  "+20 % overshoot" is gone; a small *under*shoot remains (aug-lmax 4 vs Elk lmaxo 6, and gw's own
  k still drifting — see the convergence table).
- **The campaign hypothesis is refuted.** Light cations do NOT over-capture on-site l=2 density via a
  radial mechanism and do NOT need a contracted/low-E "anti-HELO" p-radial. **No lever built.**
- **Li's residual full overshoot is a boundary/lattice term, not on-site density.** At matched k444
  Li's on-site is *under* Elk but its Weinert boundary term is 1.7–2.2× Elk (the D4 near-field for
  tiny close-packed spheres) — a separate lattice-assembly question, not a radial-density one.
- **Systemic k-mismatch flag:** the whole #352/#353 set (cations *and* the anion HELO gains) was
  gw k222 vs Elk k444; those ratios are k-unconverged and should be re-taken at a matched, converged
  mesh (≥k666) before further inference.

## Reproducing

- `experiments/autoapw/lc_run.py` (MAT=hbn|li3n, KMESH env; prints on-site / boundary /
  sphere_charge), `elk_onsite.py` (STATE.OUT rhomt on-site decomposition, general nspecies),
  `elk_cum.py` (Elk l=2 radial buildup), `lc_radial.py` (gw ρ₂(r) dump via warm efg pass).
- Elk dirs `~/efg_mgroup/{hbn,li3n}_elk` (k444) and `hbn_elk_k6` (k666); rmt forced to gw spheres.
