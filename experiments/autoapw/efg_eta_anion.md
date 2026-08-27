# Anion EFG asymmetry (η) — dataset & basis-parameter fronts (2026-08)

Goal: close the anion η gap without giving back the C_Q magnitude (96–106% of Elk). A wrong η
visibly distorts second-order MAS lineshapes, so this is the accuracy lever for quadrupolar
spectra. Two independent fronts, both measured on asus.

**Reference correction — pinned across three independent repo sources.** The corundum-O η used to
anchor this task (0.74) was a **mis-label of the rutile value**. The correct **corundum O Elk η =
0.51**: `efg_converged_k_validation.md` (k6/k8 gated) gives Elk 0.51, `efg_multimaterial_validation.md`
gives Elk 0.512, and the **experiment** (Brun, Derighetti, Hundt & Niebuhr, *Phys. Lett. A* **31**,
416 (1970)) gives η 0.517 — Elk and experiment agree at ~0.51. The corundum O site is Wyckoff 18e,
site symmetry C₂ (twofold), so η is symmetry-unconstrained and 0.51 is allowed. gw already sits at
**0.48** (FLAPW HELO) / **0.481** (PW/PAW #394) — 94 % of Elk, within ~7 % of experiment:
**corundum O η is essentially solved, not a 65 % gap.** The genuine anion-η gap is **rutile O: gw
0.654 (FLAPW HELO) vs Elk 0.740** — rutile's planar-tricoordinate O is the stringent biaxial case.
Front B therefore targets rutile O; corundum O is the *transferability control* (an already-good η
that a raised HELO energy must not overshoot).

## Front A — PW/PAW: O-dataset completeness (corundum, `efg_eta_paw_datasets.py`)

Scoping facts (measured before any SCF):

- `pseudo.upf_paw.parse_upf_paw` accepts psl-style kjpaw PAW only (`q_with_l`, `nqf=0`,
  scalar/none relativistic). JTH/ATOMPAW ship XML-PAW (never parsed); RRKJ-refit USPPs
  (`nqf>0`) are rejected by design.
- GBRV / rrkjus USPP datasets parse but carry `is_paw=False` and **no AE/PS partial waves** —
  the Petrilli–Blöchl on-site term (`EFGOnSite.from_paw`) cannot be built from them at all.
  A non-PAW dataset is structurally unable to produce the on-site EFG in this code path.
- Every parseable O PAW generation ships the **same 4 projectors** (2S ×2, 2P ×2,
  `l_max_rho=2`): psl 1.0.0, psl 0.1, and the old GIPAW-era kjpaw. So the scan varies dataset
  *generation* (cutoff radii, reference energies, partial-wave shapes), not projector count —
  "more projectors per channel" is not purchasable in the UPF-PAW ecosystem for O.

Datasets are fetched, not committed: `experiments/autoapw/efg_fetch_o_datasets.sh`.

### Results (corundum Al₂O₃, ecut 60 Ry / ecutrho 480 Ry, k 2×2×2, PBE; Al fixed psl 1.0.0)

Elk 11 anchor (`efg_converged_k_validation.md`, k6/k8): O **η 0.51**, C_Q(¹⁷O) 2.20 MHz,
on-site V_zz +27.08. (Not the 0.74 rutile value — see the reference correction above.)

| O dataset | gen. | O η | O C_Q(¹⁷O) MHz | O on-site V_zz | Al C_Q MHz | conv |
|---|---|---|---|---|---|---|
| O.pbe-n-kjpaw_psl.1.0.0 (baseline) | psl 1.0.0 | 0.481 | 2.314 | +26.67 | 2.097 | ✅ 21 it |
| O.pbe-n-kjpaw_psl.0.1 | psl 0.1 | 0.480 | 2.291 | +26.42 | 2.095 | ✅ 22 it |
| O.pbe-kjpaw (GIPAW-era) | psl 0.x | 0.475 | 2.293 | +28.94 | 2.017 | ✅ 18 it |
| O.pbe-n-rrkjus_psl.1.0.0 (USPP) | — | — | — | — | — | ❌ IndexError |

**Front A verdict: measured NULL.** Three parseable O PAW *generations* spanning ~a decade give
**η ∈ [0.475, 0.481]** — indistinguishable, and all already **93–94 % of the correct Elk 0.51**;
C_Q sits at 104–105 % of Elk. Dataset generation does not move the anion η, because every
parseable O PAW ships the same 4 projectors (2s ×2, 2p ×2, `l_max_rho=2`) — the on-site l=2
completeness is fixed by the UPF-PAW format, not purchasable. The rrkjus **USPP fails structurally**
(`is_paw=False` → zero AE/PS partial waves → the Petrilli–Blöchl `EFGOnSite.from_paw` throws
`IndexError`): a non-PAW / GBRV-style dataset cannot produce the on-site EFG at all. JTH/ATOMPAW
XML-PAW is unparsed and RRKJ-refit (`nqf>0`) USPPs are rejected, so no richer set is reachable
without a new parser. **Recommendation shipped as a warning** (see below): flag when a non-PAW
(no-partial-wave) dataset is handed to `efg_paw`, since it silently loses the on-site term.

## Front B — FLAPW: finite-difference scan of the l=1 HELO energy (`efg_eta_helo_scan.py`)

The EFG forward path is numpy/float end-to-end — not autograd-differentiable w.r.t. basis
parameters (efg_accuracy_plan.md). The one scalar the shipped anion recipe exposes is the
unconfined l=1 HELO energy E₂ (`efg_anion_basis(helo_e=…)`, default 90 eV). This is the
never-executed "differentiable-basis fit, FD tier": dη/dE₂ by re-converging the fullpot SCF
per point (warm-started from the no-HELO O-2s-LO rutile state) and running one exact EFG pass,
watching **both** η and V_zz/C_Q — the known FLAPW trade-off (aug-lmax-6 lifts |V_zz|, drops η)
must not be repeated in reverse.

### Coarse scan (rutile TiO₂ O site, ecut 300, lmax 4, fp-lmax 4, k 2×2×2, kerker 0.7, shift-invert)

Elk 11 anchor: O V_zz −19.10, η 0.740.

| E₂ (eV) | V_zz (eV/Å²) | % Elk | η | C_Q(¹⁷O) MHz | gate |
|---|---|---|---|---|---|
| 60 | +17.61 | −92 (wrong sign) | 0.168 | 1.089 | GATED (12 it) |
| 90 | −15.39 | 80.6 | 0.611 | 0.952 | MARGINAL (83 it, cap) |
| 120 | −17.17 | 89.9 | 0.478 | 1.062 | MARGINAL (80 it, cap) |
| 160 | −13.49 | 70.6 | 0.918 | 0.834 | MARGINAL (83 it, cap) |
| 220 | +13.40 | −70 (wrong sign) | 0.720 | 0.829 | MARGINAL (89 it, cap) |

**The coarse scan is NOT a clean win, and its numbers are only semi-trustworthy.** Three
problems: (1) η is **non-monotonic** in E₂ (0.168→0.611→0.478→0.918→0.720) — the eigenvalue
ordering that names the principal axis re-sorts as the aspherical p-radial changes; (2) V_zz
**flips sign** (wrong-sign, negative %Elk) at E₂=60 and E₂=220, i.e. the [001]-vs-[110] frame
handoff; (3) **every point past 60 eV hit the MAXIT=89 cap without converging** and the shift-
invert secular certificate failed 56× (silent dense fallback), so each η/V_zz carries run-to-run
noise (baseline E₂=90 gave η 0.611 here vs 0.654 in the gated #370 validation — ~0.04 η drift).
Critically, where η is best (E₂=220, η 0.720 ≈ Elk) the magnitude is **worst** (−70%, wrong sign):
the known FLAPW magnitude↔η trade-off, now reappearing in the HELO-energy coordinate rather than
decoupling from it.

### Tight crossing-region rerun (E₂ ∈ {110,120,130,140,150}, dense eigensolve, maxit 200, +newton_polish)

To decide honestly whether a *well-converged* HELO energy reaches Elk's η without wrecking V_zz,
the crossing region is rerun with the forced dense eigensolve (no certificate noise), a far higher
iteration cap, and the efg_status.md robust recipe (Kerker + Anderson + newton_polish fallback),
reporting the actual residual and converged flag per point.

| E₂ (eV) | V_zz (eV/Å²) | % Elk | η | C_Q(¹⁷O) MHz | resid | converged |
|---|---|---|---|---|---|---|
| 110 | −14.87 | 77.8 | 0.689 | 0.920 | 1.1e−2 | ✅ (14 it, gate) |
| 120 | −14.66 | 76.7 | **0.724** | 0.906 | 1.0e−2 | ✅ (37 it, gate) |
| 130 | −14.36 | 75.2 | 0.773 | 0.888 | 8.4e−1 | ❌ diverged |
| 140 | −14.09 | 73.8 | 0.817 | 0.872 | 4.2e−5 | ✅ (32 it, tight) |
| 150 | +14.45 | −75.7 | 0.880 | 0.894 | 5.1e+1 | ❌ blew up (sign flip) |

**With trustworthy convergence the story flips from the coarse scan.** The three cleanly-
converged points (110, 120, 140 — 140 polished to a tight 4e−5 residual) are strictly
**monotonic**: as E₂ rises, η climbs (0.689 → 0.724 → 0.817) and |V_zz| falls (77.8 →
76.7 → 73.8 %). The coarse scan's non-monotonic η and V_zz sign-flips were **convergence
artifacts** (MAXIT cap + 56 shift-invert certificate failures), not physics — e.g. coarse
E₂=120 gave η 0.478, the converged value is 0.724. So there is **no decoupling**: E₂ is a
clean magnitude↔η trade-off coordinate, dη/dE₂ > 0 and d|V_zz|/dE₂ < 0.

Convergence is fragile above E₂≈120 (130 and 150 diverge even with newton_polish); the
reliably-gating window is E₂ ≤ ~120–140.

**Where it crosses Elk.** Interpolating the converged points, η = 0.740 (Elk) at **E₂ ≈ 123 eV**,
with |V_zz| ≈ 76.2 % of Elk there. Versus the shipped default E₂=90 (η 0.654, |V_zz| 79 % — the
#370 gated validation), reaching Elk's η costs **Δη +0.086 for −2.8 % magnitude**. That is a
*shallow, favorable* slope — the opposite sense and a gentler rate than the aug-lmax-6 lever
(which bought +10 % magnitude for −0.30 η). But it is still the trade-off, not a free η fix: you
buy a correct asymmetry with a small, known magnitude sacrifice. The 76–79 % rutile-O magnitude
floor itself is the separate structural residual (O–Ti spheres nearly touch, support ratio 0.98;
elk_efg_forensics D4) and is *not* what E₂ governs.

### Transferability control (corundum Al₂O₃ O, `efg_eta_helo_corundum.py`, cold converge_efg, dense)

Does the raised E₂ that helps rutile *overshoot* the anion whose η is already good? Corundum O
Elk η = 0.51, gw ≈ 0.46–0.48. Two points, both gated:

| E₂ (eV) | V_zz (eV/Å²) | \|r\| vs Elk | η | C_Q(¹⁷O) MHz | gate |
|---|---|---|---|---|---|
| 90 | +34.16 | 0.959 | 0.460 | 2.113 | ✅ GATED |
| 120 | +33.90 | 0.951 | **0.462** | 2.097 | ✅ GATED |

**Corundum O η is E₂-insensitive: Δη = +0.002 over E₂ 90→120** (vs rutile's +0.07 over the same
step), magnitude flat at 95–96 %. So the raised E₂ neither helps nor *overshoots* the already-good
anion — it simply does nothing there. The E₂→η coupling strength is **structure-dependent**: strong
exactly on the stringent biaxial site where η is hard (rutile's planar-tricoordinate O), null on
the well-described one (corundum's C₂ O).

## Verdict

**Front B — a real but structure-specific η lever, shipped as documented guidance on
`efg_anion_basis(helo_e=…)`, default kept at 90.** Measured facts:

- E₂ is a genuine, monotonic η knob on the hard case (rutile O: dη/dE₂ ≈ +0.007 eV⁻¹; η reaches
  Elk's 0.74 at E₂ ≈ 123, at a ~3 % |V_zz| cost — the same magnitude↔η trade-off as aug-lmax-6 but
  in the opposite, gentler sense).
- It is **null on the easy case** (corundum O flat 0.46→0.462) — so it does not overshoot an
  already-correct η; it is safe to raise where needed.
- It is **not a free η fix or a decoupling**, and it does **not** touch the separate rutile-O
  magnitude floor (76–79 %, the O–Ti touching-sphere D4 residual).
- Convergence turns fragile above E₂ ≈ 120 on rutile (130/150 diverge even with newton_polish), so
  a raised *default* would trade the validated, robustly-gating E₂ = 90 recipe (and its MgF₂ /
  corundum / rutile validations) for a fragile one for a modest, single-material gain.

Decision: **keep `helo_e=90.0` as the default** (robust, validated) and document E₂ as the η lever
so a user chasing a correct second-order MAS *lineshape* on a stringent biaxial anion can dial it
toward ~120 knowingly, accepting the ~3 % magnitude cost and the convergence caveat. The coarse
scan's earlier "non-monotonic / sign-flipping" read was pure convergence noise — corrected here.

## Front A verdict (summary)

**Measured NULL.** PAW O-dataset generation does not move the anion η (corundum O η 0.475–0.481
across psl 1.0.0 / 0.1 / GIPAW-era kjpaw; all already 93–94 % of Elk 0.51), because every parseable
O PAW ships the same 4 projectors — on-site l=2 completeness is fixed by the UPF-PAW format. A
non-PAW / GBRV dataset carries no partial waves and cannot produce the on-site EFG at all (now a
clear error, not an IndexError — `postscf.efg_paw`). JTH XML-PAW and RRKJ-refit USPPs are
unreachable by the parser. So "richer dataset" is not an available anion-η lever in this ecosystem.

## What shipped

- **Front A**: a clear `ValueError` in `EFGOnSite.from_paw` when handed a non-PAW dataset (was an
  opaque IndexError), with a unit test — the actionable form of "don't use USPP/GBRV for EFG".
- **Front B**: `efg_anion_basis(helo_e=…)` now accepts a **per-species** `{species: eV}` mapping
  (opt-in; absent species fall back to the robust 90 eV default), so the E₂ lever can be raised on
  a hard/near-touching anion site alone — a no-op elsewhere. Measured E₂→η guidance added to the
  docstring; default `helo_e` unchanged (90 eV). A unit test covers the per-species path. Both scan
  drivers + the dataset fetch script live in `experiments/autoapw/`.
- **Honest nulls** documented above: dataset generation (Front A) and a universal raised-E₂ default
  (Front B) both fail to close η beyond what the shipped recipe already achieves.
