# Anion EFG asymmetry (η) — dataset & basis-parameter fronts (2026-08)

Goal: close the anion η gap (corundum O η = 0.48 PW/PAW (#394), 0.65 FLAPW vs Elk 0.74;
rutile O η = 0.65 FLAPW vs Elk 0.74) without giving back the C_Q magnitude (96–106% of Elk).
A wrong η visibly distorts second-order MAS lineshapes, so this is the accuracy lever for
quadrupolar spectra. Two independent fronts, both measured on asus.

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

| O dataset | O η | O C_Q(¹⁷O) MHz | O on-site V_zz | Al C_Q MHz | note |
|---|---|---|---|---|---|
| (pending) | | | | | |

Elk 11 anchor: O η 0.740, C_Q(¹⁷O) 2.19 MHz, on-site V_zz +27.08.

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
| (running) | | | | | | |

## Verdict

(pending the tight rerun + Front A)
