# The rutile EFG deficit is a BASIS/angular problem, not a convergence one

Status: **finding — CORRECTED** (2026-08-23). An earlier version of this doc (same date,
overnight EFG-accuracy hunt) concluded the rutile EFG was *SCF-convergence-limited* — that the
fullpot continuation "diverges (r_v → 100+)" and the EFG "swings ~7× with the marginal
convergence state". A direct reproduction on the documented recipe **refutes that**. This
version records what actually happens.

## TL;DR

There is **no fullpot convergence bug**. On the documented rutile recipe (O-2s-LO conditioned to
2p, ecut 300, fp-lmax 4, k 2×2×2, kerker 0.7, shift-invert) the plain joint-Anderson loop
**converges cleanly to gated=True** (`r_v = 2.9e-3`, `r_nsph = 3.8e-4`, 32 iterations). The
convergence machinery — Kerker preconditioning + metric-weighted joint Anderson +
`newton_polish` (the validated sledgehammer for the hard-config r_v plateau) + the
`anderson_stalled` detector — is **complete**. The earlier "divergence" came from *pathological
basis configs* (frozen Ti-3p semicore → `best_rv=37`, garbage) and *under-converged / non-polished*
runs, not a methodology gap.

At genuine convergence the residual vs Elk is **basis/angular, not convergence**:
- **Ti**: `V_zz = +17.10` = **88%** of Elk's `+19.34` (magnitude good); `η = 0.014` vs Elk 0.36
  (asymmetry wrong).
- **O**: `V_zz = +13.89` on `[001]`, `η = 0.168` vs Elk `V_zz = −19.10` on `[110]`, `η = 0.740`
  (the known rutile-O **biaxiality** problem — wrong principal axis and asymmetry).

This is exactly what the original 4-agent plan (`efg_accuracy_plan.md`, PR #363) named *before* the
overnight run: semicore + aspherical/angular completeness, **not** core and **not** convergence.

## Evidence

**1. The documented recipe converges (main, no ramp, no newton needed).** Cold SCF, chunked
warm-restart to the gate:

| config | outcome | Ti V_zz (Elk +19.34) | O V_zz / η (Elk −19.10 / 0.74) |
|---|---|---|---|
| OLEL=2p, ecut300, **k222** | **gated=True** r_v=2.9e-3 (32 it) | **+17.10** (88%), η=0.014 | +13.89 [001], η=0.168 |
| OLEL=2p, ecut300, **k444** | Anderson **stalls** r_v=0.69, gated=False (118 it) | +10.74 (un-converged) | +9.17, η=0.143 |
| Ti-only, ecut200, k222 | gated (done, 37 it) r_v=1.3e-2 | — | — |

The k444 row is **not a k-swing** — it is the documented Anderson *stall* on a hard, high-dof
config (`flapw/newton.py` docstring: "on the production-hard TiO₂ config … Anderson STALLS (r_v
plateau at 0.46 …) while Newton-Krylov converges in 3 Newton steps"). The `olo_accept` harness
does chunked Anderson **without** invoking `newton_polish`, so it reports gated=False; the
trustworthy k444 number requires the (already-built) newton polish. The k444 un-gated EFG is a
stalled-state artifact, not the converged value.

**2. The frozen core is a red herring** (unchanged, 4-agent research). Elk's own EFG
(`writeefg.f90`, task 115) removes l=0 and adds core only spherically → no aspherical core term;
Petrilli–Blöchl (PRB 57 14690) puts 3d core polarization at ~10–20%. The Ti factor is not a
missing core term.

**3. Why the earlier "convergence-limited" verdict was wrong.** It conflated three different
things into one nonexistent bug: (a) a *pathological basis config* (Ti-3p **frozen** → `best_rv=37`)
that is a basis-conditioning failure, not a mixing failure; (b) the *known* hard-config Anderson
stall, whose solution is the shipped `newton_polish`; and (c) *under-converged early stops*
(reading the EFG before the gate). None of these is a fullpot-continuation methodology gap. A
λ-continuation ("ramp the aspherical potential on gradually") was prototyped against the claimed
divergence and **measured net-negative**: on the config that Anderson already converges, the
discrete λ steps poison the joint-Anderson history and *induce* an r_v≈0.4 limit cycle — the very
stall it was meant to prevent. It was dropped, not shipped.

## Verdict

- **Convergence is solved.** Kerker + joint Anderson + `newton_polish` + `anderson_stalled`
  reach a genuine gate; k444 needs the newton polish (documented), k222 does not.
- **The rutile residual is basis/angular**, not convergence: Ti η and the O biaxiality
  (η 0.17 vs 0.74, wrong principal axis). The Ti *magnitude* is already 88% at k222.
- **Next (the real accuracy work):** the rutile-O biaxiality — an m-projected 2p-hole
  angular-population / aspherical-assembly diagnostic vs Elk — assessed on a **newton-converged**
  reference so the residual is a clean basis signal, not convergence noise. The Mg²⁺ 2p-in-valence
  sign fix (#58) and the differentiable-basis autodiff track are **not** blocked on a convergence
  fix (there is none to make); they proceed on the converged reference.
