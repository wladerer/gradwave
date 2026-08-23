# The rutile EFG deficit is SCF-convergence-limited, not a basis problem

Status: **finding** (2026-08-23, overnight EFG accuracy hunt). Answers the question opened by
the converged-k validation (#362) and the accuracy plan (#363): *why* are the rutile cation
EFGs (Ti 51%, O biaxiality) wrong, and what fixes them.

## TL;DR

The rutile deficits are dominated by **SCF convergence quality**, not the basis/core. Getting a
genuinely-converged fullpot rutile-TiO₂ SCF is the unsolved blocker; until it's solved, the
rutile-Ti EFG is effectively non-reproducible (it swings ~7× with the marginal convergence
state), and the basis levers (3p-semicore, d-HELO) **cannot be cleanly assessed**. The fix is a
convergence-methodology investment, not an LO. The cells that *do* converge (corundum O/Al,
MgF₂ F) are genuinely accurate (0.95–0.98 of Elk) — the problem is specific to the hard
fullpot cells (rutile, anatase).

## Evidence

**1. The frozen core is a red herring** (4-agent research, `efg_accuracy_plan.md` / PR #363).
Elk's own EFG (`writeefg.f90`, task 115) removes the l=0 component and adds the core only to l=0
(strictly spherical), so Elk carries no aspherical core term either; the literature
(Petrilli–Blöchl PRB 57 14690) puts core polarization at only ~10–20% for a 3d cation. So the
factor-of-2 Ti miss is not a missing core term.

**2. The fullpot SCF oscillates for rutile Ti.** The muffin-tin phase converges, then the
fullpot continuation (phase B) diverges: r_v converges to ~0.02–0.26, then jumps to 100+. The
`best_rv` mechanism saves the best state, but it's only MARGINAL (best_rv ~0.02 at best, often
0.9–37, `converged=False`). The recipe *phase-B-diverges → hand best state to `newton`* gates
`res ~ 7e-4` but does **not** reach genuine convergence.

**3. At marginal convergence the rutile-Ti EFG is not reproducible.** Across configs/settings it
swings by ~7×:

| setting / config | C_Q(⁴⁹Ti) | note |
|---|---|---|
| k2 / ecut200, baseline | 10.19 MHz | MARGINAL |
| k4 / ecut225, baseline | 6.64 MHz | MARGINAL(best_rv=2.2e-2) |
| k4, Ti 3p **frozen** | 77.6 MHz | best_rv=37 — never converged (garbage) |
| k4, Ti 3p **valence** | 15.77 MHz | MARGINAL — tantalizing but untrusted |
| k4, 3p-valence + **d-HELO** | 6.46 MHz | MARGINAL |
| k6, "converged" study | 2.19 MHz | V_zz components [-5.4,+4.6,+0.8] vs Elk [+19.3,-13.2,-6.2] |

Experiment is C_Q(⁴⁹Ti) ≈ 13.4 MHz. There is no stable computed value — the spread (2.2–15.8)
is the convergence state, not a basis lever. The rutile-O η is equally unstable (0.70 at
k2/ecut200 vs 0.11 at the k666 validation vs Elk 0.74).

**4. So the basis-lever ablation is confounded.** The 3p-valence config *hints* it helps
(C_Q 15.77, nearest exp) — consistent with Blaha PRB 46 1321 (Ti 3p is valence-sized) — but the
baseline itself swings 50% with a tiny k/ecut change, so the lever signal cannot be separated
from the convergence noise. A clean 3p-vs-d-HELO verdict requires a genuinely-converged SCF
first.

## Verdict (the accuracy-hunt answer)

- **The rutile (and by extension anatase) EFG deficit is a CONVERGENCE-quality problem.** The
  fullpot continuation does not reach genuine convergence for these cells at feasible cost.
- **The fix is a methodology investment**, not a basis/semicore/core lever: a more robust
  fullpot continuation / mixing / preconditioning that reaches `converged=True` with a small,
  stable r_v. (The campaign's Kerker+Newton recipe, #16, stabilizes the muffin-tin phase but the
  *fullpot* continuation for these aspherical-heavy cells remains the open problem.)
- **Not everything is broken:** the cells whose SCF *does* converge — corundum O/Al, MgF₂ F —
  are genuinely accurate (0.95–0.98 of Elk, matching experiment). The differentiable-basis /
  autodiff track (a clean FD lever signal) is **gated on this convergence fix**, since a stable
  ∂V_zz/∂E_l requires a stable converged EFG.
- **Mg²⁺ 2p-in-valence** (the sign-fix candidate) hit the *same* convergence wall (the 2p LO run
  didn't converge, best_rv~0.97), so it too is blocked on this.

## Next steps (for the convergence fix)

1. Instrument which channel drives the phase-B divergence (aspherical potential feedback
   ρ(K_Hxc·χ₀) → 1 for the aspherical mode is the prime suspect).
2. A gentler fullpot continuation: ramp the l>0 potential on gradually; stronger damping /
   Anderson regularization on the non-spherical channel; a trust-region newton.
3. Only once `converged=True` with stable r_v: re-run the 3p-vs-d-HELO ablation and Mg-2p for the
   clean basis verdict, and re-measure rutile Ti C_Q / O η vs Elk.
