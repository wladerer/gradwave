# Plan: an energy-metric SCF stopping rule

## Status

Phases (a) and (b) landed (2026-07-30, branch `feat/energy-metric-convergence`).
The gate is the kernel-only contraction `(1/2)<r|K_Hxc|r>` rather than the
chi0-dressed response metric, since one chi0 application per iteration needs a
Sternheimer solve per band restricted to insulators, which is neither cheap nor
applicable to the metallic magnets the gate targets. The kernel term alone
validated cleanly, terminating the stagnating Ni PAW pulay arm at F within
2e-8 eV of the johnson reference fixed point while the density gate sat at the
120-iteration cap, and agreeing with the density gate on Ni/Fe PAW johnson and
Si NC. The Harris-Foulkes/KS gap is recorded alongside on the NC path as the
zero-machinery bracket. Phases (c) (spinor) and (d) (default flip) remain open.
The plan adds an opt-in convergence gate on the
estimated energy error `<drho|K|drho>`, formed from the exact response operators
`scf/implicit.py` already exposes and resolved per channel. It replaces the raw
density-residual norm as the stopping test on the systems where that norm has a
physical floor the energy does not.

## Motivation

The spinor SCF has a residual floor the energy does not share. The noncollinear
campaign (`research/noncollinear-convergence`) measures the magnetization channel
on Ni+SOC flooring near 2e-3 while the charge channel sits 5 to 7 times lower, and
the collinear study (`research/uspp-spin-channel`) measures the same
mid-to-high-|G| m-channel floor on Ni PAW, mixer-independent, with the moment
direction precessing at the floor rather than settling. The spinor driver's default
`rhotol=1e-7` is unreachable on a metallic magnet for that reason, and
`tests/integration/test_ni_soc_convergence.py` already gates at a deliberately loose
`rhotol=5e-3` to work around it. A raw-residual gate cannot separate a converged
energy from a precessing-moment floor, because both report the same norm.

The energy is second order in the density residual, so a residual that floors at
2e-3 can leave the free energy correct to microelectronvolts. Other codes gate on
the energy for this reason. VASP stops on `EDIFF`, the free-energy change between
steps, and Quantum ESPRESSO stops on `conv_thr`, an approximate energy-weighted
residual (the estimated SCF accuracy `<delta rho | delta V>`). gradwave can form the
same energy-weighted metric from the exact response operators and resolve it per
channel. VASP and QE gate on a single scalar.

## The metric

At the fixed point the energy is stationary, so stopping at a finite residual
`r = rho_out - rho_in` leaves an error second order in `r`. The exact form is
`(1/2)<x|(K_Hxc - chi0^-1)|x>` with `x = (1 - chi0 K_Hxc)^-1 r` the
dielectric-dressed residual. `postscf/convergence_error.py` already forms the
screened response metric `(1/2)<r|K_Hxc (1 - chi0 K_Hxc)^-1|r>` as the post-hoc
diagnostic `denergy_response`, built from `apply_chi0` and `apply_k_hxc` in
`scf/implicit.py`. That metric omits the near-singular `chi0^-1` kinetic-response
term and is not sign-definite, which is why it is a diagnostic today and never the
reported estimate. A stopping rule needs only a quantity that crosses threshold when
the energy is converged and stays crossed. The per-channel response metric
down-weights exactly the high-|G| precessing m-channel mode that carries little
energy, so it can serve as that quantity even without the absolute term. Whether it
holds is what phase (a) measures.

## Phased scope

### (a) Collinear opt-in gate

Wire the response energy metric as a live per-iteration stopping test on the
collinear NC and USPP/PAW drivers, triggered by an input token
(`convergence: energy` against the default `density`), leaving the residual gate the
default. The ground truth is the geometric energy-tail extrapolation
`estimate_scf_error(res).denergy`, already validated by truncating a converged
history and recovering the final energy. The gate must agree with that ground truth
on every archived campaign trace, stopping within a fixed tolerance of the iteration
where the tail extrapolation reports the energy converged, and never stopping early
on a still-descending energy. Starting now on `feat/energy-metric-convergence`.

### (b) Per-channel decomposition

Split the metric into charge and magnetization contributions for the collinear
nspin=2 vector, and into the four spinor channels (charge plus three Cartesian m)
for the noncollinear vector. The total-residual gate is dominated by the m-channel
floor, whereas the energy metric weights each channel by its response, so the
mid-to-high-|G| precessing m-channel mode contributes little to the reported error.
Report the per-channel energy contributions alongside the scalar so the reason a run
stops is visible. Starting now, alongside (a).

### (c) Spinor wiring and the Ni+SOC re-run

Thread the metric through `scf_noncollinear`, which the collinear response
primitives do not yet reach, and re-run the `test_ni_soc_convergence.py` matrix arms
against the energy gate. The arms that pass today only at `rhotol=5e-3` should pass
at microelectronvolt estimated energy error under the new gate, replacing the loose
residual tolerance with a physical one. Gated on (a) and (b) landing.

### (d) Docs and a default flip

Document the gate in `docs/manual/convergence.md` and the noncollinear pages, then
flip the default only after a soak across the magnetic and metallic battery.
Changing the default stopping test is a behavior change for every run, so it waits
for measured agreement on the full standard tier, not just the archived traces.

## Gates and kill criteria

- Phase (a) is a usable gate only if the response metric tracks the energy-tail
  `denergy` on every archived trace without a sign flip or a disagreement larger
  than one decade. If the missing `chi0^-1` term makes it oscillate or mis-order the
  stop, the fallback is to gate on the energy-tail extrapolation itself, which is
  already validated and system-agnostic but assumes a geometric convergence basin.
- Phase (b) is a kill point for the premise. If no channel-resolved energy metric
  separates a converged energy from the precessing m-channel floor, that is, the
  floor carries more than microelectronvolt-scale energy after all, then the raw
  residual was reporting a real error and the energy gate buys nothing. Record the
  negative in `docs/ideas.md`.
- Phase (c) fails if the Ni+SOC arms do not reach microelectronvolt estimated error
  under the gate. The likely cause would be the spinor `chi0` path being too
  ill-conditioned for the metric to converge, which is itself a measurement worth
  recording against the response operators.

## Validation

The archived campaign traces are the oracle for phases (a) and (b), so validating
the gate against them costs no new SCF. The Ni+SOC re-run in phase (c) is the only
heavy compute, and it goes to `asus` through `scripts/gwq`. The geometric-tail
`denergy` is the independent ground truth throughout, so the gate is checked against
a validated estimate rather than against itself.

## References

- Barat, Levitt, Torrent, arXiv:2606.26693 (June 2026), Stoner-susceptibility
  preconditioner for collinear ferromagnets near the magnetic transition, cited here
  for the response structure the metric shares.
