# WORKLOG (delete before final PR)

Branch: perf/test-runtime. Goal: cut standard-tier runtime, no weakened assertions.

## STATUS
- All test-code + config edits DONE. .test_durations patched (10 changed keys, format preserved).
- Serial -n0 combined run COMPLETED: 1264.93s (21:04), all pass after cohp fix. Used for durations + step-7 total.
- -n2 loadscope combined run: KILLED twice under other-agent contention (resource pressure). Excluded; report serial 21:04 instead with caveat.
- fast-gate `make check FAST_JOBS=2` running (bd6c7dy16 -> /tmp/check_after.out).
- PENDING: fast-gate before (marker-revert A/B or estimate), delete WORKLOG, final commit+push, open ONE PR.

## CHANGES MADE (final state)
1. CI (.github/workflows/ci.yml) + Makefile test-standard: add `--dist loadscope`. Fast gate kept default `load` (surgical, not via pyproject addopts). No separate fast-gate CI job exists.
2. Markers: `pytestmark = pytest.mark.standard` added to test_hybrid_kmesh.py + test_learned_hybrid.py (verified: fast gate collects 0, standard collects all 9).
3. uspp (test_uspp_batched_equality.py): ecut 25->15, ecutrho 100->60, kmesh (2,2,2)->(2,2,1). Tolerances UNCHANGED.
4. bands_nc: ecut 28->20, kmesh (2,2,2)->(2,2,1).
5. cohp k_band_resolved: ecut 40->28. o2_gamma fixture: REVERTED to 40 (see marginality below).
6. gamma o2_gamma fixture: ecut 35->25.
7. metal_forces FD test: etol 1e-11->1e-9, rhotol 1e-10->1e-8. Regenerated fd_reference.json via gen_al_forces_fd.py (also patched to 1e-9/1e-8). Ref shift max 8.3e-11 eV, FD-derivative shift <7e-7 (tol 2e-4).
8. learned_hybrid _converge: etol 1e-10->1e-8 (rhotol kept 1e-9).

## MARGINALITY EXCLUSION
cohp o2_gamma ecut 40->28 REVERTED: assertion `iao.charge_spilling < 1e-3` (test_cohp_resolve_images_and_iao_o2) is only 2.12x from tol even at ecut 40. Scan (iao.charge_spilling):
  ecut40=4.722e-4(2.12x, main status quo) | 36=6.145e-4(1.63x) | 34=7.206e-4 | 32=8.415e-4 | 30=1.019e-3(FAIL) | 28=1.240e-3(FAIL)
Any cheapening moves it within 2x -> reverted per step-5 discipline. The 3 o2_gamma-sharing cohp tests get no savings (documented). k_band_resolved kept at 28 (its identities are basis-independent, passes with margin).

## ASSERTION MARGINS (actual deviation / tolerance) -- all PASS
- uspp nspin=1: dF=2.27e-13/1e-9, de=5.70e-11/1e-7, dforce=1.58e-10/1e-8
- uspp nspin=2: dF=0/1e-9, de=6.28e-10/1e-7, dforce=1.91e-9/1e-8
- bands_nc: max|e_scf-e_band|=4.07e-8 / 1e-3
- gamma eig: de=5.33e-14/1e-8, ortho=1.78e-15/1e-10
- cohp collinear sumrule ratio=0.97082 (at kept ecut40); iao ratio=1.00000 (bounds 0.90..1.001)
- metal_forces |analytic-FD|: mp1 c0=3.80e-7 c1=3.69e-7 ; cold c0=2.54e-7 c1=3.79e-7 (tol 2e-4)
- learned_hybrid: alpha rel=5.83e-7/1e-3 ; omega rel=1.95e-3/5e-3

## DURATIONS (s): CI-baseline (.test_durations) -> measured this-laptop after (contention-variable, NOT CI-calibrated)
- uspp[1]: CI 72  -> laptop after ~139 (uncontended-ish); laptop baseline was 315 uncontended => ratio ~0.44
- uspp[2]: CI 150 -> laptop only contended (197-1764); use ratio ~0.44 => est 66
- bands_nc: CI 134 -> 22.8 (measured, even under contention)
- cohp k_band: CI 48.1 -> 34.5 (clean low-contention rerun)
- gamma basis_closure setup: CI 33.5 -> 11.7 ; gamma eig: CI 5.15 -> 1.93
- metal FD mp1: CI 193 -> 16.6 ; cold: CI 192 -> 20.0 (NOTE: CI 193 likely STALE pre-FD-cache; tol relaxation itself saves only a few s)
- learned_hybrid alpha: CI 15.5 -> 6.2 ; omega: CI 28.7 -> 9.0 ; diff/training small
Caveat: laptop under other-agent contention; use for RELATIVE ordering only.

## Fast-gate hybrid tests removed (were unmarked=fast): hybrid_kmesh ~61s + learned_hybrid ~51s of SCF work left the <2min gate.
