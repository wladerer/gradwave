# SCF cycle-count study — worklog

Append-only. Mission: reduce SCF iteration counts; benchmark-driven; end in a #133
findings comment and AT MOST one options.py default PR if a robust win exists.

## Prior-art absorbed (docs/manual/wisdom.md, docs/manual/performance.md, #133)
- "The mixing scheme sets the iteration count on a metal": fcc Pt PAW johnson 13,
  pulay 17, broyden 20. johnson = right default for a smeared metal. Free energy
  bit-identical across schemes.
- johnson bit-identical to pulay on nspin=1 (same fixed point); auto-default in
  loop.py is johnson for nspin==2 else pulay (`_resolve_mixing_scheme`). USPP
  (options.py) defaults scheme="pulay" for ALL nspin.
- "A better initial WF does not cut the SCF iteration count" (guess axis already
  settled: O2 28->26, Ni 12->12; wall neutral-to-worse). 
- local_tf helps inhomogeneous cells only (Al slab 21->17), bulk unchanged (9==9).
- adaptive_diago_tol already adaptive (quadratic, first_tol 1e-3); expect "no lever".
- Learned MultipoleKerkerPrecond: bench_learned_precond.py already implements the
  full probe->fit->deploy loop (al ties, cu wins expected, cu3al/fe/pt/ni).

## Plan
1. NC scheme×history×alpha + kerker-q0 + precond + diago sweep -> scf_cycle_study.py
2. USPP pulay-vs-johnson (options.py PR question) -> scf_cycle_uspp.py
3. Learned precond -> run existing bench_learned_precond.py
4. All on asus CPU via gwq (iteration counts are device-independent & deterministic).

## Resource model
asus canonical .venv/bin/python has torch 2.12.1+cu130 + gradwave. Clone branch to
/tmp/gw-scfcycles, run with that python + PYTHONPATH=src (deps identical, no uv sync).
Submit via gwq run --group bench, write to /tmp/*.log, poll gwq status.

## Log
- (setup) worktree perf/scf-cycle-study from origin/main @0c367dd. Read all owned
  files + infra. Wrote drivers. Pushed. Cloned to asus /tmp/gw-scfcycles (from
  github; asus canonical lacked branch). Drivers import OK.
- GOTCHA: `gwq run -- bash -c "cmd > log"` breaks — gwq space-joins argv, so
  `bash -c bash /tmp/x.sh > log` runs `bash -c bash` (nothing) and log is 0 bytes.
  FIX: put `exec > /tmp/x.log 2>&1` inside each launcher, submit `bash /tmp/x.sh`.
- Launched jobs 46 (nc_small: scheme+alpha+q0+precond+diago), 47 (uspp), 48
  (learned) on asus default group.
- EARLY DATA (streaming):
  - si_insulator (NC): baseline=pulay/h8=pulay/h12 = 9 iters, E/at bit-identical.
  - si_paw (USPP insulator): pulay=18, broyden=21 (johnson pending).
- Next: collect metals (johnson-vs-pulay is the PR crux), learned verdicts,
  then medium confirm on the winner.

## RESUME (worker 2, 2026-07-27)
- gwq status: jobs 46/47/48 (nc_small/uspp/learned) + 49/50 all Done:Success.
  Harvested all JSON+logs from asus /tmp. No resubmits needed.
- HEADLINE: on USPP/PAW, johnson CRUSHES pulay (the scf_uspp default), nspin=1:
    si_paw 18->12 (-33%), cu_paw 19->13 (-32%), pt_paw 21->12 (-43%).
    E/atom identical (<5e-9 eV/atom). Broyden worse everywhere. This is the win.
- NC (nspin=1) sweep: NO lever. si 9=9, cu 11=11, al 10=10 across pulay/johnson;
    mgo 11->10 & si_m 17->16 johnson (marginal, <10%). Broyden worse on metals
    (cu 11->13). history h8/h12/h16 irrelevant. q0 kerker flat at optimum, q0=2.0
    hurts. kerker==local_tf on bulk. diago 3 variants flat on count (no lever,
    as predicted). alpha=0.9 johnson: al 10->8, cu 11->10 (aggressive, single-class,
    not pursued). fe_fm (nspin=2) baseline already johnson (14).
- NC medium: si_m johnson 16 vs pulay 17; cu_m 22=22. Confirms NC has no default lever.
- Learned MultipoleKerkerPrecond (bench_learned_precond both): TIES tuned Kerker on
    every real cell (al/cu/Fe/Pt 7/10/12/9 iters, same fixed point), LOSES on
    Cu3Al (11 kerker -> 12-13 learned). Synthetic 2-scale toy shows 3.5x but does
    not transfer. Honest negative -> no deploy.
- PR wiring: scf_uspp default lives in TWO spots: uspp_loop.py:735 kwarg
    mixing_scheme="pulay" (+ _flat_defaults:798 guard) AND options.py
    MixerOptions.scheme="pulay". NC scf() ignores MixerOptions (own None default +
    _resolve_mixing_scheme, johnson iff nspin==2). So flip is PAW-only, safe for NC.
- De-risk before unconditional flip: nspin=2 PAW must not regress. Submitting
    scf_cycle_uspp_confirm.py: fe_fm_paw + ni_fm_paw (nspin=2) + si_paw_m (8-atom
    medium insulator), pulay vs johnson each. -> /tmp/scf_uspp_confirm.json.
</content>
</invoke>
