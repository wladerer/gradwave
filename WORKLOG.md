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
</content>
</invoke>
