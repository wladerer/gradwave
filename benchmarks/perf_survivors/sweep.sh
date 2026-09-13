#!/usr/bin/env bash
# Perf-survivors A/B sweeps. Run on an otherwise idle box (asus), 8 threads.
# Each arm is its own process (equal warmup cost); reps=3, read the min.
# Usage: ./sweep.sh <phase>   phase in {mdf, nadd, ritz, chord}
set -u
cd "$(dirname "$0")/../.."
L="benchmarks/perf_survivors/ladder.py"
run() { echo "### $*"; env "$@" ; }

phase="${1:?phase}"
case "$phase" in
  mdf)  # S4 glue retune: subspace depth (restart cadence)
    for c in si2 al4 fe1 al32; do
      for s in davidson davidson-native; do
        [ "$c" = al32 ] && [ "$s" = davidson ] && continue  # 390 s/rep — skip
        for f in 2 3 4 6; do
          echo "### case=$c solver=$s mdf=$f"
          GRADWAVE_MAX_DIM_FACTOR=$f uv run python $L time $c $s 3
        done
      done
    done ;;
  nadd)  # S4 glue retune: expansion-width cap (eager only)
    for c in si2 al4 fe1; do
      for cap in 0 2 4 8; do
        echo "### case=$c solver=davidson nadd_cap=$cap"
        if [ "$cap" = 0 ]; then uv run python $L time $c davidson 3
        else GRADWAVE_DAV_NADD_CAP=$cap uv run python $L time $c davidson 3; fi
      done
    done ;;
  ritz)  # S5 thick Ritz buffer (eager all-k path only)
    for c in si2 al4 fe1; do
      for t in 0 2 4 8; do
        echo "### case=$c solver=davidson ritz_tail=$t"
        if [ "$t" = 0 ]; then uv run python $L time $c davidson 3
        else GRADWAVE_RITZ_TAIL=$t uv run python $L time $c davidson 3; fi
      done
    done ;;
  chord)  # S6 chord / frozen-C tail
    for c in si2 al4 fe1 al32; do
      for s in davidson-native davidson; do
        [ "$c" = al32 ] && [ "$s" = davidson ] && continue  # 390 s/rep — skip
        for m in 0 30 100; do
          echo "### case=$c solver=$s chord=$m"
          if [ "$m" = 0 ]; then uv run python $L time $c $s 3
          else GRADWAVE_CHORD_TAIL=$m uv run python $L time $c $s 3; fi
        done
      done
    done ;;
  *) echo "unknown phase $phase"; exit 2 ;;
esac
echo "EXIT=$?"
