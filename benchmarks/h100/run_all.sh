#!/usr/bin/env bash
# run_all.sh -- launch the four benchmark lanes, one pinned GPU each, in
# parallel and fully unattended. Each lane runs under its own timeout and writes
# to its own log ending in a LANE<n>_EXIT=<rc> marker, so a crashing lane never
# takes the others down and progress is greppable. When every lane has finished
# the master log gets a RUN_ALL_EXIT marker.
#
#   bash run_all.sh                # 4 GPUs, one lane each
#   H100_NGPU=2 bash run_all.sh    # fewer GPUs: lanes share by (lane % NGPU)
#
# Lanes:
#   0  bench_scf + minerals, GPU and CPU
#   1  delta-gauge Al/Cu/Si/Ge on the GPU
#   2  A/B matrix of the 3050-era optimizations (QR offload, solver, RR-GEMM)
#   3  Si memory-ceiling ladder + the multi-GPU curiosity probe
set -uo pipefail   # NOT -e: a lane's nonzero exit must not abort the launcher

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HOST="$(hostname)"
RESDIR="$REPO/benchmarks/results/$HOST"
mkdir -p "$RESDIR"

NGPU="${H100_NGPU:-4}"
# whole-lane wall ceilings (seconds); a lane past this is killed by `timeout`.
LANE_TIMEOUT="${H100_LANE_TIMEOUT:-21600}"   # 6 h

echo "[run_all] host=$HOST repo=$REPO ngpu=$NGPU results=$RESDIR"
echo "[run_all] git sha $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

pids=()
for lane in 0 1 2 3; do
    gpu=$(( lane % NGPU ))
    log="$RESDIR/lane${lane}.log"
    echo "[run_all] lane $lane -> GPU $gpu, log $log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        cd "$REPO"
        timeout "$LANE_TIMEOUT" uv run python benchmarks/h100/orchestrator.py --lane "$lane"
        echo "LANE${lane}_EXIT=$?"
    ) > "$log" 2>&1 &
    pids+=("$!")
done

echo "[run_all] launched PIDs: ${pids[*]} -- waiting"
rc_all=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc_all=1
done

echo "[run_all] all lanes finished. per-lane exit markers:"
grep -h "LANE._EXIT" "$RESDIR"/lane*.log 2>/dev/null || echo "  (no markers found)"
echo "RUN_ALL_EXIT=$rc_all"
