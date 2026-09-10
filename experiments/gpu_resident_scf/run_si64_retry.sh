#!/usr/bin/env bash
set -u
cd ~/gradwave-fftemu
P=experiments/gpu_resident_scf/profile_scf.py
LOG=/tmp/fftemu_si64retry.log
while ! grep -q "LADDER-EXIT" /tmp/fftemu_ladder.log 2>/dev/null; do sleep 120; done
load_ok() { awk "{exit !(\$1 < 0.6)}" /proc/loadavg; }
wait_clear() { while ls /tmp/BENCH_LOCK.* >/dev/null 2>&1 || ! load_ok; do sleep 120; done; touch /tmp/BENCH_LOCK.fftemu; }
release() { rm -f /tmp/BENCH_LOCK.fftemu; }
trap release EXIT
wait_clear
echo "=== si64 gpu time RETRY (projectors_b per-k patch; MDF=2 KCHUNK=1 DB=1e8) [$(date +%H:%M:%S)] ===" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=1 GRADWAVE_GPU_DENSE_BUDGET=1e8 uv run python $P si64 cuda time 8 >> $LOG 2>&1
echo "--- retry exit=$? [$(date +%H:%M:%S)] ---" >> $LOG
echo "SI64-RETRY-EXIT=0" >> $LOG
