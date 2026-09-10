#!/usr/bin/env bash
set -u
cd ~/gradwave-fftemu
P=experiments/gpu_resident_scf/profile_scf.py
LOG=/tmp/fftemu_final.log
while ! grep -q "SI64-RETRY2-EXIT" /tmp/fftemu_si64retry.log 2>/dev/null; do sleep 60; done
load_ok() { awk "{exit !(\$1 < 0.6)}" /proc/loadavg; }
wait_clear() { while ls /tmp/BENCH_LOCK.* >/dev/null 2>&1 || ! load_ok; do sleep 90; done; touch /tmp/BENCH_LOCK.fftemu; }
release() { rm -f /tmp/BENCH_LOCK.fftemu; }
trap release EXIT
run_arm() { local name=$1; shift; wait_clear; echo "=== $name [$(date +%H:%M:%S)] ===" >> $LOG; "$@" >> $LOG 2>&1; echo "--- arm exit=$? [$(date +%H:%M:%S)] ---" >> $LOG; release; }
G="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=2 GRADWAVE_GPU_DENSE_BUDGET=2e8"
run_arm "al32 gpu fp32-expansion (K2 DB2e8)" $G GRADWAVE_FP32_EXPANSION=on uv run python $P al32 cuda time 8
run_arm "al32 gpu MP-draft (K2 DB2e8)" $G GW_PROBE_MP=1 uv run python $P al32 cuda time 8
echo "FINAL-EXIT=0" >> $LOG
