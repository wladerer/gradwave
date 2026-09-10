#!/usr/bin/env bash
# Consolidated remaining arms, strictly sequential, one lock holder.
set -u
cd ~/gradwave-fftemu
P=experiments/gpu_resident_scf/profile_scf.py
LOG=/tmp/fftemu_master.log
load_ok() { awk "{exit !(\$1 < 0.6)}" /proc/loadavg; }
wait_clear() { while ls /tmp/BENCH_LOCK.* >/dev/null 2>&1 || ! load_ok; do sleep 120; done; touch /tmp/BENCH_LOCK.fftemu; }
release() { rm -f /tmp/BENCH_LOCK.fftemu; }
trap release EXIT
run_arm() { local name=$1; shift; wait_clear; echo "=== $name [$(date +%H:%M:%S)] ===" >> $LOG; "$@" >> $LOG 2>&1; echo "--- arm exit=$? [$(date +%H:%M:%S)] ---" >> $LOG; release; }
GPUL="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=1 GRADWAVE_GPU_DENSE_BUDGET=1e8"
GPUL2="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=2 GRADWAVE_GPU_DENSE_BUDGET=2e8"
run_arm "si64 gpu time RETRY (projectors per-k; K1 DB1e8)" $GPUL uv run python $P si64 cuda time 8
run_arm "al4 gpu MP-draft" env GW_PROBE_MP=1 uv run python $P al4 cuda time 8
run_arm "si8 gpu MP-draft" env GW_PROBE_MP=1 uv run python $P si8 cuda time 8
run_arm "al4 gpu fp32-expansion" env GRADWAVE_FP32_EXPANSION=on uv run python $P al4 cuda time 8
run_arm "si8 gpu fp32-expansion" env GRADWAVE_FP32_EXPANSION=on uv run python $P si8 cuda time 8
run_arm "al4 gpu subspace-c64" env GRADWAVE_SUBSPACE_STORAGE=complex64 uv run python $P al4 cuda time 8
run_arm "si64 gpu MP-draft (K1 DB1e8)" $GPUL GW_PROBE_MP=1 uv run python $P si64 cuda time 8
run_arm "si64 gpu fp32-expansion (K1 DB1e8)" $GPUL GRADWAVE_FP32_EXPANSION=on uv run python $P si64 cuda time 8
run_arm "al32 gpu profile3 cuda-only" $GPUL2 uv run python $P al32 cuda profile3 8
run_arm "si64 gpu profile3 cuda-only (K1 DB1e8)" $GPUL uv run python $P si64 cuda profile3 8
echo "MASTER-EXIT=0" >> $LOG
