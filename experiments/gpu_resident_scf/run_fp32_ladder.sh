#!/usr/bin/env bash
# fp32-draft probe arms. Chained: starts only after the large-N ladder is done.
set -u
cd ~/gradwave-fftemu
P=experiments/gpu_resident_scf/profile_scf.py
LOG=/tmp/fftemu_fp32.log
while ! grep -q "LADDER-EXIT" /tmp/fftemu_ladder.log 2>/dev/null; do sleep 120; done
load_ok() { awk "{exit !(\$1 < 0.6)}" /proc/loadavg; }
wait_clear() { while ls /tmp/BENCH_LOCK.* >/dev/null 2>&1 || ! load_ok; do sleep 120; done; touch /tmp/BENCH_LOCK.fftemu; }
release() { rm -f /tmp/BENCH_LOCK.fftemu; }
trap release EXIT
echo "fp32 ladder started $(date)" >> $LOG
run_arm() { local name=$1; shift; wait_clear; echo "=== $name [$(date +%H:%M:%S)] ===" >> $LOG; "$@" >> $LOG 2>&1; echo "--- arm exit=$? [$(date +%H:%M:%S)] ---" >> $LOG; release; }
GPUL="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=2 GRADWAVE_GPU_DENSE_BUDGET=2e8"
# small systems: same harness as Phase A (fp64 GPU baselines already measured: al4 5.77s, si8 6.36s)
run_arm "al4 gpu MP-draft" env GW_PROBE_MP=1 uv run python $P al4 cuda time 8
run_arm "si8 gpu MP-draft" env GW_PROBE_MP=1 uv run python $P si8 cuda time 8
run_arm "al4 gpu fp32-expansion" env GRADWAVE_FP32_EXPANSION=on uv run python $P al4 cuda time 8
run_arm "si8 gpu fp32-expansion" env GRADWAVE_FP32_EXPANSION=on uv run python $P si8 cuda time 8
run_arm "al4 gpu subspace-c64" env GRADWAVE_SUBSPACE_STORAGE=complex64 uv run python $P al4 cuda time 8
# large-N: the only regime where the composed gate could pass
run_arm "si64 gpu MP-draft (mem knobs)" $GPUL GW_PROBE_MP=1 uv run python $P si64 cuda time 8
run_arm "si64 gpu fp32-expansion (mem knobs)" $GPUL GRADWAVE_FP32_EXPANSION=on uv run python $P si64 cuda time 8
echo "FP32-LADDER-EXIT=0" >> $LOG
