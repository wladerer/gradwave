#!/usr/bin/env bash
# Large-N GPU-resident ladder. FULLY SERIALIZED behind the band-parallel agent:
# no arm starts until /tmp/BENCH_LOCK.bandpar is gone, all locks are clear, and
# 1-min load < 0.6. CPU baselines are cited from /tmp/bandpar.log (not re-run).
set -u
cd ~/gradwave-fftemu
P=experiments/gpu_resident_scf/profile_scf.py
LOG=/tmp/fftemu_ladder.log
load_ok() { awk "{exit !(\$1 < 0.6)}" /proc/loadavg; }
wait_clear() {
  while ls /tmp/BENCH_LOCK.* >/dev/null 2>&1 || ! load_ok; do sleep 120; done
  touch /tmp/BENCH_LOCK.fftemu
}
release() { rm -f /tmp/BENCH_LOCK.fftemu; }
trap release EXIT
echo "ladder started $(date), waiting for locks to clear" >> $LOG
run_arm() {
  local name=$1; shift
  wait_clear
  echo "=== $name [$(date +%H:%M:%S)] ===" >> $LOG
  "$@" >> $LOG 2>&1
  echo "--- arm exit=$? [$(date +%H:%M:%S)] ---" >> $LOG
  release
}
GPUENV="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=2 GRADWAVE_GPU_DENSE_BUDGET=2e8"
run_arm "al32 gpu time (MDF=2 KCHUNK=2 DB=2e8)" $GPUENV uv run python $P al32 cuda time 8
run_arm "si64 gpu time (MDF=2 KCHUNK=2 DB=2e8)" $GPUENV uv run python $P si64 cuda time 8
run_arm "al32 gpu profile3" $GPUENV uv run python $P al32 cuda profile3 8
run_arm "si64 gpu profile3" $GPUENV uv run python $P si64 cuda profile3 8 # patched projectors_b active
echo "LADDER-EXIT=0" >> $LOG
