#!/usr/bin/env bash
# A/B battery: fp64 baseline vs fp64-certified fp32-expansion Davidson
# (GRADWAVE_FP32_EXPANSION). GPU-targeted — run on asus via the queue:
#
#   ./scripts/gwq --host asus bench ab_fp32_expansion cuda 8
#
# or directly:  bash benchmarks/ab_fp32_expansion.sh cuda 8
#
# Pending (2026-08-19): not yet measured on a GPU. CPU numbers only show
# correctness overhead — complex64 FFTs on CPU are not the win the RTX-class
# fp64 penalty (~1/64 fp32 rate) makes them on CUDA.

set -euo pipefail
cd "$(dirname "$0")/.."

dev=${1:-cuda}
threads=${2:-8}

# Per-solve A/B with the apply-count split (bounds the possible speedup)
uv run python benchmarks/bench_fp32_expansion.py "$dev" "$threads" ab

# Whole-SCF wall on the standard bench, env-toggled (no code changes)
for mode in off on; do
  echo "== bench_scf GRADWAVE_FP32_EXPANSION=$mode =="
  GRADWAVE_FP32_EXPANSION=$mode uv run python benchmarks/bench_scf.py "$dev" "$threads" nosym
done
