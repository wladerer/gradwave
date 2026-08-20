#!/usr/bin/env bash
# A/B battery for the two solver-path speedups shipped from the physics-blind
# PW roadmap: CholQR2 orthonormalization (GRADWAVE_CHOLQR) and the measured
# Toeplitz local-apply auto-gate (GRADWAVE_TOEPLITZ). Runs the bench_matrix
# battery once per configuration; compare wall/iteration columns across the
# "===" sections. Iteration-count parity on the metals (al, cu) is the kill
# signal — a +1 outer iteration there means a path is perturbing the SCF.
#
# Usage: [CASES="si2 al ..."] [THREADS=8] benchmarks/ab_speedups.sh [device]
set -u
cd "$(dirname "$0")/.."
DEV="${1:-cpu}"
THREADS="${THREADS:-8}"
CASES="${CASES:-si2 c2 gaas al cu mgo si8 si64}"
export OMP_NUM_THREADS="$THREADS"
for cfg in "off off" "on off" "off auto" "on auto"; do
  read -r chol toep <<<"$cfg"
  export GRADWAVE_CHOLQR="$chol" GRADWAVE_TOEPLITZ="$toep"
  echo "=== CHOLQR=$chol TOEPLITZ=$toep ==="
  # shellcheck disable=SC2086
  uv run python benchmarks/bench_matrix.py "$DEV" "$THREADS" $CASES
done
echo "EXIT=$?"
