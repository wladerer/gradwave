#!/usr/bin/env bash
# Run the QE cross-check inputs and collect free energies.
# Usage: run_qe.sh [case ...]   (default: all inputs present in qe_inputs/)
set -u
cd "$(dirname "$0")/qe_inputs"
sel=("$@")
[ ${#sel[@]} -eq 0 ] && sel=($(ls *.in | sed 's/[0-9]*\.in//' | sort -u))
for case in "${sel[@]}"; do
  for f in "$case"[0-9].in; do
    out="${f%.in}.out"
    [ -s "$out" ] && grep -q "!" "$out" && { echo "skip $out"; continue; }
    echo "running $f"
    pw.x -in "$f" > "$out" 2>&1
  done
done
grep -H "^!" *.out | sort > ../results/eos_qe_energies.txt
echo "energies -> results/eos_qe_energies.txt"
