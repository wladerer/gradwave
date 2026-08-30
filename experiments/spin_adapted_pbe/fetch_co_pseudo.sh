#!/usr/bin/env bash
# Fetch the Co pseudopotential used by the spin-adapted-PBE training set.
# SG15 scalar-relativistic ONCV PBE (not committed — large UPF).
set -euo pipefail
DEST="${1:-tests/fixtures/qe/pseudos}"
URL="http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Co_ONCV_PBE-1.0.upf"
mkdir -p "$DEST"
curl -fsSL -o "$DEST/Co_ONCV_PBE-1.0.upf" "$URL"
echo "fetched Co_ONCV_PBE-1.0.upf -> $DEST"
