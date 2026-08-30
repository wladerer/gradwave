#!/usr/bin/env bash
# Fetch the pseudopotentials the FSM campaign needs beyond the committed set.
# Not committed (repo convention for fetched training pseudos — see
# ../fetch_co_pseudo.sh):
#   Co: SG15 scalar-relativistic ONCV PBE (same file #408 trained on)
#   Mn: PseudoDojo NC-SR v0.4 PBE standard (for Co2MnSi)
set -euo pipefail
DEST="${1:-tests/fixtures/qe/pseudos}"
mkdir -p "$DEST"
if [ ! -f "$DEST/Co_ONCV_PBE-1.0.upf" ]; then
  curl -fsSL -o "$DEST/Co_ONCV_PBE-1.0.upf" \
    "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Co_ONCV_PBE-1.0.upf"
  echo "fetched Co_ONCV_PBE-1.0.upf -> $DEST"
fi
if [ ! -f "$DEST/PD_Mn_PBE.upf" ]; then
  curl -fsSL -o /tmp/Mn.upf.gz \
    "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard/Mn.upf.gz"
  gunzip -c /tmp/Mn.upf.gz > "$DEST/PD_Mn_PBE.upf"
  rm -f /tmp/Mn.upf.gz
  echo "fetched PD_Mn_PBE.upf (PseudoDojo nc-sr-04 standard) -> $DEST"
fi
echo "pseudos ready in $DEST"
