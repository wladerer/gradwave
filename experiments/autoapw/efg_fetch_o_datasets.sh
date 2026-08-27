#!/usr/bin/env bash
# Fetch alternative O PAW datasets for the Front A EFG-eta dataset scan (efg_eta_paw_datasets.py).
# UPFs are NOT committed; this downloads them into $DEST (default /tmp/efg_pseudos).
#
# Only psl-style kjpaw PAW parses in gradwave (pseudo.upf_paw: q_with_l, nqf=0, scalar/no rel).
# All O PAW generations ship the same 4 projectors (2s x2, 2p x2) -> this varies the GENERATION,
# not the projector count. The rrkjus USPP is fetched only to demonstrate that a non-PAW dataset
# carries no AE/PS partial waves and so cannot feed the Petrilli-Blochl on-site term.
set -euo pipefail
DEST="${DEST:-/tmp/efg_pseudos}"
BASE="https://pseudopotentials.quantum-espresso.org/upf_files"
mkdir -p "$DEST"
for u in \
  O.pbe-n-kjpaw_psl.1.0.0.UPF \
  O.pbe-n-kjpaw_psl.0.1.UPF \
  O.pbe-kjpaw.UPF \
  O.pbe-n-rrkjus_psl.1.0.0.UPF ; do
  echo "fetch $u"
  curl -fsSL -o "$DEST/$u" "$BASE/$u"
done
echo "done -> $DEST"
ls -la "$DEST"
