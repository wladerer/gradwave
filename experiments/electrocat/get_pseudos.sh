#!/usr/bin/env bash
# Reproduce the pseudopotential sets for the electrocatalysis run.
# PAW psl.1.0.0 (production) + ONCV (r2SCAN/ISDF stretch). Run once after clone.
set -euo pipefail
cd "$(dirname "$0")"
REPO=../..
mkdir -p pseudos pseudos_nc

# --- PAW psl.1.0.0 PBE: Pt/C/O from the repo fixtures, Au/H from the QE repo ---
for f in Pt.pbe-n-kjpaw_psl.1.0.0.UPF C.pbe-n-kjpaw_psl.1.0.0.UPF O.pbe-n-kjpaw_psl.1.0.0.UPF; do
  cp "$REPO/tests/fixtures/qe/pseudos/$f" pseudos/
done
for f in Au.pbe-n-kjpaw_psl.1.0.0.UPF H.pbe-kjpaw_psl.1.0.0.UPF; do
  [ -f "pseudos/$f" ] || curl -sfL "https://pseudopotentials.quantum-espresso.org/upf_files/$f" -o "pseudos/$f"
done

# --- ONCV PBE for the meta-GGA (NC-only) stretch ---
cp "$REPO/benchmarks/delta_gauge/pseudos/Pt.upf" pseudos_nc/Pt_ONCV.upf
cp "$REPO/benchmarks/delta_gauge/pseudos/Au.upf" pseudos_nc/Au_ONCV.upf
for f in C_ONCV_PBE-1.2.upf O_ONCV_PBE-1.2.upf H_ONCV_PBE-1.2.upf; do
  cp "$REPO/tests/fixtures/qe/pseudos/$f" pseudos_nc/
done

echo "pseudos ready: $(ls pseudos | wc -l) PAW, $(ls pseudos_nc | wc -l) NC"
