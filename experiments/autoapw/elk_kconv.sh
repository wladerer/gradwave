#!/usr/bin/env bash
# elk_kconv.sh MAT n  — build Elk dir at k=n, run elk (tasks 0,115), print on-site decomposition.
set -e
MAT=$1; N=$2
W=~/github/gradwave/.claude/worktrees/efg-k666-reread
cd $W/experiments/autoapw
case $MAT in
  rutile)   SPEC=Ti:2,O:4; IAS=0,2;;
  anatase)  SPEC=Ti:2,O:4; IAS=0,2;;
  corundum) SPEC=Al:4,O:6; IAS=0,4;;
  mgf2)     SPEC=Mg:2,F:4; IAS=0,2;;
  hbn)      SPEC=B:2,N:2;  IAS=0,2;;
  li3n)     SPEC=Li:3,N:1; IAS=0,1,3;;
  *) echo unknown MAT $MAT; exit 1;;
esac
MAT=$MAT NGRIDK=$N uv run --project $W python setup_elk_kconv.py
DIR=~/efg_kconv/${MAT}_k${N}_elk
cd $DIR
export OMP_NUM_THREADS=${OMP:-8}
t0=$(date +%s)
~/github/elk-11.0.2/src/elk > elk.log 2>&1
echo "[elk $MAT k$N done $(( $(date +%s)-t0 ))s]"
cd $W/experiments/autoapw
echo "=== ELK ONSITE $MAT k$N ==="
uv run --project $W python elk_onsite.py $DIR/STATE.OUT $SPEC $IAS
echo "--- EFG.OUT tail ---"
tail -25 $DIR/EFG.OUT 2>/dev/null || echo no EFG.OUT
