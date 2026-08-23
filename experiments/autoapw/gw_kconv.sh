#!/usr/bin/env bash
# gw_kconv.sh MAT N [HELO]
MAT=$1; N=$2; HELO=$3
W=~/github/gradwave/.claude/worktrees/efg-k666-reread
A=$W/experiments/autoapw
L=~/efg_kconv/logs; mkdir -p $L
tag=${MAT}_k${N}
cd $A
if [ -n "$HELO" ]; then export HELO=$HELO; fi
env OMP_NUM_THREADS=2 MAT=$MAT KMESH=$N KWORKERS=4 uv run --project $W python kconv_efg.py > $L/gw_${tag}.log 2>&1
echo EXIT=$? >> $L/gw_${tag}.log
