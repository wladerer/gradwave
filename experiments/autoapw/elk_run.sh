#!/usr/bin/env bash
# elk_run.sh MAT N
MAT=$1; N=$2
A=~/github/gradwave/.claude/worktrees/efg-k666-reread/experiments/autoapw
L=~/efg_kconv/logs; mkdir -p $L
bash $A/elk_kconv.sh $MAT $N > $L/elk_${MAT}_k${N}.log 2>&1
echo EXIT=$? >> $L/elk_${MAT}_k${N}.log
