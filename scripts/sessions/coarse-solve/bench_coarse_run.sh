#!/bin/bash
# usage: bench_coarse_run.sh NP CELL VARIANT  -> one line: BENCH ... | coarse count / max time
. /mnt/data/lmoresi/hyperion-env.sh
B=/mnt/data/lmoresi; W=$B/underworld3/.claude/worktrees/custom-mg-empty-rank
cd $W
tag="np$1-c$2-$3"; log=$B/tmp/bench/$tag.log; lv=$B/tmp/bench/$tag.logview
mkdir -p $B/tmp/bench
timeout 1800 pixi run -e dev mpirun -np $1 python $B/tmp/bench_coarse_solve.py -uw_cell_size $2 -uw_coarse $3 -uw_logview $lv > $log 2>&1
rc=$?
line=$(grep -E "BENCH" $log | sed 's/^\[[0-9]*\] *//' | head -1)
mg=$(grep -E "^MGSmooth Level 0 " $lv 2>/dev/null | awk '{print "coarse_applies " $4 " coarse_t " $6}')
ks=$(grep -E "^KSPSolve " $lv 2>/dev/null | head -1 | awk '{print "ksp_t " $4}')
echo "rc=$rc $line | $mg $ks"
