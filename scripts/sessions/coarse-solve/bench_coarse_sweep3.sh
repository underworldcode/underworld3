#!/bin/bash
# Sweep 3: the coarse solve chosen before any setup (no SVD built first).
OUT=/mnt/data/lmoresi/tmp/bench/sweep3.txt; : > $OUT
for cell in 0.2 0.1; do for np in 8 16 32; do for v in svd mumps mumps_nsp; do
  r=$(/mnt/data/lmoresi/bin/bench_coarse_run.sh $np $cell $v)
  lv=/mnt/data/lmoresi/tmp/bench/np$np-c$cell-$v.logview
  setup=$(grep -E "^MGSetup Level 0 " $lv 2>/dev/null | awk '{print "coarse_setup " $6 " x" $4}')
  echo "$r | $setup" >> $OUT
done; done; done
echo SWEEP3 DONE >> $OUT
