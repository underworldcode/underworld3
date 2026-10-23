#!/bin/bash
OUT=/mnt/data/lmoresi/tmp/bench/sweep.txt; : > $OUT
for cell in 0.2 0.1; do for np in 8 16 32; do for v in svd svd_nsp mumps redundant_mumps lu_shift; do
  /mnt/data/lmoresi/bin/bench_coarse_run.sh $np $cell $v >> $OUT
done; done; done
echo SWEEP DONE >> $OUT
