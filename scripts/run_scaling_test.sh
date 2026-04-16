#!/bin/bash
# Parallel scaling test: runs the Stokes benchmark at 1,2,4,8,12 ranks
# Usage: pixi run -e amr-dev bash scripts/run_scaling_test.sh

SCRIPT="scripts/scaling_benchmark.py"
RANKS="1 2 4 8 12"

echo ""
echo "============================================================"
echo "  OpenMPI Parallel Scaling Test — $(date)"
echo "  Machine: $(uname -m) / $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p)"
echo "  MPI: $(mpirun --version 2>&1 | head -1)"
echo "============================================================"
echo ""

for N in $RANKS; do
    echo ">>> Running with $N rank(s) ..."
    if [ "$N" -eq 1 ]; then
        python "$SCRIPT"
    else
        mpirun --oversubscribe -n "$N" python "$SCRIPT"
    fi
    echo ""
done

echo "All scaling runs complete."
