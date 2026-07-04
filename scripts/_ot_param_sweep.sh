#!/bin/bash
# Sequential parameter sweep on the validated ot-reset baseline.
# Each run: mode-1, Ra=1e7, Δη=1e2, 100 steps from perturbation IC.

set -e
cd /Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/winslow-mesh-smoother
export PIXI_PROJECT_MANIFEST="$(pwd)/pixi.toml"

LOG_ROOT="$HOME/+Simulations/StagnantLid"

run_case() {
    local tag="$1"
    local ref="$2"
    local coar="$3"
    local mc="$4"
    local logfile="$LOG_ROOT/ot_sweep_${tag}_baseline.log"
    echo "=== START sweep case: $tag (R=$ref, coar=$coar, mc=$mc) ===" \
        | tee -a "$LOG_ROOT/ot_sweep_master.log"
    date | tee -a "$LOG_ROOT/ot_sweep_master.log"
    pixi run -e amr-dev python -u scripts/stagnant_lid_adapt_loop.py \
        --from-perturbation --Ra 1e7 --delta-eta 1e2 --pert-mode 1 \
        --n-steps 100 --snapshot-every 10 \
        --adapt-method ot-reset --refinement "$ref" \
        --coarsening "$coar" --metric-choice "$mc" --dt-mult 3.0 \
        --out-tag "ot_sweep_${tag}" \
        2>&1 | tee "$logfile" \
        | grep -E '^[[:space:]]+[0-9]+[[:space:]]+0\.|ABORT|overshoot|done;' \
        | tee -a "$LOG_ROOT/ot_sweep_master.log"
    echo "=== END sweep case: $tag ===" \
        | tee -a "$LOG_ROOT/ot_sweep_master.log"
    date | tee -a "$LOG_ROOT/ot_sweep_master.log"
}

run_case "R1.5_coar1.0_ff"   1.5 1.0  front-following
run_case "R5.0_coar1.0_ff"   5.0 1.0  front-following
run_case "R3.0_coarauto_ff"  3.0 auto front-following
run_case "R3.0_coar1.0_grad" 3.0 1.0  gradient-uniform

echo "=== SWEEP COMPLETE ===" | tee -a "$LOG_ROOT/ot_sweep_master.log"
date | tee -a "$LOG_ROOT/ot_sweep_master.log"
