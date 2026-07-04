#!/bin/bash
cd /Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/winslow-mesh-smoother
export PIXI_PROJECT_MANIFEST=$(pwd)/pixi.toml
pixi run -e amr-dev python scripts/stagnant_lid_adapt_loop.py \
  --from-perturbation \
  --refinement 5.0 \
  --adapt-every 5 \
  --skip-threshold 99 \
  --dt-mult 3.0 \
  --n-steps 500 \
  --max-t 0.04 \
  --snapshot-every 5 \
  --out-tag adapt_loop_movie_ref5
