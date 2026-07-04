#!/bin/bash
cd /Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/winslow-mesh-smoother
OUT=/Users/lmoresi/+Simulations/StagnantLid/adapt_loop_movie_ref5
export PIXI_PROJECT_MANIFEST=$(pwd)/pixi.toml
export SL_MOVIE_OUT=$OUT
prev=$(ls $OUT/step*.mesh.00000.h5 2>/dev/null | wc -l)
echo "watcher started: $prev snapshots"
while true; do
  cur=$(ls $OUT/step*.mesh.00000.h5 2>/dev/null | wc -l)
  if [ "$cur" -gt "$prev" ]; then
    pixi run -e amr-dev python -u scripts/_sl_movie_render_partial.py 2>&1 \
      | grep -E "Rendering|frame" | tail -1
    echo "snapshots: $cur"
    prev=$cur
  fi
  pgrep -f "stagnant_lid_adapt_loop.*movie_ref5" >/dev/null 2>&1 \
    || { echo "sim exited"; break; }
  sleep 30
done
