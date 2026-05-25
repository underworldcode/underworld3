"""Per-OT-step timing breakdown. Want to know:
  (a) first call vs subsequent (JIT amortization)
  (b) where the dominant cost is — assembly? factor? solve?
  (c) what cellSize / element count we're actually running at
"""
import os
import sys
import time
import numpy as np
import underworld3 as uw

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


t0 = time.time()
m = build_uniform_mesh()
print(f"mesh build:          {time.time()-t0:6.2f} s   "
      f"({m.dm.getDepthStratum(0)[1]-m.dm.getDepthStratum(0)[0]} verts, "
      f"{m.dm.getHeightStratum(0)[1]-m.dm.getHeightStratum(0)[0]} cells)")

t0 = time.time()
rho = analytic_rho(m)
print(f"analytic_rho symb:   {time.time()-t0:6.2f} s")

# First OT call — full setup + JIT compile
t0 = time.time()
uw.meshing.smooth_mesh_interior(
    m, metric=rho, method="ot", verbose=False,
    boundary_slip="box",
    method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3))
print(f"OT call #1 (setup):  {time.time()-t0:6.2f} s")

# Subsequent OT calls — should be cached
for k in range(2, 7):
    t0 = time.time()
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, method="ot", verbose=False,
        boundary_slip="box",
        method_kwargs=dict(n_outer=1, relax=0.1, step_frac=0.3))
    print(f"OT call #{k}:          {time.time()-t0:6.2f} s")
