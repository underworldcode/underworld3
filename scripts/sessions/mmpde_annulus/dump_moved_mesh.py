"""Move an annulus with a sharp thermal layer once with the default mover and save
the triangles, vertex coordinates and T at the vertices, for rendering."""
import sys
import time

import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import graph

out, move = sys.argv[1], int(sys.argv[2])
cell = float(sys.argv[3]) if len(sys.argv) > 3 else 1 / 12
refinement = int(sys.argv[4]) if len(sys.argv) > 4 else None
mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=cell, refinement=refinement, qdegree=3)
x, y = mesh.X
r = sympy.sqrt(x ** 2 + y ** 2)
field = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * sympy.atan2(y, x)))
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
seconds = 0.0
if move:
    density = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=5, coarsening="auto", metric_choice="front-following")
    start = time.time()
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   verbose=True)
    seconds = time.time() - start
xy = np.array(mesh.dm.getCoordinatesLocal().array).reshape(-1, 2)
cells = graph._tri_cells(mesh.dm)
T_vertices = uw.function.evaluate(field, xy).reshape(-1)
np.savez(out, xy=xy, cells=cells, T=T_vertices, seconds=seconds)
print("saved", out, "cells", len(cells), "seconds", round(seconds, 1))
