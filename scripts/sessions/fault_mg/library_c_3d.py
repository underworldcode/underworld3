"""Split-level fault multigrid in 3-D through the library: a planar fault patch in a
box in simple shear, on an add-on mesh of k levels.

python library_c_3d.py <A|C> <k>
"""
import json
import os
import sys
import time

import numpy as np
import underworld3 as uw

strategy, k = sys.argv[1], int(sys.argv[2])
h_base = 0.075
T = {}


def tic(name):
    T["_t0"], T["_n"] = time.perf_counter(), name


def toc():
    T[T["_n"]] = T.get(T["_n"], 0.0) + time.perf_counter() - T["_t0"]


base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1),
                                         cellSize=2 * h_base, refinement=1, qdegree=2)
# the patch: x = 0.5, y and z in [0.3, 0.7], a 7 x 7 grid of points (five cells
# across at the base spacing of 0.075; the split needs about four, and the
# placement's cavity must clear the walls). Each quad takes the
# diagonal that leaves no triangle with all three vertices on the rim.
n = 7
yy, zz = np.meshgrid(np.linspace(0.3, 0.7, n), np.linspace(0.3, 0.7, n), indexing="ij")
pts = np.column_stack([np.full(n * n, 0.5), yy.ravel(), zz.ravel()])
rim = lambda i, j: i in (0, n - 1) or j in (0, n - 1)
tris = []
for i in range(n - 1):
    for j in range(n - 1):
        a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
        ra, rb, rc, rd = rim(i, j), rim(i + 1, j), rim(i + 1, j + 1), rim(i, j + 1)
        if (ra and rb and rc) or (ra and rc and rd):
            tris += [[a, b, d], [b, c, d]]
        else:
            tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)


def metric(X):
    d_plane = np.abs(X[:, 0] - 0.5)
    d_rect = np.maximum(0.0, np.maximum(np.abs(X[:, 1] - 0.5) - 0.2,
                                        np.abs(X[:, 2] - 0.5) - 0.2))
    d = np.sqrt(d_plane**2 + d_rect**2)
    h_near, h_far, width = h_base / 2**k, h_base, 0.15
    h = np.minimum(h_far, h_near + (h_far - h_near) * d / width)
    return 1.0 / h**2


tic("adapt")
child = base.adapt(metric, max_levels=k)
toc()
tic("add_fault")
finest = child.add_fault([("Flt", pts, tris)], split_levels=(strategy == "C"), clearance=0.9)
toc()
tail = finest._custom_mg_coarse_meshes
nsplit = sum(1 for m in tail if getattr(m, "_fault_point_pairs", None))
x, y, z = finest.X
s = uw.systems.Stokes(finest)
s.constitutive_model = uw.constitutive_models.ViscousFlowModel
s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
s.tolerance = 1e-6
s.petsc_use_pressure_nullspace = True
for side in ("Top", "Bottom", "Left", "Right", "Front", "Back"):
    s.add_dirichlet_bc((y - 0.5, 0.0, 0.0), side)
s.add_fault_bc(0, "Flt")
tic("solver_build")
s._build()
toc()
tic("solve")
s.solve()
toc()
info = s._rotated_freeslip_info
from underworld3.utilities.fault_contact import fault_pair_jumps
_c, jumps, normals = fault_pair_jumps(s, "Flt", info, gather=True)
tangential = jumps - np.einsum("ij,ij->i", jumps, normals)[:, None] * normals
peak = float(np.linalg.norm(tangential, axis=1).max()) if len(jumps) else 0.0
rec = {"strategy": strategy, "k": k, "np": uw.mpi.size,
       "cells": int(uw.mpi.comm.allreduce(finest.dm.getHeightStratum(0)[1])),
       "levels": len(tail) + 1, "split_levels": nsplit, "pc": info["velocity_pc"],
       "vel_its": info["vel_its_last"], "pres_its": info["pres_its_last"],
       "conv": bool(info["converged"]), "peak_slip": peak,
       "times": {kk: round(v, 2) for kk, v in T.items() if not kk.startswith("_")}}
uw.pprint("STEP " + json.dumps(rec))
out = os.path.expanduser(
    f"~/+Simulations/fault_split_mg/lib3d_{strategy}_k{k}_np{uw.mpi.size}.json")
if uw.mpi.rank == 0:
    json.dump(rec, open(out, "w"), indent=1)
