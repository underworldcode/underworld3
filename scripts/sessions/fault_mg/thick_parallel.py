"""The faulted thick plate (test_1083's model) through the parallel extrusion:
pair count, multigrid levels, velocity iterations and peak slip at any np.

mpirun -np N python thick_parallel.py [k]      (k add-on levels; 0 = plain refinement)
"""
import json
import os
import sys
import time

import numpy as np
import sympy
import underworld3 as uw

k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
L, ETA, C = 0.04, 1, 1
t0 = time.perf_counter()
base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1, qdegree=3)
trace = np.array([[0.25, 0.5], [0.75, 0.5]])
mesh = base
if k:
    pts3 = np.column_stack([np.linspace(0.25, 0.75, 9), np.full(9, 0.5), np.zeros(9)])
    surf = uw.meshing.Surface("g", base, pts3)
    surf.discretize()
    mesh = base.adapt(surf.refinement_metric_function(h_near=0.05 / 2**k, h_far=0.05,
                                                      width=0.08, profile="linear"),
                      max_levels=k)
sheet_mesh = mesh.add_fault([("Flt", trace)])
t_sheet = time.perf_counter() - t0
t0 = time.perf_counter()
layer_mesh = sheet_mesh.extrude(L, layers=2)
t_extrude = time.perf_counter() - t0
n_pairs = int(uw.mpi.comm.allreduce(sum(len(p) for p in layer_mesh._fault_point_pairs.values())))
tail = layer_mesh._custom_mg_coarse_meshes or []
n_split = sum(1 for m in tail if getattr(m, "_fault_point_pairs", None))


def model(mesh, Solver):
    solver = Solver(mesh)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = ETA
    y = mesh.X[1]
    base_velocity = [y - sympy.Rational(1, 2), 0] + [0] * (mesh.cdim - 2)
    solver.add_basal_drag(C, "Base", reference_velocity=base_velocity)
    solver.add_fault_bc(0, "Flt")
    solver.tolerance = 1.0e-5
    return solver


layer = model(layer_mesh, uw.systems.Stokes)
t0 = time.perf_counter()
layer.solve()
t_solve = time.perf_counter() - t0
info = layer._rotated_freeslip_info
from underworld3.utilities.fault_contact import fault_pair_jumps
_c, jumps, normals = fault_pair_jumps(layer, "Flt", info, gather=True)
tangential = jumps - np.einsum("ij,ij->i", jumps, normals)[:, None] * normals
peak = float(np.linalg.norm(tangential, axis=1).max()) if len(jumps) else 0.0
leak = float(np.abs(np.einsum("ij,ij->i", jumps, normals)).max()) if len(jumps) else 0.0
cells = int(uw.mpi.comm.allreduce(layer_mesh.dm.getHeightStratum(0)[1]))
rec = {"np": uw.mpi.size, "k": k, "cells": cells, "pairs": n_pairs, "levels": len(tail) + 1,
       "split_levels": n_split, "pc": info["velocity_pc"], "vel_its": info["vel_its_last"],
       "pres_its": info["pres_its_last"], "conv": bool(info["converged"]), "peak_slip": peak, "leak": leak,
       "t_sheet": round(t_sheet, 2), "t_extrude": round(t_extrude, 2), "t_solve": round(t_solve, 2)}
uw.pprint("STEP " + json.dumps(rec))
out = os.path.expanduser(f"~/+Simulations/fault_split_mg/thick_k{k}_np{uw.mpi.size}.json")
if uw.mpi.rank == 0:
    json.dump(rec, open(out, "w"), indent=1)
