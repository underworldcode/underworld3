"""Where the work sits in parallel: per-rank cells on every level of the split-level
hierarchy, the fault star's share, the transfer's rank-local hit rate, and the
time of each stage, at any rank count.

mpirun -np N python parallel_load.py <k> [positions=1] [base_cell=0.1]
"""
import json
import os
import sys
import time

import numpy as np
import underworld3 as uw
from mpi4py import MPI

k = int(sys.argv[1])
positions = int(sys.argv[2]) if len(sys.argv) > 2 else 1
base_cell = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
comm = uw.mpi.comm
HALF = np.array([0.20, 0.05])
h_base = base_cell / 2


def owned_cells(mesh):
    dm = mesh.dm
    cS, cE = dm.getHeightStratum(0)
    try:
        _n, ileaf, _r = dm.getPointSF().getGraph()
        ghost = set(int(q) for q in ileaf if cS <= q < cE)
    except (ValueError, TypeError):
        ghost = set()
    return cE - cS - len(ghost)


def fault_star_cells(mesh, name):
    """cells incident to a vertex of the fault's Plus or Minus facets (local)."""
    dm = mesh.dm
    cS, cE = dm.getHeightStratum(0)
    fS, fE = dm.getHeightStratum(1)
    vS, vE = dm.getDepthStratum(0)
    verts = set()
    for side in ("Plus", "Minus"):
        lab = dm.getLabel(f"{name}{side}")
        value = mesh.boundaries[f"{name}{side}"].value
        if lab and lab.getStratumSize(value) > 0:
            for f in lab.getStratumIS(value).getIndices():
                if fS <= int(f) < fE:
                    verts.update(int(q) for q in dm.getTransitiveClosure(int(f))[0]
                                 if vS <= int(q) < vE)
    cells = set()
    for v in verts:
        cells.update(int(q) for q in dm.getTransitiveClosure(v, useCone=False)[0]
                     if cS <= int(q) < cE)
    return len(cells)


base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                         cellSize=base_cell, refinement=1, qdegree=2)
records = []
centres = np.linspace(0.3, 0.7, positions) if positions > 1 else [0.5]
for step, cx in enumerate(centres):
    T = {}
    A_ = np.array([cx, 0.5]) - HALF
    B_ = np.array([cx, 0.5]) + HALF
    trace = np.vstack([A_, B_])
    t0 = time.perf_counter()
    pts3 = np.column_stack([np.linspace(A_[0], B_[0], 9), np.linspace(A_[1], B_[1], 9),
                            np.zeros(9)])
    surf = uw.meshing.Surface(f"g{step}", base, pts3)
    surf.discretize()
    child = base.adapt(surf.refinement_metric_function(h_near=h_base / 2**k, h_far=h_base,
                                                       width=0.08, profile="linear"),
                       max_levels=k)
    T["adapt"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    unsplit_finest = child.add_fault([("Flt", trace)], split_levels=False)
    T["split_finest_only"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    finest = child.add_fault([("Flt", trace)])
    T["split_all_levels"] = time.perf_counter() - t0
    tail = finest._custom_mg_coarse_meshes
    levels = list(tail) + [finest]
    load = []
    for li, m in enumerate(levels):
        own = comm.allgather(owned_cells(m))
        split = bool(getattr(m, "_fault_point_pairs", None))
        star = comm.allgather(fault_star_cells(m, "Flt") if split else 0)
        load.append({"level": li, "split": split, "owned_cells": own, "fault_star": star,
                     "static": bool(getattr(m, "_mg_static", False))})
    # the unsplit adapt child's balance, for comparison
    load_child = comm.allgather(owned_cells(child))

    x, y = finest.X
    s = uw.systems.Stokes(finest)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.tolerance = 1e-6
    s.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), side)
    s.add_fault_bc(0, "Flt")
    t0 = time.perf_counter()
    s._build()
    T["solver_build"] = time.perf_counter() - t0
    from underworld3.utilities import custom_mg, rotated_bc
    _orig_build = custom_mg.CustomMGHierarchy.build
    _orig_pl = rotated_bc._build_rotated_custom_Pl

    def timed_build(self, solver):
        t = time.perf_counter()
        out = _orig_build(self, solver)
        T["transfers"] = T.get("transfers", 0.0) + time.perf_counter() - t
        T["_stats"] = self.level_transfer_stats
        return out

    def timed_pl(solver, Q, normal_rows):
        t = time.perf_counter()
        out = _orig_pl(solver, Q, normal_rows)
        T["prolongations_total"] = T.get("prolongations_total", 0.0) + time.perf_counter() - t
        return out
    custom_mg.CustomMGHierarchy.build = timed_build
    rotated_bc._build_rotated_custom_Pl = timed_pl
    t0 = time.perf_counter()
    s.solve()
    T["solve_total"] = time.perf_counter() - t0
    custom_mg.CustomMGHierarchy.build = _orig_build
    rotated_bc._build_rotated_custom_Pl = _orig_pl
    info = s._rotated_freeslip_info
    stats = T.pop("_stats", None)
    rec = {"np": comm.size, "k": k, "step": step, "cx": float(cx),
           "cells_total": int(sum(load[-1]["owned_cells"])),
           "levels": len(levels), "pc": info["velocity_pc"], "vel_its": info["vel_its_last"],
           "converged": bool(info["converged"]), "load": load, "load_unsplit_child": load_child,
           "transfer_stats": stats,
           "times": {kk: round(v, 2) for kk, v in T.items()}}
    records.append(rec)
    uw.pprint("STEP " + json.dumps({kk: v for kk, v in rec.items() if kk != "load"}))
tag = "" if base_cell == 0.1 else f"_b{base_cell}"
out = os.path.expanduser(f"~/+Simulations/fault_split_mg/load_k{k}{tag}_np{comm.size}.json")
if comm.rank == 0:
    json.dump(records, open(out, "w"), indent=1)
uw.pprint("DONE " + out)
