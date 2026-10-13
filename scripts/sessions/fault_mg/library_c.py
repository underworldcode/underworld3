"""Split-level fault multigrid through the LIBRARY path (no monkeypatching).

Box in simple shear with a straight frictionless (or viscous / Coulomb) fault on an
add-on mesh of k levels. `mesh.add_fault` splits the add-on levels by default
(strategy C); `split_levels=False` keeps the welded tail (A). Reports the velocity
iterations, the peak slip and the stage timings. Runs in serial or under mpirun.

python library_c.py <A|C> <k> [positions=1] [law=frictionless|viscous|coulomb]
"""
import json
import os
import sys
import time

import numpy as np
import underworld3 as uw

strategy, k = sys.argv[1], int(sys.argv[2])
positions = int(sys.argv[3]) if len(sys.argv) > 3 else 1
law = sys.argv[4] if len(sys.argv) > 4 else "frictionless"
HALF = np.array([0.20, 0.05])
h_base = 0.05
T = {}


def tic(name):
    T["_t0"], T["_n"] = time.perf_counter(), name


def toc():
    T[T["_n"]] = T.get(T["_n"], 0.0) + time.perf_counter() - T["_t0"]


base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                         cellSize=0.1, refinement=1, qdegree=2)


def stokes_on(m):
    x, y = m.X
    s = uw.systems.Stokes(m)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.tolerance = 1e-6
    s.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), side)
    if law == "frictionless":
        s.add_fault_bc(0, "Flt")
    elif law == "viscous":
        s.add_fault_bc(1.0, "Flt")
    else:
        from underworld3.utilities.fault_contact import add_coulomb_fault_bc
        add_coulomb_fault_bc(s, 0.6, "Flt", sigma_n=0.8)
    return s


records = []
for step, cx in enumerate(np.linspace(0.3, 0.7, positions) if positions > 1 else [0.5]):
    A_ = np.array([cx, 0.5]) - HALF
    B_ = np.array([cx, 0.5]) + HALF
    trace = np.vstack([A_, B_])
    tic("adapt")
    pts3 = np.column_stack([np.linspace(A_[0], B_[0], 9), np.linspace(A_[1], B_[1], 9),
                            np.zeros(9)])
    surf = uw.meshing.Surface(f"g{step}", base, pts3)
    surf.discretize()
    child = base.adapt(surf.refinement_metric_function(h_near=h_base / 2**k, h_far=h_base,
                                                       width=0.08, profile="linear"),
                       max_levels=k)
    toc()
    tic("add_fault")
    finest = child.add_fault([("Flt", trace)], split_levels=(strategy == "C"))
    toc()
    tail = finest._custom_mg_coarse_meshes
    nsplit = sum(1 for m in tail if getattr(m, "_fault_point_pairs", None))
    tic("solver_build")
    stokes = stokes_on(finest)
    stokes._build()
    toc()
    tic("solve")
    stokes.solve()
    toc()
    info = stokes._rotated_freeslip_info
    from underworld3.utilities.fault_contact import fault_slip
    s_, V, leak = fault_slip(stokes, "Flt", info)
    peak = float(uw.mpi.comm.allreduce(float(np.abs(V).max()) if len(V) else 0.0, op=max))
    rec = {"strategy": strategy, "k": k, "law": law, "step": step, "cx": float(cx),
           "np": uw.mpi.size,
           "cells": int(uw.mpi.comm.allreduce(finest.dm.getHeightStratum(0)[1])),
           "levels": len(tail) + 1, "split_levels": nsplit, "pc": info["velocity_pc"],
           "vel_its": info["vel_its_last"], "pres_its": info["pres_its_last"],
           "newton": info.get("nonlinear_iterations"),
           "conv": bool(info["converged"]), "peak_slip": peak,
           "times": {kk: round(v, 2) for kk, v in T.items() if not kk.startswith("_")}}
    T.clear()
    records.append(rec)
    uw.pprint("STEP " + json.dumps(rec))
out = os.path.expanduser(
    f"~/+Simulations/fault_split_mg/lib_{strategy}_k{k}_{law}_np{uw.mpi.size}.json")
if uw.mpi.rank == 0:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(records, open(out, "w"), indent=1)
uw.pprint("DONE " + out)
