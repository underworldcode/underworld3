"""Memory over repeated faulted-mesh rebuilds, with steps between them.

The box in simple shear with a straight frictionless fault of
fault-split-multigrid.md (the ``library_c.py`` route). Each rebuild moves the
fault, rebuilds the add-on part from the one background (``adapt``), cuts and
splits every level (``add_fault``) and builds a Stokes solver; ``-uw_steps``
solves follow, each with the viscosity changed slightly (a step). Then every
reference is dropped and the garbage collected, as a production driver does.
After every solve the process's resident set and the count of live meshes,
mesh variables and PETSc objects (after a collection) are written, one JSON
line each, so growth per step and growth per rebuild can be told apart.

    python probe_rebuild_memory.py -uw_rebuilds 20 -uw_steps 3
    mpirun -np 2 python probe_rebuild_memory.py -uw_rebuilds 20 -uw_steps 3
"""
import gc
import json
import os

import numpy as np
from petsc4py import PETSc

import underworld3 as uw

params = uw.Params(
    uw_rebuilds=20,          # faulted meshes built in this process
    uw_steps=3,              # solves on each mesh
    uw_depth=3,              # add-on levels k
    uw_log="rebuild_memory.jsonl",
    uw_logview="",           # write PETSc's object table here at the end
    uw_petsc_cleanup=0,      # free PETSc's deferred garbage after each rebuild
)

HALF = np.array([0.20, 0.05])                 # half the fault, centre to tip
PETSC_TYPES = {"DM": PETSc.DM, "Mat": PETSc.Mat, "Vec": PETSc.Vec,
               "KSP": PETSc.KSP, "SNES": PETSc.SNES, "IS": PETSc.IS,
               "SF": PETSc.SF}


def resident_mb():
    """This process's resident set, MB (Linux /proc)."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 20


def live_objects():
    """Counts of live meshes, variables and PETSc objects after collection."""
    gc.collect()
    objects = gc.get_objects()
    counts = {"Mesh": sum(isinstance(o, uw.discretisation.Mesh) for o in objects),
              "MeshVariable": sum(isinstance(o, uw.discretisation.MeshVariable)
                                  for o in objects)}
    for name, kind in PETSC_TYPES.items():
        counts[name] = sum(isinstance(o, kind) for o in objects)
    return counts


if str(params.uw_logview):
    PETSc.Log.begin()

base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
    refinement=1, qdegree=2)
h_base = 0.05
log = []

for rebuild in range(int(params.uw_rebuilds)):
    centre = 0.35 + 0.3 * (rebuild % 7) / 6          # the fault moves
    tip_a = np.array([centre, 0.5]) - HALF
    tip_b = np.array([centre, 0.5]) + HALF
    samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                               np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface(f"guide{rebuild}", base, samples)
    guide.discretize()
    metric = guide.refinement_metric_function(
        h_near=h_base / 2 ** int(params.uw_depth), h_far=h_base, width=0.08,
        profile="linear")
    child = base.adapt(metric, max_levels=int(params.uw_depth))
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])

    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    viscosity = uw.function.expression(r"\eta", 1.0, "viscosity")
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, "Flt")

    for step in range(int(params.uw_steps)):
        viscosity.sym = 1.0 + 0.01 * step              # a step: new coefficient
        stokes.solve(zero_init_guess=(step == 0))
        record = {"rebuild": rebuild, "step": step, "np": uw.mpi.size,
                  "rss_mb": round(resident_mb(), 1), **live_objects()}
        log.append(record)
        uw.pprint("MEM " + json.dumps(record))

    # drop everything this rebuild made, as a driver does
    del stokes, viscosity, mesh, child, guide, metric, x, y
    gc.collect()
    if int(params.uw_petsc_cleanup):
        # in parallel petsc4py defers destroying collected objects until a
        # collective cleanup; without one they are never freed
        PETSc.garbage_cleanup()

with uw.selective_ranks(0) as writer:
    if writer:
        with open(params.uw_log, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in log)

if str(params.uw_logview):
    gc.collect()
    viewer = PETSc.Viewer().createASCII(str(params.uw_logview),
                                        comm=PETSc.COMM_WORLD)
    PETSc.Log.view(viewer)
    viewer.destroy()
