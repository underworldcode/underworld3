"""Rebuilding a faulted mesh solved by rotated free slip and the custom FMG does not
grow memory, in parallel.

#882's rebuild test (test_0078) covers a plain Stokes solve with Dirichlet walls. This
test is its fixture on the path the zg95 fault runs take, which test_0078 does not
reach. Each rebuild cuts a new fault into one refined base, as a driver's rebuild does,
and solves with rotated free slip on every wall, the fault's own constraint, and the
custom FMG on the child's coarse levels with the FAC patch keyed on the fault's cells.

Written for #948. Those runs grew about 190 MB per rank at every fault rebuild, but the
cause was outside the library: a monitoring hook kept every rebuild's solver in a
module-level list. On this path the library keeps nothing per rebuild, at any rank
count measured: in CI's environment, 1.0 MB per rebuild at np 2 and 0.7 MB at np 4.
The test guards against a library leak appearing here.

In serial, PETSc's object table balances on this path at any rebuild count. A parallel
leak would live where petsc4py defers destruction, so the measure is the resident set,
which tracks freed memory on Linux only, as in test_0078.

Run with at least two ranks.
"""
import gc
import sys

import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import custom_mg, memprobe

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1200),
              pytest.mark.skipif(sys.platform != "linux",
                                 reason="the resident set tracks freed memory on Linux only")]

WARMUP, MEASURED = 2, 5
BOUND_MB = 10.0              # per rebuild, as test_0078


def _resident_mb():
    return memprobe.snapshot()["rss_mb"]


def _rebuild_and_drop(base, k, centre):
    half = np.array([0.20, 0.05])
    trace = np.vstack([np.array([centre, 0.5]) - half, np.array([centre, 0.5]) + half])
    mesh = base.add_fault([("Flt", trace)])
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)])
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.add_fault_bc(0, "Flt")
    custom_mg.set_custom_fmg(stokes, mesh._custom_mg_coarse_meshes, field_id=0,
                             fac_zone=mesh.cells_supporting("Flt"))
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    del stokes, mesh, info, x, y
    gc.collect()


def test_rebuilding_a_rotated_fmg_faulted_mesh_does_not_grow_memory():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05,
        refinement=1, qdegree=2)
    centres = 0.35 + 0.3 * np.arange(WARMUP + MEASURED + 1) / (WARMUP + MEASURED)
    for k in range(WARMUP):
        _rebuild_and_drop(base, k, centres[k])
    start = _resident_mb()
    for k in range(WARMUP, WARMUP + MEASURED):
        _rebuild_and_drop(base, k, centres[k])
    # one more mesh: its construction frees what the last rebuild dropped
    uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                      cellSize=0.5)
    growth = (_resident_mb() - start) / MEASURED
    worst = MPI.COMM_WORLD.allreduce(growth, op=MPI.MAX)
    uw.pprint(f"REBUILD GROWTH worst rank {worst:.2f} MB per rebuild")
    assert worst < BOUND_MB, f"{worst:.1f} MB per rebuild on the worst rank"
