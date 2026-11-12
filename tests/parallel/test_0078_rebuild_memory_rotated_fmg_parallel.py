"""Rebuilding a faulted mesh solved by rotated free slip and the custom FMG does not
grow memory, in parallel.

#882's rebuild test (test_0078) covers a plain Stokes solve with Dirichlet walls. This
test is its fixture on the path the zg95 fault runs take, which test_0078 does not
reach. Each rebuild cuts a new fault into one refined base, as a driver's rebuild does,
and solves with rotated free slip on every wall, the fault's own constraint, and the
custom FMG on the child's coarse levels with the FAC patch keyed on the fault's cells.

Written for #948. Those runs grew about 190 MB per rank at every fault rebuild, but the
cause was outside the library: a monitoring hook kept every rebuild's solver in a
module-level list. On this path the library keeps no mesh or solver per rebuild, at any rank
count measured: in CI's environment on kaiju (2026-11-06), 1.0 MB per rebuild at np 2
and 0.7 MB at np 4, on a two-level fixture.
The test guards against a library leak appearing here.

Two checks share one run of rebuilds:
- every measured rebuild's mesh and solver are gone once dropped (weak references).
  A Python-held pin, #948's kind, fails this on any platform, at any size;
- the resident set grows by less than 10 MB per rebuild on the worst rank, Linux only
  (macOS keeps freed memory in it). This sees a PETSc-side leak too, where petsc4py
  defers destruction in parallel.
Measured at np 2 in CI's environment on hyperion (2026-11-06): 2.61 MB per rebuild
and no object alive; with every Stokes solver held in a module-level list, as the
hook did, 49.4 MB per rebuild and 20 objects alive, and both tests fail.
Before each reading of the resident set a mesh is built, which empties petsc4py's
deferred-destruction list, so both readings see the same state.

Run with at least two ranks.
"""
import gc
import sys
import weakref

import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import custom_mg, memprobe

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(600)]

WARMUP, MEASURED = 2, 5
BOUND_MB = 10.0              # per rebuild: 2.6 measured, 49 with a pinned solver (np 2)


def _resident_mb():
    return memprobe.snapshot()["rss_mb"]


def _free_deferred():
    """Build a mesh: its construction empties petsc4py's deferred-destruction list."""
    uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)


def _rebuild_and_drop(base, centre):
    """One rebuild: a new fault cut into ``base`` and a solve on it. Returns weak
    references to the rebuild's mesh and solver."""
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
    refs = (weakref.ref(mesh), weakref.ref(stokes))
    del stokes, mesh, info, x, y
    gc.collect()
    return refs


@pytest.fixture(scope="module")
def rebuilds():
    """Two warm-up and five measured rebuilds on one base with three levels: the
    weak references of the measured ones and the worst rank's growth per rebuild."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
        refinement=2, qdegree=2)
    centres = 0.35 + 0.3 * np.arange(WARMUP + MEASURED) / (WARMUP + MEASURED)
    for k in range(WARMUP):
        _rebuild_and_drop(base, centres[k])
    _free_deferred()
    start = _resident_mb()
    refs = [_rebuild_and_drop(base, centres[k]) for k in range(WARMUP, WARMUP + MEASURED)]
    _free_deferred()
    gc.collect()
    growth = (_resident_mb() - start) / MEASURED
    worst = MPI.COMM_WORLD.allreduce(growth, op=MPI.MAX)
    alive = MPI.COMM_WORLD.allreduce(sum(r() is not None for pair in refs for r in pair))
    uw.pprint(f"REBUILD GROWTH worst rank {worst:.2f} MB per rebuild; {alive} objects alive")
    return alive, worst


@pytest.mark.tier_b
def test_no_rebuild_keeps_its_mesh_or_solver_alive(rebuilds):
    alive, _ = rebuilds
    assert alive == 0, f"{alive} meshes or solvers of dropped rebuilds are still alive"


@pytest.mark.tier_b
@pytest.mark.skipif(sys.platform != "linux",
                    reason="the resident set tracks freed memory on Linux only")
def test_rebuilding_does_not_grow_the_resident_set(rebuilds):
    _, worst = rebuilds
    assert worst < BOUND_MB, f"{worst:.1f} MB per rebuild on the worst rank"
