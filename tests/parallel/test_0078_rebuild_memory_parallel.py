"""Rebuilding a faulted mesh does not grow memory, in parallel.

In parallel petsc4py does not destroy a PETSc object when Python collects
its wrapper: the ranks may collect in different orders and the destroy is
collective, so the object goes on a deferred list that only a collective
``PETSc.garbage_cleanup()`` empties. Nothing called it, so every mesh a
driver dropped kept its DM hierarchy, matrices and vectors for the rest of
the process even after the Python objects were freed: measured on the
moving-fault fixture at np = 2, 52.7 MB per rebuild with every Python
object freed (2217 DMs alive in PETSc's table after five rebuilds, 98 in
serial). A new mesh now empties the list (``Mesh.__init__`` is collective).

Run with at least two ranks.
"""
import gc
import sys

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import memprobe

# The resident set measures the leak only where freed memory leaves it. On
# macOS it does not: with the fix, PETSc's object table is the same after 4
# and after 8 rebuilds (284 DMs, 221 Mats, 298 Vecs on rank 0 at np 2) while
# the resident set still rises 14-26 MB per rebuild (2026-10-28).
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900),
              pytest.mark.skipif(sys.platform != "linux",
                                 reason="the resident set tracks freed memory on Linux only")]

WARMUP, MEASURED = 2, 5
BOUND_MB = 10.0              # per rebuild; the leak was ~50 at this size


def _resident_mb():
    return memprobe.snapshot()["rss_mb"]


def _rebuild_and_drop(base, k, centre):
    half = np.array([0.20, 0.05])
    tip_a, tip_b = np.array([centre, 0.5]) - half, np.array([centre, 0.5]) + half
    samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                               np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface(f"guide{k}", base, samples)
    guide.discretize()
    metric = guide.refinement_metric_function(
        h_near=0.0125, h_far=0.05, width=0.08, profile="linear")
    child = base.adapt(metric, max_levels=2)
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])
    _x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()
    del stokes, mesh, child, guide, metric, y
    gc.collect()


def test_rebuilding_a_faulted_mesh_does_not_grow_memory():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
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
    assert worst < BOUND_MB, f"{worst:.1f} MB per rebuild on the worst rank"
