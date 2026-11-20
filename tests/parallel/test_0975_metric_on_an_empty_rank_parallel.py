"""An isotropic metric made on a mesh with a rank that holds no cells leaves
the ranks in step (#975).

When the indicator is not the mesh's first field, PETSc's
``DMPlexMetricCreateIsotropic`` projects it onto the metric's P1 space
(``DMProjectFieldLocal`` on the indicator's DM). The projection needs the
DM's coordinate field, which PETSc creates lazily, the first time a cell's
geometry is needed, so only on ranks holding cells. Creating an object on the
communicator takes a tag from it, so the empty rank's tag counter fell one
behind, and the next two-sided exchange (a later ``distribute``) hung. The
metric is now made after the coordinate field is created on every rank.

This PETSc has no metric remesher, so ``remesh`` and ``mesh_adapt_meshVar``
stop at ``adaptMetric`` with their plain error, after the metric is made. The
ranks' next tag on COMM_WORLD is then compared, and a new mesh is
distributed.
"""
import ctypes

import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from empty_rank_fixtures import cells_per_rank, right_triangle_of_cells

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300),
              pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture's one cell is "
                                 "redistributed onto both ranks, so no rank is empty")]

HAS_REMESHER = any(PETSc.Sys.hasExternalPackage(p) for p in ("mmg", "parmmg", "pragmatic"))


def _next_tag():
    with open("/proc/self/maps") as maps:
        path = next(line.split()[-1] for line in maps if "/libpetsc.so" in line)
    lib = ctypes.CDLL(path)
    handle = ctypes.c_int if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int) else ctypes.c_void_p
    inner, tag = handle(), ctypes.c_int()
    assert lib.PetscCommDuplicate(handle(MPI._handleof(MPI.COMM_WORLD)), ctypes.byref(inner),
                                  ctypes.byref(tag)) == 0
    assert lib.PetscCommDestroy(ctypes.byref(inner)) == 0
    return tag.value


def _mesh_with_an_empty_rank_and_an_indicator():
    """Cells [8, 8, 0] at np 3; the indicator H is the mesh's second field."""
    size = uw.mpi.size
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(4, size - 1), degree=1, qdegree=3)
    assert 0 in cells_per_rank(mesh)
    uw.discretisation.MeshVariable("S0975", mesh, 1, degree=2)        # defined before H
    H = uw.discretisation.MeshVariable("H0975", mesh, 1, degree=1)
    H.array[:, 0, 0] = 36.0
    assert H.field_id != 0
    return mesh, H


def _in_step_after(route):
    mesh, H = _mesh_with_an_empty_rank_and_an_indicator()
    if HAS_REMESHER:
        pytest.skip("this PETSc has a metric remesher; the route would adapt")
    with pytest.raises(RuntimeError, match="metric remesher"):
        route(mesh, H)
    tags = uw.mpi.comm.allgather(_next_tag())
    assert len(set(tags)) == 1, tags
    # the next distribute: it hung on the empty rank before
    new = right_triangle_of_cells(6, uw.mpi.size)
    assert uw.mpi.comm.allreduce(new.getHeightStratum(0)[1] - new.getHeightStratum(0)[0]) > 0


def test_remesh_leaves_the_ranks_in_step():
    _in_step_after(lambda mesh, H: mesh.remesh(H))


def test_mesh_adapt_meshvar_leaves_the_ranks_in_step():
    from underworld3 import adaptivity
    _in_step_after(lambda mesh, H: adaptivity.mesh_adapt_meshVar(mesh, H, H))
