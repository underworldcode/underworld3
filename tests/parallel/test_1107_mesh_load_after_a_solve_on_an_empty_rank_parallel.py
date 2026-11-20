"""A mesh load after a solve with a rank that holds no cells (review of #1232).

Every PETSc object created on a communicator takes a tag from it, and PETSc's
two-sided exchanges take theirs from the same counter, so the counter must
advance the same way on every rank. A solver works on clones of the mesh DM,
and a clone does not carry the coordinate field: PETSc created it lazily, the
first time an assembly needed a cell's geometry. A rank with no cells did not
need it in the residual, so after a solve its counter was behind the others'.
The next gmsh mesh load then hung in its label read (ReadLabelStratumHDF5),
with the empty rank's exchange under a tag the other ranks never use. The
clones now create the field when they are made.

At np 3 with one rank empty, after a viscoelastic Stokes solve, the communicator's next tag
is compared over the ranks, and a gmsh mesh is loaded.
"""
import ctypes

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

# the right-triangle fixture leaves no rank empty at np 2 (its one cell is
# redistributed onto both ranks); the gmsh cases below run from np 2
needs_three = pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture's one cell is "
                                 "redistributed onto both ranks, so no rank is empty")


def _libpetsc():
    with open("/proc/self/maps") as maps:
        path = next(line.split()[-1] for line in maps if "/libpetsc.so" in line)
    return ctypes.CDLL(path)


def _next_tag():
    """The next tag PETSc would hand out on COMM_WORLD (one is used up)."""
    lib = _libpetsc()
    handle = ctypes.c_int if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int) else ctypes.c_void_p
    inner, tag = handle(), ctypes.c_int()
    assert lib.PetscCommDuplicate(handle(MPI._handleof(MPI.COMM_WORLD)), ctypes.byref(inner),
                                  ctypes.byref(tag)) == 0
    assert lib.PetscCommDestroy(ctypes.byref(inner)) == 0
    return tag.value


def _ve_stokes_on_a_mesh_with_an_empty_rank():
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size), size - 1)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)     # collective
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    assert 0 in cells_per_rank(mesh)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v1107load", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p1107load", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.dt_elastic = 0.1
    stokes.tolerance = 1e-6
    stokes.add_dirichlet_bc((y - 0.5, 0.0), "All_Boundaries")
    stokes.solve(timestep=0.1)
    return mesh


@needs_three
def test_the_ranks_agree_on_the_next_tag_after_a_solve():
    _ve_stokes_on_a_mesh_with_an_empty_rank()
    tags = uw.mpi.comm.allgather(_next_tag())
    assert len(set(tags)) == 1, tags


@needs_three
def test_a_gmsh_mesh_loads_after_a_solve():
    _ve_stokes_on_a_mesh_with_an_empty_rank()
    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                            cellSize=1 / 8, qdegree=2)
    cells = uw.mpi.comm.allreduce(box.dm.getHeightStratum(0)[1] - box.dm.getHeightStratum(0)[0])
    assert cells > 0


# At np 2, as CI runs the parallel files: a gmsh mesh whose cells all go to
# rank 0 (a shell partition set on the loaded DM before it is distributed), so
# every other rank holds none.

@pytest.fixture
def every_cell_on_rank_0(monkeypatch):
    from petsc4py import PETSc
    from underworld3.discretisation import discretisation_mesh as dmod
    original = dmod._from_plexh5

    def loaded(*a, **k):
        out = original(*a, **k)
        dm = out[1] if isinstance(out, tuple) else out
        cS, cE = dm.getHeightStratum(0)
        sizes = np.zeros(uw.mpi.size, dtype=np.int32)
        sizes[0] = cE - cS
        part = dm.getPartitioner()
        part.setType(PETSc.Partitioner.Type.SHELL)
        part.setShellPartition(uw.mpi.size, sizes=sizes,
                               points=np.arange(cE - cS, dtype=np.int32))
        return out

    monkeypatch.setattr(dmod, "_from_plexh5", loaded)


def _gmsh_box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8, qdegree=3)


def _ve_solve_on(mesh):
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v1107g", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p1107g", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.dt_elastic = 0.1
    stokes.tolerance = 1e-6
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.solve(timestep=0.1)


def test_on_a_gmsh_mesh_with_every_cell_on_rank_0_the_ranks_stay_in_step(every_cell_on_rank_0):
    mesh = _gmsh_box()
    held = uw.mpi.comm.allgather(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0])
    assert held[0] > 0 and not any(held[1:]), held
    _ve_solve_on(mesh)
    tags = uw.mpi.comm.allgather(_next_tag())
    assert len(set(tags)) == 1, tags
    # the next gmsh load: its label read hung here before
    again = _gmsh_box()
    assert uw.mpi.comm.allreduce(again.dm.getHeightStratum(0)[1] - again.dm.getHeightStratum(0)[0]) > 0
