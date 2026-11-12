"""A mesh with a rank holding no cells: degree 1 builds, any other degree is
refused on every rank, and an error code on one rank raises on all (#908, #978).

A degree-2 coordinate field made ``DMPlexComputeGeometryFVM`` fail on a rank's
first face, while a rank with no cells has no faces and went on through the
routine: MPI_ERR_BUFFER in the next ``DMClone`` and a hang, at np 3 on a 4 x 4
grid with cells [16, 16, 0]. A mesh now refuses ``degree != 1`` before any
collective, from an argument every rank holds alike, and a ``uw.meshing``
constructor refuses before rank 0 starts gmsh.

Power: exactly one rank holds no cells at every np >= 2 (see
``empty_rank_fixtures``), and the cell-size test asserts that premise. The
refusals depend on the argument alone, so what they can show is that every rank
raises the same error and is still in step afterwards. A regression hangs and
the MPI supervisor's silence limit ends it, so every assert follows the
collectives it needs.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0782_mesh_degree_refused_on_every_rank.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

from empty_rank_fixtures import (cells_along_a_side_leaving_one_rank_empty,
                                 right_triangle_of_cells)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def _cells_per_rank(mesh):
    start, end = mesh.dm.getHeightStratum(0)
    return uw.mpi.comm.allgather(end - start)


def _grid():
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    return n, right_triangle_of_cells(n, max(size - 1, 1))


def test_a_mesh_with_an_empty_rank_has_the_true_cell_size():
    """The constructor returns on every rank, every cell has the closed-form
    radius, and the ranks are still in step for a later collective."""
    n, dm = _grid()
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    counts = _cells_per_rank(mesh)

    # A cell has area 1 / (2 n^2), and its radius is area**(1/dim).
    radius = (0.5 / n ** 2) ** 0.5
    worst = uw.mpi.comm.allreduce(
        float(np.abs(np.asarray(mesh._cell_radii) / radius - 1.0).max(initial=0.0)),
        op=MPI.MAX)
    r_min, r_max = mesh.get_min_radius(), mesh.get_max_radius()

    field = uw.discretisation.MeshVariable("T0782", mesh, 1, degree=2)
    field.array[:, 0, 0] = 1.0
    area = uw.maths.Integral(mesh, field.sym[0]).evaluate()

    assert counts.count(0) == 1 and sum(counts) == n ** 2, counts
    assert worst < 1.0e-10, (worst, counts)
    assert np.isclose(r_min, radius, rtol=1e-10), (r_min, radius)
    assert np.isclose(r_max, radius, rtol=1e-10), (r_max, radius)
    assert np.isclose(area, 0.5, rtol=1e-12), area


def test_a_degree_2_mesh_is_refused_on_every_rank():
    """Every rank raises the same error, the empty one included, and the ranks
    are still in step for the gather that follows."""
    _n, dm = _grid()
    try:
        uw.discretisation.Mesh(dm, degree=2, qdegree=3)
        message = "built"
    except ValueError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)

    assert all(m == messages[0] for m in messages), messages
    assert "mesh coordinates are degree 1" in messages[0], messages


def test_a_constructor_refuses_degree_2_on_every_rank():
    """Rank 0 alone runs gmsh, so a refusal that came after it would leave the
    other ranks ahead of rank 0."""
    try:
        uw.meshing.UnstructuredSimplexBox(cellSize=0.5, degree=2)
        message = "built"
    except ValueError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)

    assert all(m == messages[0] for m in messages), messages
    assert "mesh coordinates are degree 1" in messages[0], messages


def test_an_error_code_on_one_rank_raises_on_every_rank():
    """The check itself: a rank with a good code raises too, and every rank is
    told which rank failed and how."""
    from underworld3.cython import petsc_discretisation

    failing_rank = uw.mpi.size - 1
    ierr = 73 if uw.mpi.rank == failing_rank else 0
    try:
        petsc_discretisation._check_error_code_collectively(ierr, "TheRoutine")
        message = "returned"
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)

    expected = f"TheRoutine failed on rank {failing_rank} with error code 73"
    assert messages == [expected] * uw.mpi.size, messages


def test_a_parent_extracts_a_region_some_ranks_hold_none_of():
    """How many ranks hold none of the region is the partitioner's business, so
    the count is reported and not asserted; the area of the region is the
    baseline."""
    parent = uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12), zelementRes=(10, 2), zintCoord=0.85,
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), simplex=True, qdegree=3)
    region = parent.extract_region("Outer")
    counts = _cells_per_rank(region)
    uw.mpi.pprint(f"SUBMESH_CELL_SIZE ranks={uw.mpi.size} cells={counts}")

    area = uw.mpi.comm.allreduce(float(np.sum(np.asarray(region._cell_radii) ** 2)))

    assert sum(counts) > 0, counts
    assert np.isclose(area, 0.15, rtol=1e-10), (area, counts)
