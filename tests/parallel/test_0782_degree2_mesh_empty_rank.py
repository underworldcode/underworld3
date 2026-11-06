"""A degree-2 mesh takes its cell sizes from a computation that supports it (#908).

``Mesh._get_cell_radii`` took each cell's volume from
``DMPlexComputeGeometryFVM``, which expects affine coordinates. On a degree-2
coordinate field the routine stops at its first face with "Edge has 6
coordinates != 4", and the error code was ignored. A rank with cells stopped
there; a rank with no cells has no faces and went on through the routine, so the
ranks disagreed from then on: MPI_ERR_BUFFER in the next ``DMClone`` and a hang,
at np 3 on a 4 x 4 grid with cells [16, 16, 0].

Power: exactly one rank holds no cells at every np >= 2 (see
``empty_rank_fixtures``), and the test asserts that premise. At np 2 that is the
one-cell mesh, and from np 3 the last rank of a shell partition. A regression hangs
and the MPI supervisor's silence limit ends it, so every assert follows the
collectives it needs.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0782_degree2_mesh_empty_rank.py
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


def test_a_degree_2_mesh_with_an_empty_rank_has_the_true_cell_size():
    """The constructor returns on every rank, every cell has the closed-form
    radius, and the ranks are still in step for a later collective."""
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(n, max(size - 1, 1)),
                                  degree=2, qdegree=3)
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


def test_the_fvm_routine_refuses_a_degree_2_mesh_on_every_rank():
    """Called directly on such a mesh, the routine would fail where there are
    cells and hang where there are none, so every rank must refuse up front."""
    from underworld3.cython import petsc_discretisation

    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(n, max(size - 1, 1)),
                                  degree=2, qdegree=3)
    counts = _cells_per_rank(mesh)

    outcomes = []
    for routine in (petsc_discretisation.petsc_fvm_get_local_cell_sizes,
                    petsc_discretisation.petsc_fvm_get_min_radius):
        try:
            routine(mesh)
            outcomes.append("returned")
        except ValueError as error:
            outcomes.append(str(error))
    all_outcomes = uw.mpi.comm.allgather(outcomes)

    assert counts.count(0) == 1, counts
    assert all(o == all_outcomes[0] for o in all_outcomes), all_outcomes
    assert all("needs affine coordinates" in o for o in all_outcomes[0]), all_outcomes


def test_an_error_code_on_one_rank_raises_on_every_rank():
    """The check itself: a rank with a good code raises too, and every rank is
    told which rank failed and how."""
    from underworld3.cython import petsc_discretisation

    failing_rank = uw.mpi.size - 1
    ierr = 73 if uw.mpi.rank == failing_rank else 0
    try:
        petsc_discretisation._check_error_code_collectively(ierr, "TheRoutine", uw.mpi.comm)
        message = "returned"
    except RuntimeError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)

    expected = f"TheRoutine failed; error code by rank: {{{failing_rank}: 73}}"
    assert messages == [expected] * uw.mpi.size, messages


def test_a_degree_2_parent_extracts_a_region_some_ranks_hold_none_of():
    """The second report: at np 4 the child's constructor died with the same
    message, exit 142. How many ranks hold none of the region is the
    partitioner's business, so the count is reported and not asserted; the area
    of the region is the baseline."""
    parent = uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12), zelementRes=(10, 2), zintCoord=0.85,
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), simplex=True, qdegree=3, degree=2)
    region = parent.extract_region("Outer")
    counts = _cells_per_rank(region)
    uw.mpi.pprint(f"DEGREE2_SUBMESH ranks={uw.mpi.size} cells={counts}")

    area = uw.mpi.comm.allreduce(float(np.sum(np.asarray(region._cell_radii) ** 2)))

    assert sum(counts) > 0, counts
    assert np.isclose(area, 0.15, rtol=1e-10), (area, counts)
