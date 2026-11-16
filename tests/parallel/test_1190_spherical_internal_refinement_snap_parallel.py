"""Refinement keeps the internal sphere of a shell on its sphere, on every rank (#949 item 3).

The refinement hierarchy is built on the distributed mesh, so the refinement callback
runs on each rank's part of it. A rank can hold a vertex of the ``Internal`` surface
without holding every ``Internal`` facet around it; the hierarchy gives every copy the
position the callback moved it to. Contracts, on every level and every rank:

- every local copy, owned or not, of a vertex of ``Lower``, ``Internal`` or ``Upper``
  lies on its sphere to 64 machine epsilon relative to the radius;
- every copy of every vertex has its owner's coordinates;
- the mesh declares the ``Internal`` sphere, with radius ``radiusInternal``.

Every assert follows the reductions it needs, so a failure fails on every rank.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1190_spherical_internal_refinement_snap_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]

RADII = {"Lower": 0.5, "Internal": 0.8, "Upper": 1.0}
ON_SURFACE = 64 * np.finfo(float).eps


@pytest.fixture(scope="module")
def refined_shell():
    return uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=1.0, radiusInternal=0.8, radiusInner=0.5, cellSize=0.3, refinement=1)


def test_every_copy_of_each_surface_vertex_is_on_its_sphere(refined_shell):
    worst = {}
    for level, dm in enumerate(refined_shell.dm_hierarchy):
        radius = np.linalg.norm(dm.getCoordinatesLocal().array.reshape(-1, 3), axis=1)
        for label, expected in RADII.items():
            vertices = petsc_dm_find_labeled_points_local(dm, label)
            local = np.max(np.abs(radius[vertices] - expected), initial=0.0) / expected
            worst[(level, label)] = uw.mpi.comm.allreduce(local, op=MPI.MAX)
    off = {key: value for key, value in worst.items() if value > ON_SURFACE}
    assert not off, f"(level, surface): largest distance from the sphere / radius: {off}"


def test_every_copy_has_its_owners_coordinates(refined_shell):
    disagree = 0
    for dm in refined_shell.dm_hierarchy:
        local = dm.getCoordinatesLocal()
        owners = local.duplicate()
        dm.getCoordinateDM().globalToLocal(dm.getCoordinates(), owners)
        disagree += int(np.count_nonzero(owners.array != local.array))
        owners.destroy()
    disagree = uw.mpi.comm.allreduce(disagree)
    assert disagree == 0


def test_the_refined_shell_declares_its_internal_sphere_on_every_rank(refined_shell):
    try:
        radius = refined_shell.analytic_surface("Internal").radius
    except ValueError:
        # a refusal on one rank must still reach the gather, or the others wait in it:
        # it is recorded as a radius no sphere has
        radius = -1.0
    radii = uw.mpi.comm.allgather(radius)
    assert radii == [0.8] * len(radii), radii
