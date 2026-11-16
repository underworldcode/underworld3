"""A declared wall gives one answer on every rank, and on every copy of a vertex.

Refinement: on this cubed sphere the refinement callback would invert cells at a
few vertices of the inner wall, which stay where refinement placed them (#949), so
the mesh releases the sphere it would have declared for ``Lower``. The held-back
vertices sit in one corner of the mesh: a rank whose partition holds none of them
must release the wall all the same, or a method reading ``mesh.analytic_surface``
would proceed on that rank and refuse on another.

Slip: ``project_to_slip_surface`` returns every local copy of a wall vertex on the
wall, the copies a rank holds of another rank's vertices included.
"""
import warnings

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing.graph import _pinned_mask, _reduce_over_sharing_ranks

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(uw.mpi.size == 1,
                                 reason="the ranks' verdicts are compared with each other")]

ON_SURFACE = 64 * np.finfo(float).eps


def test_a_wall_held_back_on_one_rank_is_released_on_every_rank():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mesh = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.547, numElements=3,
                                      simplex=True, refinement=1)

    def refused(label):
        try:
            mesh.analytic_surface(label)
        except ValueError:
            return True
        return False

    X = np.asarray(mesh.X.coords, dtype=float)
    lower = np.linalg.norm(X[_pinned_mask(mesh.dm, ("Lower",))], axis=1)
    holds_one = bool(np.any(np.abs(lower - 0.547) > ON_SURFACE * 0.547))

    holders = uw.mpi.comm.allgather(holds_one)
    lower_refused = uw.mpi.comm.allgather(refused("Lower"))
    upper_refused = uw.mpi.comm.allgather(refused("Upper"))

    assert any(holders), (
        "on this host's triangulation refinement held back no Lower vertex, so the "
        "test has nothing to release: choose another fixture")
    assert all(lower_refused), f"Lower refused by rank: {lower_refused}, holders: {holders}"
    assert not any(upper_refused), f"Upper refused by rank: {upper_refused}"
    # What the case exists for is a rank holding none of the held-back vertices.
    # Measured on Linux (MPICH), 2026-11-14: at two ranks both hold one, so the case
    # can only check the verdicts and skips; at four, one rank holds none, and
    # scripts/test.sh runs this file at four ranks for that. At four or more ranks a
    # partition in which every rank holds one is a failure, not a skip, so that line
    # cannot pass without showing the reduction.
    if all(holders):
        if uw.mpi.size >= 4:
            pytest.fail(f"every one of the {uw.mpi.size} ranks holds a held-back vertex on "
                        "this host's partition, so the case shows nothing about a rank "
                        "learning of it from another: choose a fixture where one does not")
        pytest.skip(f"every one of the {uw.mpi.size} ranks holds a held-back vertex, so "
                    "this rank count does not show a rank learning of it from another")


def test_slip_projection_puts_every_local_copy_of_a_wall_vertex_on_the_wall():
    """The vertices are displaced by a field of position, the same for every copy
    of a vertex, that takes the walls up to 2 % of a radius off their circles."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
    X = np.array(mesh.X.coords, dtype=float)
    angle = np.arctan2(X[:, 1], X[:, 0])
    moved = X * (1.0 + 0.02 * np.cos(3.0 * angle))[:, None]
    moved += 0.02 * np.column_stack([-X[:, 1], X[:, 0]])
    projected = mesh.project_to_slip_surface(moved, True, reference_coords=X)

    worst = {}
    for label in ("Lower", "Upper"):
        # every local copy of the label's vertices, also on a rank holding a vertex
        # of the wall and none of its facets
        on_label = _reduce_over_sharing_ranks(mesh.dm, _pinned_mask(mesh.dm, (label,)), MPI.BOR)
        radius = mesh.analytic_surface(label).radius
        off = np.abs(np.linalg.norm(projected[on_label], axis=1) - radius) / radius
        worst[label] = uw.mpi.comm.allgather(float(np.max(off, initial=0.0)))
    for label, by_rank in worst.items():
        assert max(by_rank) < ON_SURFACE, f"{label} off its circle, by rank: {by_rank}"
