"""Whether a point is outside the mesh does not depend on which rank asks (#1176).

On a deformed mesh, and on every mesh read from a file, ``return_coords_to_bounds``
is the facet restore: the nearest point on the boundary facets, with a test against
that facet's outward normal. Before the fix each rank used only the boundary facets
it held, so at more than one rank a point outside could be left outside or sent to
the wrong wall, and a point inside could be moved onto a wall. Measured on
development bc3d5362 at np 4 (H1, the parallel review desk): the annulus below,
walls at 0.45 and 0.90, a point at r = 0.95 went to 0.95, 0.90, 0.45 and 0.45 on
ranks 0 to 3, and a point at r = 0.47 went to 0.45 on ranks 1 to 3.

The baselines: an interior point is returned unchanged, bit for bit; a point
outside lands just inside the nearest wall, within that wall's largest chord
sagitta of its circle; and every rank returns the same coordinates for the same
query, bit for bit.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing.smoothing.graph import _pinned_mask

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(uw.mpi.size == 1,
                                 reason="the ranks' answers are compared with each other")]

R_IN, R_OUT, SCALE = 0.5, 1.0, 0.9


def _scaled_annulus(cell_size):
    """An annulus moved by deform, so the restore is the facet restore and the
    walls are at 0.45 and 0.90."""
    mesh = uw.meshing.Annulus(radiusOuter=R_OUT, radiusInner=R_IN, cellSize=cell_size)
    mesh.deform(SCALE * np.asarray(mesh.X.coords, dtype=float))
    assert mesh.return_coords_to_bounds == mesh._facet_return_coords_to_bounds
    return mesh


def _largest_sagitta(mesh, label, radius):
    """The largest distance, over every rank, of a chord of wall ``label`` from
    its circle of ``radius``: a point restored onto that wall is within it."""
    X = np.asarray(mesh.X.coords, dtype=float)[_pinned_mask(mesh.dm, (label,))]
    angles = np.concatenate(uw.mpi.comm.allgather(np.arctan2(X[:, 1], X[:, 0])))
    angles = np.unique(angles)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    return radius * (1.0 - np.cos(0.5 * gaps.max()))


def _ring(radius, n=32, phase=0.1):
    theta = phase + 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _same_on_every_rank(restored):
    every = uw.mpi.comm.allgather(restored)
    return all(np.array_equal(every[0], other) for other in every[1:])


def test_points_inside_and_outside_get_one_answer_on_every_rank():
    mesh = _scaled_annulus(0.1)
    inside = np.vstack([_ring(0.47), _ring(0.60), _ring(0.75), _ring(0.88)])
    beyond_outer = _ring(0.95)
    query = np.vstack([inside, beyond_outer])

    restored = np.asarray(mesh.return_coords_to_bounds(query.copy()))

    assert _same_on_every_rank(restored), (
        "the same query points were restored differently on different ranks")
    moved = np.any(restored[: len(inside)] != inside, axis=1)
    assert not np.any(moved), (
        f"rank {uw.mpi.rank}: {int(moved.sum())} of {len(inside)} interior points were "
        f"moved, to radii {np.linalg.norm(restored[: len(inside)][moved], axis=1)}")
    r = np.linalg.norm(restored[len(inside):], axis=1)
    sagitta = _largest_sagitta(mesh, "Upper", SCALE * R_OUT)
    assert np.all(r <= SCALE * R_OUT) and np.all(r >= SCALE * R_OUT - 2.0 * sagitta), (
        f"rank {uw.mpi.rank}: points beyond the outer wall went to radii "
        f"{r.min():.6f} to {r.max():.6f}, not to the wall at {SCALE * R_OUT}")


def test_rotated_interior_departure_points_are_left_where_they_are():
    """The semi-Lagrangian trace-back clamps every departure point with this
    restore. Departure points from an exact rotation keep their radius, so all of
    them are inside; H1 measured 254 of them moved onto a wall at np 4 for a
    rotation of 0.8 rad on development."""
    mesh = _scaled_annulus(0.05)
    X = np.asarray(mesh.X.coords, dtype=float)
    r = np.linalg.norm(X, axis=1)
    X = X[(r > 0.47) & (r < 0.88)]
    c, s = np.cos(0.8), np.sin(0.8)
    departure = X @ np.array([[c, s], [-s, c]])

    restored = np.asarray(mesh.return_coords_to_bounds(departure.copy()))

    moved = int(np.count_nonzero(np.any(restored != departure, axis=1)))
    every = uw.mpi.comm.allgather(moved)
    assert sum(every) == 0, f"interior departure points moved, by rank: {every}"


def test_a_rank_with_no_points_still_takes_part():
    """The restore is collective (the boundary is gathered from every rank), so a
    rank asking about no points must still enter it, and the others get their
    answer."""
    mesh = _scaled_annulus(0.1)
    sagitta = _largest_sagitta(mesh, "Upper", SCALE * R_OUT)
    query = _ring(0.95) if uw.mpi.rank % 2 == 0 else np.zeros((0, 2))

    restored = np.asarray(mesh.return_coords_to_bounds(query.copy()))

    assert restored.shape == query.shape
    if len(query):
        r = np.linalg.norm(restored, axis=1)
        assert np.all(r <= SCALE * R_OUT) and np.all(r >= SCALE * R_OUT - 2.0 * sagitta)

