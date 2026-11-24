"""global_evaluate answers a point from the rank whose cell contains it (#1178).

A point within the evaluation face tolerance (``Mesh._EVAL_FACE_TOL``) of a
partition seam was claimed by the ranks on both sides. If it stayed on the rank
whose cells do not contain it, that rank evaluated it in its nearest cell with the
reference coordinates clamped onto the face: an error of about |grad f| times the
point's distance from that cell. Measured by the parallel review desk on #1123's
sheared box, at points 1e-8 to 1e-4 off interior edges: up to 2.1e-4, 1.0e-4 and
4.1e-4 at np 2, 3 and 4, where serial is 1.8e-15.

The baseline is exact: the field is P1 and linear in x, y, z, so the element
represents it exactly and its interpolant at a point inside the mesh is the
linear function there, to round-off. Every rank asks for every point, so every
rank's answer is checked.
"""
import itertools

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

from sheared_box_fixture import SHEAR, SHEAR_WAVE, sheared_box

pytestmark = [pytest.mark.level_2,
              pytest.mark.timeout(300, method="thread")]

N = 6


def _linear(points):
    return 1.0 + points[:, 0] + 2.0 * points[:, 1] - 0.5 * points[:, 2]


def _sheared(reference):
    """The fixture's map applied to reference-box points."""
    mapped = reference @ SHEAR.T
    mapped[:, 0] += SHEAR_WAVE * np.sin(2 * np.pi * reference[:, 1])
    mapped[:, 1] += SHEAR_WAVE * np.sin(2 * np.pi * reference[:, 2])
    mapped[:, 2] += SHEAR_WAVE * np.sin(2 * np.pi * reference[:, 0])
    return mapped


def _points_off_interior_edges(offset, seed):
    """One point on each interior edge of the sheared box (neither end on the
    boundary), at a random fraction along it, moved by ``offset`` in a random
    direction. The edges are those of the fixture's tetrahedra; a mesh edge is
    straight between its mapped ends (the mesh is P1). Interior edges are at
    least one cell from the boundary, so every point is inside the mesh."""
    edges = set()
    for i, j, k in itertools.product(range(N), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = [i, j, k]
            path = [tuple(corner)]
            for axis in order:
                corner[axis] += 1
                path.append(tuple(corner))
            for a, b in itertools.combinations(path, 2):
                edges.add((a, b) if a < b else (b, a))
    interior = sorted(e for e in edges
                      if all(0 < c < N for end in e for c in end))
    ends = np.array(interior, dtype=float) / N
    a, b = _sheared(ends[:, 0]), _sheared(ends[:, 1])
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.1, 0.9, (len(interior), 1))
    direction = rng.normal(size=a.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return t * a + (1 - t) * b + offset * direction


@pytest.mark.tier_b
@pytest.mark.parametrize("offset", [1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4])
def test_a_point_near_a_seam_is_answered_by_the_rank_that_contains_it(offset):
    mesh = sheared_box(N)
    T = uw.discretisation.MeshVariable("T1178", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear(np.asarray(T.coords))
    points = _points_off_interior_edges(offset, seed=1178)

    values = np.asarray(uw.function.global_evaluate(T.sym[0], points)).reshape(-1)

    # Round-off of a barycentric interpolation of an O(1) field: serial gives
    # 1.8e-15 on this set; 1e-12 is far below the 1e-8 to 4e-4 of a point
    # evaluated in a cell that does not contain it.
    error = np.abs(values - _linear(points))
    wrong = uw.mpi.comm.allgather(int((error > 1.0e-12).sum()))
    worst = uw.mpi.comm.allreduce(float(error.max()), op=MPI.MAX)
    assert sum(wrong) == 0, (
        f"{wrong} of {len(points)} points per asking rank are off the exact field "
        f"(worst {worst:.2e}) at offset {offset:.0e}")


def _linear_2d(points):
    return 1.0 + points[:, 0] + 2.0 * points[:, 1]


@pytest.mark.tier_b
def test_a_rank_with_no_cells_takes_part_and_every_point_is_exact():
    """A rank that holds no cells still joins every exchange of the hand-over
    (it asks about nothing and is asked about nothing), and every point inside
    the mesh is answered exactly. Every rank asks for every point."""
    from empty_rank_fixtures import (cells_along_a_side_leaving_one_rank_empty,
                                     cells_per_rank, right_triangle_of_cells)

    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size) if size > 1 else 4
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(n, max(size - 1, 1)), degree=1,
                                  qdegree=2)
    if size > 1:
        assert 0 in cells_per_rank(mesh), "the fixture should leave one rank empty"
    T = uw.discretisation.MeshVariable("T1178e", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear_2d(np.asarray(T.coords))
    rng = np.random.default_rng(1178)
    points = rng.uniform(0.01, 0.98, (500, 2))
    points = points[points.sum(axis=1) < 0.99]

    values = np.asarray(uw.function.global_evaluate(T.sym[0], points)).reshape(-1)

    error = np.abs(values - _linear_2d(points))
    worst = uw.mpi.comm.allreduce(float(error.max()), op=MPI.MAX)
    assert worst < 1.0e-12, f"worst error {worst:.2e} with a rank holding no cells"


@pytest.mark.tier_b
def test_every_rank_asking_for_no_points_returns_nothing():
    """All ranks ask for zero points: every rank still reaches every collective
    and gets an empty answer."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.125)
    T = uw.discretisation.MeshVariable("T1178z", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear_2d(np.asarray(T.coords))

    values = np.asarray(uw.function.global_evaluate(T.sym[0], np.zeros((0, 2))))

    assert values.size == 0


@pytest.mark.tier_b
def test_points_just_outside_keep_the_face_tolerance_answer_on_every_rank():
    """Points 1e-6 outside the boundary are contained by no rank, so they keep
    the face-tolerance answer: the point is clamped onto the boundary cell and
    the field read there, an error of about |grad f| times the distance (3.0e-6
    measured here, 2026-11-21, with |grad f| up to 3 along the clamp). Every
    rank asks for every point and gets the same answer."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.125)
    T = uw.discretisation.MeshVariable("T1178o", mesh, 1, degree=1)
    T.array[:, 0, 0] = _linear_2d(np.asarray(T.coords))
    t = np.linspace(0.05, 0.95, 40)
    eps = 1.0e-6
    points = np.concatenate([np.c_[t, np.full_like(t, -eps)], np.c_[np.full_like(t, 1 + eps), t],
                             np.c_[t, np.full_like(t, 1 + eps)], np.c_[np.full_like(t, -eps), t]])

    values = np.asarray(uw.function.global_evaluate(T.sym[0], points)).reshape(-1)

    error = np.abs(values - _linear_2d(points))
    worst = uw.mpi.comm.allreduce(float(error.max()), op=MPI.MAX)
    # Bounded by |grad f| (at most 3 along any clamp of this field) times a few
    # times the distance; a point answered from a cell far from it would be off
    # by the cell size times |grad f|, about 0.4 here.
    assert worst < 4 * 3 * eps, f"worst error {worst:.2e} just outside the boundary"
    answers = uw.mpi.comm.allgather(values)
    assert all(np.array_equal(answers[0], a) for a in answers), "ranks disagree"


TINY = 1.0e-6


def _wavy(points):
    """A field P1 does not represent exactly, on the tiny box's scale: an answer
    from the wrong cell is off by order one, an answer from the containing cell
    by |grad f| times the distance outside, a few times 1e-7 here."""
    return np.sin(3.0 * points[:, 0] / TINY) * np.cos(2.0 * points[:, 1] / TINY)


@pytest.mark.tier_b
def test_points_just_outside_a_tiny_box_are_answered_by_the_cell_that_contains_them():
    """On a quad box 1e-6 across, boundary vertices pushed 1.1e-14 and 3e-14
    outside are contained, to rounding, by the cell beside them (the cell-wall
    test, used for quads, hexes and 2-manifolds, accepts points a few times
    1e-14 outside and rejects 1e-13), and take that cell's value:
    within 1e-6 of the field at the vertex (field of order one). The rank boxes
    are widened so that the rank holding that cell is asked about the point; a
    widening of 1e-8 of the extent alone is 1e-14 here, below the cell-wall
    test's rounding slab (a few times 1e-14 absolute), and the parallel review
    desk measured answers up to 0.12 off on this box at np 2 to 4. Only the
    pushed points some rank's cell contains are checked: the others are
    contained by no cell and take the face-tolerance answer, which at this
    scale is already about 0.1 off on development, and is not this test's."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(12, 12), minCoords=(0.0, 0.0),
                                        maxCoords=(TINY, TINY))
    T = uw.discretisation.MeshVariable("TTinyQuad", mesh, 1, degree=1)
    T.array[:, 0, 0] = _wavy(np.asarray(T.coords))

    vertices = np.unique(np.concatenate(uw.mpi.comm.allgather(np.asarray(T.coords))), axis=0)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    pushed, nearest = [], []
    for axis, side, sign in itertools.product(range(2), (lo, hi), (-1.0, 1.0)):
        if (sign < 0) != (side is lo):
            continue
        on_wall = vertices[vertices[:, axis] == side[axis]]
        other = 1 - axis
        on_wall = on_wall[(on_wall[:, other] > lo[other]) & (on_wall[:, other] < hi[other])]
        for distance in (1.1e-14, 3.0e-14):
            points = on_wall.copy()
            points[:, axis] += sign * distance
            pushed.append(points)
            nearest.append(on_wall)
    pushed, nearest = np.concatenate(pushed), np.concatenate(nearest)
    # The points some rank's cell contains to rounding: the ones the widened
    # boxes must route to that rank. The others take the face-tolerance answer.
    here = np.asarray(mesh._contained_in_local_cells(pushed), dtype=np.int32)
    anywhere = np.zeros_like(here)
    uw.mpi.comm.Allreduce(here, anywhere, op=MPI.MAX)
    contained = anywhere.astype(bool)
    assert contained.any(), "no pushed point is contained by any cell: the fixture misses"

    values = np.asarray(uw.function.global_evaluate(T.sym, pushed)).reshape(-1)
    worst = np.max(np.abs(values[contained] - _wavy(nearest[contained])))
    assert worst < 1.0e-6, (
        f"rank {uw.mpi.rank}: a point just outside the box is {worst:.3e} off the value "
        f"of the cell beside it")
