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
