"""The facet restore's side test at a corner, at more than one rank.

The corner normals are built from the boundary facets gathered from every rank,
matched by their vertices' global numbers, each facet once. Every rank asks about
the same points beside and beyond the corners of a graded box, in 2-D and 3-D.

Baselines, exact: each point comes back inside the box (a point on a wall counts as
inside), and every rank returns the same coordinates for it, bit for bit.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(uw.mpi.size == 1,
                                 reason="the ranks' answers are compared with each other")]


def _graded_box(dim, cell, power):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
                                             cellSize=cell)
    X = np.asarray(mesh.X.coords, dtype=float).copy()
    X[:, 1] = X[:, 1] ** power
    mesh.deform(X)
    return mesh


def _probe(dim):
    a = np.linspace(0.0, 0.05, 11)[1:]
    s = np.linspace(0.0005, 0.14, 40)
    if dim == 2:
        return np.vstack([
            np.column_stack([np.full_like(s, 1.02), 1.0 - s]),   # beside the corner
            np.column_stack([1.0 + a, 1.0 + a]),                  # beyond it
            np.column_stack([1.0 + a, np.ones(10)]),
            np.column_stack([np.ones(10), 1.0 + a]),
        ])
    z = np.linspace(0.1, 0.9, 10)
    return np.vstack([
        np.column_stack([1.0 + a, 1.0 + a, z]),
        np.column_stack([np.ones(10), 1.0 + a, z]),
        np.column_stack([1.0 + a, np.ones(10), z]),
        np.column_stack([1.0 + a, 1.0 + a, 1.0 + a]),
        np.column_stack([1.0 + a, np.ones(10), np.ones(10)]),
    ])


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dim, cell, power", [(2, 0.05, 3), (3, 0.2, 2)])
def test_points_at_a_corner_come_back_inside_and_the_same_on_every_rank(dim, cell, power):
    mesh = _graded_box(dim, cell, power)
    probe = _probe(dim)
    out = np.asarray(mesh.return_coords_to_bounds(probe.copy()))

    outside = np.any(out > 1.0, axis=1)
    assert not outside.any(), (
        f"rank {uw.mpi.rank}: {int(outside.sum())} of {len(probe)} points left outside: "
        f"{probe[outside]}")
    answers = uw.mpi.comm.allgather(out)
    differ = [r for r, other in enumerate(answers) if not np.array_equal(other, answers[0])]
    assert not differ, f"ranks {differ} restore the points differently from rank 0"


@pytest.mark.mpi(min_size=2)
def test_points_beyond_a_corner_along_an_edge_line_come_back_inside_on_every_rank():
    """Beyond each corner of the graded cube along its edges' lines, where the
    nearest boundary point is the corner vertex: every point comes back inside on
    every rank, so the in/out answer is the one-rank answer, and the coordinates
    agree across ranks bit for bit."""
    mesh = _graded_box(3, 0.2, 2)
    h = float(mesh.get_min_radius())
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T
    pts = []
    for c in corners:
        for axis in range(3):
            d = np.zeros(3)
            d[axis] = -1.0 if c[axis] == 0 else 1.0
            for f in (1e-12, 1e-9, 1e-6, 1e-3):
                pts.append(c + f * h * d)
    probe = np.array(pts, dtype=float)
    out = np.asarray(mesh.return_coords_to_bounds(probe.copy()))

    outside = np.any((out < 0.0) | (out > 1.0), axis=1)
    assert not outside.any(), (
        f"rank {uw.mpi.rank}: {int(outside.sum())} of {len(probe)} points left outside: "
        f"{probe[outside]}")
    every = uw.mpi.comm.allgather(out)
    assert all(np.array_equal(every[0], o) for o in every), (
        f"rank {uw.mpi.rank}: the ranks return different coordinates for the same points")
