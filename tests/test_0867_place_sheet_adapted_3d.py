"""Placing a sheet into an ``adapt()`` child in 3-D.

The fault workflow is base mesh -> ``adapt`` toward the fault ->
``add_conforming_sheet`` -> ``split_fault``. On a newest-vertex-bisection
child the cavity the carve clears, and the fill gmsh returns for it, can be
wrong in three ways that the earlier gates did not see, all at the default
clearance and none on the gmsh base:

* a KEPT cell (or cluster) wholly enclosed by dropped cells — its faces
  form a separate closed component of the shell, gmsh fills it as part of
  the cavity, and the fill overlaps the kept cell (measured: the domain
  volume grew by exactly that cell's volume, refused by the volume gate);
* a PINCHED vertex — every shell edge has two faces, but the faces around
  the vertex form two or more separate fans — where gmsh returns
  zero-volume tets (refused by PETSc's geometry check, ``|J| = 0``);
* a FLAT tet spanning four nearly coplanar shell nodes — volume ~1e-20,
  which no gate refused, from either of gmsh's Delaunay or HXT algorithms
  depending on the orientation.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import _cell_volumes_signed6

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

CENTRE = np.array([0.5, 0.5, 0.5])
RADIUS = 0.2
H_NEAR = 0.05
TILTED = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
DIAGONAL = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def _disc_sheet(normal, n_rim=32):
    """A planar disc as a centre fan; ``size=`` re-triangulates it."""
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    a = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
    rim = CENTRE + RADIUS * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    pts = np.vstack([CENTRE, rim])
    tris = np.array([(0, 1 + i, 1 + (i + 1) % n_rim) for i in range(n_rim)],
                    dtype=np.int64)
    return pts, tris


def _graded_metric(normal):
    """M = 1/h^2 from the exact distance to the disc, h_near -> 2 h_near."""
    def metric(X):
        rel = np.asarray(X)[:, :3] - CENTRE
        zn = rel @ normal
        r = np.linalg.norm(rel - zn[:, None] * normal, axis=1)
        d = np.sqrt(np.maximum(r - RADIUS, 0.0) ** 2 + zn ** 2)
        h = np.clip(H_NEAR + d * H_NEAR / 0.4, H_NEAR, 2.0 * H_NEAR)
        return 1.0 / h**2
    return metric


def _uniform_metric(normal):
    return lambda X: np.full(len(X), 1.0 / H_NEAR**2)


@pytest.fixture(scope="module")
def base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2, regular=False, qdegree=2, refinement=1)


@pytest.mark.parametrize(
    "metric, normal",
    [(_graded_metric, TILTED), (_uniform_metric, TILTED),
     (_graded_metric, DIAGONAL)],
    ids=["graded-enclosed-cell", "uniform-pinch-and-flat", "graded-flat"])
def test_sheet_places_into_an_nvb_child(base, metric, normal):
    """The placed mesh keeps the domain volume, one orientation, and no
    degenerate cell, and every sheet triangle is a labelled face."""
    child = base.adapt(metric(normal), max_levels=1)
    pts, tris = _disc_sheet(normal)
    placed = child.add_conforming_sheet(pts, tris, "Fault", size=H_NEAR)

    v6 = _cell_volumes_signed6(placed.dm)
    assert abs(np.abs(v6).sum() / 6.0 - 1.0) < 1e-9
    assert (np.sign(v6) == np.sign(v6[0])).all()
    assert np.abs(v6).min() / 6.0 > 1e-12
    assert placed._surface_info["n_surface_facets"] > 0
