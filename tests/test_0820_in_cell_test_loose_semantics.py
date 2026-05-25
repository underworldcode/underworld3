"""Regression test for `Mesh._test_if_points_in_cells_internal` on_boundary modes.

Locks in the contract for the `on_boundary` kwarg added to
`_test_if_points_in_cells_internal` (and forwarded through
`_get_closest_local_cells_internal`, `get_closest_local_cells`, and the
public `test_if_points_in_cells`):

- on_boundary=True (default): a point exactly on a cell face counts as
  inside the cell — the natural semantics for FE evaluation, where the
  basis at a shared face/vertex is consistent across the adjacent cells.
- on_boundary=False: strict-inside semantics — a point on the face is
  reported as NOT inside. Useful when uniqueness matters.
"""

import numpy as np
import pytest

import underworld3 as uw


pytestmark = pytest.mark.level_1


def test_default_accepts_vertices_simplex_2d():
    """Default (on_boundary=True): every 2D simplex vertex resolves to a containing cell."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
    )
    verts = np.asarray(mesh.X.coords)
    cells = mesh._get_closest_local_cells_internal(verts)
    assert (cells == -1).sum() == 0, (
        f"default loose mode rejected {(cells == -1).sum()}/{len(verts)} vertices"
    )


def test_default_accepts_vertices_simplex_3d():
    """Default (on_boundary=True): every 3D simplex vertex resolves to a containing cell."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4
    )
    verts = np.asarray(mesh.X.coords)
    cells = mesh._get_closest_local_cells_internal(verts)
    assert (cells == -1).sum() == 0, (
        f"default loose mode rejected {(cells == -1).sum()}/{len(verts)} vertices"
    )


def test_default_accepts_vertices_quad():
    """Default (on_boundary=True): every structured-quad vertex resolves to a containing cell."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    verts = np.asarray(mesh.X.coords)
    cells = mesh._get_closest_local_cells_internal(verts)
    assert (cells == -1).sum() == 0, (
        f"default loose mode rejected {(cells == -1).sum()}/{len(verts)} vertices"
    )


def test_on_boundary_false_rejects_vertices_simplex_3d():
    """on_boundary=False reproduces strict-inside semantics — most 3D simplex vertices come back -1."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4
    )
    verts = np.asarray(mesh.X.coords)
    cells = mesh._get_closest_local_cells_internal(verts, on_boundary=False)
    assert (cells == -1).sum() > 0, (
        "expected strict mode to reject at least some boundary-vertex queries"
    )


def test_on_boundary_modes_diverge_at_face_queries():
    """Strict and loose mode must give a distinguishable result on vertex queries."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4
    )
    verts = np.asarray(mesh.X.coords)
    hint = mesh.get_closest_cells(verts)
    inside_strict = mesh._test_if_points_in_cells_internal(verts, hint, on_boundary=False)
    inside_loose = mesh._test_if_points_in_cells_internal(verts, hint, on_boundary=True)
    assert (~inside_strict).sum() > (~inside_loose).sum(), (
        f"strict-vs-loose distinction lost: strict rejected {(~inside_strict).sum()}, "
        f"loose rejected {(~inside_loose).sum()}"
    )
    assert (~inside_loose).sum() == 0, (
        f"loose mode rejected {(~inside_loose).sum()} kdtree-nearest cells of vertices"
    )


def test_default_returns_containing_cells():
    """The cell id returned by the default must have the query in its closure."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4
    )
    verts = np.asarray(mesh.X.coords)
    cells = mesh._get_closest_local_cells_internal(verts)
    assert (cells == -1).sum() == 0

    cStart, _ = mesh.dm.getHeightStratum(0)
    pStart, pEnd = mesh.dm.getDepthStratum(0)
    for v, c in zip(verts, cells):
        closure = mesh.dm.getTransitiveClosure(int(c) + cStart)[0]
        vp = closure[(closure >= pStart) & (closure < pEnd)]
        vc = mesh._coords[vp - pStart]
        assert np.linalg.norm(vc - v, axis=1).min() < 1e-10, (
            f"vertex {v} returned cell {c} whose closure does not contain it"
        )


def test_get_closest_local_cells_public_forwards_kwarg():
    """The public `get_closest_local_cells` wrapper forwards on_boundary."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
    )
    verts = np.asarray(mesh.X.coords)
    # Default (True): no vertices rejected
    cells_loose = mesh.get_closest_local_cells(verts)
    # Opt out (False): expect some vertices rejected
    cells_strict = mesh.get_closest_local_cells(verts, on_boundary=False)
    assert (cells_loose == -1).sum() == 0
    assert (cells_strict == -1).sum() > 0
