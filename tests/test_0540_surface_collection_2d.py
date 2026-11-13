"""A SurfaceCollection works in 2-D: nearest-trace normals and distances (#540).

In 2-D a ``Surface`` is a polyline. The collection's nearest-surface director
failed there in three places: ``Surface.face_normals`` raised KeyError (pyvista
has no cell normals for lines), ``transfer_normals`` made a 3-component variable
(which a 2-D mesh refuses), and ``compute_distance_field`` handed 2-D
coordinates to pyvista, which wants 3-D. Checked against an analytic oracle:
two straight traces whose normals differ materially, each probe point taking
its own trace's normal, and the distance to the nearer segment.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(120)]

TRACES = {"f1": np.array([[0.2, 0.2], [0.4, 0.8]]),
          "f2": np.array([[0.6, 0.2], [0.9, 0.7]])}


def _unit_normal(seg):
    t = seg[1] - seg[0]
    n = np.array([-t[1], t[0]])
    return n / np.linalg.norm(n)


def _segment_distance(X, seg):
    a, b = seg
    t = np.clip(((X - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(X - (a + t[:, None] * (b - a)), axis=1)


@pytest.fixture(scope="module")
def mesh_and_collection():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.05)
    collection = uw.meshing.SurfaceCollection()
    for name, seg in TRACES.items():
        surface = uw.meshing.Surface(f"s0540_{name}", mesh,
                                     np.column_stack([seg, np.zeros(2)]))
        surface.discretize()
        collection.add(surface)
    return mesh, collection


def test_face_normals_of_a_polyline(mesh_and_collection):
    _, collection = mesh_and_collection
    for name, surface in collection.surfaces.items():
        normals = np.asarray(surface.face_normals)
        assert normals.ndim == 2 and len(normals) > 0
        along = np.abs(normals[:, :2] @ _unit_normal(TRACES[name.split("_")[-1]]))
        np.testing.assert_allclose(along, 1.0, atol=1e-12)


def test_transferred_normals_follow_the_nearer_trace(mesh_and_collection):
    mesh, collection = mesh_and_collection
    var = collection.transfer_normals(mesh, variable_name="n0540_nodes")
    X = np.asarray(var.coords)
    d1, d2 = (_segment_distance(X, TRACES[k]) for k in ("f1", "f2"))
    clear = np.abs(d1 - d2) > 0.05                          # not near the bisector
    expect = np.where((d1 < d2)[:, None], _unit_normal(TRACES["f1"]), _unit_normal(TRACES["f2"]))
    along = np.abs(np.sum(np.asarray(var.data)[:, :2] * expect, axis=1))
    np.testing.assert_allclose(along[clear], 1.0, atol=1e-12)


def test_distance_field_is_the_distance_to_the_nearer_trace(mesh_and_collection):
    mesh, collection = mesh_and_collection
    var = collection.compute_distance_field(mesh, variable_name="d0540")
    X = np.asarray(var.coords)
    expect = np.minimum(*(_segment_distance(X, TRACES[k]) for k in ("f1", "f2")))
    np.testing.assert_allclose(np.asarray(var.data)[:, 0], expect, rtol=0, atol=1e-10)
