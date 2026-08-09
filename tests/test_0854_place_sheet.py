"""The 3-D placed sheet (:func:`underworld3.utilities.place_surface.place_sheet`).

The 3-D form of the placed surface: the sheet's points become mesh vertices
and every sheet triangle becomes a labelled interior face, with the rim free
inside the mesh — the case a cut cannot represent and a two-sub-volume trick
cannot fake. The cavity fill is delegated to gmsh and GATED (constraint
surfaces bit-identical, conformity, Euler, no inversion); these tests assert
the contract the fault module consumes — a labelled facet patch — plus the
guards the 2-D work paid for: earlier surfaces survive placement, and their
survival is read off the RESULT mesh, never off accumulated bookkeeping.

Measured basis: ``~/+Simulations/mesh_reconnection_study/sheet_cavity_spike.py``
(6/6 across tilt, resolution, clearance and sub-h spacing).
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_along_lines, place_sheet

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(cell_size=0.11):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)


def _sheet(centre, tilt=0.25, half=0.2, n=5):
    """A structured tilted patch: the triangulation is exactly known."""
    u = np.array([1.0, 0.0, tilt])
    u = u / np.linalg.norm(u)
    v = np.array([0.0, 1.0, 0.0])
    s = np.linspace(-half, half, n)
    pts = np.array([np.asarray(centre) + a * u + b * v for a in s for b in s])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    return pts, np.array(tris, dtype=np.int64)


def _face_vertex_sets(dm, name, value):
    vS, _vE = dm.getDepthStratum(0)
    out = []
    for f in dm.getLabel(name).getStratumIS(value).getIndices():
        out.append(tuple(sorted(
            int(p) - vS for p in dm.getTransitiveClosure(int(f))[0]
            if dm.getPointDepth(int(p)) == 0)))
    return out


def test_the_sheet_becomes_a_labelled_interior_manifold_patch():
    """The contract the fault module consumes: a facet patch under one name.

    Every sheet triangle a face, every one interior (two cells — the split's
    precondition), and the patch an open manifold: each interior edge in two
    patch faces, the rim's edges in one. That last is what "the rim is free
    inside the mesh" means topologically, and it is the property a fan-walked
    2-D tip has at a point, here along a whole curve.
    """
    from collections import Counter

    mesh = _box()
    pts, tris = _sheet((0.5, 0.5, 0.5))
    dm, info = place_sheet(mesh.dm, pts, tris, label="Fault", label_value=44)

    assert info["n_surface_facets"] == len(tris)
    assert dm.getLabel("Fault").getStratumSize(44) == len(tris)

    fS, fE = dm.getHeightStratum(1)
    for f in dm.getLabel("Fault").getStratumIS(44).getIndices():
        assert fS <= int(f) < fE
        assert len(dm.getSupport(int(f))) == 2, "a sheet face is not interior"

    edges = Counter()
    for verts in _face_vertex_sets(dm, "Fault", 44):
        a, b, c = verts
        for e in ((a, b), (a, c), (b, c)):
            edges[e] += 1
    assert set(edges.values()) == {1, 2}, "the patch is not an open manifold"
    assert sum(1 for k in edges.values() if k == 1) > 0, "the rim vanished"


def test_placed_points_are_vertices_exactly():
    mesh = _box()
    pts, tris = _sheet((0.5, 0.5, 0.5), tilt=0.4)
    dm, _info = place_sheet(mesh.dm, pts, tris, label="Fault", label_value=44)

    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(pts)
    assert d.max() == 0.0, "a sheet point is not a vertex of the result"


def test_walls_survive_and_the_base_is_untouched():
    mesh = _box()
    walls = {b.name: mesh.dm.getLabel(b.name).getStratumSize(b.value)
             for b in mesh.boundaries if mesh.dm.hasLabel(b.name)}
    cells_before = mesh.dm.getHeightStratum(0)[1]
    coords_before = np.array(mesh.dm.getCoordinatesLocal().array)

    pts, tris = _sheet((0.5, 0.5, 0.5))
    dm, _info = place_sheet(mesh.dm, pts, tris, label="Fault", label_value=44)

    for name, before in walls.items():
        after = dm.getLabel(name).getStratumSize(mesh.boundaries[name].value)
        assert after == before, f"wall {name}: {before} -> {after}"
    assert mesh.dm.getHeightStratum(0)[1] == cells_before
    assert np.array_equal(np.array(mesh.dm.getCoordinatesLocal().array),
                          coords_before)


def test_a_second_sheet_cannot_break_the_first():
    """The 2-D lesson, ported: counts come off the RESULT mesh.

    A well-separated second sheet places and the first keeps every face; a
    second sheet needing the first's cells is refused, never silently
    accommodated at the first's expense.
    """
    mesh = _box()
    pts1, tris1 = _sheet((0.5, 0.5, 0.35), tilt=0.0)
    one, info1 = place_sheet(mesh.dm, pts1, tris1, label="A", label_value=41)

    pts2, tris2 = _sheet((0.5, 0.5, 0.72), tilt=0.0)
    two, _info2 = place_sheet(one, pts2, tris2, label="B", label_value=42)
    assert two.getLabel("A").getStratumSize(41) == info1["n_surface_facets"], (
        "the second sheet's cavity consumed faces of the first")
    assert two.getLabel("B").getStratumSize(42) == len(tris2)

    # A third sheet inside A's held band must be REFUSED — by whichever guard
    # trips first (the held-cell check, or the shell-manifold check that the
    # held cells make fail). Which one fires depends on the geometry; that it
    # fires, and that A is intact afterwards, is the contract.
    pts3, tris3 = _sheet((0.5, 0.5, 0.38), tilt=0.0)
    with pytest.raises(RuntimeError,
                       match="already embedded|not a closed manifold"):
        place_sheet(one, pts3, tris3, label="C", label_value=43)
    assert one.getLabel("A").getStratumSize(41) == info1["n_surface_facets"]


def test_dimension_dispatch_is_refused_both_ways():
    mesh3 = _box(0.3)
    mesh2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
        regular=False, qdegree=2)
    pts, tris = _sheet((0.5, 0.5, 0.5))
    with pytest.raises(NotImplementedError, match="place_along_lines"):
        place_sheet(mesh2.dm, pts, tris)
    with pytest.raises(NotImplementedError, match="cavity of a placed sheet"):
        place_along_lines(mesh3.dm, [np.array([[0.2, 0.5], [0.8, 0.5]])])
