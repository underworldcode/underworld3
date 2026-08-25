"""The NETWORK mesher (:func:`place_surface.place_thin_volume`,
``mesher="network"``): fused ribbons with EMBEDDED spines.

Two properties the earlier paths had only one of each: touching
strands are resolved as one fused region (junctions free — the plain
fuse), and every polyline's interior sample points are vertices of the
mesh so a split cut walks them exactly (#595 — the sequential ladder,
which could not touch). Asserted on a KISSING pair: a main line and a
splay leaving its side at a gap below the band width.

The plain fuse is the negative control: it must NOT carry the spine
vertices, or this test is not measuring the embedding it claims to.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite")]

WIDTH = 0.02


def _box2(cell):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell,
        regular=False, qdegree=2)


def _kissing_pair(spacing=0.01):
    """A straight main line and a splay leaving its side at gap WIDTH/2
    (kissing: inside the band), both sampled at the rung scale."""
    s = np.arange(0.0, 0.5 + 1e-9, spacing)
    main = np.column_stack([0.25 + s, 0.4 + 0.3 * s])
    t = np.array([1.0, 0.3]) / np.hypot(1.0, 0.3)
    n = np.array([-t[1], t[0]])
    d = np.cos(np.deg2rad(25)) * t + np.sin(np.deg2rad(25)) * n
    start = main[len(main) // 2] + 0.5 * WIDTH * n
    u = np.arange(0.0, 0.15 + 1e-9, spacing)
    splay = start + u[:, None] * d
    return [main, splay]


def _vertex_distance(dm, points):
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    return cKDTree(X).query(points)[0]


def test_a_kissing_pair_embeds_its_spines_as_vertices():
    from underworld3.utilities.line_cut import cell_areas

    mesh = _box2(0.05)
    before = float(cell_areas(mesh.dm).sum())
    pair = _kissing_pair()
    new, info = place_thin_volume(mesh.dm, pair, width=WIDTH,
                                  label="Net", label_value=9,
                                  size=0.01, mesher="network")
    assert info["n_zone_cells"] > 0
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)
    assert info["min_angle"] > 10.0
    # every INTERIOR spine point is a mesh vertex (the end points sit on
    # the caps and are not promised)
    for P in pair:
        assert _vertex_distance(new, P[1:-1]).max() < 1e-9

    # negative control: the plain fuse places the same pair but does NOT
    # carry the spine vertices
    plain, _ = place_thin_volume(mesh.dm, pair, width=WIDTH, label="Net",
                                 label_value=9, size=0.01)
    assert _vertex_distance(plain, pair[0][1:-1]).max() > 1e-4, (
        "the plain fuse now carries the spine vertices; the control no "
        "longer distinguishes the network path")


def test_a_kissing_pair_can_be_cut_along_both_spines():
    """The point of the embedding: a split cut along each strand of a
    touching pair, one network add_fault call, on the fused mesh."""
    base = _box2(0.05)
    pair = _kissing_pair()
    dm, _ = place_thin_volume(base.dm, pair, width=WIDTH, label="Net",
                              label_value=9, size=0.01, mesher="network")
    mesh = uw.discretisation.Mesh(
        dm, simplex=True, qdegree=2,
        coordinate_system_type=base.CoordinateSystem.coordinate_type,
        boundaries=base.boundaries, verbose=False)
    # cut the interior chains (the cap points are not vertices)
    cut = mesh.add_fault([("Main", pair[0][2:-2]),
                          ("Splay", pair[1][2:-2])])
    assert cut.dm.getDepthStratum(0)[1] > mesh.dm.getDepthStratum(0)[1], (
        "the cut duplicated no vertices")


def test_an_unknown_mesher_is_refused():
    mesh = _box2(0.2)
    with pytest.raises(ValueError, match="network"):
        place_thin_volume(mesh.dm, _kissing_pair(), width=WIDTH,
                          mesher="lattice")
