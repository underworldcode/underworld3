"""Refining a spherical mesh does not fold cells (#946); a refined ball keeps finite
coordinates (#903).

gmsh meshes an OCC sphere in its (longitude, latitude) parameter plane. Near a pole its
surface triangulation can join four neighbouring wall nodes along the long diagonal
instead of the convex hull's, which creases the wall toward the centre. On the inner wall
of a shell, the pocket under such a crease lies inside the domain, and the volume mesher
fills it with a cell whose four vertices are all on the sphere. Refinement is a
subdivision and cannot fold that cell. The refinement callback then snaps the midpoint of
the crease edge outward onto the sphere, past the unlabelled midpoint of the cell's other
diagonal, and turns the children between them inside out. On a shell at cell size 0.5,
three children folded on one host and five on another, and the velocity multigrid ran to
its iteration cap on every solve.

The spherical generators now flip creased wall edges to the hull before the volume mesh.
The refinement hierarchy also puts back, where refinement placed it, any vertex whose move
by a callback inverts a cell, and warns: a callback need not come from a gmsh mesh.

Contracts, all exactness properties that hold on any host:
- the wall triangulation of a shell is the convex hull of its nodes, with no creased edge;
- no coarse cell has all four vertices on one wall;
- no refined cell is inverted: one orientation throughout;
- a callback that folds cells is held back for those vertices, with a warning, and
  leaves no inverted cell; a callback that folds nothing is applied exactly;
- a refined ball has finite coordinates, with its wall vertices on the sphere;
- every vertex a wall label carries lies on that wall: the segmented shell and ball
  tagged their volume with Upper's value, so Upper took interior vertices and the
  callback snapped them onto the sphere.
"""
import itertools
import warnings
from collections import defaultdict

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing.smoothing import graph

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

R_INNER, R_OUTER = 0.5, 1.0


def _shell(refinement):
    return uw.meshing.SphericalShell(radiusOuter=R_OUTER, radiusInner=R_INNER, cellSize=0.5,
                                     refinement=refinement, qdegree=2)


def _internal_shell(refinement):
    # cell size 0.3: the generator tells its volumes apart by radius to within half a
    # cell size, and at 0.5 that does not separate 0.75 from 1.0
    return uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=R_OUTER, radiusInternal=0.75, radiusInner=R_INNER, cellSize=0.3,
        refinement=refinement, qdegree=2)


MESHES = {"shell": (_shell, {"Lower": R_INNER, "Upper": R_OUTER}),
          "internal": (_internal_shell, {"Lower": R_INNER, "Internal": 0.75, "Upper": R_OUTER})}


def _vertices(dm, point):
    vs, ve = dm.getDepthStratum(0)
    return [q - vs for q in dm.getTransitiveClosure(point)[0] if vs <= q < ve]


def _wall_triangles(mesh, name):
    """The faces carrying boundary label ``name``, as vertex triples."""
    dm = mesh.dm
    fs, fe = dm.getHeightStratum(1)
    points = dm.getStratumIS(name, mesh.boundaries[name].value)
    if not points:
        return []
    return [_vertices(dm, f) for f in points.getIndices() if fs <= f < fe]


def _creased_edges(triangles, coords):
    """Wall edges where the two triangles fold toward the centre: the far vertex of one
    lies outside the plane of the other. None on a convex-hull triangulation."""
    opposite = defaultdict(list)
    for t in triangles:
        for a, b in itertools.combinations(t, 2):
            opposite[(min(a, b), max(a, b))].append((set(t) - {a, b}).pop())
    creased = 0
    for (a, b), far in opposite.items():
        if len(far) != 2:
            continue
        xa, xb, xc, xd = coords[[a, b, far[0], far[1]]]
        n = np.cross(xb - xa, xc - xa)
        if n @ (xa + xb + xc) < 0:
            n = -n
        if n @ (xd - xa) > 1.0e-12 * np.linalg.norm(n):
            creased += 1
    return creased


def _folded_cells(dm):
    """Cells whose signed volume, in DMPlex vertex order, has the opposite sign to the
    mesh's: cells turned inside out."""
    cells = graph._tet_cells(dm)
    if cells is None:
        return 0
    vol = graph._signed_volumes(dm.getCoordinatesLocal().array.reshape(-1, 3), cells)
    return int(np.count_nonzero(np.sign(vol) != np.sign(np.median(vol))))


@pytest.mark.parametrize("kind", MESHES)
def test_the_walls_are_convex_hull_triangulations(kind):
    build, walls = MESHES[kind]
    mesh = build(None)
    coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    for name, radius in walls.items():
        triangles = _wall_triangles(mesh, name)
        assert triangles, name
        assert _creased_edges(triangles, coords) == 0, name
        on_wall = np.isclose(np.linalg.norm(coords, axis=1), radius, atol=1.0e-9)
        cs, ce = mesh.dm.getHeightStratum(0)
        pockets = sum(on_wall[_vertices(mesh.dm, c)].all() for c in range(cs, ce))
        assert pockets == 0, name


@pytest.mark.parametrize("kind", MESHES)
def test_a_refined_shell_has_no_inverted_cell(kind):
    build, walls = MESHES[kind]
    mesh = build(1)
    assert _folded_cells(mesh.dm) == 0
    coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    for name, radius in walls.items():
        if name == "Internal":
            continue   # not snapped on refinement (a TODO in the generator)
        wall = np.unique(np.concatenate(_wall_triangles(mesh, name)))
        assert np.allclose(np.linalg.norm(coords[wall], axis=1), radius, rtol=1.0e-12), name


def _refined_box(refinement_callback):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5, regular=True)
    return uw.discretisation.Mesh(base.dm.clone(), simplex=True, boundaries=base.boundaries,
                                  refinement=1, refinement_callback=refinement_callback)


def _signed_areas(dm):
    return graph._signed_areas(dm.getCoordinatesLocal().array.reshape(-1, 2), graph._tri_cells(dm))


def test_a_callback_that_folds_cells_is_held_back():
    """The new top-wall vertex at (0.25, 1) is pushed down to y = 0.55, below the
    refined row at y = 0.75: the cells between turn inside out. That vertex stays
    where refinement put it; every other move of the callback stands."""
    def push_down(dm):
        c = dm.getCoordinatesLocal()
        x = c.array.reshape(-1, 2)
        target = np.isclose(x[:, 0], 0.25) & np.isclose(x[:, 1], 1.0)
        x[target, 1] = 0.55
        x[np.isclose(x[:, 0], 0.75) & np.isclose(x[:, 1], 1.0), 1] = 0.99
        c.array[...] = x.ravel()
        dm.setCoordinatesLocal(c)

    with pytest.warns(UserWarning, match="refinement callback"):
        mesh = _refined_box(push_down)
    area = _signed_areas(mesh.dm)
    assert np.all(np.sign(area) == np.sign(area[0]))
    x = mesh.dm.getCoordinatesLocal().array.reshape(-1, 2)
    assert np.any(np.isclose(x[:, 0], 0.25) & np.isclose(x[:, 1], 1.0))
    assert not np.any(np.isclose(x[:, 1], 0.55))
    assert np.any(np.isclose(x[:, 0], 0.75) & np.isclose(x[:, 1], 0.99))


def test_a_callback_that_folds_nothing_is_applied_exactly():
    """The annulus callback snaps every new wall vertex onto its circle; nothing folds,
    so nothing is held back and nothing is reported."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4, refinement=2)
    x = mesh.dm.getCoordinatesLocal().array.reshape(-1, 2)
    for name, radius in (("Lower", 0.5), ("Upper", 1.0)):
        points = mesh.dm.getStratumIS(name, mesh.boundaries[name].value).getIndices()
        vs, ve = mesh.dm.getDepthStratum(0)
        wall = [p - vs for p in points if vs <= p < ve]
        assert np.allclose(np.linalg.norm(x[wall], axis=1), radius, rtol=1.0e-12), name


def test_a_refined_ball_keeps_finite_coordinates():
    """A ball has no inner wall. The callback's lookup of the missing label returned
    vertex 0, which it moved to the origin and then divided by its zero radius (#903)."""
    with warnings.catch_warnings():
        # the hierarchy would hold the moved vertex back and hide the lookup's error
        warnings.simplefilter("error", UserWarning)
        mesh = uw.meshing.SphericalShell(radiusOuter=R_OUTER, radiusInner=0.0, cellSize=0.5,
                                         refinement=1, qdegree=2)
    coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    assert np.all(np.isfinite(coords))
    assert _folded_cells(mesh.dm) == 0
    wall = np.unique(np.concatenate(_wall_triangles(mesh, "Upper")))
    assert np.allclose(np.linalg.norm(coords[wall], axis=1), R_OUTER, rtol=1.0e-12)


SEGMENTED = {
    "segmented_shell": (lambda refinement: uw.meshing.SegmentedSphericalShell(
        radiusOuter=R_OUTER, radiusInner=0.547, cellSize=0.4, numSegments=6,
        refinement=refinement), {"Lower": 0.547, "Upper": R_OUTER}),
    "segmented_ball": (lambda refinement: uw.meshing.SegmentedSphericalBall(
        radius=R_OUTER, cellSize=0.4, numSegments=6, refinement=refinement), {"Upper": R_OUTER}),
}


@pytest.mark.parametrize("kind", SEGMENTED)
def test_a_wall_label_holds_only_wall_vertices(kind):
    build, walls = SEGMENTED[kind]
    mesh = build(None)
    dm = mesh.dm
    vs, ve = dm.getDepthStratum(0)
    coords = dm.getCoordinatesLocal().array.reshape(-1, 3)
    for name, radius in walls.items():
        points = dm.getStratumIS(name, mesh.boundaries[name].value).getIndices()
        wall = [p - vs for p in points if vs <= p < ve]
        assert wall, name
        assert np.allclose(np.linalg.norm(coords[wall], axis=1), radius, rtol=1.0e-9), name


@pytest.mark.parametrize("kind", SEGMENTED)
def test_a_refined_segmented_mesh_has_no_inverted_cell(kind):
    build, _ = SEGMENTED[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mesh = build(1)
    assert _folded_cells(mesh.dm) == 0
