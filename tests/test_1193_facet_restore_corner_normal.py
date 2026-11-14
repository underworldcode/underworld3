"""The facet restore's side test at a corner.

The restore takes the nearest point on the boundary facets and decides whether the
query is outside by the sign of (query - nearest point) along an outward normal.
The normal must be that of the facet the nearest point lies on. It was taken from
the facet whose centroid is nearest the nearest point, which is another facet where
the two walls' facets at a corner differ in length: the integration desk's review of
#1192 found 10 of 49 points beside a graded box corner left outside.

Baselines, all exact: a point beyond a straight wall of the box lands inside the box
(0 < x, y < 1); a point inside the mesh is returned unchanged, bit for bit.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _graded_box():
    """The unit box, cell 0.05, with y -> y^3: at the top right corner the right
    wall's facets are much longer than the top wall's."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.05)
    X = np.asarray(mesh.X.coords, dtype=float).copy()
    X[:, 1] = X[:, 1] ** 3
    mesh.deform(X)
    return mesh


def _corner_facet_lengths(mesh):
    fpts, _ = mesh._boundary_facet_geometry()
    right = fpts[np.all(fpts[:, :, 0] == 1.0, axis=1)]
    top = fpts[np.all(fpts[:, :, 1] == 1.0, axis=1)]
    L = np.abs(right[:, 1, 1] - right[:, 0, 1])[np.argmax(right[:, :, 1].max(axis=1))]
    l = np.abs(top[:, 1, 0] - top[:, 0, 0])[np.argmax(top[:, :, 0].max(axis=1))]
    return L, l


def test_points_beside_a_graded_corner_are_returned_inside():
    mesh = _graded_box()
    L, l = _corner_facet_lengths(mesh)
    assert L > 2 * l, f"fixture: the corner facets should differ in length, got {L} and {l}"

    # 49 points just beyond the right wall, beside its top facet
    s = np.linspace(0.02, 0.98, 49) * L
    probe = np.column_stack([np.full_like(s, 1.02), 1.0 - s])
    out = np.asarray(mesh.return_coords_to_bounds(probe.copy()))

    left_outside = (out[:, 0] > 1.0) | (out[:, 1] > 1.0)
    assert not left_outside.any(), (
        f"{int(left_outside.sum())} of 49 points beside the corner left outside the box, "
        f"at depths below the corner {s[left_outside]}")
    # each lands on the right wall it is beside, just inside (the restore's offset)
    eps = 1.0e-3 * float(mesh.get_min_radius())
    assert np.all(np.abs(out[:, 0] - (1.0 - eps)) <= 4 * np.finfo(float).eps)
    assert np.array_equal(out[:, 1], probe[:, 1])


def test_points_beyond_a_corner_are_returned_inside():
    """Beyond the corner itself the nearest point is the corner vertex, which two
    facets share. Points along each wall's line, past the corner, included."""
    mesh = _graded_box()
    a = np.linspace(0.0, 0.05, 11)
    probe = np.vstack([
        np.column_stack([1.0 + a[1:], 1.0 + a[1:]]),          # the diagonal
        np.column_stack([np.full(10, 1.0), 1.0 + a[1:]]),     # the right wall's line
        np.column_stack([1.0 + a[1:], np.full(10, 1.0)]),     # the top wall's line
        np.column_stack([1.0 + a[1:], 1.0 + 0.3 * a[1:]]),
        np.column_stack([1.0 + 0.3 * a[1:], 1.0 + a[1:]]),
    ])
    out = np.asarray(mesh.return_coords_to_bounds(probe.copy()))
    left_outside = (out[:, 0] > 1.0) | (out[:, 1] > 1.0)
    assert not left_outside.any(), (
        f"{int(left_outside.sum())} of {len(probe)} points beyond the corner left outside: "
        f"{probe[left_outside]}")


def test_points_inside_near_a_concave_corner_are_left_alone():
    """An L-shaped region (the unit box less its upper right quarter), graded so
    the facets at the reentrant corner (0.5, 0.5) differ in length. A point inside
    near that corner, whose nearest boundary point is the corner or a facet beside
    it, must not be moved."""
    pytest.importorskip("gmsh")
    import gmsh
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "l_shape.msh")
    if uw.mpi.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("L")
        pts = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
        sizes = [0.05, 0.05, 0.05, 0.01, 0.05, 0.05]
        tags = [gmsh.model.geo.addPoint(x, y, 0, h) for (x, y), h in zip(pts, sizes)]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % 6]) for i in range(6)]
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, lines, 1, name="Walls")
        gmsh.model.addPhysicalGroup(2, [surface], 2, name="Elements")
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
        gmsh.finalize()
    path = uw.mpi.comm.bcast(path, root=0)
    lshape = uw.discretisation.Mesh(path, degree=1)

    rng = np.random.default_rng(1193)
    r = 0.04 * np.sqrt(rng.uniform(0.0, 1.0, 400))
    theta = rng.uniform(np.pi, 2.5 * np.pi, 400)         # the three inside quadrants
    probe = np.column_stack([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
    inside = ~((probe[:, 0] >= 0.5) & (probe[:, 1] >= 0.5))
    probe = probe[inside]
    out = np.asarray(lshape.return_coords_to_bounds(probe.copy()))
    moved = np.any(out != probe, axis=1)
    assert not moved.any(), (
        f"{int(moved.sum())} of {len(probe)} points inside near the reentrant corner moved")


def test_points_beyond_an_edge_and_a_corner_of_a_box_are_returned_inside():
    """3-D: the unit cube, graded by y -> y^2. Beyond the edge x = y = 1 and the
    corner (1, 1, 1) the nearest point is on an edge or at a vertex that several
    facets share."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0),
                                             maxCoords=(1.0, 1.0, 1.0), cellSize=0.2)
    X = np.asarray(mesh.X.coords, dtype=float).copy()
    X[:, 1] = X[:, 1] ** 2
    mesh.deform(X)
    a = np.linspace(0.0, 0.05, 11)[1:]
    z = np.linspace(0.1, 0.9, 10)
    probe = np.vstack([
        np.column_stack([1.0 + a, 1.0 + a, z]),                  # beyond the edge
        np.column_stack([np.ones(10), 1.0 + a, z]),              # along one face's plane
        np.column_stack([1.0 + a, np.ones(10), z]),              # along the other's
        np.column_stack([1.0 + a, 1.0 + a, 1.0 + a]),            # beyond the corner
        np.column_stack([1.0 + a, np.ones(10), np.ones(10)]),
        np.column_stack([1.0 + 0.2 * a, 1.0 + a, 1.0 + 0.5 * a]),
    ])
    out = np.asarray(mesh.return_coords_to_bounds(probe.copy()))
    left_outside = np.any(out > 1.0, axis=1)
    assert not left_outside.any(), (
        f"{int(left_outside.sum())} of {len(probe)} points beyond an edge or corner left "
        f"outside or on a wall: {probe[left_outside]}")
