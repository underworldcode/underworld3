"""The removal primitive (:func:`place_surface.remove_embedded`) — the other
half of the fault lifecycle.

The design ruling this serves: a fault is ADDED to the running mesh and a
spent one DELETED, with no redistribution ever — the same gather-first
surgery as the placements, so the rest of the mesh never moves. Removal is
the label-seeded carve (the object's cells, or the region around its faces)
followed by a PLAIN fill at the background scale, which erases the
layer-scale grading; the object's labels vanish with their points.

What is asserted: the labels are globally empty afterwards (the point of the
operation), the domain volume is conserved to round-off, every OTHER
embedded surface comes through intact, the mesh stays FE-exact (the P2
quadratic oracle — the probe class every topological gate is blind to), and
refusals name their cause. Cell counts after a refill are NOT pinned — the
fill is gmsh's and partition-sensitive in ordering; the invariants are what
the contract promises.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.place_surface import (place_sheet,
                                                 place_thin_volume,
                                                 remove_embedded)

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel form is "
                                        "ptest_0856")]


def _box(cell=0.11):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell, regular=False, qdegree=2)


def _sheet(centre, half=0.15, n=4):
    u = np.array([1.0, 0.0, 0.25])
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


PATCH = np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                  [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]])


def _volume_of(dm):
    from underworld3.utilities.place_surface import _owned_cell_volume
    return _owned_cell_volume(dm)


def _fe_exact(dm, base, bounds, tag):
    mesh = uw.discretisation.Mesh(
        dm.clone(), simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y, z = mesh.X
    exact = x**2 + y**2 + z**2
    t = uw.discretisation.MeshVariable(f"T_rm_{tag}", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -6.0
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0])
                 - (X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2))
    return float(err.max())


def test_a_zone_is_removed_and_the_mesh_returns_to_health():
    base = _box()
    bounds = base._boundaries_with("Zone")
    before = _volume_of(base.dm)
    zoned, info = place_thin_volume(base.dm, [PATCH], width=0.045,
                                    label="Zone",
                                    label_value=bounds["Zone"].value)
    assert zoned.getLabel("Zone").getStratumSize(bounds["Zone"].value) > 0

    cleared, rinfo = remove_embedded(zoned, "Zone",
                                     label_value=bounds["Zone"].value)
    assert rinfo["n_removed_cells"] >= info["n_zone_cells"]
    for name in ("Zone", "Zone_skin"):
        if cleared.hasLabel(name):
            assert cleared.getLabel(name).getStratumSize(
                bounds["Zone"].value) == 0
    assert _volume_of(cleared) == pytest.approx(before, rel=1e-12)
    assert _fe_exact(cleared, base, bounds, "zone") < 1e-8


def test_a_sheet_is_removed_the_same_way():
    base = _box()
    bounds = base._boundaries_with("Rupture")
    pts, tris = _sheet((0.5, 0.5, 0.5))
    placed, _ = place_sheet(base.dm, pts, tris, label="Rupture",
                            label_value=bounds["Rupture"].value)
    cleared, rinfo = remove_embedded(placed, "Rupture",
                                     label_value=bounds["Rupture"].value)
    assert rinfo["n_removed_cells"] > 0
    assert cleared.getLabel("Rupture").getStratumSize(
        bounds["Rupture"].value) == 0
    assert _volume_of(cleared) == pytest.approx(_volume_of(base.dm),
                                                rel=1e-12)


def test_removal_leaves_an_unrelated_surface_intact():
    # cellSize 0.09 gives the margins room: the sheet's carve (victims,
    # stars, and the centroid rule) stays off the floor below and off the
    # zone's held cells above.
    base = _box(0.09)
    bounds = base._boundaries_with("Zone")
    zoned, zinfo = place_thin_volume(base.dm, [PATCH], width=0.04,
                                     label="Zone",
                                     label_value=bounds["Zone"].value)
    pts, tris = _sheet((0.5, 0.5, 0.3), half=0.12)
    both, sinfo = place_sheet(zoned, pts, tris, label="Keep", label_value=9)

    cleared, _ = remove_embedded(both, "Zone",
                                 label_value=bounds["Zone"].value)
    assert cleared.getLabel("Keep").getStratumSize(9) == \
        sinfo["n_surface_facets"]
    assert cleared.getLabel("Zone").getStratumSize(
        bounds["Zone"].value) == 0


def test_removing_nothing_is_refused_with_the_reason():
    base = _box(0.2)
    with pytest.raises((ValueError, RuntimeError),
                       match="nothing is embedded"):
        remove_embedded(base.dm, "Ghost", label_value=3)


def test_add_remove_add_composes():
    """The lifecycle in miniature: a new object goes where an old one was."""
    base = _box()
    bounds = base._boundaries_with("Zone")
    v0 = _volume_of(base.dm)
    zoned, _ = place_thin_volume(base.dm, [PATCH], width=0.045,
                                 label="Zone",
                                 label_value=bounds["Zone"].value)
    cleared, _ = remove_embedded(zoned, "Zone",
                                 label_value=bounds["Zone"].value)
    # The SAME region accepts a new object after the refill.
    pts, tris = _sheet((0.5, 0.5, 0.5))
    again, info = place_sheet(cleared, pts, tris, label="Zone",
                              label_value=bounds["Zone"].value)
    assert info["n_surface_facets"] == len(tris)
    assert _volume_of(again) == pytest.approx(v0, rel=1e-12)
    assert _fe_exact(again, base, bounds, "again") < 1e-8


def test_the_two_dimensional_lifecycle_composes():
    """Zone -> remove -> line -> remove, FE-exact at every stage (2-D)."""
    from underworld3.utilities.place_surface import place_along_lines
    from underworld3.utilities.line_cut import cell_areas

    base2 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05,
        regular=False, qdegree=2)
    bounds = base2._boundaries_with("Zone")

    def fe2(dm, tag):
        mesh = uw.discretisation.Mesh(
            dm.clone(), simplex=True, qdegree=3, boundaries=bounds,
            coordinate_system_type=base2.CoordinateSystem.coordinate_type)
        x, y = mesh.X
        exact = x**2 + y**2
        t = uw.discretisation.MeshVariable(f"T2d_{tag}", mesh, 1, degree=2)
        poisson = uw.systems.Poisson(mesh, u_Field=t)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = -4.0
        for wall in ("Bottom", "Top", "Left", "Right"):
            poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
        poisson.tolerance = 1e-11
        poisson.solve()
        X = np.asarray(t.coords)
        return float(np.abs(np.asarray(t.data[:, 0])
                            - (X[:, 0]**2 + X[:, 1]**2)).max())

    a0 = float(cell_areas(base2.dm).sum())
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    l2 = np.array([[0.3, 0.65], [0.7, 0.35]])
    zoned, _ = place_thin_volume(base2.dm, [l1, l2], width=0.02,
                                 label="Zone",
                                 label_value=bounds["Zone"].value)
    assert fe2(zoned, "a") < 1e-8
    cleared, _ = remove_embedded(zoned, "Zone",
                                 label_value=bounds["Zone"].value)
    assert cleared.getLabel("Zone").getStratumSize(
        bounds["Zone"].value) == 0
    assert fe2(cleared, "b") < 1e-8
    line = np.array([[0.35, 0.45], [0.65, 0.55]])
    again, _ = place_along_lines(cleared, [line], label="Zone",
                                 label_value=bounds["Zone"].value)
    assert fe2(again, "c") < 1e-8
    gone, _ = remove_embedded(again, "Zone",
                              label_value=bounds["Zone"].value)
    assert fe2(gone, "d") < 1e-8
    assert float(cell_areas(gone).sum()) == pytest.approx(a0, rel=1e-12)
