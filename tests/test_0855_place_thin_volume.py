"""The embedded thin-volume fault mesh (:func:`place_surface.place_thin_volume`).

The finite-width representation: patches are thickened by ±width/2 in OCC and
resolved against one another in CAD — the only kernel that can, by ``fuse``
into one region or ``fragment`` into the overlap pieces — the assembly is
meshed standalone at layer scale, and the meshed
assembly is embedded whole into the existing mesh: cavity carved, annular gap
filled by gmsh with the assembly's skin as an interior HOLE, both constraint
surfaces verbatim. Junctions need no geometric treatment: they are ordinary
cells of the union, and the rheology decides (the design ruling this exists
to serve).

What is asserted is what the construction promises: the zone's CELLS carry
the label (that is the point of a volume representation), the skin is a
separate face/edge label (a cell-bearing stratum is a volume, invisible to
the interface machinery — the skin label is what protects the zone from later
surgery), domain volume/area is conserved to round-off, sub-h widths work,
and every refusal names its cause. Absolute cell counts are never pinned —
they are gmsh-version-dependent (the test_0842 lesson).

Measured basis: ~/+Simulations/mesh_reconnection_study/thin_volume_spike.py
(widths h, h/2, h/4; junction angles to 10 degrees).
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel form is "
                                        "ptest_0855")]


def _box3(cell):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell, regular=False, qdegree=2)


def _box2(cell):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell,
        regular=False, qdegree=2)


CROSS_3D = [np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                      [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]]),
            np.array([[0.3, 0.5, 0.3], [0.7, 0.5, 0.3],
                      [0.7, 0.5, 0.7], [0.3, 0.5, 0.7]])]


def _volume(dm):
    from underworld3.utilities.place_surface import _owned_cell_volume
    return _owned_cell_volume(dm)


# ------------------------------------------------------------------------ 3-D

def test_a_crossing_pair_embeds_with_the_junction_in_the_volume():
    mesh = _box3(0.09)
    before = _volume(mesh.dm)
    new, info = place_thin_volume(mesh.dm, CROSS_3D, width=0.045,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert new.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert new.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]
    assert _volume(new) == pytest.approx(before, rel=1e-12)

    # Every zone point is a CELL and every skin point a FACE — the label
    # split that keeps the skin visible to the interface machinery.
    cS, cE = new.getHeightStratum(0)
    fS, fE = new.getHeightStratum(1)
    zone = new.getLabel("Zone").getStratumIS(5).getIndices()
    assert all(cS <= p < cE for p in zone)
    skin = new.getLabel("Zone_skin").getStratumIS(5).getIndices()
    assert all(fS <= p < fE for p in skin)
    # Skin faces are interior: two cells each.
    assert all(len(new.getSupport(int(p))) == 2 for p in skin)

    # No isolated vertex: the carve's growth can swallow the whole star of a
    # non-victim vertex, which then rides through the rebuild referenced by
    # nothing (global Euler 2 — caught by CI, gmsh-version-dependent). The
    # carve promotes such orphans to victims; this is the direct probe.
    vS, vE = new.getDepthStratum(0)
    assert all(len(new.getSupport(int(v))) > 0 for v in range(vS, vE))


def test_sub_h_width_is_supported():
    """Width below the background h is the whole point (V = 2 edot w)."""
    mesh = _box3(0.12)
    new, info = place_thin_volume(
        mesh.dm, [CROSS_3D[0]], width=0.03, label="Thin", label_value=2)
    assert info["n_zone_cells"] > 0
    assert _volume(new) == pytest.approx(_volume(mesh.dm), rel=1e-12)


def test_the_zone_survives_a_later_sheet_and_vice_versa():
    """The skin label is the guard: later surgery must hold clear of it."""
    from underworld3.utilities.place_surface import place_sheet

    # The margins are arithmetic, not luck: at cellSize 0.07 the cell
    # diameter is ~0.11, the sheet's reach 0.6*0.11 = 0.066, so a sheet at
    # z = 0.25 keeps its cavity off the floor (0.25 - 0.066 - 0.11 > 0) and
    # its victim stars off the zone's held cells (0.48 - 0.316 > 0.11).
    mesh = _box3(0.07)
    new, info = place_thin_volume(mesh.dm, [CROSS_3D[0]], width=0.04,
                                  label="Zone", label_value=5)

    # A sheet placed well away embeds; the zone comes through intact.
    n = 4
    s = np.linspace(-0.12, 0.12, n)
    pts = np.array([[0.5 + a, 0.5 + b, 0.25] for a in s for b in s])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    two, _ = place_sheet(new, pts, np.array(tris, dtype=np.int64),
                         label="Sheet", label_value=9)
    assert two.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert two.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]

    # A sheet THROUGH the zone must refuse, naming the embedded surface.
    mid = np.array([[0.5 + a, 0.5 + b, 0.5] for a in s for b in s])
    with pytest.raises(RuntimeError,
                       match="already embedded|held|hole|clearance"):
        place_sheet(two, mid, np.array(tris, dtype=np.int64), label="Bad")


def test_the_wall_is_refused_with_the_reason():
    mesh = _box3(0.12)
    low = np.array([[0.3, 0.3, 0.08], [0.7, 0.3, 0.08],
                    [0.7, 0.7, 0.08], [0.3, 0.7, 0.08]])
    with pytest.raises(RuntimeError, match="domain wall"):
        place_thin_volume(mesh.dm, [low], width=0.04)


def test_a_non_planar_patch_is_refused():
    mesh = _box3(0.2)
    warped = np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                       [0.7, 0.7, 0.6], [0.3, 0.7, 0.5]])
    with pytest.raises(RuntimeError, match="planar"):
        place_thin_volume(mesh.dm, [warped], width=0.04)


# ------------------------------------------------------------------------ 2-D

def test_a_ribbon_x_junction_embeds_in_two_dimensions():
    from underworld3.utilities.line_cut import cell_areas

    mesh = _box2(0.05)
    before = float(cell_areas(mesh.dm).sum())
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    l2 = np.array([[0.3, 0.65], [0.7, 0.35]])
    new, info = place_thin_volume(mesh.dm, [l1, l2], width=0.02,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert new.getLabel("Zone").getStratumSize(5) == info["n_zone_cells"]
    assert new.getLabel("Zone_skin").getStratumSize(5) == info["n_skin_faces"]
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)
    assert info["min_angle"] > 10.0


def test_a_kinked_ribbon_does_not_sliver():
    """The mitre-join lesson: per-segment quads fragmented together sliver at
    every kink (the inner-side overlap lens meshes at ~2 degrees, measured);
    one mitred outline per polyline has no internal seam to lens."""
    mesh = _box2(0.05)
    kinked = np.array([[0.15, 0.15], [0.4, 0.18], [0.6, 0.12]])
    new, info = place_thin_volume(mesh.dm, [kinked], width=0.02,
                                  label="Kink", label_value=3)
    assert info["min_angle"] > 10.0


def test_a_tangential_merge_fuses_instead_of_slivering():
    """The sole: a ribbon converging onto another until the two coincide.

    Where two zones overlap, ``fragment`` cuts along the boundary of the
    overlap, and for a tangential merge that boundary is a lens whose tips
    close at the convergence angle — the mesh must resolve a chain of
    slivers, and does (measured: 22 cells under 5 degrees, one of them
    degenerate). ``fuse`` returns the union as one face with no such
    boundary. The zone carries one label either way, so nothing downstream
    can tell the difference except the conditioning.

    The ``fragment`` branch is the negative control: it must show the
    slivers, or this test is not measuring what it claims to.
    """
    from underworld3.utilities.line_cut import min_angles

    sole = [np.array([[0.20, 0.50], [0.80, 0.50]]),
            np.array([[0.20, 0.56], [0.50, 0.505], [0.80, 0.50]])]

    mesh = _box2(0.05)
    fused, _ = place_thin_volume(mesh.dm, sole, width=0.02, label="Sole",
                                 label_value=7)
    assert float(min_angles(fused).min()) > 5.0

    torn, _ = place_thin_volume(mesh.dm, sole, width=0.02, label="Sole",
                                label_value=7, assembly="fragment")
    assert int((min_angles(torn) < 5.0).sum()) > 5, (
        "the fragmented merge did not sliver; the geometry no longer "
        "exercises the defect this test exists for")


def test_an_unknown_assembly_boolean_is_refused():
    mesh = _box2(0.2)
    with pytest.raises(ValueError, match="fuse.*fragment"):
        place_thin_volume(mesh.dm, [np.array([[0.3, 0.5], [0.7, 0.5]])],
                          width=0.02, assembly="union")


def test_a_second_zone_leaves_the_first_intact():
    mesh = _box2(0.05)
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    one, info1 = place_thin_volume(mesh.dm, [l1], width=0.02,
                                   label="Z1", label_value=1)
    l2 = np.array([[0.2, 0.8], [0.8, 0.8]])
    two, _info2 = place_thin_volume(one, [l2], width=0.02,
                                    label="Z2", label_value=2)
    assert two.getLabel("Z1").getStratumSize(1) == info1["n_zone_cells"]
    assert two.getLabel("Z1_skin").getStratumSize(1) == info1["n_skin_faces"]


def test_a_sharp_kink_is_refused_with_the_reason():
    mesh = _box2(0.1)
    hairpin = np.array([[0.3, 0.5], [0.6, 0.5], [0.32, 0.55]])
    with pytest.raises((RuntimeError, ValueError), match="mitre|sharply"):
        place_thin_volume(mesh.dm, [hairpin], width=0.02)


def test_zero_width_is_refused():
    mesh = _box2(0.2)
    with pytest.raises(ValueError, match="width"):
        place_thin_volume(mesh.dm, [np.array([[0.3, 0.5], [0.7, 0.5]])],
                          width=0.0)


def test_finite_elements_are_exact_on_the_embedded_mesh():
    """P2 Poisson reproduces a quadratic through the embedded zone exactly.

    The same oracle as the sheet's (test_0854): the embed shares the sewing
    skeleton whose mixed-handedness and seam-cone defects (#518 review
    finding 1, issue #520) were invisible to every topological gate. The
    layer's diffusivity is 1 like the background, so the exact solution
    crosses the zone untouched and any assembly disorder shows as error.
    """
    import sympy

    base = _box3(0.11)
    bounds = base._boundaries_with("Zone")
    new, _ = place_thin_volume(base.dm, CROSS_3D, width=0.045,
                               label="Zone",
                               label_value=bounds["Zone"].value)
    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y, z = mesh.X
    exact = x**2 + y**2 + z**2
    t = uw.discretisation.MeshVariable("T_fe_zone", mesh, 1, degree=2)
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
    assert float(err.max()) < 1e-8, (
        f"the embedded mesh assembles a wrong operator: max |u - exact| = "
        f"{float(err.max()):.3e}")


def test_an_outcropping_zone_leaves_a_band_on_the_surface():
    """The zone form of the outcrop: a fault ZONE meeting the top surface.

    Specify-long: the patch protrudes past the box and the assembly is
    clipped in OCC; its clipped face is the BAND — boundary faces carrying
    BOTH the zone's skin label and the wall's labels — and the cap over the
    bowl is an annulus (2-D fill with the band outline as its hole). The
    P2 oracle with Dirichlet on the remeshed Top is the whole-chain check.
    """
    import sympy

    base = _box3(0.12)
    bounds = base._boundaries_with("Zone")
    patch = np.array([[0.3, 0.45, 1.1], [0.7, 0.45, 1.1],
                      [0.72, 0.55, 0.55], [0.32, 0.55, 0.55]])
    new, info = place_thin_volume(base.dm, [patch], width=0.05,
                                  label="Zone",
                                  label_value=bounds["Zone"].value)
    assert info["n_zone_cells"] > 0
    zv = bounds["Zone"].value

    # Band faces: boundary (support 1), on the top plane, carrying BOTH the
    # skin label and the Top label. And every top-plane boundary face is
    # labelled Top — the cap was relabelled, not stripped.
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array
                    ).reshape(-1, 3)[: vE - vS]
    top_label = new.getLabel("Top")
    skin_label = new.getLabel("Zone_skin")
    trace_label = new.getLabel("Zone_trace")
    tv = bounds["Top"].value
    n_band = n_trace = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(Xn[v][2] == 1.0 for v in verts):
            assert top_label.getValue(f) == tv
            if skin_label.getValue(f) == zv:
                n_band += 1
            if trace_label.getValue(f) == zv:
                n_trace += 1
    assert n_band > 0, "the zone left no band on the surface"
    assert n_trace == n_band, (
        "the trace label does not coincide with the band")
    assert info["n_trace_facets"] == n_trace

    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y, z = mesh.X
    exact = x**2 + y**2 + z**2
    t = uw.discretisation.MeshVariable("T_band", mesh, 1, degree=2)
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
    assert float(err.max()) < 1e-8


# ------------------------------------------------------------ 2-D outcrop

def _top_edge_census_2d(new, skin_value):
    """Boundary edges in the top wall line: (total, band, missing Top,
    trace).

    The band is an edge carrying BOTH the zone's skin label and the wall's;
    an edge missing the Top label is a hole the relabel left in the wall.
    The trace label marks the intersection itself and must coincide with
    the band.
    """
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array).reshape(-1, 2)[: vE - vS]
    top = new.getLabel("Top")
    skin = new.getLabel("Zone_skin")
    trace = new.getLabel("Zone_trace")
    n_top = n_band = n_bare = n_trace = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(Xn[v][1] == 1.0 for v in verts):
            n_top += 1
            if top.getValue(f) < 0:
                n_bare += 1
            if skin.getValue(f) == skin_value:
                n_band += 1
            if trace is not None and trace.getValue(f) == skin_value:
                n_trace += 1
    return n_top, n_band, n_bare, n_trace


def test_an_outcropping_ribbon_leaves_a_band_on_the_surface():
    """The 2-D zone outcrop: a ribbon specified past the top wall embeds.

    Specify-long: the polyline protrudes, the assembly is clipped in OCC,
    and the clipped edge is the BAND — boundary edges carrying both the
    skin label and the wall's. The 2-D cap is the two splice segments, so
    every top-line boundary edge must still be labelled Top (relabelled,
    not stripped) and the domain area is conserved.

    The negative control comes first: the identical census on an INTERIOR
    twin counts no band, so the band counted here is produced by the
    outcrop, not by a probe that fires on anything.
    """
    import sympy
    from underworld3.utilities.line_cut import cell_areas

    base = _box2(0.05)
    bounds = base._boundaries_with("Zone")
    zv = bounds["Zone"].value

    interior, info0 = place_thin_volume(
        base.dm, [np.array([[0.35, 0.40], [0.60, 0.80]])], width=0.03,
        label="Zone", label_value=zv)
    n_top0, n_band0, _bare0, n_trace0 = _top_edge_census_2d(interior, zv)
    assert n_top0 > 0 and n_band0 == 0, (
        "the census counted a band on an interior ribbon; it cannot "
        "validate the outcrop")
    assert n_trace0 == 0 and info0["n_trace_facets"] == 0, (
        "an interior ribbon carries a trace; the trace label cannot "
        "validate the outcrop")

    before = float(cell_areas(base.dm).sum())
    line = np.array([[0.35, 0.40], [0.60, 1.10]])   # past the top wall
    new, info = place_thin_volume(base.dm, [line], width=0.03,
                                  label="Zone", label_value=zv)
    assert info["n_zone_cells"] > 0
    n_top, n_band, n_bare, n_trace = _top_edge_census_2d(new, zv)
    assert n_band > 0, "the ribbon left no band on the surface"
    assert n_bare == 0, "the relabel left top-wall edges without Top"
    assert n_trace == n_band, (
        "the trace label does not coincide with the band")
    assert info["n_trace_facets"] == n_trace
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)

    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y = mesh.X
    exact = x**2 + y**2
    t = uw.discretisation.MeshVariable("T_band2", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -4.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0]) - (X[:, 0]**2 + X[:, 1]**2))
    assert float(err.max()) < 1e-8, (
        f"the outcropped mesh assembles a wrong operator: max |u - exact| "
        f"= {float(err.max()):.3e}")


def _annulus_boundary_census(new, zone_value):
    """Boundary edges of the annulus: (total, trace, missing a wall label).

    An edge missing both Lower and Upper is a hole the relabel left in the
    boundary; the trace marks the outcrop band.
    """
    fS, fE = new.getHeightStratum(1)
    lower = new.getLabel("Lower")
    upper = new.getLabel("Upper")
    trace = new.getLabel("Zone_trace")
    n_bound = n_trace = n_bare = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        n_bound += 1
        if lower.getValue(f) < 0 and upper.getValue(f) < 0:
            n_bare += 1
        if trace.getValue(f) == zone_value:
            n_trace += 1
    return n_bound, n_trace, n_bare


def test_an_outcropping_ribbon_on_the_annulus_leaves_a_trace():
    """The general boundary: a radial ribbon out of the annulus's outer
    boundary. There is no circle the annulus is failing to be — the clip
    must land on the mesh's own chords, so the domain area is conserved to
    round-off, the trace edges carry the Upper label as well as the trace,
    and the P2 oracle stays exact through the zone.

    The negative control comes first: an interior twin of the same ribbon
    carries no trace, so the trace counted on the outcrop is produced by
    the outcrop.
    """
    import sympy
    from underworld3.utilities.line_cut import cell_areas

    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.06, qdegree=3)
    bounds = base._boundaries_with("Zone")
    zv = bounds["Zone"].value
    theta = 0.4
    ray = np.array([np.cos(theta), np.sin(theta)])

    interior, info0 = place_thin_volume(
        base.dm, [np.array([0.62 * ray, 0.88 * ray])], width=0.03,
        label="Zone", label_value=zv)
    _nb0, n_trace0, _bare0 = _annulus_boundary_census(interior, zv)
    assert info0["n_trace_facets"] == 0 and n_trace0 == 0, (
        "an interior ribbon carries a trace; the census cannot validate "
        "the outcrop")

    before = float(cell_areas(base.dm).sum())
    line = np.array([0.62 * ray, 1.30 * ray])       # past the outer boundary
    new, info = place_thin_volume(base.dm, [line], width=0.03,
                                  label="Zone", label_value=zv)
    assert info["n_zone_cells"] > 0
    assert info["n_trace_facets"] > 0, "the ribbon left no trace"
    n_bound, n_trace, n_bare = _annulus_boundary_census(new, zv)
    assert n_trace == info["n_trace_facets"]
    assert n_bare == 0, "the relabel left boundary edges without a label"
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)
    # A clipped corner landing near an imprinted domain vertex once meshed
    # a 0.18-degree sliver; the imprint collapse restores the interior
    # twin's quality (measured 29.4 against 29.6 degrees). The floor is
    # far above any sliver and far below the healthy fill.
    assert info["min_angle"] > 15.0

    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y = mesh.X
    exact = x**2 + y**2
    t = uw.discretisation.MeshVariable("T_ann", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -4.0
    for wall in ("Lower", "Upper"):
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0]) - (X[:, 0]**2 + X[:, 1]**2))
    assert float(err.max()) < 1e-8, (
        f"the outcropped annulus assembles a wrong operator: "
        f"max |u - exact| = {float(err.max()):.3e}")


def test_a_ribbon_out_through_a_box_corner_embeds():
    """A single ribbon exiting diagonally through the (1,1) corner.

    The wall-code frame used to refuse this as \"more than one domain
    wall\"; the general carve/sew subsumes it. The corner vertex lies ON
    the band, so it is deletable and the band re-provides its exact
    position — one contiguous band across the corner, area conserved,
    both walls' labels restored. Two SEPARATE bands stay refused
    (the two-ribbon and arch tests below).
    """
    from underworld3.utilities.line_cut import cell_areas

    base = _box2(0.05)
    bounds = base._boundaries_with("Zone")
    zv = bounds["Zone"].value
    before = float(cell_areas(base.dm).sum())
    corner = np.array([[0.6, 0.6], [1.1, 1.1]])
    new, info = place_thin_volume(base.dm, [corner], width=0.03,
                                  label="Zone", label_value=zv)
    assert info["n_zone_cells"] > 0
    assert info["n_trace_facets"] > 0, "the ribbon left no trace"
    assert info["min_angle"] > 15.0
    assert float(cell_areas(new).sum()) == pytest.approx(before, rel=1e-12)

    # The trace edges carry a wall label each — Top or Right — and no
    # boundary edge in either wall line is left bare.
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array).reshape(-1, 2)[: vE - vS]
    top = new.getLabel("Top")
    right = new.getLabel("Right")
    trace = new.getLabel("Zone_trace")
    n_trace = n_bare = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        on_top = all(Xn[v][1] == 1.0 for v in verts)
        on_right = all(Xn[v][0] == 1.0 for v in verts)
        if not (on_top or on_right):
            continue
        if top.getValue(f) < 0 and right.getValue(f) < 0:
            n_bare += 1
        if trace.getValue(f) == zv:
            n_trace += 1
    assert n_trace == info["n_trace_facets"]
    assert n_bare == 0, "the relabel left a wall edge without its label"


def test_a_ribbon_stopping_short_of_the_wall_still_refuses():
    """No band, no outcrop: the interior contract keeps its refusal.

    A ribbon ending just inside the wall has no clipped edge, so the carve
    must still refuse when its cavity reaches the wall — the outcrop path
    must not have widened what interior ribbons may do.
    """
    mesh = _box2(0.05)
    short = np.array([[0.35, 0.40], [0.60, 0.98]])
    with pytest.raises(RuntimeError, match="domain wall"):
        place_thin_volume(mesh.dm, [short], width=0.03, label="Zone")


def test_an_outcrop_through_two_walls_is_refused():
    mesh = _box2(0.05)
    out_top = np.array([[0.30, 0.50], [0.30, 1.10]])
    out_right = np.array([[0.70, 0.50], [1.10, 0.50]])
    with pytest.raises(NotImplementedError, match="more than one"):
        place_thin_volume(mesh.dm, [out_top, out_right], width=0.03,
                          label="Zone")


def test_a_ribbon_meeting_the_wall_in_two_bands_is_refused():
    """An arch out the top twice: one skin loop, two bands — refused."""
    mesh = _box2(0.05)
    arch = np.array([[0.30, 1.10], [0.35, 0.50], [0.65, 0.50],
                     [0.70, 1.10]])
    with pytest.raises(NotImplementedError, match="more than one band"):
        place_thin_volume(mesh.dm, [arch], width=0.03, label="Zone")
