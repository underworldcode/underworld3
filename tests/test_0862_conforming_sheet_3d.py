"""Mesh-level 3-D conforming sheet (:meth:`Mesh.add_conforming_sheet`).

The 3-D twin of the 2-D contract asserted in ``test_0844_line_cut.py``:
the sheet is cut at the FINEST level only, the child inherits the
multigrid tail (the coarse levels carry neither the cut nor the label —
Galerkin coarse operators inherit the material contrast from the fine
operator), the sheet is a named boundary whose facet support is the fault
zone, and the base mesh is untouched. The placement mechanics themselves
are covered by ``test_0854_place_sheet.py``; these tests cover what the
Mesh-level wrapper ADDS — the adoption and the tail.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(cell_size=0.2, refinement=1):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2,
        refinement=refinement)


def _sheet(centre, tilt=0.25, half=0.2, n=5):
    """A structured tilted patch: the triangulation is exactly known."""
    u = np.array([1.0, 0.0, tilt])
    u = u / np.linalg.norm(u)
    v = np.array([0.0, 1.0, 0.0])
    s = np.linspace(-half, half, n)
    pts = np.array([np.asarray(centre) + a * u + b * v
                    for a in s for b in s])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    return pts, np.array(tris, dtype=np.int64)


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)


def test_the_sheet_exists_on_the_finest_level_only():
    """The stack-on invariant, unchanged from 2-D: nothing below is cut.

    The sheet's position is a design variable, so the base and the
    hierarchy resting on it stay fixed while the sheet moves. The child's
    coarse tail is the base's OWN levels, carrying neither cut nor label;
    and the cut REPLACES the base finest in the tail rather than sitting
    on top of it, since it re-represents the same grid with the sheet
    conformed and coarsens nothing.
    """
    base = _box()
    tail_before = base._coarse_level_meshes()
    counts_before = [m.dm.getHeightStratum(0)[1] - m.dm.getHeightStratum(0)[0]
                     for m in tail_before]

    pts, tris = _sheet((0.5, 0.5, 0.5))
    child = base.add_conforming_sheet(pts, tris, "Fault")

    assert child.dm.hasLabel("Fault")
    for level in child._custom_mg_coarse_meshes:
        assert not level.dm.hasLabel("Fault"), (
            "a coarse level carries the sheet; the base hierarchy must be "
            "reusable unchanged when the sheet moves")
    counts_after = [m.dm.getHeightStratum(0)[1] - m.dm.getHeightStratum(0)[0]
                    for m in base._coarse_level_meshes()]
    assert counts_after == counts_before, "a coarse level gained cells"

    assert len(child._custom_mg_coarse_meshes) == len(tail_before) - 1, (
        "the cut was kept as a level of its own; it does not coarsen the "
        "mesh it was cut from, so it should have replaced it")
    finest = child._custom_mg_coarse_meshes[-1]
    assert np.array_equal(_coords(finest.dm),
                          _coords(base.dm_hierarchy[-2])), (
        "the coarse tail's finest level is not an uncut base level")
    assert child._custom_mg_builder == base._custom_mg_builder


def test_the_sheet_becomes_a_named_boundary_and_a_zone():
    """The delivered feature: named facets a solver can resolve, and the
    fault zone as their support."""
    base = _box()
    pts, tris = _sheet((0.5, 0.5, 0.5))
    child = base.add_conforming_sheet(pts, tris, "Fault")

    assert child.parent is base
    assert "Fault" in [b.name for b in child.boundaries]
    value = child.boundaries["Fault"].value
    # The wrap COMPLETES the label with each face's closure (edges and
    # vertices), as every boundary label gets, so the stratum is larger
    # than the face count: count the height-1 points.
    fS, fE = child.dm.getHeightStratum(1)
    faces = [int(f) for f in
             child.dm.getLabel("Fault").getStratumIS(value).getIndices()
             if fS <= int(f) < fE]
    assert len(faces) == len(tris)
    # UW_Boundaries is what the solver reads when resolving by name.
    assert child.dm.getLabel("UW_Boundaries").getStratumSize(value) > 0

    zone = child.cells_supporting("Fault")
    assert zone.sum() > 0, "the fault zone is empty"
    # Each interior sheet face contributes its two support cells.
    cells = set()
    for f in faces:
        cells.update(int(c) for c in child.dm.getSupport(f))
    assert zone.sum() == len(cells)

    # Placement metadata reaches the child for downstream passes.
    info = child._surface_info
    assert info["n_surface_facets"] == len(tris)
    assert info["min_volume"] > 0.0


def test_the_base_mesh_is_not_modified():
    base = _box()
    before_cells = (base.dm.getHeightStratum(0)[1]
                    - base.dm.getHeightStratum(0)[0])
    before_coords = _coords(base.dm).copy()

    pts, tris = _sheet((0.5, 0.5, 0.5))
    base.add_conforming_sheet(pts, tris, "Fault")

    after_cells = (base.dm.getHeightStratum(0)[1]
                   - base.dm.getHeightStratum(0)[0])
    assert after_cells == before_cells
    assert np.array_equal(_coords(base.dm), before_coords)
    assert not base.dm.hasLabel("Fault")


def test_a_second_sheet_chains_and_a_duplicate_name_is_refused():
    # A finer box than the other tests: TWO stacked sheets need their
    # carve cavities clear of the walls and of each other.
    base = _box(cell_size=0.15)
    one = base.add_conforming_sheet(
        *_sheet((0.5, 0.5, 0.4), half=0.15), "Fault")
    two = one.add_conforming_sheet(
        *_sheet((0.5, 0.5, 0.65), half=0.15), "Moho")

    names = [b.name for b in two.boundaries]
    assert "Fault" in names and "Moho" in names
    assert two.dm.getLabel("Fault").getStratumSize(
        two.boundaries["Fault"].value) > 0, "the first sheet was lost"
    # Neither cut coarsens anything, so each replaced its parent in the
    # tail: two cuts on a refinement=1 base leave the base L0 alone.
    assert len(two._custom_mg_coarse_meshes) == 1

    with pytest.raises(ValueError, match="already has a boundary"):
        two.add_conforming_sheet(*_sheet((0.5, 0.5, 0.5)), "Fault")


def test_two_dimensions_are_refused():
    flat = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        regular=False, qdegree=2)
    pts, tris = _sheet((0.5, 0.5, 0.5))
    with pytest.raises(NotImplementedError, match="3-D"):
        flat.add_conforming_sheet(pts, tris, "Fault")


def test_a_solver_consumes_the_cut_mesh():
    """The point of the tail: a solve on the cut child works as it stands
    (a material contrast across the sheet, no condition on its facets)."""
    base = _box()
    pts, tris = _sheet((0.5, 0.5, 0.5))
    child = base.add_conforming_sheet(pts, tris, "Fault")
    assert child._custom_mg_coarse_meshes, "no tail to consume"

    u = uw.discretisation.MeshVariable("u_s", child, 1, degree=1)
    poisson = uw.systems.Poisson(child, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    zone = child.cells_supporting("Fault")
    k = uw.discretisation.MeshVariable("k_s", child, 1, degree=0)
    k.array[:, 0, 0] = np.where(zone, 1.0e-2, 1.0)
    poisson.constitutive_model.Parameters.diffusivity = k.sym[0]
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.solve()

    vals = np.asarray(u.data[:, 0])
    assert np.isfinite(vals).all()
    assert 0.0 <= vals.min() and vals.max() <= 1.0 + 1e-8
