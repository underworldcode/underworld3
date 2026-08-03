"""The standard hierarchy view: one actor per multigrid level, plus each fault.

``plot_mesh_hierarchy`` is a figure routine, so what can be asserted is what it
DREW, not what it looks like: the level count, that a fault the mesh carries
becomes its own actor, and that it degrades sensibly on a mesh with no tail and
in 3-D. Those are the ways it can silently draw the wrong thing — a hierarchy
missing a level reads as a shallower mesh, and a fault that quietly contributed
no actor reads as a mesh with no fault in it.
"""
import numpy as np
import pytest

import underworld3 as uw
import underworld3.visualisation as vis

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _actors(pl):
    return len(pl.renderer.actors)


def _plain_box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.35,
        regular=False, qdegree=2)


def _fault_mesh():
    """An adapt child with a conforming surface — a real hierarchy and a label."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 6,
        regular=False, qdegree=2, refinement=2)
    line = np.array([[-0.1, 0.37], [1.1, 0.63]])
    surf = uw.meshing.Surface("Grade", base, line)
    surf.discretize()
    child = base.adapt(surf.refinement_metric_function(
        h_near=1 / 48, h_far=1 / 6, width=1 / 12), max_levels=3)
    return child.add_conforming_surface(
        uw.meshing.Surface("Flt", child, line), snap_frac=0.30)


def test_one_actor_per_level_and_per_fault():
    mesh = _fault_mesh()
    levels = len(getattr(mesh, "_custom_mg_coarse_meshes", []) or []) + 1
    assert levels > 1, "fixture has no multigrid tail, so nothing is being tested"

    plain = vis.plot_mesh_hierarchy(mesh)
    assert _actors(plain) == levels

    withfault = vis.plot_mesh_hierarchy(mesh, faults=("Flt",))
    assert _actors(withfault) == levels + 1, (
        "the fault contributed no actor; it would be invisible in the figure")
    plain.close()
    withfault.close()


def test_facets_are_the_fault_and_cells_are_the_zone():
    """The default must draw the fault, not the zone — they are different sets.

    ``cells_supporting`` is every cell with a labelled facet, which is one
    element on EACH side, so filling it makes a one-element fault look two or
    three elements thick. The facets are the fault as the mesh represents it.
    This asserts the two really do differ, so the default cannot quietly revert
    to the fat one without failing.
    """
    mesh = _fault_mesh()
    value = int(mesh.boundaries["Flt"].value)
    n_facets = mesh.dm.getLabel("Flt").getStratumSize(value)
    zone = np.asarray(mesh.cells_supporting("Flt"))
    assert n_facets > 0 and zone.any()

    facets = vis.labelled_facets_to_pv_mesh(mesh, "Flt")
    # n_cells is the wrong counter: `pv.PolyData(points)` gives every point its
    # own vertex cell, so n_cells is n_points + the lines. Count the lines (2-D)
    # or faces (3-D) instead.
    assert facets.n_lines + facets.n_faces_strict == n_facets

    # The zone is strictly bigger: a facet has a cell on each side of it.
    assert int(zone.sum()) > n_facets, (
        "the zone is no larger than the facet chain, so this fixture cannot "
        "show that the default picks the narrower set")


def test_an_unknown_fault_style_is_refused():
    mesh = _fault_mesh()
    with pytest.raises(ValueError):
        vis.plot_mesh_hierarchy(mesh, faults=("Flt",), fault_style="zone")


def test_a_mesh_without_a_tail_is_drawn_alone():
    """No hierarchy is not an error — a base mesh is a one-level hierarchy."""
    pl = vis.plot_mesh_hierarchy(_plain_box())
    assert _actors(pl) == 1
    pl.close()


def test_a_missing_fault_label_is_skipped_not_fatal():
    mesh = _plain_box()
    pl = vis.plot_mesh_hierarchy(mesh, faults=("All_Boundaries",))
    assert _actors(pl) >= 1
    pl.close()


def test_three_dimensions_and_clipping():
    """The 3-D path: surface wireframes, and a clip that opens the model.

    Not a picture test — only that the dimension-dependent branches run and
    still produce one actor per level, since that is what will be exercised the
    moment there is a 3-D fault to look at.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4,
        regular=False, qdegree=2)
    pl = vis.plot_mesh_hierarchy(mesh)
    assert _actors(pl) == 1
    pl.close()

    clipped = vis.plot_mesh_hierarchy(
        mesh, clip=((1.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    assert _actors(clipped) == 1
    clipped.close()
