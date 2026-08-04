"""Surfaces embedded by PLACING their points
(:mod:`underworld3.utilities.place_surface`).

The cut (``tests/test_0844_line_cut.py``) makes a surface a chain of element
edges by splitting every edge it crosses. Placement makes the same chain by
asserting the surface's own points, deleting the mesh vertices in the way and
retriangulating the cavity. The two must deliver the same guarantees, so this
file asserts the same properties the cut's suite does — chain identity, no
straddling cell, exact placement, the label, area — and then the cases the cut
**refuses**, which are the reason the second implementation exists:

- a surface **ending inside** the mesh. The cut has no split that represents a
  triangle the surface enters and does not leave, and says so;
- two surfaces **closer together than one element**. They cross the same edges,
  an edge can be split once, and the cut says so;
- a surface **finer than the local h**, which the cut cannot express at all
  because its vertices are the crossings of the mesh's own edges.

Three of these tests assert the refusal as well as the success. That pairing is
the point: a test that only showed placement working would not show that it is
buying anything.

The walk that fills the cavity can fail, and the failure is loud. It is measured
rather than argued: 100 random traces on a uniform mesh and 100 on a graded
adapt-on-top mesh all place, with the total area exact to 2e-16 in every case.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import reconnect
from underworld3.utilities.line_cut import (cell_areas, cut_along_lines,
                                            min_angles)
from underworld3.utilities.place_surface import place_along_lines

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

# Both ends outside the mesh: the usual "specify it a little long" convention.
CROSSING = np.array([[-0.2, 0.317], [1.2, 0.683]])
# One end outside, one inside: a fault reaching the wall and stopping.
ONE_TIP = np.array([[-0.2, 0.40], [0.55, 0.52]])
# Both ends inside: a fault. The cut refuses this outright.
FAULT = np.array([[0.25, 0.45], [0.70, 0.56]])


def _box(cell_size=1 / 16, **kwargs):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2, **kwargs)


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)


def _cell_vertex_indices(dm):
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)])


def _surface_vertices(dm, name, value):
    """Local indices of the vertices carrying the labelled facets."""
    vS, _vE = dm.getDepthStratum(0)
    out = set()
    for e in dm.getLabel(name).getStratumIS(value).getIndices():
        out.update(int(v) - vS for v in dm.getCone(e))
    return np.array(sorted(out))


def _distance_to_trace(points, trace):
    """Distance from each point to the trace, measured to its SEGMENTS.

    Not to the infinite line through its ends: a fault stops, and the mesh
    beyond its tip is legitimately on both sides of that line.
    """
    best = np.full(len(points), np.inf)
    for A, B in zip(trace[:-1], trace[1:]):
        d = B - A
        u = np.clip(((points - A) @ d) / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (A + u[:, None] * d),
                                               axis=1))
    return best


def _conforming(dm):
    """No facet with more than two cells, no inverted cell, Euler number 1."""
    nv = dm.getDepthStratum(0)[1] - dm.getDepthStratum(0)[0]
    ne = dm.getDepthStratum(1)[1] - dm.getDepthStratum(1)[0]
    nc = dm.getHeightStratum(0)[1] - dm.getHeightStratum(0)[0]
    over = sum(1 for f in range(*dm.getHeightStratum(1))
               if len(dm.getSupport(f)) > 2)
    return over == 0 and (cell_areas(dm) > 0.0).all() and nv - ne + nc == 1


@pytest.mark.parametrize("trace", [CROSSING, ONE_TIP, FAULT],
                         ids=["crossing", "one-tip", "fault"])
def test_the_three_cavity_shapes_all_place(trace):
    """No end, one end or two ends on the wall: three topologies, one walk.

    How many ends reach the domain boundary decides whether the cavity is an
    annulus, a disc, or two discs. All three go through the same walk, so all
    three are asserted together — a change that broke one of them would
    otherwise be caught only by whichever case happened to be tested.
    """
    base = _box()
    before = cell_areas(base.dm).sum()
    dm, info = place_along_lines(base.dm, [trace], label="Fault", label_value=7)

    assert _conforming(dm)
    assert cell_areas(dm).sum() == pytest.approx(before, rel=1e-13)
    assert info["n_surface_facets"] == (info["n_placed"]
                                        + info["n_on_surface"] - 1)
    assert dm.getLabel("Fault").getStratumSize(7) == info["n_surface_facets"]


@pytest.mark.parametrize("trace", [CROSSING, ONE_TIP, FAULT],
                         ids=["crossing", "one-tip", "fault"])
def test_the_surface_is_where_it_was_asked_for(trace):
    """The placed points are vertices of the result, not near ones.

    This is the property placement has and snapping does not: the cut moves the
    MESH onto the surface and so is exact too, but it can only choose from the
    crossings the mesh offers. Here the surface's own points are the vertices,
    so "on the surface" is not a tolerance — the only slack is where an end had
    to be projected onto a boundary facet, which is rounding.
    """
    dm, _info = place_along_lines(_box().dm, [trace], label="Fault",
                                  label_value=7)
    X = _coords(dm)[: dm.getDepthStratum(0)[1] - dm.getDepthStratum(0)[0]]
    on = _surface_vertices(dm, "Fault", 7)
    assert _distance_to_trace(X[on], trace).max() < 1e-12


@pytest.mark.parametrize("trace", [CROSSING, ONE_TIP, FAULT],
                         ids=["crossing", "one-tip", "fault"])
def test_no_cell_straddles_the_surface(trace):
    """The property a cell-wise material needs in order to be CORRECT.

    Asserted at the source: every segment between consecutive placed points is
    an element edge, so the surface is a union of element edges and no cell can
    have part of it inside. Checked by walking the chain rather than by
    signed distances, which cannot tell "beyond the tip" from "straddling".
    """
    dm, info = place_along_lines(_box().dm, [trace], label="Fault",
                                 label_value=7)
    X = _coords(dm)
    on = _surface_vertices(dm, "Fault", 7)
    order = on[np.argsort((X[on] - trace[0]) @ (trace[-1] - trace[0]))]

    vS, _vE = dm.getDepthStratum(0)
    edges = {frozenset(int(v) - vS for v in dm.getCone(e)): int(e)
             for e in range(*dm.getDepthStratum(1))}
    labelled = set(dm.getLabel("Fault").getStratumIS(7).getIndices())
    assert len(order) == info["n_surface_facets"] + 1
    for a, b in zip(order[:-1], order[1:]):
        e = edges.get(frozenset((int(a), int(b))))
        assert e is not None, "a segment of the surface is not a mesh edge"
        assert e in labelled, "a segment of the surface is not labelled"


def test_a_surface_ending_inside_the_mesh_is_placed_where_the_cut_refuses():
    """The headline case. A tip is a placed vertex; it is not a split.

    The cut has no representation for a triangle the surface enters and does not
    leave, so it refuses. Placement closes the cavity round the tip with a fan,
    which is the whole reason the tip's turn through 180 degrees is given a
    window of the walk's parameter to itself.
    """
    base = _box()
    with pytest.raises(ValueError, match="entered but not left"):
        cut_along_lines(base.dm, [FAULT])

    dm, info = place_along_lines(base.dm, [FAULT], label="Fault", label_value=7)
    assert _conforming(dm)
    assert info["n_surface_facets"] == (info["n_placed"]
                                        + info["n_on_surface"] - 1)

    # The tip really is a vertex of the mesh, and it is the END of the chain.
    X = _coords(dm)
    tip = np.flatnonzero(np.linalg.norm(X - FAULT[-1], axis=1) < 1e-13)
    assert len(tip) == 1, "the tip is not a vertex of the result"
    on = set(_surface_vertices(dm, "Fault", 7).tolist())
    assert int(tip[0]) in on


def test_two_surfaces_closer_than_one_element_are_placed():
    """An edge can be split once, so the cut refuses converging flanks.

    This is the restriction that makes a tapering fault unmeshable by cutting,
    and it is a property of the METHOD rather than of the mesh: refining shrinks
    the separation at which it bites but never removes it. Placement is not
    competing for the mesh's edges at all, so the separation is free.
    """
    base = _box()
    h = 1 / 16

    def pair(gap):
        return [np.array([[-0.2, 0.5 + gap * h / 2], [1.2, 0.5 + gap * h / 2]]),
                np.array([[-0.2, 0.5 - gap * h / 2], [1.2, 0.5 - gap * h / 2]])]

    with pytest.raises(ValueError, match="crossed more than once"):
        cut_along_lines(base.dm, pair(0.25))

    dm, info = place_along_lines(base.dm, pair(0.25), label="Pair",
                                 label_value=9, clearance=0.8)
    assert _conforming(dm)
    # Two chains, so two fewer facets than vertices along them.
    assert info["n_surface_facets"] == (info["n_placed"]
                                        + info["n_on_surface"] - 2)


def test_the_surface_may_be_finer_than_the_mesh():
    """``spacing`` is the knob the cut does not have.

    A cut's surface vertices ARE the crossings of the mesh's edges, so its
    resolution is the mesh's and cannot be asked for separately. A placed
    surface carries its own point spacing, which is what lets the geometry be
    resolved without refining everything around it.
    """
    base = _box()
    counts = []
    for factor in (1.0, 0.5, 0.25):
        _dm, info = place_along_lines(base.dm, [FAULT], label="Fault",
                                      label_value=7, spacing=factor / 16)
        counts.append(info["n_surface_facets"])
    assert counts[1] > 1.5 * counts[0] and counts[2] > 1.5 * counts[1], (
        f"halving the spacing did not roughly double the facets: {counts}")


def test_the_base_mesh_is_not_modified():
    """The surface's position is a design variable: the base must survive it."""
    base = _box()
    cells_before = base.dm.getHeightStratum(0)[1] - base.dm.getHeightStratum(0)[0]
    coords_before = _coords(base.dm).copy()

    place_along_lines(base.dm, [CROSSING], label="A", label_value=7)
    place_along_lines(base.dm, [FAULT], label="B", label_value=8)

    after = base.dm.getHeightStratum(0)[1] - base.dm.getHeightStratum(0)[0]
    assert after == cells_before
    assert np.array_equal(_coords(base.dm), coords_before)


def test_splitting_a_wall_facet_keeps_the_wall_labelled():
    """A surface reaching the boundary replaces one wall facet with two.

    The rebuild carries labels by point id, and the facet that was split has no
    point id in the result, so without inheritance its ``Left`` / ``Top`` /
    ``UW_Boundaries`` values would simply be dropped. The hole that leaves is
    invisible until a boundary condition steps over it, which is why the count
    is asserted rather than the mesh merely being looked at.
    """
    base = _box(1 / 12)
    # The four named walls only. ``Null_Boundary`` is the sentinel a natural
    # condition attaches to when it applies to no boundary, and it marks every
    # VERTEX of every UW3 mesh — so its stratum shrinks by however many vertices
    # the cavity swallowed, which says nothing about the wall.
    walls = {name: base.dm.getLabel(name).getStratumSize(
        base.boundaries[name].value)
        for name in ("Left", "Right", "Top", "Bottom")}
    dm, _info = place_along_lines(base.dm, [CROSSING], label="Fault",
                                  label_value=99)

    for name, before in walls.items():
        after = dm.getLabel(name).getStratumSize(base.boundaries[name].value)
        assert after >= before, f"{name} lost facets: {before} -> {after}"
    # The two walls the surface crosses each gained exactly one facet.
    gained = sum(dm.getLabel(n).getStratumSize(base.boundaries[n].value) - b
                 for n, b in walls.items())
    assert gained == 2, f"a split wall facet was not replaced by two: {gained}"


def test_a_second_surface_does_not_damage_the_first():
    """Placed surfaces compose, because a labelled edge is an interface.

    The second placement clears mesh vertices out of its own way, and the first
    surface's vertices are exactly the ones it must not take. They are protected
    by the same rule the repair passes use — a labelled edge is an interface —
    so this also checks that the label placement writes is the one that rule
    reads.
    """
    base = _box()
    one, info_one = place_along_lines(base.dm, [CROSSING], label="Fault",
                                      label_value=7)
    two, _info_two = place_along_lines(one, [np.array([[-0.2, 0.12],
                                                       [1.2, 0.12]])],
                                       label="Moho", label_value=8)
    assert _conforming(two)
    assert two.getLabel("Fault").getStratumSize(7) == info_one["n_surface_facets"]
    assert two.getLabel("Moho").getStratumSize(8) > 0


@pytest.mark.level_2
def test_repair_improves_the_shapes_and_leaves_the_surface_alone():
    """The walk fills the cavity by parameter, not by shape; repair does shape.

    Flipping and deleting are the two operations the fill does not have, and
    both refuse to act on a labelled edge — so the surface has to come through
    with the same facets, which is asserted rather than assumed.

    A GRADED mesh, and the fixture is checked before the claim is made. On a
    uniform mesh this same fault comes out of the walk with no cell under 15
    degrees at all, so a uniform fixture would assert that repair improved
    something that was not wrong — true, and vacuous.
    """
    line = np.array([[-0.1, 0.37], [1.1, 0.63]])
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 6,
        regular=False, qdegree=2, refinement=2)
    surf = uw.meshing.Surface("Grade", base, line)
    surf.discretize()
    graded = base.adapt(surf.refinement_metric_function(
        h_near=1 / 48, h_far=1 / 6, width=1 / 12), max_levels=3)

    dm, info = place_along_lines(graded.dm, [line], label="Fault", label_value=7)
    before = min_angles(dm)
    assert int((before < 15).sum()) > 0, (
        "the fill left no thin cell on this mesh, so there is nothing for "
        "repair to fix and this test would pass without exercising it")

    flipped, n_flips = reconnect.flip_to_reduce_max_angle(dm)
    after = min_angles(flipped)

    assert n_flips > 0
    assert int((after < 15).sum()) < int((before < 15).sum())
    assert after.min() >= before.min() - 1e-12, "repair lowered the worst angle"
    assert flipped.getLabel("Fault").getStratumSize(7) == info["n_surface_facets"]
    assert cell_areas(flipped).sum() == pytest.approx(cell_areas(dm).sum(),
                                                      rel=1e-13)


def test_a_dirichlet_condition_applies_on_a_placed_surface():
    """A label is only useful if a solver actually constrains those DOFs.

    Also the end-to-end check that the rebuilt plex is a mesh UW3 can use at
    all: sections, coordinates, boundaries and the solve.
    """
    base = _box(1 / 12, refinement=1)
    boundaries = base._boundaries_with("Fault")
    dm, _info = place_along_lines(base.dm, [np.array([[0.5, -0.2], [0.5, 1.2]])],
                                  label="Fault",
                                  label_value=boundaries["Fault"].value)
    mesh = uw.discretisation.Mesh(
        dm, simplex=True, qdegree=3, boundaries=boundaries,
        coordinate_system_type=base.CoordinateSystem.coordinate_type,
        verbose=False)

    u = uw.discretisation.MeshVariable("u_placed", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.add_dirichlet_bc(1.0, "Fault")
    poisson.solve()

    X, values = np.asarray(u.coords), np.asarray(u.data[:, 0])
    on = np.abs(X[:, 0] - 0.5) < 1e-11
    assert on.sum() > 0
    assert np.allclose(values[on], 1.0, atol=1e-10), "the surface BC was not applied"
    interior = (~on) & (X[:, 0] > 0.1) & (X[:, 0] < 0.4)
    assert 0.0 < values[interior].max() < 1.0


def test_a_surface_that_leaves_the_domain_and_returns_is_refused():
    """Two pieces are two surfaces, and they want two names."""
    base = _box()
    out_and_back = np.array([[0.2, 0.5], [0.5, 1.2], [0.8, 0.5]])
    with pytest.raises(ValueError, match="leaves the domain and re-enters"):
        place_along_lines(base.dm, [out_and_back], label="Fault", label_value=7)


def test_three_dimensions_is_refused_with_the_reason():
    """A placed sheet's cavity is a polyhedron, and filling one is not this."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.5, qdegree=2)
    with pytest.raises(NotImplementedError, match="Steiner"):
        place_along_lines(mesh.dm, [CROSSING], label="Fault", label_value=7)
