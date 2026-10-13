"""Longest-edge refinement without a conforming closure (``engine="edge_split"``).

The engine (:mod:`underworld3.utilities.edge_split`) splits the longest edge of
every cell coarser than the metric asks for. Because splitting an edge divides
*every* cell incident on it at the same new vertex there is no hanging node and
no closure, so — unlike bisection — refinement cannot escape the marked region.
It drives the compiled ``uwnvb_bisect`` :c:type:`DMPlexTransform`, the same
primitive the newest-vertex engine uses for each of its sub-passes, so topology,
coordinates, labels and the parallel star-forest are PETSc's.

What is asserted, and why each test would have caught a real defect found while
building this:

- **conformity** — no over-shared facet, at every generation;
- **the diameter is what converges** — the size field is expressed as a
  diameter, and the volume proxy ``(dim!·vol)^(1/dim)`` is NOT a substitute: it
  reported the target met while the mesh was 3.2x coarser across the feature on
  a non-bisection engine. A regression to the proxy passes a naive cell-count
  check and fails this one;
- **no halo** — cells far from the feature are untouched. This is the property
  the engine exists for, and the one a conforming closure gives up;
- **the exact prolongation survives** — every inserted vertex is the exact float
  midpoint of a parent edge, so the recorded 1/2,1/2 transfer applies and the
  child carries one MG level per generation;
- **partition independence** — the refined mesh is the same at any communicator
  size. Three separate defects during development (a collective inside a
  rank-local branch, an order-dependent greedy edge selection, and a mis-sized
  star-forest reduce) all showed up here and nowhere else, so this is the
  load-bearing test. The np>1 half lives in
  ``tests/parallel/ptest_0843_edge_split_parallel.py``; this file records the
  serial reference the parallel run must reproduce.
"""
import numpy as np
import pytest
import underworld3 as uw
from underworld3.utilities import edge_split

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(dim, cell_size, refinement=1):
    lo = tuple([0.0] * dim)
    hi = tuple([1.0] * dim)
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=lo, maxCoords=hi, cellSize=cell_size,
        refinement=refinement, qdegree=2)


def _centroids(dm):
    cS, cE = dm.getHeightStratum(0)
    if cE == cS:
        return np.zeros((0, dm.getCoordinateDim()))
    return np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cS, cE)])


def _over_shared_facets(dm):
    fS, fE = dm.getHeightStratum(1)
    return sum(1 for f in range(fS, fE) if len(dm.getSupport(f)) > 2)


def _refine_to(dm, h_of_centroid, max_passes=40):
    """Drive the engine until the diameter target is met everywhere."""
    passes = 0
    while passes < max_passes:
        cS, _cE = dm.getHeightStratum(0)
        cen = _centroids(dm)
        if cen.shape[0] == 0:
            break
        sel = np.flatnonzero(edge_split.cell_diameters(dm) > h_of_centroid(cen)) + cS
        dm, n_split = edge_split.bisect_longest_edges(dm, sel)
        if n_split == 0:
            break
        assert _over_shared_facets(dm) == 0, (
            f"pass {passes} left a facet shared by more than two cells")
        passes += 1
    return dm, passes


def _disc_target(centre, radius, h_near, h_far):
    def h(cen):
        d = np.linalg.norm(cen - np.asarray(centre), axis=1)
        return np.where(d < radius, h_near, h_far)
    return h


@pytest.mark.parametrize("dim", [2, 3])
def test_conforming_and_diameter_target_met(dim):
    """The mesh stays conforming and the DIAMETER reaches the target."""
    centre = np.array([0.35, 0.5] if dim == 2 else [0.35, 0.5, 0.6])
    h_near, h_far = (0.06, 0.4) if dim == 2 else (0.15, 0.5)
    target = _disc_target(centre, 0.25, h_near, h_far)

    dm = _box(dim, 0.35 if dim == 2 else 0.5).dm_hierarchy[-1]
    n0 = dm.getHeightStratum(0)[1]
    dm, passes = _refine_to(dm, target)

    assert dm.getHeightStratum(0)[1] > n0, "no refinement happened"
    assert _over_shared_facets(dm) == 0

    cen = _centroids(dm)
    inside = np.linalg.norm(cen - centre, axis=1) < 0.25
    diameter = edge_split.cell_diameters(dm)
    # The engine converges the DIAMETER. The volume proxy is systematically
    # smaller, so a regression to marking on it would leave these cells long.
    assert diameter[inside].max() <= h_near * 1.001, (
        f"largest diameter in the target region is {diameter[inside].max():.4f}, "
        f"target {h_near}")
    assert passes >= 1


def test_refinement_does_not_escape_the_marked_region():
    """No halo: cells far from the feature keep their original size.

    A conforming closure necessarily refines beyond the marked set; this engine
    must not. Measured against the coarsest cell size of the unrefined base, so
    the test states a property rather than a magic number.
    """
    centre = np.array([0.3, 0.3])
    target = _disc_target(centre, 0.15, 0.04, 1.0)

    base = _box(2, 0.3)
    dm0 = base.dm_hierarchy[-1]
    far0 = _far_field_diameter(dm0, centre, 0.45)
    dm, _passes = _refine_to(dm0, target)
    far1 = _far_field_diameter(dm, centre, 0.45)

    assert far1 == pytest.approx(far0, rel=1e-12), (
        f"cells beyond r=0.45 changed size ({far0:.5f} -> {far1:.5f}); "
        f"refinement escaped the marked region")


def _far_field_diameter(dm, centre, radius):
    cen = _centroids(dm)
    far = np.linalg.norm(cen - np.asarray(centre), axis=1) > radius
    return float(edge_split.cell_diameters(dm)[far].max()) if far.any() else 0.0


def test_adapt_returns_child_with_graded_mg_tail():
    """``mesh.adapt(engine="edge_split")`` carries the hierarchy and the exact
    prolongation for every generation."""
    base = _box(2, 0.2, refinement=2)
    centre = np.array([0.4, 0.55])

    def metric(cen):
        d = np.linalg.norm(np.asarray(cen) - centre, axis=1)
        h = np.where(d < 0.2, 0.03, 0.12)
        return 1.0 / h**2

    child = base.adapt(metric, max_levels=2, engine="edge_split")

    assert child.parent is base
    n_child = child.dm.getHeightStratum(0)[1]
    assert n_child > base.dm_hierarchy[-1].getHeightStratum(0)[1]

    tail = child._custom_mg_coarse_meshes
    assert tail is not None and len(tail) >= 3, (
        "the child must carry one MG level per refinement generation on top of "
        "the base tail; without it the V-cycle count triples")

    # Every inserted vertex is an exact float edge midpoint, so the recorded
    # 1/2,1/2 transfer must be available for EVERY generation — a None here means
    # coordinate identity was lost and the geometric builder would be used.
    recorded = child._adapt_prolongation
    assert recorded and all(P is not None for P in recorded), (
        "exact prolongation missing for at least one generation")


def test_unknown_engine_is_refused():
    """The engine name is validated, so a typo cannot silently fall back."""
    base = _box(2, 0.4, refinement=1)
    with pytest.raises(ValueError, match="edge_split"):
        base.adapt(lambda cen: np.ones(len(cen)), max_levels=1,
                   engine="edgesplit")


def test_serial_reference_for_parallel_confluence():
    """Record the serial result the parallel test must reproduce exactly.

    Kept in the serial file deliberately: the parallel counterpart asserts
    equality against these numbers, and a change here is then visible as a
    change to the contract rather than as a mysterious parallel failure.
    """
    centre = np.array([0.35, 0.6])
    target = _disc_target(centre, 0.2, 0.05, 0.3)
    dm = _box(2, 0.35).dm_hierarchy[-1]
    assert dm.getHeightStratum(0)[1] == 104
    dm, passes = _refine_to(dm, target)
    assert (dm.getHeightStratum(0)[1], passes) == (412, 7)
