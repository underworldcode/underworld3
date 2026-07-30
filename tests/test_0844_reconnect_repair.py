"""Reconnection (Lawson flip) repair of a refined 2-D mesh.

The load-bearing checks here are the ones that would pass for the wrong reason if
they were written loosely:

* the maximum angle must **improve**, and this is the check that earned its place.
  The pass originally accepted a flip on the Delaunay criterion, and this
  assertion is what caught that Delaunay maximises the *minimum* angle while P1
  interpolation depends on the *maximum* — flipping a gmsh mesh towards Delaunay
  raised the 99th-percentile maximum angle. Assert the quantity the method claims
  to improve, not a proxy for it;
* **volume conservation**, not orientation. Checking that the new cells are
  positively oriented is worthless when they were built anticlockwise by
  construction: the check can never fail. Equal total area is the real test;
* the **point chart is unchanged**, which is the invariant that lets the parallel
  path reuse the star-forest verbatim rather than reconstructing it.

Note what the idempotence check below does *not* prove. A second pass flipping
nothing shows the acceptance test is self-consistent, but an **inverted** criterion
is equally idempotent — it would flip every good edge once and then find nothing
more to do. That is exactly how the Delaunay criterion passed here while degrading
the mesh. Idempotence catches oscillation, not a wrong objective.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import edge_split, reconnect

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _max_angles(dm):
    """Largest interior angle of every cell, in degrees.

    The maximum angle is the quantity a P1 interpolation bound depends on
    (Babuska-Aziz); the minimum angle is not, so it is the wrong thing to assert.
    """
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    out = []
    for c in range(cS, cE):
        v = [int(p) for p in dm.getTransitiveClosure(c)[0] if vS <= p < vE]
        P = X[np.array(v) - vS]
        angles = []
        for i in range(3):
            u1 = P[(i + 1) % 3] - P[i]
            u2 = P[(i + 2) % 3] - P[i]
            cos = np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))
            angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        out.append(max(angles))
    return np.array(out)


def _signed_areas(dm):
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    out = []
    for c in range(cS, cE):
        v = [int(p) for p in dm.getTransitiveClosure(c)[0] if vS <= p < vE]
        a, b, d = X[np.array(v) - vS]
        out.append(0.5 * ((b[0] - a[0]) * (d[1] - a[1])
                          - (d[0] - a[0]) * (b[1] - a[1])))
    return np.array(out)


def _over_shared_facets(dm):
    fS, fE = dm.getHeightStratum(1)
    return sum(1 for f in range(fS, fE) if len(dm.getSupport(f)) > 2)


def _refined_dm(cell_size=0.3, h_near=0.05, centre=(0.35, 0.6), radius=0.2):
    """A box mesh refined by ``edge_split`` — the mesh repair is meant to fix."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=False, qdegree=2)
    dm = base.dm
    for _ in range(20):
        cS, cE = dm.getHeightStratum(0)
        cen = np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cS, cE)])
        d = np.linalg.norm(cen - np.array(centre), axis=1)
        target = np.where(d < radius, h_near, 0.4)
        sel = np.flatnonzero(edge_split.cell_diameters(dm) > target) + cS
        dm, n = edge_split.bisect_longest_edges(dm, sel)
        if n == 0:
            break
    return dm


def test_repair_conserves_area_and_conformity():
    dm = _refined_dm()
    ncells = dm.getHeightStratum(0)[1] - dm.getHeightStratum(0)[0]
    chart = dm.getChart()
    area = _signed_areas(dm).sum()

    out, nflips = reconnect.flip_to_reduce_max_angle(dm)

    assert nflips > 0, "nothing to repair — the fixture is not exercising the pass"
    # A flip replaces two cells by two cells and adds no points, so both the cell
    # count and the whole chart are invariant. The chart being invariant is what
    # makes the parallel star-forest reusable.
    assert out.getChart() == chart
    assert out.getHeightStratum(0)[1] - out.getHeightStratum(0)[0] == ncells
    assert _over_shared_facets(out) == 0
    new_areas = _signed_areas(out)
    assert (new_areas > 0).all(), "repair inverted a cell"
    assert new_areas.sum() == pytest.approx(area, rel=1e-13)


def test_repair_improves_the_maximum_angle():
    dm = _refined_dm()
    before = _max_angles(dm)
    out, _ = reconnect.flip_to_reduce_max_angle(dm)
    after = _max_angles(out)

    assert np.percentile(after, 99) < np.percentile(before, 99)
    assert after.max() <= before.max()


def test_second_pass_flips_nothing():
    """Idempotence: the control that catches an inconsistently signed predicate.

    A kernel that keeps finding improvements in an already-repaired mesh is
    reporting a predicate bug, and that bug would be invisible in every other
    check here.
    """
    dm = _refined_dm()
    once, n1 = reconnect.flip_to_reduce_max_angle(dm)
    assert n1 > 0
    twice, n2 = reconnect.flip_to_reduce_max_angle(once)
    assert n2 == 0
    assert twice is once, "a no-op pass must return the mesh it was given"


def test_labels_survive_and_remain_usable():
    dm = _refined_dm()
    names = sorted(dm.getLabelName(i) for i in range(dm.getNumLabels()))
    sizes = {}
    for name in names:
        if name in ("depth", "celltype"):
            continue
        sizes[name] = dm.getLabel(name).getStratumSize(
            int(dm.getLabel(name).getValueIS().getIndices()[0]))

    out, _ = reconnect.flip_to_reduce_max_angle(dm)

    assert sorted(out.getLabelName(i) for i in range(out.getNumLabels())) == names
    for name, size in sizes.items():
        label = out.getLabel(name)
        assert label.getStratumSize(
            int(label.getValueIS().getIndices()[0])) == size

    # Labels surviving as point sets is not the same as being usable: the real
    # test is that a Dirichlet condition can still be imposed on one.
    mesh = uw.discretisation.Mesh(out, qdegree=2)
    u = uw.discretisation.MeshVariable("u_rec", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "All_Boundaries")
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    assert u.array[:, 0, 0].max() > 0.0


def test_labelled_interior_edges_are_never_flipped():
    """A labelled interior edge is an interface and must survive untouched.

    This is what protects a fault or a material boundary from being reconnected
    across.
    """
    dm = _refined_dm()
    eS, eE = dm.getDepthStratum(1)
    interior = [e for e in range(eS, eE) if len(dm.getSupport(e)) == 2]
    # Lock a slice of interior edges, including ones the pass would otherwise flip.
    dm.createLabel("test_interface")
    label = dm.getLabel("test_interface")
    locked = interior[::7]
    for e in locked:
        label.setValue(e, 1)
    cones = {e: tuple(int(v) for v in dm.getCone(e)) for e in locked}

    out, _ = reconnect.flip_to_reduce_max_angle(dm)

    for e, cone in cones.items():
        assert tuple(int(v) for v in out.getCone(e)) == cone, (
            f"locked interface edge {e} was flipped")


def test_three_dimensions_is_refused():
    """3-D must fail loudly: Delaunay is the wrong criterion, not merely untested."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5,
        regular=False, qdegree=2)
    with pytest.raises(NotImplementedError, match="2-D only"):
        reconnect.flip_to_reduce_max_angle(mesh.dm)
