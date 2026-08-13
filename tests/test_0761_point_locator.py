"""Point location: the cell hint must contain the point, and a point no local
cell can own must be rejected in O(1) (#551 items 1 and 3; #432, a recurrence
of #390).

Two properties are pinned here, both of which have been broken before.

**The hint contains the point.** ``petsc_interpolate`` may bypass PETSc's
``DMLocatePoints`` and evaluate the basis directly in a cell UW3 nominates.
Serial simplex meshes used to nominate the cell owning the nearest kd-tree
CONTROL POINT, with no containment test at all. On a tetrahedron nothing
downstream can rescue that: the reference-coordinate guard is a componentwise
box clamp and the reference tet is not the reference box, so a query on a
shared edge was answered by extrapolating the basis of a cell that does not
contain it. The oracle here is the closed-form P1 value ``(1-t)*u_a + t*u_b``
along an edge — the same reference
``test_0753_nested_mg_prolongation.py::test_reproduces_an_arbitrary_coarse_field``
uses, and for the same reason: it is computed WITHOUT ``uw.function.evaluate``,
so it cannot inherit the defect it is testing for.

**Rejection is cheap.** A point outside the local mesh used to enter a
50-nearest-centroid walk whose only exit was every lost point being found —
51 containment tests against 1 for an owned point. The guard here counts
containment tests rather than seconds: an instrumented counter says what the
algorithm does, where a stopwatch says what the machine was doing at the time.
The parallel wall-clock table lives in the PR, not in a brittle timing assert.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

# High enough in frequency that neighbouring nodal values are uncorrelated: a
# smooth low-frequency field would be reproduced by an interpolant anchored to
# the WRONG cell almost as well as by the right one, and the test would pass
# while the locator was broken (the "linear field cannot test neighbour
# selection" trap). test_nodal_signal_discriminates_between_cells measures
# this rather than assuming it.
def _nodal_signal(coords):
    coords = np.asarray(coords)
    signal = np.sin(97.0 * coords[:, 0]) * np.cos(89.0 * coords[:, 1])
    if coords.shape[1] == 3:
        signal = signal * np.sin(83.0 * coords[:, 2])
    return signal


def _box(dim, cell_size):
    if dim == 2:
        return uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
            cellSize=cell_size, regular=False, qdegree=2)
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)


def _vertex_coords(mesh):
    """Local mesh vertex coordinates, in DM point order."""
    pStart, pEnd = mesh.dm.getDepthStratum(0)
    raw = mesh.dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
    return raw[: pEnd - pStart], pStart


def _interior(points, margin):
    """Keep points at least ``margin`` inside the unit box, so the query is an
    interior FE evaluation and not an RBF extrapolation of a boundary point."""
    points = np.asarray(points)
    keep = np.all((points > margin) & (points < 1.0 - margin), axis=1)
    return np.ascontiguousarray(points[keep]), keep


class _CountedContainment:
    """Count the containment tests a locator call performs.

    ``points`` counts point-tests, which is the quantity that blew up: the walk
    re-tested the whole lost set on every one of its 50 rounds.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.points = 0
        self.calls = 0
        self._wrapped = mesh._test_if_points_in_cells_internal

    def __enter__(self):
        def counted(points, cells, **kwargs):
            self.calls += 1
            self.points += len(points)
            return self._wrapped(points, cells, **kwargs)

        self.mesh._test_if_points_in_cells_internal = counted
        return self

    def __exit__(self, *exc):
        self.mesh._test_if_points_in_cells_internal = self._wrapped
        return False


@pytest.fixture(scope="module", params=[2, 3], ids=["2d", "3d"])
def located_box(request):
    """Mesh plus the ONE P1 variable every test shares.

    Adding a mesh variable rebuilds the DM (#492), so a fixture that handed
    out a bare mesh and let each test add its own variable invalidated the
    mesh under the tests that ran before it.
    """
    dim = request.param
    mesh = _box(dim, 1.0 / 16 if dim == 2 else 1.0 / 8)
    field = uw.discretisation.MeshVariable(f"u_p1_{dim}", mesh, 1, degree=1)
    field.data[:, 0] = _nodal_signal(field.coords)
    mesh._build_kd_tree_index()
    mesh._mark_faces_inside_and_out()
    mesh._mark_local_boundary_faces_inside_and_out()
    return dim, mesh, field


def _edge_endpoints(mesh, limit=1500):
    """Vertex-coordinate pairs of local mesh edges."""
    verts, pStart = _vertex_coords(mesh)
    eStart, eEnd = mesh.dm.getDepthStratum(1)
    a, b = [], []
    for edge in range(eStart, min(eEnd, eStart + limit)):
        cone = mesh.dm.getCone(edge)
        a.append(verts[cone[0] - pStart])
        b.append(verts[cone[1] - pStart])
    return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)


# ---------------------------------------------------------------------------
# The hint contains the point (#432)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.0, 0.5, 0.25])
def test_p1_field_on_a_shared_edge_matches_the_closed_form(located_box, t):
    """``evaluate`` at a point on a cell edge must return ``(1-t)u_a + t u_b``.

    ``t = 0`` is a shared VERTEX, ``t = 0.5`` an edge midpoint. Both are shared
    by every incident cell, so a locator that returns any cell CONTAINING the
    point gives the exact answer, and a locator that returns a cell merely
    NEAR it does not.
    """
    dim, mesh, u = located_box

    a, b = _edge_endpoints(mesh)
    query = (1.0 - t) * a + t * b
    query, keep = _interior(query, 1.0e-4)
    assert query.shape[0] > 100, "not enough interior edges to test with"

    expected = (1.0 - t) * _nodal_signal(a)[keep] + t * _nodal_signal(b)[keep]
    got = np.asarray(uw.function.evaluate(u.sym[0], query)).reshape(-1)

    worst = float(np.abs(got - expected).max())
    assert worst < 1.0e-11, (
        f"{dim}-D P1 evaluation at t={t} along cell edges is off by {worst:.3e} "
        f"at {int(np.count_nonzero(np.abs(got - expected) > 1.0e-11))} of "
        f"{query.shape[0]} points — the query was answered in a cell that does "
        f"not contain it (#432)")


def test_p1_field_on_a_shared_face_matches_the_closed_form(located_box):
    """3-D face centroids: the P1 value is the mean of the three face vertices.

    A face is shared by exactly two tets and lies in the interior of neither;
    it is the case the box clamp on reference coordinates cannot express.
    """
    dim, mesh, u = located_box
    if dim != 3:
        pytest.skip("faces are edges in 2-D and are covered by the edge test")

    verts, pStart = _vertex_coords(mesh)
    fStart, fEnd = mesh.dm.getHeightStratum(1)
    corners = []
    for face in range(fStart, min(fEnd, fStart + 1500)):
        closure = mesh.dm.getTransitiveClosure(face)[0][-3:]
        corners.append(verts[closure - pStart])
    corners = np.array(corners, dtype=np.float64)  # (n, 3 vertices, 3 coords)

    query = corners.mean(axis=1)
    query, keep = _interior(query, 1.0e-4)
    assert query.shape[0] > 100

    expected = np.mean(
        [_nodal_signal(corners[:, k, :])[keep] for k in range(3)], axis=0)
    got = np.asarray(uw.function.evaluate(u.sym[0], query)).reshape(-1)

    worst = float(np.abs(got - expected).max())
    assert worst < 1.0e-11, (
        f"P1 evaluation at tet face centroids is off by {worst:.3e} — the "
        f"query was answered in a cell that does not contain it (#432)")


def test_nodal_signal_discriminates_between_cells(located_box):
    """NEGATIVE CONTROL for the two tests above.

    If the nodal field were smooth on the cell scale, the P1 interpolant
    anchored to a neighbouring cell would agree with the right one and the
    oracle would pass on a broken locator. Measure that the field is NOT
    smooth on that scale: the closed-form edge value must differ from the
    signal evaluated at the same point by far more than the tolerance the
    tests assert.
    """
    dim, mesh, _ = located_box
    a, b = _edge_endpoints(mesh)
    midpoint = 0.5 * (a + b)
    interpolated = 0.5 * (_nodal_signal(a) + _nodal_signal(b))
    pointwise = _nodal_signal(midpoint)
    spread = float(np.abs(interpolated - pointwise).max())
    assert spread > 0.1, (
        f"the {dim}-D nodal field varies by only {spread:.3e} between a cell "
        f"edge and its midpoint — too smooth to detect a wrong cell")


def test_the_unchecked_nearest_control_point_hint_misses(located_box):
    """NEGATIVE CONTROL for the fix in ``petsc_interpolate``.

    ``get_closest_cells`` is a nearest-CONTROL-POINT lookup with no containment
    test; it was the hint the serial simplex path handed to the
    DMLocatePoints bypass. Show it really does nominate cells that do not
    contain the query — otherwise the edge test above would be pinning
    nothing — and that the containment-checked locator does not.

    Quarter-points, not midpoints. A midpoint is equidistant from both ends of
    the edge and the nearest control point is (measured) always one belonging
    to a cell that does contain the edge; a quarter-point leans towards one
    vertex, and the nearest control point is then any cell around THAT vertex,
    most of which do not contain the far end. In 2-D the vertex neighbourhood
    is small enough that even the quarter-point never misses on these meshes,
    which is why #432 is a 3-D report.
    """
    dim, mesh, _ = located_box
    a, b = _edge_endpoints(mesh)
    query, _ = _interior(0.75 * a + 0.25 * b, 1.0e-4)

    unchecked = np.asarray(mesh.get_closest_cells(query)).reshape(-1)
    contained = mesh._test_if_points_in_cells_internal(query, unchecked)

    checked = np.asarray(mesh._robust_owning_cells(query)).reshape(-1)
    assert (checked >= 0).all(), "an interior edge point was not located at all"
    assert mesh._test_if_points_in_cells_internal(
        query, checked, tol=mesh._EVAL_FACE_TOL).all(), (
        "the containment-checked locator returned a cell that does not "
        "contain the point")

    if dim == 2:
        pytest.skip(
            f"the unchecked hint contains {int(contained.sum())} of "
            f"{query.shape[0]} 2-D queries — nothing to demonstrate here; the "
            f"3-D case carries the control")

    assert not contained.all(), (
        "the unchecked nearest-control-point hint happens to contain every "
        "query on this mesh, so it cannot demonstrate the defect — pick a "
        "harsher query set before trusting the edge test above")


def test_every_located_cell_contains_its_point(located_box):
    """The locator's contract, which the shrinking working set must preserve.

    Which of several qualifying cells is returned for a point on a shared face
    is not defined (and used to depend on whether some unrelated point in the
    same batch was findable). That the returned cell CONTAINS the point is.
    """
    dim, mesh, _ = located_box
    rng = np.random.default_rng(4)
    query = np.ascontiguousarray(rng.uniform(0.02, 0.98, size=(2000, dim)))

    for tol in (0.0, mesh._EVAL_FACE_TOL):
        cells = np.asarray(
            mesh._get_closest_local_cells_internal(query, tol=tol)).reshape(-1)
        found = cells >= 0
        assert found.any(), "nothing was located at all"
        assert mesh._test_if_points_in_cells_internal(
            query[found], cells[found], tol=tol).all(), (
            f"a cell returned at tol={tol} does not contain its point")


# ---------------------------------------------------------------------------
# Rejection is cheap (#551 item 1)
# ---------------------------------------------------------------------------

def test_a_point_outside_the_mesh_costs_a_bounded_number_of_tests(located_box):
    """A point no local cell can own must be rejected in O(1) containment
    tests, not by walking 50 nearest centroids.

    Structural, not a stopwatch: the count is what the algorithm does. Before
    the fix this was exactly 51 tests per point at every mesh size.
    """
    dim, mesh, _ = located_box
    rng = np.random.default_rng(11)
    outside = np.ascontiguousarray(rng.uniform(1.5, 2.5, size=(1000, dim)))
    owned = np.ascontiguousarray(mesh._centroids[:1000])

    with _CountedContainment(mesh) as counter:
        cells = np.asarray(mesh._robust_owning_cells(outside)).reshape(-1)
    assert (cells < 0).all(), "a point well outside the mesh was located"
    per_outside_point = counter.points / outside.shape[0]

    with _CountedContainment(mesh) as counter:
        cells = np.asarray(mesh._robust_owning_cells(owned)).reshape(-1)
    assert (cells >= 0).all(), "a cell centroid was not located in its own cell"
    per_owned_point = counter.points / owned.shape[0]

    # The counter fires at all: an owned point costs exactly the one test that
    # confirms the nearest control point's cell. A zero here would mean the
    # instrumentation missed the call and the bound above proved nothing.
    assert per_owned_point >= 1.0
    assert per_owned_point <= 2.0

    assert per_outside_point <= 4.0, (
        f"a point outside the {dim}-D mesh costs {per_outside_point:.1f} "
        f"containment tests per point (was 51 before the rejection path)")


def test_one_unlocatable_point_does_not_cost_the_whole_batch(located_box):
    """The working set has to shrink.

    The walk used to re-test every already-located point on every round, and
    to keep going until the LAST point was found — so a single point that
    could never be found charged the whole batch 50 extra rounds.
    """
    dim, mesh, _ = located_box
    rng = np.random.default_rng(12)
    interior = np.ascontiguousarray(rng.uniform(0.02, 0.98, size=(1000, dim)))
    poisoned = np.ascontiguousarray(
        np.vstack([interior, rng.uniform(1.5, 2.5, size=(1, dim))]))

    with _CountedContainment(mesh) as counter:
        mesh._robust_owning_cells(interior)
    clean = counter.points

    with _CountedContainment(mesh) as counter:
        mesh._robust_owning_cells(poisoned)
    with_poison = counter.points

    assert with_poison <= clean + 60, (
        f"one unlocatable point added {with_poison - clean} containment "
        f"point-tests to a batch of {interior.shape[0]}")


def test_the_classifier_hands_over_the_cells_it_located(located_box):
    """#551 item 2: the classification and the interpolation share one
    location pass, so the cells the classifier looked up come back with the
    in/out mask instead of being thrown away."""
    dim, mesh, _ = located_box
    rng = np.random.default_rng(13)
    query = np.ascontiguousarray(
        np.vstack([rng.uniform(0.02, 0.98, size=(500, dim)),
                   rng.uniform(1.5, 2.5, size=(50, dim))]))

    in_or_not, cells = mesh._classify_points_in_domain(
        query, strict_validation=False)

    assert np.array_equal(
        in_or_not, mesh.points_in_domain(query, strict_validation=False)), (
        "splitting the classifier changed points_in_domain's answer")
    assert cells.shape[0] == query.shape[0]
    # An exterior point is never given a hint: it goes to RBF extrapolation.
    assert (cells[~in_or_not] == -1).all()
    # Every hint that IS offered must be a cell containing its point.
    offered = in_or_not & (cells >= 0)
    if offered.any():
        assert mesh._test_if_points_in_cells_internal(
            query[offered], cells[offered], tol=mesh._EVAL_FACE_TOL).all()
