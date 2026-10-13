"""Point location: the cell hint must contain the point, and a point no local
cell can own must be rejected in O(1) (#551 items 1 and 3; #432, a recurrence
of #390).

Five properties are pinned here, all of which have been broken before.

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

**Rejection never costs a point the mesh owns.** The cheap direction of the
rejection radius is easy to see; the expensive one is silent. A brute-force
oracle (every point against every local cell) says which points a local cell
really contains, and a sweep over the reach margin is the negative control:
0 false rejections at the shipped margin and at half of it, and a growing
count below that, so a clean result is a measurement and not a coincidence.

**Which containing cell you get is allowed to change; a NON-containing cell
is not.** #551 moved the tie-break for a point several cells share from "last
cell to claim it across up to 50 walk rounds" (which depended on the rest of
the batch) to "containing cell with the nearest centroid". For a discontinuous
field the cell is the answer, so this is user-visible; the test pins the
property that survives — the value belongs to a cell containing the query —
rather than a specific cell, which was never part of the contract.

**A point no cell owns gets a value, not a NaN.** The RBF fallback rung in
``petsc_interpolate`` used to iterate an exhausted generator and fill nothing.
The loss is injected rather than waited for, so the rung is exercised on every
mesh and every platform.
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


def _cells_containing(mesh, points, tol):
    """Brute-force oracle: for every point, which local cells contain it?

    Tests every point against every local cell with the SAME containment
    predicate the locator uses, so a disagreement is a locator disagreement
    and not a difference of geometric opinion. Returns a boolean
    ``(n_points, n_cells)`` table.
    """
    nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
    cStart, cEnd = nav_dm.getHeightStratum(0)
    table = np.zeros((points.shape[0], cEnd - cStart), dtype=bool)
    column = np.empty(points.shape[0], dtype=np.int64)
    for cell in range(cEnd - cStart):
        column[:] = cell
        table[:, cell] = mesh._test_if_points_in_cells_internal(
            points, column, tol=tol)
    return table


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


def test_a_point_a_local_cell_contains_is_never_rejected(located_box):
    """The dangerous direction of the rejection radius.

    Rejecting a foreign point early is the point of the radius; rejecting a
    point that a local cell really does contain would be silent — the point
    disappears into the RBF fallback (serial) or is claimed by nobody
    (parallel), and no other test in this file would notice.

    The oracle is brute force: every point against every local cell, with the
    same containment predicate the locator uses. The sweep over
    ``_LOCATOR_REACH_MARGIN`` is the negative control — it shows the probe
    detects false rejections when the radius is deliberately too tight, so a
    clean result at the shipped margin means something.
    """
    dim, mesh, _ = located_box
    rng = np.random.default_rng(21)
    query = np.ascontiguousarray(rng.uniform(0.02, 0.98, size=(800, dim)))
    tol = mesh._EVAL_FACE_TOL

    # Rank-local: in parallel a rank only holds its share of the query, so the
    # bar scales with the partition (measured 179-434 of 800 per rank at
    # np2/np4, 800 of 800 in serial).
    contained = _cells_containing(mesh, query, tol).any(axis=1)
    assert contained.sum() > 0.4 * query.shape[0] / uw.mpi.size, (
        f"only {int(contained.sum())} of {query.shape[0]} interior points are "
        f"contained by a cell of this rank — the oracle is not exercising the "
        f"local mesh")

    shipped = mesh._LOCATOR_REACH_MARGIN
    counts = {}
    try:
        for margin in (shipped, 1.0, 0.5, 0.25, 0.1):
            mesh._LOCATOR_REACH_MARGIN = margin
            got = np.asarray(mesh._get_closest_local_cells_internal(
                query, tol=tol)).reshape(-1)
            counts[margin] = int(np.count_nonzero(contained & (got < 0)))
    finally:
        del mesh._LOCATOR_REACH_MARGIN
    assert mesh._LOCATOR_REACH_MARGIN == shipped

    # Measured serial: 2-D {2.0: 0, 1.0: 0, 0.5: 5, 0.25: 16, 0.1: 16},
    # 3-D {2.0: 0, 1.0: 0, 0.5: 4, 0.25: 168, 0.1: 237} — the bound is tight
    # at about 1.0 and the shipped 2.0 keeps a factor of two over the first
    # observable loss. 0 at 2.0 and 1.0 on every rank at np2 and np4 too.
    assert counts[shipped] == 0, (
        f"the shipped reach margin ({shipped}) rejected {counts[shipped]} of "
        f"{int(contained.sum())} {dim}-D points that a local cell contains "
        f"(sweep: {counts})")
    assert counts[1.0] == 0

    # NEGATIVE CONTROL. A quarter of the true reach must lose points, or the
    # clean result above is the probe failing to fire rather than the radius
    # being right.
    assert counts[0.25] > 0, (
        f"a reach margin of 0.25 rejected nothing in {dim}-D, so this test "
        f"cannot tell a correct radius from a broken one (sweep: {counts})")


def test_the_rejection_radius_is_rebuilt_when_the_mesh_moves():
    """A stale SMALL reach is the one way the rejection silently loses points.

    ``_local_cell_reach`` is measured in :meth:`_build_kd_tree_index` and
    invalidated with ``_index``. That is an invariant, not a comment: deform
    the mesh so its cells grow, and both the stored reach and the locations it
    admits must follow.
    """
    mesh = _box(2, 1.0 / 10)
    mesh._build_kd_tree_index()
    mesh._mark_faces_inside_and_out()
    before = mesh._local_cell_reach
    assert before > 0.0

    mesh.deform(np.asarray(mesh.X.coords, dtype=np.float64) * 50.0)
    mesh._build_kd_tree_index()
    mesh._mark_faces_inside_and_out()
    after = mesh._local_cell_reach

    assert after / before == pytest.approx(50.0, rel=1e-9), (
        f"reach {before:.6g} -> {after:.6g} after deform(x50): the rejection "
        f"radius did not follow the geometry")

    # And the expanded domain is actually usable: every point a cell contains
    # is still located, none rejected by a radius measured on the old mesh.
    rng = np.random.default_rng(22)
    query = np.ascontiguousarray(rng.uniform(1.0, 49.0, size=(400, 2)))
    tol = mesh._EVAL_FACE_TOL
    contained = _cells_containing(mesh, query, tol).any(axis=1)
    got = np.asarray(mesh._get_closest_local_cells_internal(
        query, tol=tol)).reshape(-1)
    assert contained.any(), "no query landed in the deformed mesh"
    assert not (contained & (got < 0)).any(), (
        f"{int((contained & (got < 0)).sum())} points inside the deformed mesh "
        f"were rejected — the reach did not grow with the cells")

    del mesh


def test_a_discontinuous_field_at_a_shared_vertex_takes_a_containing_cell():
    """#551 changed WHICH containing cell a shared point gets, and for a
    discontinuous field the cell IS the answer.

    A mesh vertex is contained by every cell around it. The locator returns
    one of them, and the tie-break moved from "last cell to claim it across up
    to 50 walk rounds" (batch-dependent) to "containing cell with the nearest
    centroid". Pinning a SPECIFIC cell would over-pin: it is not part of the
    contract, and the ambiguity is genuine. What is pinned is that the value
    comes from a cell that CONTAINS the point — so a future change that
    answers from a merely nearby cell fails here.

    The P2/P0-discontinuous pressure space the fault work uses is exactly this
    configuration, which is why it is worth a test rather than a paragraph.
    """
    if uw.mpi.size > 1:
        pytest.skip(
            "serial only: the oracle is the LOCAL cell table, and in parallel "
            "the rank that answers a seam vertex need not be the rank whose "
            "cells this rank can see")

    mesh = _box(3, 1.0 / 6)
    p0 = uw.discretisation.MeshVariable(
        "p0_jump", mesh, 1, degree=0, continuous=False)
    mesh._build_kd_tree_index()
    mesh._mark_faces_inside_and_out()
    mesh._mark_local_boundary_faces_inside_and_out()

    rng = np.random.default_rng(23)
    p0.data[:, 0] = rng.uniform(-1.0, 1.0, size=p0.data.shape[0])

    # Map nav-DM cell index -> P0 degree of freedom by centroid identity, and
    # check the map is a bijection before trusting it.
    centroids = np.asarray(mesh._nav_centroids, dtype=np.float64)
    dof_coords = np.asarray(p0.coords, dtype=np.float64)
    separation = np.linalg.norm(
        centroids[:, None, :] - dof_coords[None, :, :], axis=-1)
    dof_of_cell = separation.argmin(axis=1)
    assert separation.min(axis=1).max() < 1.0e-10
    assert len(set(dof_of_cell.tolist())) == centroids.shape[0]

    verts, _ = _vertex_coords(mesh)
    query, _ = _interior(verts, 0.05)
    query = np.ascontiguousarray(query[:200])
    assert query.shape[0] > 50, "not enough interior vertices to test with"

    tol = mesh._EVAL_FACE_TOL
    table = _cells_containing(mesh, query, tol)
    shared = table.sum(axis=1)
    assert (shared >= 2).mean() > 0.9, (
        f"only {(shared >= 2).mean():.2f} of the vertex queries are contained "
        f"by more than one cell — nothing here is ambiguous, so the test "
        f"cannot see a tie-break at all")

    got = np.asarray(uw.function.evaluate(p0.sym[0], query)).reshape(-1)
    assert np.isfinite(got).all()

    # NEGATIVE CONTROL: the containing cells must actually disagree, otherwise
    # "the value is one of them" is satisfied by any locator at all.
    spread = np.array([
        np.ptp(p0.data[dof_of_cell[np.flatnonzero(row)], 0]) for row in table])
    assert spread.max() > 0.5, (
        f"the containing cells of every vertex agree to within "
        f"{spread.max():.3e} — a wrong cell would be undetectable here")

    for i, row in enumerate(table):
        allowed = p0.data[dof_of_cell[np.flatnonzero(row)], 0]
        assert np.abs(allowed - got[i]).min() < 1.0e-12, (
            f"P0 evaluation at a shared vertex returned {got[i]:.6g}, which is "
            f"not the value of any of the {int(row.sum())} cells containing it "
            f"({np.sort(allowed)}) — the query was answered in a cell that does "
            f"not contain it")

    del mesh


def test_evaluate_fills_a_point_no_cell_owns_instead_of_returning_nan():
    """The RBF fallback rung has to be reachable.

    ``petsc_interpolate`` may hand ``DMInterpolation`` a cell hint of ``-1``
    for a point it could not place; ``DMInterpolationEvaluate_UW`` writes NaN
    there, and the rung below is what replaces it with the bounded, topology-
    free RBF value. The rung iterated ``mesh.vars.values()``, a WeakValueDict
    GENERATOR that an earlier loop had already exhausted, so it filled nothing
    and the NaN was returned to the caller.

    The loss is injected rather than waited for: a graded mesh does lose the
    occasional interior point to the 50-neighbour cap, but how many is a
    property of whatever gmsh produced today. Injecting it makes the test
    exercise the rung on every mesh, every platform, every run.
    """
    if uw.mpi.size > 1:
        pytest.skip(
            "serial only: with more than one rank the injected loss re-routes "
            "the point to whichever rank still claims it, so the value is not "
            "this rank's RBF interpolant. Serial is where #551 item 3 opened "
            "the door onto this rung")

    mesh = _box(3, 1.0 / 8)
    u = uw.discretisation.MeshVariable("u_fallback", mesh, 1, degree=1)
    coords = np.asarray(mesh.X.coords, dtype=np.float64).copy()
    coords[:, 0] = coords[:, 0] ** 4          # strong grading, as #551 reports
    mesh.deform(coords)
    mesh._build_kd_tree_index()
    mesh._mark_faces_inside_and_out()
    mesh._mark_local_boundary_faces_inside_and_out()
    u.data[:, 0] = _nodal_signal(np.asarray(u.coords, dtype=np.float64))

    rng = np.random.default_rng(24)
    query = np.ascontiguousarray(rng.uniform(0.02, 0.98, size=(400, 3)))
    intact = np.asarray(uw.function.evaluate(u.sym[0], query)).reshape(-1)
    assert np.isfinite(intact).all(), (
        f"{int((~np.isfinite(intact)).sum())} of {query.shape[0]} interior "
        f"points came back NaN with the locator untouched")

    victims = np.array([3, 17, 61, 128, 349])
    victim_coords = query[victims]
    located = mesh._robust_owning_cells
    mesh._dminterpolation_cache.invalidate_all("fault injection")

    def loses_the_victims(points):
        cells = np.asarray(located(points), dtype=np.int64).copy()
        for victim in victim_coords:
            cells[np.all(points == victim, axis=1)] = -1
        return cells

    mesh._robust_owning_cells = loses_the_victims
    try:
        got = np.asarray(uw.function.evaluate(u.sym[0], query)).reshape(-1)
    finally:
        del mesh._robust_owning_cells
        mesh._dminterpolation_cache.invalidate_all("fault injection")

    assert np.isfinite(got).all(), (
        f"{int((~np.isfinite(got)).sum())} points the locator could not place "
        f"came back NaN: the RBF fallback rung did not run")

    expected = np.asarray(
        u.rbf_interpolate(np.ascontiguousarray(victim_coords))).reshape(-1)
    assert np.abs(got[victims] - expected).max() < 1.0e-12, (
        "the unlocated points were filled with something other than the RBF "
        "interpolant")

    # NEGATIVE CONTROL: the fallback value must be distinguishable from the FE
    # value, or "no NaN" could be passing on an untouched array.
    assert np.abs(got[victims] - intact[victims]).max() > 1.0e-6, (
        "the injected loss changed nothing — the fault injection missed and "
        "the rung was never asked to run")
    others = np.setdiff1d(np.arange(query.shape[0]), victims)
    assert np.abs(got[others] - intact[others]).max() < 1.0e-12, (
        "injecting a loss at five points changed the answer elsewhere")

    del mesh


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

    # Which points get a hint is decided by _eval_use_robust_location(), and
    # that is False in serial by design: the serial classifier keeps the
    # validated cell-wall path bit-for-bit and offers NO cells at all. Assert
    # that reality rather than wrapping the real check in `if offered.any():`
    # — measured 0 hints for 1465 interior points in serial against 462 of 462
    # at np4, so the guarded version asserted nothing in the default test run.
    offered = in_or_not & (cells >= 0)
    if uw.mpi.size == 1:
        assert not mesh._eval_use_robust_location()
        assert not offered.any(), (
            f"the serial classifier handed over {int(offered.sum())} cells; it "
            f"is supposed to hand over none, and the evaluator locates the "
            f"points itself")
        return

    assert mesh._eval_use_robust_location()
    assert offered.any(), (
        f"the parallel classifier located points in the domain "
        f"({int(in_or_not.sum())} of {query.shape[0]}) but handed over no "
        f"cells, so item 2 saves nothing")
    # Every hint that IS offered must be a cell containing its point.
    assert mesh._test_if_points_in_cells_internal(
        query[offered], cells[offered], tol=mesh._EVAL_FACE_TOL).all()
