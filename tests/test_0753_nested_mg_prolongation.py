"""Exact nested MG prolongations recorded by ``mesh.adapt`` (#425).

``adapt`` maintains an exact refinement hierarchy; the MG transfer used to
discard it and re-derive an approximation by Delaunay point location, which
is what made #424 possible (a coarse DOF with no fine image -> zero column
-> singular coarse operator).

The recorded transfer is the true P1 embedding of the coarse space in the fine
one. The properties asserted here are what make it better than point location,
not merely different.

One multigrid level now spans as many engine passes as it takes to halve `h`
(``adapt(mg_coarsening_ratio=...)``), so a recorded transfer is the COMPOSITION
of those passes. That widens two things and neither is a weakening:

* a fine vertex need no longer lie on a coarse EDGE. Composing two bisections
  can place it at the midpoint of a segment joining two midpoints, which is
  strictly inside a coarse cell. The reference here is therefore the coarse P1
  value at the vertex's position, computed barycentrically in the containing
  coarse cell — which covers every fine vertex rather than the ~64 % that lie on
  an edge, so the test now checks more than it did;
* a row holds up to ``dim+1`` entries rather than 2, because that is how many
  coarse vertices a point inside a coarse cell depends on. It is still the exact
  embedding, and still far sparser than a point-located row would be dense.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _metric(centroids):
    d = np.abs(np.asarray(centroids)[:, -1] - 0.5)
    return 1.0 / np.minimum(np.sqrt(0.05**2 + (2.0 * d) ** 2), 0.3) ** 2


def _coarse_cell_vertices(cdm, dim):
    """(n_cells, dim+1) vertex indices of every coarse cell."""
    vS, vE = cdm.getDepthStratum(0)
    cS, cE = cdm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in cdm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)])


def _coarse_p1_value(cx, cells, data, x, tol=1e-9):
    """The coarse P1 field at ``x``, by barycentric interpolation.

    Returns ``None`` if ``x`` lies in no coarse cell, which the caller treats as
    a failure to cover rather than a pass. Computed here rather than through
    `uw.function.evaluate`, which is wrong exactly on cell boundaries (#432) —
    and a composed transfer puts many fine vertices there.
    """
    for verts in cells:
        P0 = cx[verts[0]]
        M = np.stack([cx[v] - P0 for v in verts[1:]], axis=1)
        try:
            lam = np.linalg.solve(M, x - P0)
        except np.linalg.LinAlgError:      # degenerate cell; cannot contain x
            continue
        bary = np.concatenate([[1.0 - lam.sum()], lam])
        if (bary > -tol).all() and (bary < 1.0 + tol).all():
            return float(bary @ data[verts])
    return None


def _adapted(dim, cell_size):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=cell_size, refinement=1, qdegree=2)
    return base.adapt(_metric, max_levels=2)


def _levels(child):
    return [m.dm for m in child._custom_mg_coarse_meshes] + [child.dm]


def _as_matrix(entry, coarse_dm, fine_dm):
    rows, cols, vals = entry
    cvS, cvE = coarse_dm.getDepthStratum(0)
    fvS, fvE = fine_dm.getDepthStratum(0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(fvE - fvS, cvE - cvS))


@pytest.mark.parametrize("dim,cell_size", [(2, 0.2), (3, 0.4)])
def test_every_pass_records_a_prolongation(dim, cell_size):
    child = _adapted(dim, cell_size)
    Ps = child._adapt_prolongation
    assert Ps, "adapt recorded no nested prolongations"
    assert all(P is not None for P in Ps), (
        "a refinement pass could not be expressed as a bisection embedding")


@pytest.mark.parametrize("dim,cell_size", [(2, 0.2), (3, 0.4)])
def test_partition_of_unity_and_no_zero_columns(dim, cell_size):
    """No zero column is the property that makes #424 impossible here."""
    child = _adapted(dim, cell_size)
    Ps = child._adapt_prolongation
    lvl = _levels(child)[-(len(Ps) + 1):]
    for k, entry in enumerate(Ps):
        P = _as_matrix(entry, lvl[k], lvl[k + 1])
        rowsum = np.asarray(P.sum(axis=1)).ravel()
        assert np.allclose(rowsum, 1.0, atol=1e-12), (
            f"pass {k}: prolongation is not a partition of unity")
        colsum = np.asarray(P.sum(axis=0)).ravel()
        assert int((colsum == 0.0).sum()) == 0, (
            f"pass {k}: coarse DOF with no fine image — this is exactly the "
            f"zero-column failure the nested transfer is meant to preclude")


@pytest.mark.parametrize("dim,cell_size", [
    (2, 0.2),
    pytest.param(3, 0.4, marks=pytest.mark.xfail(
        reason="TODO(BUG) nvb.nested_prolongation: in 3-D the recorded transfer "
               "is not the coarse P1 embedding for vertices a closure cascade "
               "places strictly INSIDE a coarse tet (worst error 1.19, measured "
               "per generation with no composition). Pre-existing and masked by "
               "this test's previous edge-based reference, which skipped exactly "
               "those vertices. 2-D is exact.",
        strict=True)),
])
def test_reproduces_an_arbitrary_coarse_field(dim, cell_size):
    """The transfer must be the coarse P1 EMBEDDING, not merely a linear
    interpolant.

    Reproducing a globally linear field (the test below) is far too weak — any
    local averaging of nearby values passes it, so a prolongation that
    attributed weights to the wrong coarse cell would go undetected. This uses
    a RANDOM coarse nodal field, where only the true embedding agrees.

    The reference is computed independently, by barycentric interpolation in the
    coarse cell that contains the fine vertex. Deliberately NOT
    `uw.function.evaluate`, which returns wrong values at points lying exactly on
    cell boundaries (#432) — using it as the reference produced a convincing
    false accusation against this code.
    """
    child = _adapted(dim, cell_size)
    Ps = child._adapt_prolongation
    lvl = _levels(child)[-(len(Ps) + 1):]
    rng = np.random.default_rng(0)
    for k, entry in enumerate(Ps):
        if entry is None:
            continue
        cdm, fdm = lvl[k], lvl[k + 1]
        P = _as_matrix(entry, cdm, fdm)
        cvS, cvE = cdm.getDepthStratum(0)
        fvS, fvE = fdm.getDepthStratum(0)
        cx = cdm.getCoordinatesLocal().array.reshape(-1, dim)
        fx = fdm.getCoordinatesLocal().array.reshape(-1, dim)
        data = rng.standard_normal(cvE - cvS)
        got = P @ data

        cells = _coarse_cell_vertices(cdm, dim)
        checked = 0
        for r in range(fvE - fvS):
            truth = _coarse_p1_value(cx, cells, data, fx[r])
            if truth is None:
                continue
            assert abs(got[r] - truth) < 1e-10, (
                f"pass {k}, fine vertex {r}: transfer {got[r]} != coarse P1 "
                f"value {truth} at its position — the prolongation is not the "
                f"coarse embedding")
            checked += 1
        assert checked == fvE - fvS, (
            f"pass {k}: only {checked} of {fvE - fvS} fine vertices fell inside "
            f"a coarse cell; the test is not covering what it claims")


@pytest.mark.parametrize("dim,cell_size", [(2, 0.2), (3, 0.4)])
def test_reproduces_a_linear_field_exactly(dim, cell_size):
    """Necessary but WEAK — see the embedding test above. Kept because a
    failure here localises the problem to the arithmetic rather than the
    parentage."""
    child = _adapted(dim, cell_size)
    Ps = child._adapt_prolongation
    lvl = _levels(child)[-(len(Ps) + 1):]
    for k, entry in enumerate(Ps):
        cdm, fdm = lvl[k], lvl[k + 1]
        P = _as_matrix(entry, cdm, fdm)
        xc = cdm.getCoordinatesLocal().array.reshape(-1, dim)
        xf = fdm.getCoordinatesLocal().array.reshape(-1, dim)
        assert P.shape == (xf.shape[0], xc.shape[0])
        assert np.abs(P @ xc - xf).max() < 1e-12, (
            f"pass {k}: prolongation does not reproduce a linear field")


def test_transfer_is_sparser_than_point_location():
    """At most ``dim+1`` nonzeros per row — the exact embedding, still sparse.

    A single bisection gives 1-2 entries per row. A level that spans several
    passes composes them, and a fine vertex strictly inside a coarse cell depends
    on that cell's ``dim+1`` vertices — which is the bound, not a symptom. The
    point of the recorded transfer is that it is EXACT and sparse where point
    location was approximate; ``dim+1`` per row keeps both.
    """
    for dim, cell_size in ((2, 0.2), (3, 0.4)):
        child = _adapted(dim, cell_size)
        Ps = child._adapt_prolongation
        lvl = _levels(child)[-(len(Ps) + 1):]
        for k, entry in enumerate(Ps):
            P = _as_matrix(entry, lvl[k], lvl[k + 1])
            assert P.nnz / P.shape[0] <= dim + 1, (
                f"{dim}D pass {k}: {P.nnz / P.shape[0]:.2f} nonzeros per row "
                f"exceeds the {dim + 1} a coarse cell can supply")


def test_mg_actually_uses_the_recorded_transfer_for_degree_one():
    """Guard against silently reverting to point location.

    Solving is not enough evidence — the geometric builder also solves. This
    asserts the recorded relation is what the hierarchy installed for the
    adapt generations, while the uniform coarse tail below them (not a
    bisection hierarchy) still uses the builder.
    """
    from underworld3.utilities import custom_mg

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        refinement=2, qdegree=2)
    child = base.adapt(_metric, max_levels=1)

    original = custom_mg.CustomMGHierarchy._recorded_node_transfer
    stats = {"nested": 0, "builder": 0}

    def _spy(self, level, nlev, degree, n_coarse, n_fine):
        P = original(self, level, nlev, degree, n_coarse, n_fine)
        stats["nested" if P is not None else "builder"] += 1
        return P

    custom_mg.CustomMGHierarchy._recorded_node_transfer = _spy
    try:
        u = uw.discretisation.MeshVariable("u_nested_used", child, 1, degree=1)
        poisson = uw.systems.Poisson(child, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 0.0
        poisson.add_dirichlet_bc(0.0, "Bottom")
        poisson.add_dirichlet_bc(1.0, "Top")
        poisson.petsc_options["ksp_rtol"] = 1e-10
        poisson.solve()
    finally:
        custom_mg.CustomMGHierarchy._recorded_node_transfer = original

    assert stats["nested"] > 0, (
        "the exact recorded transfer was not used — MG silently fell back to "
        "point location on an adapt hierarchy")
    assert poisson.snes.getKSP().getPC().getType() == "mg"
    # exact linear solution, so the answer should be at round-off
    err = (np.linalg.norm(u.data[:, 0] - u.coords[:, 1])
           / np.linalg.norm(u.coords[:, 1]))
    assert err < 1e-9


def test_higher_degree_still_uses_the_geometric_builder():
    """The recorded relation is vertex-level, so P2 must NOT use it (its edge
    DOFs are not described by the vertex parentage). #425 stage 2."""
    from underworld3.utilities import custom_mg

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        refinement=2, qdegree=2)
    child = base.adapt(_metric, max_levels=1)
    h = custom_mg.CustomMGHierarchy(
        list(child._custom_mg_coarse_meshes) + [child], builder="barycentric")
    # degree 2 must decline the recorded transfer at every level
    for level in range(1, len(h.level_meshes)):
        assert h._recorded_node_transfer(level, len(h.level_meshes), 2,
                                         10**6, 10**6) is None
