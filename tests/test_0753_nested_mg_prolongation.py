"""Exact nested MG prolongations recorded by ``mesh.adapt`` (#425).

``adapt`` maintains an exact refinement hierarchy; the MG transfer used to
discard it and re-derive an approximation by Delaunay point location, which
is what made #424 possible (a coarse DOF with no fine image -> zero column
-> singular coarse operator).

The recorded transfer is the true P1 embedding: every fine vertex is an
inherited coarse vertex (weight 1) or a midpoint (1/2, 1/2), composed
through any closure cascade. The properties asserted here are what make it
better than point location, not merely different.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _metric(centroids):
    d = np.abs(np.asarray(centroids)[:, -1] - 0.5)
    return 1.0 / np.minimum(np.sqrt(0.05**2 + (2.0 * d) ** 2), 0.3) ** 2


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


@pytest.mark.parametrize("dim,cell_size", [(2, 0.2), (3, 0.4)])
def test_reproduces_a_linear_field_exactly(dim, cell_size):
    """P1 interpolation of a linear function is that function."""
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
    """1-2 nonzeros per row, vs dim+1 for a barycentric point-located row."""
    child = _adapted(3, 0.4)
    Ps = child._adapt_prolongation
    lvl = _levels(child)[-(len(Ps) + 1):]
    for k, entry in enumerate(Ps):
        P = _as_matrix(entry, lvl[k], lvl[k + 1])
        assert P.nnz / P.shape[0] <= 2.0


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
