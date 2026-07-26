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
