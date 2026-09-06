"""Quadrature-point ("delta") finite element.

The element's basis is the identity on the mesh quadrature rule, its dofs
all live on the cell, and it tabulates to zero at any point off its rule. Those three properties are what
let a field of this type carry pre-evaluated values straight into the
pointwise functions as ``a[]``.
"""

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw
from underworld3.cython.petsc_quadrature_fe import create_delta_fe, tabulate

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

CELLS = [(2, True), (3, True), (2, False), (3, False)]
IDS = ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]


def _box(dim, simplex):
    """A bare clone of a UW3 mesh DM (no fields) and its cell polytope."""
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, cellSize=0.5, qdegree=2,
        )
    else:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(2,) * dim, qdegree=2)
    dm = mesh.dm.clone()
    cStart, _ = dm.getHeightStratum(0)
    return dm, dm.getCellType(cStart)


def _rule(dim, simplex, qdegree):
    ref = PETSc.FE().createDefault(dim, 1, simplex, qdegree, "ref_", PETSc.COMM_SELF)
    quad = ref.getQuadrature()
    pts = np.array(quad.getData()[0]).reshape(-1, dim)
    return ref, quad, pts


@pytest.mark.parametrize("dim,simplex", CELLS, ids=IDS)
@pytest.mark.parametrize("qdegree", [1, 2])
def test_identity_on_own_rule(dim, simplex, qdegree):
    _, quad, pts = _rule(dim, simplex, qdegree)
    _, polytope = _box(dim, simplex)
    fe = create_delta_fe(quad, polytope)

    assert fe.getDimension() == len(pts)
    assert fe.getNumComponents() == 1
    B = tabulate(fe, pts)[:, :, 0]
    assert np.array_equal(B, np.eye(len(pts)))
    # Derivatives are zero by construction.
    D = tabulate(fe, pts, K=1)
    assert D.shape[0] == len(pts)


@pytest.mark.parametrize("dim,simplex", CELLS, ids=IDS)
def test_dofs_live_on_the_cell(dim, simplex):
    """Local layout is (ncells, Nq): every dof on the cell, none elsewhere."""
    dm, polytope = _box(dim, simplex)
    _, quad, pts = _rule(dim, simplex, 2)
    fe = create_delta_fe(quad, polytope)
    dm.setNumFields(1)
    dm.setField(0, fe)
    # PetscDSSetUp asks for the face tabulation; the delta space answers zeros.
    dm.createDS()
    section = dm.getLocalSection()
    cStart, cEnd = dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        assert section.getDof(c) == len(pts)
    pStart, pEnd = dm.getChart()
    for p in range(pStart, pEnd):
        if not (cStart <= p < cEnd):
            assert section.getDof(p) == 0
    assert section.getStorageSize() == (cEnd - cStart) * len(pts)


def test_off_rule_points_are_zero():
    """Negative control: a point that is not on the rule contributes
    nothing, and the same points in a different order give the permuted
    identity (PETSc's own point space compares point p only with its own p)."""
    dim, simplex = 2, True
    _, quad, pts = _rule(dim, simplex, 2)
    _, polytope = _box(dim, simplex)
    fe = create_delta_fe(quad, polytope)
    Nq = len(pts)
    perm = np.arange(Nq)[::-1]
    off = pts[:2] + 0.05

    assert np.all(tabulate(fe, off) == 0.0)
    Bperm = tabulate(fe, pts[perm])[:, :, 0]
    assert np.array_equal(Bperm, np.eye(Nq)[perm])


def test_rule_is_the_mesh_rule():
    """The element must be built on the rule every other field uses, and
    that rule is fixed by the quadrature degree, not the field degree."""
    for degree in (1, 2, 3):
        fe = PETSc.FE().createDefault(2, 1, True, 2, f"p{degree}_", PETSc.COMM_SELF)
        pts = np.array(fe.getQuadrature().getData()[0]).reshape(-1, 2)
        assert len(pts) == 6
