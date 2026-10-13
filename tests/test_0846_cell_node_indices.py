"""``Mesh._cell_node_indices`` — which DOF rows belong to which cell (#425 stage 2).

The multigrid transfer needs more than the DOF coordinates: to evaluate a coarse
Lagrange basis inside a parent cell it needs to know WHICH rows of
``_get_coords_for_basis`` are that cell's own nodes. That mapping comes from the
local section of the same coordinate DM the coordinates are read from.

Validated GEOMETRICALLY: every node a cell claims must actually lie inside that
cell, checked with barycentric coordinates built from the cell's vertices. The
section-offset arithmetic under test contributes nothing to that check, so a
wrong offset cannot pass it.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

# nodes of P_k on a d-simplex = C(k+d, d)
NODES_PER_CELL = {(1, 1): 2, (1, 2): 3, (1, 3): 4,
                  (2, 1): 3, (2, 2): 6, (2, 3): 10,
                  (3, 1): 4, (3, 2): 10, (3, 3): 20}


def mesh(dim, cell_size=0.35):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=cell_size, qdegree=3)


def cell_vertex_coords(m):
    """``(n_cells, dim+1, dim)`` vertex coordinates, straight from the plex."""
    dm = m.dm
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    xyz = dm.getCoordinatesLocal().array.reshape(-1, dm.getCoordinateDim())
    out = []
    for c in range(cS, cE):
        verts = [q for q in dm.getTransitiveClosure(c)[0] if vS <= q < vE]
        out.append(xyz[np.asarray(verts) - vS])
    return np.asarray(out)


def barycentric(points, simplex):
    v0 = simplex[0]
    lam = np.linalg.solve((simplex[1:] - v0).T, (points - v0).T).T
    return np.column_stack([1.0 - lam.sum(axis=1), lam])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("continuous", [True, False])
def test_node_count_per_cell_is_the_polynomial_dimension(dim, degree, continuous):
    m = mesh(dim)
    idx = m._cell_node_indices(degree, continuous)
    cS, cE = m.dm.getHeightStratum(0)
    assert idx.shape == (cE - cS, NODES_PER_CELL[(dim, degree)])
    assert idx.dtype == np.int64


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("continuous", [True, False])
def test_every_claimed_node_lies_inside_its_cell(dim, degree, continuous):
    """The load-bearing check. A wrong section offset points at some other
    cell's node, which is then somewhere else in the mesh."""
    m = mesh(dim)
    idx = m._cell_node_indices(degree, continuous)
    coords = np.asarray(m._get_coords_for_basis(degree, continuous))
    verts = cell_vertex_coords(m)

    worst = 0.0
    for k in range(idx.shape[0]):
        lam = barycentric(coords[idx[k]], verts[k])
        worst = min(worst, float(lam.min()))
    assert worst > -1.0e-10, (
        f"a cell claims a node {worst:.2e} outside itself (barycentric) — the "
        f"section offsets do not line up with the coordinate array")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_continuous_nodes_are_covered_exactly_once_between_them(dim, degree):
    """Every row of the coordinate array belongs to at least one cell, and the
    cell count of a shared node matches how many cells actually touch it."""
    m = mesh(dim)
    idx = m._cell_node_indices(degree, True)
    n_nodes = np.asarray(m._get_coords_for_basis(degree, True)).shape[0]
    seen = np.bincount(idx.ravel(), minlength=n_nodes)
    assert seen.size == n_nodes, "a cell claims a row past the end of the coordinate array"
    assert int((seen == 0).sum()) == 0, (
        f"{int((seen == 0).sum())} of {n_nodes} nodes belong to no cell")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_discontinuous_nodes_belong_to_exactly_one_cell(dim, degree):
    """A DG space hangs every DOF off its own cell, so the rows partition."""
    m = mesh(dim)
    idx = m._cell_node_indices(degree, False)
    n_nodes = np.asarray(m._get_coords_for_basis(degree, False)).shape[0]
    seen = np.bincount(idx.ravel(), minlength=n_nodes)
    assert n_nodes == idx.size, "DG node count should equal cells x nodes-per-cell"
    assert np.array_equal(seen, np.ones(n_nodes, dtype=seen.dtype)), (
        "DG rows do not partition the coordinate array")
    # ... and they are laid out cell-block-contiguously, which is what
    # Mesh._build_kd_tree_index_DS already relies on.
    assert np.array_equal(np.sort(idx, axis=1),
                          np.arange(n_nodes).reshape(idx.shape))


def test_quadrilateral_mesh_is_refused():
    """Q_k carries (k+1)^dim nodes, so the total-degree monomial basis the
    transfer builds would not be square against them. Refuse rather than
    return something the caller has to second-guess."""
    m = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
    with pytest.raises(NotImplementedError, match="simplex"):
        m._cell_node_indices(2, True)


@pytest.mark.parametrize("dim", [2, 3])
def test_result_is_cached_per_basis(dim):
    m = mesh(dim)
    first = m._cell_node_indices(2, True)
    assert m._cell_node_indices(2, True) is first
    assert m._cell_node_indices(3, True) is not first
