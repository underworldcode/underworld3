"""A mesh variable of any degree is drawn on the mesh's own cells (#1057).

``meshVariable_to_native_pv_mesh`` subdivides every cell through the
variable's own nodes. The cell-to-DOF map comes from the DMPlex closure and
the variable's section, so it holds on curved and moved cells, where matching
lattice positions alone would not. Until #1057 the native path handled only
continuous P1, and every other variable fell back to a Delaunay triangulation
of its nodes, whose alpha filter drops coarse cells on a graded mesh.

Each case checks exact, closed-form properties of the result:

* every cell is kept: there are ``cells * k**dim`` sub-cells;
* point ``i`` is the variable's DOF ``i``, so ``T.data`` attaches directly. A
  field that is a polynomial of degree ``k`` reads back exactly at every
  sub-cell vertex;
* the sub-cells tile the domain conformingly: no sub-cell is inverted, no
  edge is shared by more than two sub-cells (in 2D), and the boundary has
  exactly ``k**(dim-1)`` sub-facets for each boundary facet of the mesh. A
  node misplaced on an edge shared by two cells breaks that count;
* within each cell, the sub-cells fill it exactly, and every cell is cut
  into the same set of size ratios (the cells here are straight-sided, so all
  are images of one reference lattice). That catches a node on the wrong
  lattice site even inside a cell, where the tiling stays conforming;
* on a box, the sub-cell volumes sum to the box volume exactly.
"""

import numpy as np
import pytest

import underworld3 as uw

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

pytestmark = [pytest.mark.level_2]


def _box(dim, simplex, refinement=0):
    if simplex:
        return uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
            cellSize=0.25 if dim == 2 else 0.5, refinement=refinement)
    return uw.meshing.StructuredQuadBox(elementRes=(3,) * dim if dim == 2 else (2,) * dim)


def _annulus(refinement):
    return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25,
                              refinement=refinement)


def _moved_box():
    mesh = _box(2, True)
    x = np.asarray(mesh.X.coords).copy()
    # Grade the mesh towards x = 0 (a smooth monotone map of the unit square).
    x[:, 0] = x[:, 0] ** 2
    mesh.deform(x)
    return mesh


def _n_cells(mesh):
    cs, ce = mesh.dm.getHeightStratum(0)
    return ce - cs


def _boundary_count(grid, dim):
    if dim == 2:
        return grid.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                          manifold_edges=False,
                                          non_manifold_edges=False).n_cells
    return grid.extract_surface().n_cells


def _check(mesh, degree, exact_volume=None):
    dim = mesh.dim
    name = f"T{degree}"
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=degree)
    X = np.asarray(T.coords)
    poly = (X[:, 0] + 0.5 * X[:, 1]) ** degree + (X[:, -1] ** degree if dim == 3 else 0.0)
    T.data[:, 0] = poly

    grid = vis.meshVariable_to_native_pv_mesh(T)
    assert grid is not None, f"no native grid for degree {degree}"
    assert grid.n_cells == _n_cells(mesh) * degree**dim

    # Point i is DOF i, and the field attaches by row.
    np.testing.assert_array_equal(np.asarray(grid.points)[:, :dim], X[:, :dim])
    grid.point_data[name] = np.asarray(T.data[:, 0])
    p = np.asarray(grid.points)
    expect = (p[:, 0] + 0.5 * p[:, 1]) ** degree + (p[:, 2] ** degree if dim == 3 else 0.0)
    np.testing.assert_allclose(grid.point_data[name], expect, rtol=0, atol=1e-12)

    # Conforming tiling, no inverted sub-cell.
    sizes = grid.compute_cell_sizes(length=False, area=dim == 2, volume=dim == 3)
    size = np.asarray(sizes.cell_data["Area" if dim == 2 else "Volume"])
    assert size.min() > 0.0
    base = vis.mesh_to_pv_mesh(mesh)
    assert _boundary_count(grid, dim) == _boundary_count(base, dim) * degree ** (dim - 1)
    if dim == 2:
        assert grid.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                          manifold_edges=False,
                                          non_manifold_edges=True).n_cells == 0
    # The cells here are straight-sided, so each is an affine (or, for a box's
    # quads and hexahedra, a scaled) image of the reference cell, and its
    # sub-cells are images of the reference lattice's. Within each cell they
    # must fill it exactly (an overlap, or a node on the wrong site, makes them
    # sum to more), and every cell must cut into the same set of size ratios.
    # PETSc's degree-3 nodes are not equispaced, so the ratios are not 1/k**dim.
    base_sizes = base.compute_cell_sizes(length=False, area=dim == 2, volume=dim == 3)
    parent = np.abs(np.asarray(base_sizes.cell_data["Area" if dim == 2 else "Volume"]))
    ratio = size.reshape(len(parent), degree**dim) / parent[:, None]
    np.testing.assert_allclose(ratio.sum(axis=1), 1.0, rtol=1e-9)
    pattern = np.sort(ratio, axis=1)
    np.testing.assert_allclose(pattern, np.broadcast_to(pattern[0], pattern.shape), rtol=1e-9)
    if exact_volume is not None:
        assert size.sum() == pytest.approx(exact_volume, rel=1e-12)
    return grid


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("refinement", [0, 1])
def test_simplex_box_2d(degree, refinement):
    _check(_box(2, True, refinement), degree, exact_volume=1.0)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("refinement", [0, 1])
def test_annulus_curved_boundary(degree, refinement):
    # Higher-order nodes on the boundary sit off the chord of a straight cell;
    # the map must still place them.
    _check(_annulus(refinement), degree)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_graded_moved_mesh(degree):
    _check(_moved_box(), degree, exact_volume=1.0)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_quad_box(degree):
    _check(_box(2, False), degree, exact_volume=1.0)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2])
def test_hex_box(degree):
    _check(_box(3, False), degree, exact_volume=1.0)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_tet_box(degree):
    _check(_box(3, True), degree, exact_volume=1.0)


@pytest.mark.tier_b
def test_pv_mesh_object_uses_the_native_grid_for_higher_order():
    # The public entry point no longer falls back to Delaunay for P2.
    mesh = _moved_box()
    T = uw.discretisation.MeshVariable("T_obj", mesh, 1, degree=2)
    native = vis.meshVariable_to_native_pv_mesh(T)
    assert native is not None
    grid = vis.meshVariable_to_pv_mesh_object(T)
    np.testing.assert_array_equal(np.asarray(grid.cells), np.asarray(native.cells))


@pytest.mark.tier_b
def test_discontinuous_variable_falls_back():
    mesh = _box(2, True)
    T = uw.discretisation.MeshVariable("T_dg", mesh, 1, degree=1, continuous=False)
    assert vis.meshVariable_to_native_pv_mesh(T) is None


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [2, 3])
def test_map_is_topological_off_the_chord(degree):
    # Nodes pushed off their straight-cell positions (as on a curved boundary or
    # a degree-2 coordinate mesh) by a tenth of the lattice spacing, far beyond
    # the 1e-4 of a curved annulus boundary, give the same connectivity.
    from types import SimpleNamespace
    from underworld3.visualisation.visualisation import _lagrange_subcell_connectivity

    mesh = _annulus(0)
    T = uw.discretisation.MeshVariable(f"T_chord{degree}", mesh, 1, degree=degree)
    exact = _lagrange_subcell_connectivity(T)
    rng = np.random.default_rng(1057)
    X = np.asarray(T.coords_nd)
    spacing = 0.25 / degree
    moved = SimpleNamespace(mesh=mesh, degree=degree, continuous=True,
                            coords_nd=X + 0.1 * spacing * rng.uniform(-1, 1, X.shape))
    np.testing.assert_array_equal(_lagrange_subcell_connectivity(moved), exact)
