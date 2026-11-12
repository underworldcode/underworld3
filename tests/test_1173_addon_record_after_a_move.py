"""The add-on record of a moved mesh is the one it was built with (#1091).

``addon_partition.partition_addon`` records, for every finest cell, its parent on
the level below, and ``fault_mg.provenance_transfer_parallel`` builds the
multigrid transfer of that pair from the record. A move (``mesh.deform``, the
MMPDE mover) changes the finest level's coordinates and leaves the level below
where it was. Before the fix both read the moved coordinates: a cell's parent was
the coarse cell under its moved centroid (635 of 1440 cells on this fixture, and a
refusal in parallel where that cell is on another rank), and a moved node was
weighted in its unmoved parent.

The partition is skipped in serial, so these tests call the two functions it is
made of, on a record set the way the partition sets it. The fixture is a unit box
refined twice and moved by one fine cell in the interior, 0.7 of a cell of the
level below. The parallel route is in
tests/parallel/test_1173_addon_record_after_a_move_parallel.py.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.spatial import cKDTree

import underworld3 as uw
from underworld3.utilities import addon_partition as ap
from underworld3.utilities import fault_mg
from underworld3.utilities.custom_mg import LevelLayout

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

H_FINE = 1 / 24
TRACE = np.array([[0.30, 0.45], [0.70, 0.55]])


def _box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6, refinement=2, qdegree=2)


def _shift(X):
    w = H_FINE * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    return np.column_stack([w, w])


def _move(mesh):
    """One fine cell in the interior, nothing on the walls. Returns the
    coordinates the mesh was built with."""
    X0 = np.asarray(mesh.X.coords).copy()
    mesh.deform(X0 + _shift(X0))
    return X0


def _before_the_move(X):
    """Where the move took the points ``X`` from (it is a contraction)."""
    X0 = X.copy()
    for _ in range(60):
        X0 = X - _shift(X0)
    return X0


def _record(mesh, below, parents):
    """The record the partition leaves on a level it moved."""
    mesh._addon_below = below
    mesh._addon_parent = np.column_stack(
        [np.zeros(len(parents), dtype=np.int64), parents])


def _transfer(below, lev, degree):
    """The provenance transfer of the pair, one unknown a node, with the node
    coordinates of the two levels."""
    Xc = np.asarray(below._get_coords_for_basis(degree, True))
    Xf = np.asarray(lev._get_coords_for_basis(degree, True))

    def layout(n):
        return LevelLayout(np.arange(n, dtype=np.int64), 0, n, n)

    P = fault_mg.provenance_transfer_parallel(
        below, lev, lev, degree, True, Xc, Xf, layout(len(Xc)), layout(len(Xf)),
        1, PETSc.COMM_WORLD)
    indptr, indices, data = P.getValuesCSR()
    P.destroy()
    return sp.csr_matrix((data, indices, indptr), shape=(len(Xf), len(Xc))), Xc, Xf


def test_a_moved_mesh_keeps_the_parents_it_was_refined_from():
    mesh = _box()
    below = mesh._coarse_level_meshes()[-2]
    before = ap._parent_cells_local(below, mesh)
    # four children a parent: the relation of the refinement
    assert (np.bincount(before) == 4).all()
    _move(mesh)
    after = ap._parent_cells_local(mesh._coarse_level_meshes()[-2], mesh)
    assert (after != before).sum() == 0


@pytest.mark.parametrize("degree", [1, 2])
def test_the_transfer_of_a_moved_pair_is_the_one_it_had_unmoved(degree):
    mesh = _box()
    below = mesh._coarse_level_meshes()[-2]
    _record(mesh, below, ap._parent_cells_local(below, mesh))
    P0, Xc, X0 = _transfer(below, mesh, degree)
    # unmoved, the transfer interpolates the coarse coordinates to the fine nodes
    assert np.abs(P0 @ Xc - X0).max() < 1e-12
    # the weights are the parent's basis at barycentric coordinates in halves
    # (degree 1) or quarters (degree 2)
    nested = {1: [0.0, 0.5, 1.0], 2: [-0.125, 0.0, 0.25, 0.375, 0.5, 0.75, 1.0]}
    assert np.abs(P0.data[:, None] - np.array(nested[degree])).min(axis=1).max() < 1e-9
    _move(mesh)
    P1, _Xc, _X1 = _transfer(below, mesh, degree)
    assert abs(P1 - P0).max() < 1e-12


def test_a_cut_made_on_a_moved_mesh_reaches_the_level_below_through_it():
    """The cut and the split inherit the record from the mesh they were made on.
    Their cells have no refinement relation to the level below: a node reaches
    its parent through the cell of the moved mesh it lies in. Where a node of
    the split mesh is a node of the moved mesh, the transfer has to give it the
    position that node was built at."""
    mesh = _box()
    below = mesh._coarse_level_meshes()[-2]
    _record(mesh, below, ap._parent_cells_local(below, mesh))
    X0 = _move(mesh)
    child = mesh.add_fault([("Flt", TRACE)])
    assert fault_mg.provenance_source(below, child) is child
    P, Xc, Xf = _transfer(below, child, 1)
    distance, node = cKDTree(np.asarray(mesh.X.coords)).query(Xf)
    shared = distance < 1e-12
    # the cut adds a few nodes on the trace; the rest are the moved mesh's
    assert shared.sum() >= len(X0) - 5
    assert np.abs((P @ Xc)[shared] - X0[node[shared]]).max() < 1e-10
    # A vertex the cut added or pulled onto the trace is no vertex of the moved
    # mesh. It is taken back through the cell it lies in, which undoes the move
    # to the accuracy of a straight-sided cell: 1e-4 here, where the move is
    # 0.03. (The transfer keeps such a node's weights inside its parent, moved
    # or not, so the position is checked and not the weights.)
    held_for, reference = child._addon_reference
    assert held_for is below
    vertices, _cells = fault_mg._cell_vertex_table(child)
    added = cKDTree(np.asarray(mesh.X.coords)).query(vertices)[0] > 1e-12
    assert added.sum() > 20
    assert np.abs(reference - _before_the_move(vertices))[added].max() < 2e-4
    assert np.abs(reference - vertices)[added].max() > 0.02


@pytest.mark.parametrize("child_moves", [False, True])
def test_a_cut_child_of_a_moved_mesh_is_partitioned_where_the_mesh_was_made(child_moves):
    """With no record on the mesh (serial, or a partition that moved nothing)
    the cut child takes the mesh's place over the level below, and the next
    fault's partition asks for the child's parents there. They are the parents
    of the cells of the moved mesh the child's cells lie in."""
    mesh = _box()
    below = mesh._coarse_level_meshes()[-2]
    parent_of_host = ap._parent_cells_local(below, mesh)
    _move(mesh)
    child = mesh.add_fault([("Flt", TRACE)])
    centroid, _v, _cv = ap._cell_centroids(child)
    _c, host_vertices, host_cells = ap._cell_centroids(mesh)
    host, _lam = fault_mg.locate_in_cells_exhaustive(centroid, host_vertices, host_cells)
    if child_moves:
        Xc = np.asarray(child.X.coords).copy()
        child.deform(Xc - 0.5 * _shift(Xc))
    base, _static, addon = ap._levels_of(child)
    assert base is below and not addon
    parents = ap._base_parent_addresses(base, child, child.dm.getComm().tompi4py())
    assert (parents[:, 1] != parent_of_host[host]).sum() == 0
