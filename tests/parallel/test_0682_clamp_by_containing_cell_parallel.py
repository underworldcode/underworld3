"""The monotone clamp bounds a value by the cell it was interpolated in (#682, #994).

``global_evaluate(..., monotone="clamp")`` used to bound each value by the range
of the ``dim + 1`` source DOFs nearest the point, from a kd-tree over the
asking rank's own nodes. That is wrong twice over:

- in serial, the nearest nodes need not enclose the point, so a P1 field that is
  linear in x and y, which the clamp must leave alone, was clipped at 5 of 400
  interior points by up to 1.7e-2;
- in parallel, the asking rank's nodes are the wrong neighbourhood for a point
  another rank owns: 48 and 74 of 400 points wrong at np 2, by up to 0.61;
- on a rank with fewer than ``dim + 1`` DOFs the kd-tree query fails, on that
  rank alone (#994).

The bound is now the per-component range of the source variable's DOFs in the
closure of the cell the evaluator interpolated the value in (P2 edge, face and
cell DOFs included), applied on the rank that evaluated it.

The oracle here does not use the library's locator. On a structured quad mesh
the containing cell of ``(x, y)`` is ``floor(x * nx), floor(y * ny)``, and a Q2
cell's nine nodes are at known coordinates, so the bound is computed from the
field itself.

Serial and parallel alike, no ``min_size`` mark: it runs serially in batch 3 and
under ``--p N``.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0682_clamp_by_containing_cell_parallel.py
"""
import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw

from empty_rank_fixtures import cells_along_a_side_leaving_one_rank_empty, right_triangle_of_cells

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

NX = NY = 6


def _fill(var, f):
    """Set every stored column of ``var`` to ``f(coords, i, j)`` for its entry."""
    coords = np.asarray(var.coords)
    data = np.array(var.data)
    rows, columns = var.sym.shape
    for i in range(rows):
        for j in range(columns):
            data[:, var._data_layout(i, j)] = f(coords, i, j)
    var.data[...] = data


def _step(c, i=0, j=0):
    """Steep enough that a Q2 interpolant overshoots its nodes, so the clamp acts."""
    return (1 + i + 2 * j) * np.tanh(12.0 * (c[:, 0] - 0.41)) + 0.3 * (i - j) * np.sin(4.0 * c[:, 1])


def _points(n, seed, lo=0.02, hi=0.98):
    """Random points, the same on every rank, kept off the quad mesh's cell faces."""
    pts = np.random.default_rng(seed).uniform(lo, hi, size=(4 * n, 2))
    frac = np.minimum((pts * NX) % 1.0, 1.0 - (pts * NX) % 1.0)
    return pts[(frac > 1e-3).all(axis=1)][:n]


def _q2_cell_range(pts, f):
    """Min and max of ``f`` over the nine Q2 nodes of the structured cell holding each point."""
    i = np.floor(pts[:, 0] * NX)
    j = np.floor(pts[:, 1] * NY)
    nodes = [np.stack([(i + a) / NX, (j + b) / NY], axis=1)
             for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0)]
    values = np.stack([f(node) for node in nodes], axis=1)
    return values.min(axis=1), values.max(axis=1)


def _quad_mesh():
    return uw.meshing.StructuredQuadBox(elementRes=(NX, NY), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))


def _allreduce_max(x):
    from mpi4py import MPI
    return uw.mpi.comm.allreduce(float(np.max(x, initial=0.0)), op=MPI.MAX)


def test_a_linear_p1_field_is_left_exact():
    """H6's probe: P1 scalar and vector linear in x and y, 400 interior points.

    The interpolant is a convex combination of its cell's nodes, so the cell
    bound never acts. Development clipped 5 of 400 in serial and 48 to 74 at np 2.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.125)
    T = uw.discretisation.MeshVariable("T682a", mesh, 1, degree=1)
    V = uw.discretisation.MeshVariable("V682a", mesh, 2, degree=1)
    _fill(T, lambda c, i, j: 3 * c[:, 0] + 2 * c[:, 1])
    _fill(V, lambda c, i, j: 3 * c[:, 0] + 2 * c[:, 1] if j == 0 else c[:, 0] - c[:, 1])
    pts = np.random.default_rng(5).uniform(0.15, 0.85, size=(400, 2))
    exact = np.stack([3 * pts[:, 0] + 2 * pts[:, 1], pts[:, 0] - pts[:, 1]], axis=1)

    c = np.asarray(uw.function.global_evaluate(T.sym, pts, monotone="clamp")).reshape(len(pts))
    cv = np.asarray(uw.function.global_evaluate(V.sym, pts, monotone="clamp")).reshape(len(pts), 2)

    scalar_error = _allreduce_max(np.abs(c - exact[:, 0]))
    vector_error = _allreduce_max(np.abs(cv - exact))
    assert scalar_error < 1e-10, f"rank {uw.mpi.rank}: clamped linear scalar off by {scalar_error:.2e}"
    assert vector_error < 1e-10, f"rank {uw.mpi.rank}: clamped linear vector off by {vector_error:.2e}"


@pytest.mark.parametrize("degree", [1, 2])
def test_the_bound_is_the_containing_cells_nodal_range(degree):
    """Clamped value == the unclamped value clipped to its own cell's nodal range.

    Q2: the bound is the cell's nine nodes, edge and cell DOFs included, and the
    clamp acts. Q1: the bilinear interpolant cannot leave its four corners, so
    the clamp must change nothing.
    """
    mesh = _quad_mesh()
    T = uw.discretisation.MeshVariable(f"T682b{degree}", mesh, 1, degree=degree)
    _fill(T, lambda c, i, j: _step(c))
    pts = _points(400, seed=11)

    plain = np.asarray(uw.function.global_evaluate(T.sym, pts)).reshape(len(pts))
    clamped = np.asarray(uw.function.global_evaluate(T.sym, pts, monotone="clamp")).reshape(len(pts))
    if degree == 2:
        lo, hi = _q2_cell_range(pts, _step)
        expected = np.clip(plain, lo, hi)
        acted = int(np.count_nonzero(np.abs(expected - plain) > 1e-8))
        assert acted > 0, "the Q2 step field should overshoot its nodes somewhere"
    else:
        expected = plain

    error = _allreduce_max(np.abs(clamped - expected))
    assert error < 1e-10, (
        f"rank {uw.mpi.rank}: degree {degree} clamp differs from the containing "
        f"cell's bound by {error:.2e}")


def test_a_matrix_is_bounded_entry_by_entry():
    """A 2x3 MATRIX stores entries by column (0,2,4,1,3,5): each entry is bounded
    by its own component's cell range, whole or one entry at a time."""
    mesh = _quad_mesh()
    M = uw.discretisation.MeshVariable("M682", mesh, (2, 3), vtype=uw.VarType.MATRIX, degree=2)
    _fill(M, _step)
    pts = _points(300, seed=17)

    plain = np.asarray(uw.function.global_evaluate(M.sym, pts)).reshape(len(pts), 2, 3)
    whole = np.asarray(uw.function.global_evaluate(M.sym, pts, monotone="clamp")).reshape(len(pts), 2, 3)
    error = 0.0
    for i in range(2):
        for j in range(3):
            lo, hi = _q2_cell_range(pts, lambda c: _step(c, i, j))
            expected = np.clip(plain[:, i, j], lo, hi)
            one = np.asarray(uw.function.global_evaluate(
                M.sym[i, j], pts, monotone="clamp")).reshape(len(pts))
            error = max(error, float(np.abs(whole[:, i, j] - expected).max()),
                        float(np.abs(one - expected).max()))
    error = _allreduce_max(np.array([error]))
    assert error < 1e-10, f"rank {uw.mpi.rank}: MATRIX clamp off its entry's cell bound by {error:.2e}"


def test_a_p2_bound_counts_the_edge_dofs():
    """P2 on triangles, 0 at every vertex and 1 at every edge midpoint: the
    interpolant is 4/3 at a centroid. The closure range is [0, 1], so the clamp
    gives 1; a bound over vertices alone would give 0."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    T = uw.discretisation.MeshVariable("T682c", mesh, 1, degree=2)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))
    nodes = np.asarray(T.coords)
    if vertices.shape[0]:
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(vertices).query(nodes)
        at_vertex = distance < 1e-10
    else:
        at_vertex = np.zeros(nodes.shape[0], dtype=bool)
    T.data[:, 0] = np.where(at_vertex, 0.0, 1.0)

    centroids = np.asarray(mesh._centroids).reshape(-1, 2)
    plain = np.asarray(uw.function.global_evaluate(T.sym, centroids)).reshape(-1)
    clamped = np.asarray(uw.function.global_evaluate(T.sym, centroids, monotone="clamp")).reshape(-1)
    overshoot = _allreduce_max(np.abs(plain - 4.0 / 3.0))
    assert overshoot < 1e-10, f"the fixture should overshoot to 4/3 (off by {overshoot:.2e})"
    error = _allreduce_max(np.abs(clamped - 1.0))
    assert error < 1e-10, f"rank {uw.mpi.rank}: P2 clamp at a centroid off 1 by {error:.2e}"


def test_a_rank_with_no_query_points():
    """Rank 0 asks for every point; the others ask for none and still take part."""
    mesh = _quad_mesh()
    T = uw.discretisation.MeshVariable("T682d", mesh, 1, degree=2)
    _fill(T, lambda c, i, j: _step(c))
    pts = _points(200, seed=23) if uw.mpi.rank == 0 else np.zeros((0, 2))

    plain = np.asarray(uw.function.global_evaluate(T.sym, pts)).reshape(-1)
    clamped = np.asarray(uw.function.global_evaluate(T.sym, pts, monotone="clamp")).reshape(-1)
    assert clamped.shape[0] == pts.shape[0]
    lo, hi = _q2_cell_range(pts, _step) if pts.shape[0] else (np.zeros(0), np.zeros(0))
    error = _allreduce_max(np.abs(clamped - np.clip(plain, lo, hi)))
    assert error < 1e-10, f"rank {uw.mpi.rank}: clamp off its cell bound by {error:.2e}"


def test_a_rank_with_no_cells():
    """One rank holds no cells and no DOFs. The P1 linear field is left exact."""
    size = uw.mpi.size
    if size == 2:
        pytest.skip("a DM on one rank alone is redistributed; see empty_rank_fixtures")
    n = cells_along_a_side_leaving_one_rank_empty(size)
    dm = right_triangle_of_cells(n, max(size - 1, 1))
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=2)
    T = uw.discretisation.MeshVariable("T682e", mesh, 1, degree=1)
    _fill(T, lambda c, i, j: 1.0 + c[:, 0] + 2.0 * c[:, 1])
    pts = np.random.default_rng(29).uniform(0.05, 0.45, size=(100, 2))

    clamped = np.asarray(uw.function.global_evaluate(T.sym, pts, monotone="clamp")).reshape(-1)
    error = _allreduce_max(np.abs(clamped - (1.0 + pts[:, 0] + 2.0 * pts[:, 1])))
    assert error < 1e-10, f"rank {uw.mpi.rank}: clamp on a rank without cells off by {error:.2e}"


def test_a_rank_with_fewer_dofs_than_the_old_stencil():
    """#994: DG0, one cell on rank 0. The old ``dim + 1`` nearest-DOF query failed
    on that rank alone. A DG0 value is its own cell's only DOF, so the clamp
    changes nothing."""
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    n = 4
    coords = np.zeros((0, 2))
    cells = np.zeros((0, 3), dtype=PETSc.IntType)
    if rank == 0:
        vertex = {}
        for j in range(n + 1):
            for i in range(n + 1 - j):
                vertex[i, j] = len(vertex)
        coords = np.array([[i / n, j / n] for i, j in vertex])
        triangles = []
        for j in range(n):
            for i in range(n - j):
                triangles.append([vertex[i, j], vertex[i + 1, j], vertex[i, j + 1]])
                if i + j < n - 1:
                    triangles.append([vertex[i + 1, j], vertex[i + 1, j + 1], vertex[i, j + 1]])
        cells = np.asarray(triangles, dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)
    if size > 1:
        part = dm.getPartitioner()
        part.setType(PETSc.Partitioner.Type.SHELL)
        if rank == 0:
            rest = np.array_split(np.arange(1, len(cells)), size - 1)
            part.setShellPartition(size, [1] + [len(r) for r in rest],
                                   [0] + [int(c) for r in rest for c in r])
        else:
            part.setShellPartition(size, [0] * size, [])
        dm.distribute()
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=2)
    a = uw.discretisation.MeshVariable("a682", mesh, 1, degree=0, continuous=False)
    _fill(a, lambda c, i, j: 1.0 + c[:, 0] + 2.0 * c[:, 1])

    centroids = np.asarray(a.coords).reshape(-1, 2)
    plain = np.asarray(uw.function.global_evaluate(a.sym, centroids)).reshape(-1)
    clamped = np.asarray(uw.function.global_evaluate(a.sym, centroids, monotone="clamp")).reshape(-1)
    error = _allreduce_max(np.abs(clamped - plain))
    assert error < 1e-12, f"rank {uw.mpi.rank}: DG0 clamp moved a value by {error:.2e}"
