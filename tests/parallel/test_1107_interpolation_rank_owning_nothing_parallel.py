"""Interpolation within one mesh on a rank that holds cells but owns none of their DOFs.

PETSc's ``DMPlexComputeInterpolatorNested`` builds the matrix between two DMs on
one plex (a coordinate DM and a basis DM, a variable's sub-DM and a scratch
clone). It skips its preallocation pass on a rank with no local rows and no
local columns, but runs its fill pass over that rank's cells all the same. A
shared point belongs to the highest rank that holds it, so a rank whose cells
sit wholly inside higher ranks' regions owns nothing. Its entries then reach
owners that never preallocated them: the owner raises "New nonzero caused a
malloc" inside ``MatAssemblyEnd``, and the rank that did not raise waits in the
assembly's own collective for ever. Found on kaiju at np 32 and 52 on the
124-cell coarsest level of the faulted annulus, where custom-P multigrid reads
each level's node coordinates (#902), and on a 3-D notch mesh at np 16.

A shell partition gives rank 0 one interior cell of a 4 x 4 grid of triangles
and the other ranks the rest, so rank 0 owns none of its cell's vertices and
edges. The test checks that case is set up, then that the node coordinates and
the projection between degrees come back right on every rank. Before the fix
the degree-2 coordinates hung, so each test carries a thread timeout, which
fails the run instead of stalling it.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1107_interpolation_rank_owning_nothing_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.function import project_to_degree

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(120, method="thread")]

N = 4


def _owned_nothing_mesh():
    """The 4 x 4 grid of triangles, all on rank 0, partitioned so rank 0 keeps one
    interior cell (every edge shared with a cell of another rank)."""
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    if rank == 0:
        xs = np.linspace(0.0, 1.0, N + 1)
        coords = np.array([[x, y] for y in xs for x in xs])
        cells = []
        for j in range(N):
            for i in range(N):
                a = j * (N + 1) + i
                cells += [[a, a + 1, a + N + 2], [a, a + N + 2, a + N + 1]]
        cells = np.asarray(cells, dtype=PETSc.IntType)
    else:
        coords = np.zeros((0, 2))
        cells = np.zeros((0, 3), dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)
    part = dm.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    if rank == 0:
        cS, cE = dm.getHeightStratum(0)
        interior = [c for c in range(cS, cE)
                    if all(dm.getSupportSize(e) == 2 for e in dm.getCone(c))]
        c0 = interior[len(interior) // 2]
        rest = np.array_split(np.asarray([c for c in range(cS, cE) if c != c0]), size - 1)
        part.setShellPartition(size, [1] + [len(r) for r in rest],
                               [c0] + [int(c) for r in rest for c in r])
    else:
        part.setShellPartition(size, [0] * size, [])
    dm.distribute()
    return uw.discretisation.Mesh(dm, degree=1, qdegree=2)


@pytest.fixture(scope="module")
def mesh():
    return _owned_nothing_mesh()


def _all(flag):
    return uw.mpi.comm.allreduce(bool(flag), op=MPI.LAND)


def _cell_vertex_coords(mesh):
    """Each local cell's vertex coordinates, read from the coordinate section."""
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    sec = dm.getCoordinateSection()
    x = dm.getCoordinatesLocal().array
    cS, cE = dm.getHeightStratum(0)
    out = []
    for c in range(cS, cE):
        pts = dm.getTransitiveClosure(c)[0]
        out.append(np.array([x[sec.getOffset(v):sec.getOffset(v) + 2]
                             for v in pts if vS <= v < vE]))
    return out


def _sorted_rows(a):
    a = np.round(np.asarray(a), 12)
    return a[np.lexsort(a.T[::-1])]


def test_rank_zero_holds_a_cell_and_owns_nothing(mesh):
    """The case under test: rank 0 holds one cell and owns no vertex or edge."""
    cS, cE = mesh.dm.getHeightStratum(0)
    owned = 0
    for d in (0, 1):
        pS, pE = mesh.dm.getDepthStratum(d)
        sec = mesh.dm.getPointSF()
        leaves = set(np.asarray(sec.getGraph()[1]).tolist())
        owned += sum(1 for p in range(pS, pE) if p not in leaves)
    if uw.mpi.rank == 0:
        assert cE - cS == 1 and owned == 0, (cE - cS, owned)
    assert _all(True)


@pytest.mark.parametrize("degree", [1, 2])
def test_node_coordinates_on_every_rank(mesh, degree):
    """Each cell's nodes are its vertices (degree 1), and its vertices and edge
    midpoints (degree 2)."""
    coords = mesh._get_coords_for_basis(degree, True)
    rows = mesh._cell_node_indices(degree, True)
    ok = True
    for k, verts in enumerate(_cell_vertex_coords(mesh)):
        expected = verts if degree == 1 else np.vstack(
            [verts, 0.5 * (verts[[0, 1, 2]] + verts[[1, 2, 0]])])
        got = coords[rows[k]]
        ok &= np.allclose(_sorted_rows(got), _sorted_rows(expected), atol=1e-12)
    assert _all(ok)


def test_projection_between_degrees_on_every_rank(mesh):
    """P1 to P2 of a linear field and P2 to P1 of a quadratic one are exact at the
    nodes."""
    s1 = uw.discretisation.MeshVariable("s1_1107", mesh, 1, degree=1)
    s2 = uw.discretisation.MeshVariable("s2_1107", mesh, 1, degree=2)
    x1 = np.asarray(s1.coords)
    x2 = np.asarray(s2.coords)
    s1.array[...] = (x1[:, 0] + 2.0 * x1[:, 1]).reshape(-1, 1, 1)
    s2.array[...] = (x2[:, 0] ** 2).reshape(-1, 1, 1)

    up = project_to_degree(s1, target_degree=2)
    down = project_to_degree(s2, target_degree=1)
    n2 = mesh._get_coords_for_basis(2, True)
    n1 = mesh._get_coords_for_basis(1, True)
    ok = (np.allclose(up[:, 0], n2[:, 0] + 2.0 * n2[:, 1], atol=1e-12)
          and np.allclose(down[:, 0], n1[:, 0] ** 2, atol=1e-12))
    assert _all(ok)
