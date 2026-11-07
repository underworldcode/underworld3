"""A triangle mesh with a rank that holds no cells, whatever the rank count."""
import numpy as np
from petsc4py import PETSc


def cells_along_a_side_leaving_one_rank_empty(size):
    """How finely to split the fixture so that one rank stays empty.

    PETSc calls a DM with cells on one rank alone "not distributed", and ``Mesh``
    then distributes it, so a shell partition cannot empty a rank at np 2 unless
    the mesh has one cell, which has nowhere else to go (which rank it lands on
    is not ours to say). From np 3 up, two or more ranks share the cells and the
    partition stands.
    """
    return 1 if size == 2 else 4


def right_triangle_of_cells(n, ranks_with_cells):
    r"""The triangle with corners (0,0), (1,0), (0,1), split into :math:`n^2` equal
    cells of area :math:`1/(2 n^2)`, shared among the first ``ranks_with_cells``
    ranks by a shell partition. Collective.
    """
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
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
                    triangles.append(
                        [vertex[i + 1, j], vertex[i + 1, j + 1], vertex[i, j + 1]])
        cells = np.asarray(triangles, dtype=PETSc.IntType)

    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)
    part = dm.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    if rank == 0:
        shares = np.array_split(np.arange(len(cells)), ranks_with_cells)
        part.setShellPartition(
            size, [len(s) for s in shares] + [0] * (size - ranks_with_cells),
            [int(c) for s in shares for c in s])
    else:
        part.setShellPartition(size, [0] * size, [])
    dm.distribute()
    return dm
