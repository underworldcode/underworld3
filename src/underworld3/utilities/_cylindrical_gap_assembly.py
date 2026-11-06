"""Distributed annulus block insertion and mixed graph allocation."""
import numpy as np
from petsc4py import PETSc

def _insert_blocks(matrix, velocity_ids, pressure_ids, blocks, viscosity, sign):
    for u, p, a, b in zip(
        velocity_ids, pressure_ids, blocks["A"], blocks["B"], strict=True
    ):
        matrix.setValues(
            u,
            u,
            np.ascontiguousarray(sign * viscosity * a),
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        matrix.setValues(
            p, u, np.ascontiguousarray(sign * b), addv=PETSc.InsertMode.ADD_VALUES
        )
        matrix.setValues(
            u, p, np.ascontiguousarray(sign * b.T), addv=PETSc.InsertMode.ADD_VALUES
        )

def _create_gap_matrix(
    size: int,
    velocity_ids: np.ndarray,
    pressure_ids: np.ndarray,
    wall_couplings: list[tuple[np.ndarray, np.ndarray]],
    comm: PETSc.Comm,
) -> PETSc.Mat:
    """Preallocate the distributed mixed graph, including remote circle couplings.

    PETSc's MATPREALLOCATOR routes the known cell connectivity to row owners;
    filling the graph with zeros prevents storage growth during numeric insertion.
    See https://petsc.org/release/manualpages/Mat/MatPreallocatorPreallocate/.
    The graph is the same A/B/B-transpose and circle C/C-transpose pattern used
    by the variational assembly. Stored zero diagonals support PETSc's row/column
    pinning; they add no pressure stabilization or physical matrix term.
    """
    graph = PETSc.Mat().create(comm=comm)
    graph.setSizes((size, size))
    graph.setType(PETSc.Mat.Type.PREALLOCATOR)
    graph.setUp()
    zero_a = np.zeros((12, 12))
    zero_b = np.zeros((3, 12))
    zero_bt = np.zeros((12, 3))
    for u, p in zip(velocity_ids, pressure_ids, strict=True):
        graph.setValues(u, u, zero_a)
        graph.setValues(p, u, zero_b)
        graph.setValues(u, p, zero_bt)
    for uids, lids in wall_couplings:
        zero_c = np.zeros((lids.shape[1], 12))
        zero_ct = np.zeros((12, lids.shape[1]))
        for u, l in zip(uids, lids, strict=True):
            graph.setValues(l, u, zero_c)
            graph.setValues(u, l, zero_ct)
    lo, hi = graph.getOwnershipRange()
    for row in range(lo, hi):
        graph.setValue(row, row, 0.0)
    graph.assemble()
    matrix = PETSc.Mat().create(comm=comm)
    matrix.setSizes((size, size))
    matrix.setType(PETSc.Mat.Type.AIJ)
    graph.preallocatorPreallocate(matrix, fill=True)
    graph.destroy()
    return matrix
