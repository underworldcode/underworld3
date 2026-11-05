"""Exact velocity graph/layout without globally replicated submatrix indices."""

from contextlib import ExitStack

import numpy as np
from petsc4py import PETSc
import pytest

from underworld3.utilities._spherical_gap_preconditioner import _velocity_norm_graph

pytestmark = pytest.mark.level_2


@pytest.mark.parametrize("velocity_size", [3, 24])
def test_velocity_graph_preserves_zero_connections_and_field_layout(velocity_size):
    comm = PETSc.COMM_WORLD
    size = velocity_size + 13
    with ExitStack() as cleanup:
        original = PETSc.Mat().createAIJ((size, size), nnz=3, comm=comm)
        cleanup.callback(original.destroy)
        lower, upper = original.getOwnershipRange()
        for row in range(lower, upper):
            # The velocity ring crosses owners. A stored zero must survive;
            # the last column is a pressure/multiplier coupling to exclude.
            columns = sorted({row, (row+3) % velocity_size, size-1})
            original.setValues(row, columns, np.zeros(len(columns)))
        original.assemble()
        layout = PETSc.Vec().createMPI(velocity_size//3, comm=comm)
        cleanup.callback(layout.destroy)
        first, last = (3*index for index in layout.getOwnershipRange())
        graph = _velocity_norm_graph(original, velocity_size, last-first)
        cleanup.callback(graph.destroy)
        assert graph.getOwnershipRange() == (first, last)
        assert graph.getSize() == (velocity_size, velocity_size)
        assert graph.query("ISAllGather") is None
        assert graph.norm() == 0
        entries = []
        for row in range(first, last):
            columns, values = graph.getRow(row)
            assert set(columns) == {row, (row+3) % velocity_size}
            assert not np.any(values)
            entries.append((row, columns.copy()))
        for row, columns in entries:
            graph.setValues(row, columns, columns+row+1.)
        graph.assemble()
        x = graph.createVecRight()
        y = graph.createVecLeft()
        cleanup.callback(x.destroy)
        cleanup.callback(y.destroy)
        x.set(1)
        graph.mult(x, y)
        expected = [sum(column+row+1. for column in {row, (row+3) % velocity_size})
                    for row in range(first, last)]
        np.testing.assert_array_equal(y.getArray(readonly=True), expected)
        assert original.norm() == 0, "Graph construction must not mutate mixed entries"


def test_velocity_graph_matches_original_submatrix_pattern():
    comm = PETSc.COMM_WORLD
    velocity_size, size = 3*comm.size, 4*comm.size+1
    with ExitStack() as cleanup:
        original = PETSc.Mat().createAIJ((size, size), nnz=3, comm=comm)
        cleanup.callback(original.destroy)
        lower, upper = original.getOwnershipRange()
        for row in range(lower, upper):
            columns = sorted({row, (row+1) % velocity_size, size-1})
            original.setValues(row, columns, np.zeros(len(columns)))
        original.assemble()
        rows = PETSc.IS().createStride(3, first=3*comm.rank, step=1, comm=comm)
        cleanup.callback(rows.destroy)
        previous = original.createSubMatrix(rows, rows)
        cleanup.callback(previous.destroy)
        graph = _velocity_norm_graph(original, velocity_size, 3)
        cleanup.callback(graph.destroy)
        assert graph.getOwnershipRange() == previous.getOwnershipRange()
        assert graph.getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"] == previous.getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"]
        for row in range(*graph.getOwnershipRange()):
            expected, _ = previous.getRow(row)
            columns, _ = graph.getRow(row)
            np.testing.assert_array_equal(columns, expected)
