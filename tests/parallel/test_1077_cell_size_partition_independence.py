"""Rank-count regression for the cell-local stabilization length.

The parallel result is compared with a fresh single-rank run on the same Gmsh
mesh.  This checks the complete cell geometry table, not merely a reduction or
a within-rank geometric identity.
"""

import numpy as np
import pytest

import underworld3 as uw

from serial_reference import emit, mesh_fingerprint, serial_reference


pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(300)]


def _cell_geometry_table():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.12, qdegree=2)
    mesh.cell_size()

    local = np.column_stack((mesh._centroids, mesh._cell_radii))
    local = local[mesh._get_owned_cells_mask()]
    gathered = uw.mpi.comm.allgather(local)
    table = np.vstack(gathered)
    order = np.lexsort(tuple(table[:, axis] for axis in reversed(range(mesh.dim))))
    return table[order].reshape(-1), mesh_fingerprint(mesh)


def test_cell_size_matches_single_rank_cell_by_cell():
    values, fingerprint = _cell_geometry_table()
    reference = serial_reference(__file__, "cell_size")

    assert int(fingerprint[0]) == int(reference["fingerprint"][0])
    assert np.isclose(
        fingerprint[1], reference["fingerprint"][1], rtol=1.0e-12, atol=0.0
    )

    expected = np.asarray(reference["values"])
    assert values.shape == expected.shape
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1.0e-14)


if __name__ == "__main__":
    _values, _fingerprint = _cell_geometry_table()
    emit(_values, _fingerprint)
