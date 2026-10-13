"""Parallel regression test for ``Mesh.adapt`` variable transfer.

The transfer routes through ``uw.function.evaluate`` → ``global_evaluate_nd``
→ a transient swarm. This test verifies a degree-2 field round-trips on
N>1 ranks within RBF tolerance.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0763_mesh_adapt_transfer.py
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


def _smooth_field(coords):
    return np.sin(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])


@pytest.mark.skip(
    reason="PETSc adaptMetric (MMG/pragmatic backend) raises error code 63 on "
    "N>1 ranks for the simplex meshes UW3 uses — parallel mesh adaptation "
    "itself is not supported in the current PETSc build. Once that lands, "
    "remove this skip; the transfer path on the UW3 side is already "
    "exercised end-to-end in tests/test_0830_mesh_adapt_variable_transfer.py."
)
@pytest.mark.mpi(min_size=2)
@pytest.mark.level_2
@pytest.mark.tier_a
def test_adapt_preserves_degree_2_field_in_parallel():
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16.0,
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = _smooth_field(T.coords)

    H = uw.discretisation.MeshVariable("H", mesh, 1, degree=1)
    H.array[:, 0, 0] = np.where(
        H.coords[:, 0] > 0.5, 1.0 / 0.05 ** 2, 1.0 / 0.1 ** 2
    )

    mesh.remesh(H)

    T2 = mesh.vars["T"]
    expected = _smooth_field(T2.coords)
    actual = np.asarray(T2.array[:, 0, 0])
    max_err = float(np.abs(expected - actual).max())

    # Gather max errors from all ranks for a global view
    global_max_err = uw.mpi.comm.allreduce(max_err, op=uw.mpi.MPI.MAX)

    assert np.asarray(T2.array).max() > 0.5, (
        f"[rank {uw.mpi.rank}] T appears to have been reset by adapt"
    )
    assert global_max_err < 0.1, (
        f"global max err = {global_max_err:.3e} (this rank: {max_err:.3e})"
    )
