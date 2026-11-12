"""A symmetric tensor survives a mesh move in parallel. See tests/test_0769_remap_symmetric_tensor.py."""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(120)]


@pytest.mark.mpi(min_size=2)
def test_a_symmetric_tensor_is_remapped_across_a_move():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    S = uw.discretisation.MeshVariable("S0769", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)

    def linear_tensor(coords, shape):
        x, y = coords[:, 0], coords[:, 1]
        values = np.zeros(shape)
        values[:, 0, 0] = 2.0 + x
        values[:, 1, 1] = 3.0 + 2.0 * y
        values[:, 0, 1] = values[:, 1, 0] = 5.0 + x - y
        return values

    S.array[...] = linear_tensor(np.asarray(S.coords), np.asarray(S.array).shape)
    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    coords[(radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9)] *= 1.02
    assert all(uw.mpi.comm.allgather(mesh.deform(coords)))

    remapped = np.asarray(S.array)
    exact = linear_tensor(np.asarray(S.coords), remapped.shape)
    interior = np.linalg.norm(np.asarray(S.coords), axis=1) < 0.99
    error = float(np.abs(remapped - exact)[interior].max()) if interior.any() else 0.0
    assert uw.mpi.comm.allreduce(error, op=uw.MPI.MAX) < 1.0e-9   # P1 is exact on a linear field
