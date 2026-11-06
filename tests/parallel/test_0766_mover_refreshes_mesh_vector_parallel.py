"""The mover marks the mesh vector stale on every rank, so a solve after a move
sees the remapped field at any rank count. See tests/test_0766_mover_refreshes_mesh_vector.py."""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


@pytest.mark.mpi(min_size=2)
def test_integral_after_a_move_sees_the_remapped_field():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0766", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)), T.coords).reshape(-1)
    mesh.update_lvec()

    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))

    flags = uw.mpi.comm.allgather(bool(mesh._stale_lvec))
    assert all(flags), flags
    seen_by_a_solve = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    T.array[:, 0, 0] = np.array(T.array[:, 0, 0])
    remapped = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    assert seen_by_a_solve == pytest.approx(remapped, rel=1.0e-6)
