"""A mesh move stays in its frame in a units model, in parallel.
See tests/test_0770_mesh_move_in_a_units_model.py.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(120)]


@pytest.mark.mpi(min_size=2)
def test_deform_keeps_the_mesh_in_its_frame_and_remaps_a_field_with_units():
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), temperature_diff=uw.quantity(1000, "K"),
        viscosity=uw.quantity(1e21, "Pa*s"), diffusivity=uw.quantity(1e-6, "m**2/s"))
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    T = uw.discretisation.MeshVariable("T0770", mesh, 1, degree=1, units="K")
    model_coords = lambda: np.array(mesh.dm.getCoordinatesLocal().array).reshape(-1, mesh.cdim)
    start = model_coords()
    T.data[:, 0] = 0.3 + 0.2 * start[:, 0]

    radius = np.linalg.norm(start, axis=1)
    inside = (radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9)
    target = mesh.X.coords * np.where(inside, 1.02, 1.0)[:, None]
    assert all(uw.mpi.comm.allgather(mesh.deform(target)))

    moved = model_coords()
    expected = start.copy()
    expected[inside] *= 1.02
    largest = lambda a: uw.mpi.comm.allreduce(float(np.abs(a).max()) if a.size else 0.0, op=uw.MPI.MAX)
    assert largest(moved - expected) < 1.0e-9
    interior = np.linalg.norm(moved, axis=1) < 0.99
    assert largest((np.array(T.data).reshape(-1) - (0.3 + 0.2 * moved[:, 0]))[interior]) < 1.0e-9
    uw.reset_default_model()
