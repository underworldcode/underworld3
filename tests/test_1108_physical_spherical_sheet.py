"""Physical-sphere virtual work and zero torque without owner extrapolation."""

import numpy as np
import pytest
import underworld3 as uw
from petsc4py import PETSc

from underworld3.utilities._cylindrical_gap import _layout
from underworld3.utilities._physical_spherical_sheet import _assemble_physical_spherical_sheet

pytestmark = pytest.mark.level_2


def _work(vector, velocity, values):
    mapping, _ = _layout(velocity)
    probe = vector.duplicate()
    try:
        probe.set(0)
        probe.setValues(mapping, values.ravel(), addv=PETSc.InsertMode.INSERT_VALUES)
        probe.assemble()
        return vector.dot(probe)
    finally:
        probe.destroy()


@pytest.mark.tier_b
@pytest.mark.parametrize("order,component", [(0, "cos"), (2, "sin")])
def test_true_sphere_sheet_work_migration_and_all_rotation_modes(order, component):
    radius, coefficient = 0.775, 1.3
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=0.55, radiusInternal=radius, radiusOuter=1,
        cellSize=1/8, degree=1, qdegree=3,
    )
    velocity = uw.discretisation.MeshVariable("PhysicalSheetU", mesh, 3, degree=2)
    result = _assemble_physical_spherical_sheet(
        velocity, "Internal", radius, 2, order, component, "real_orthonormal",
        coefficient, 8,
    )
    try:
        x, y, z = np.asarray(velocity.coords).T
        # Independent normalized solid harmonics H_20 and real H_22(sine).
        if order == 0:
            factor = np.sqrt(5/(16*np.pi))
            gradient = factor*np.column_stack((-2*x, -2*y, 4*z))
        else:
            factor = np.sqrt(15/(4*np.pi))
            gradient = factor*np.column_stack((y, x, np.zeros_like(z)))
        expected = 2*coefficient*radius**3
        work = _work(result.residual, velocity, gradient)
        np.testing.assert_allclose(work, expected, rtol=2e-11)
        coordinates = np.asarray(velocity.coords)
        for axis in np.eye(3):
            torque = _work(result.residual, velocity, np.cross(axis, coordinates))
            assert abs(torque)/expected < 2e-12
        checks = result.checks
        assert checks["packets_before"] == checks["packets_after"] > 0
        np.testing.assert_allclose(checks["area"], 4*np.pi*radius**2, rtol=2e-11)
        assert checks["harmonic_moment_relative_error"] < 2e-11
        assert checks["minimum_barycentric"] >= -1e-10
        if uw.mpi.rank == 0:
            print("Physical spherical sheet:", dict(work=work, expected=expected, **checks))
    finally:
        result.residual.destroy()
