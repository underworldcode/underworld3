"""Native physical-sheet residual signs, virtual work and shared RHS ownership."""

from contextlib import ExitStack

import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw

pytestmark = pytest.mark.level_2


@pytest.fixture(scope="module")
def shell():
    return uw.meshing.SphericalShellInternalBoundary(
        radiusInner=.55, radiusInternal=.775, radiusOuter=1,
        cellSize=1/8, degree=1, qdegree=2,
    )


def _stokes(mesh):
    velocity = uw.discretisation.MeshVariable("NativePhysicalSheetU", mesh, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("NativePhysicalSheetP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.zeros(1, 3)
    stokes.petsc_use_nullspace = False
    stokes.petsc_use_pressure_nullspace = False
    return stokes


def _arguments():
    return dict(interface="Internal", radius=.775, harmonic_degree=2,
                harmonic_order=0, harmonic_component="cos",
                harmonic_normalization="real_orthonormal",
                density_coefficient=1.3, quadrature_points=8)


@pytest.mark.tier_b
def test_native_physical_sheet_residual_work_and_shared_lifecycle(shell):
    stokes = _stokes(shell)
    arguments = _arguments()
    rhs = stokes.install_linear_physical_spherical_sheet_load(**arguments)
    with ExitStack() as cleanup:
        zero = rhs.duplicate(); cleanup.callback(zero.destroy); zero.set(0)
        residual = rhs.duplicate(); cleanup.callback(residual.destroy)
        linear = rhs.duplicate(); cleanup.callback(linear.destroy)
        vector = rhs.duplicate(); cleanup.callback(vector.destroy)
        independent = rhs.copy(); cleanup.callback(independent.destroy)
        saved = independent.array.copy()
        stokes.snes.computeFunction(zero, residual)
        residual.axpy(-1, rhs)
        assert residual.norm() < 1e-13*rhs.norm()
        lo, hi = vector.getOwnershipRange()
        vector.array[:] = 1e-4*np.sin(np.arange(lo, hi))
        stokes.snes.getJacobian()[0].mult(vector, linear)
        linear.axpy(1, rhs)
        stokes.snes.computeFunction(vector, residual)
        residual.axpy(-1, linear)
        assert residual.norm() < 1e-13*max(linear.norm(), rhs.norm())
        pressure_is, _ = stokes._subdict["pressure"]
        pressure_load = rhs.getSubVector(pressure_is)
        try:
            assert pressure_load.norm() == 0
        finally:
            rhs.restoreSubVector(pressure_is, pressure_load)
        velocity_is, velocity_dm = stokes._subdict["velocity"]
        velocity_load = rhs.getSubVector(velocity_is)
        try:
            local = stokes.u.vec.duplicate(); cleanup.callback(local.destroy)
            probe = velocity_dm.createGlobalVec(); cleanup.callback(probe.destroy)
            x, y, z = np.asarray(stokes.u.coords).T
            gradient = np.sqrt(5/(16*np.pi))*np.column_stack((-2*x, -2*y, 4*z))
            expected = 2*arguments["density_coefficient"]*arguments["radius"]**3
            fields = [gradient]+[np.cross(axis, np.asarray(stokes.u.coords)) for axis in np.eye(3)]
            for index, values in enumerate(fields):
                local.array.reshape(-1, 3)[:] = values
                probe.set(0)
                velocity_dm.localToGlobal(local, probe, addv=PETSc.InsertMode.INSERT_VALUES)
                work = velocity_load.dot(probe)
                if index == 0:
                    np.testing.assert_allclose(work, expected, rtol=2e-11)
                else:
                    assert abs(work) < 2e-12*expected
        finally:
            rhs.restoreSubVector(velocity_is, velocity_load)
        checks = stokes._physical_spherical_sheet_checks
        assert checks["packets_before"] == checks["packets_after"] > 0
        assert checks["minimum_barycentric"] >= -1e-10
        # Both geometries share one borrowed-vector lifecycle and callback slot.
        old = stokes.install_linear_spherical_sheet_load(
            interface="Internal", harmonic_degree=2, density_coefficient=1.3,
            harmonic_normalization="real_orthonormal",
        )
        assert not bool(rhs) and bool(old)
        second = stokes.install_linear_physical_spherical_sheet_load(**arguments)
        assert not bool(old) and bool(second)
        residual = second.copy(); cleanup.callback(residual.destroy)
        residual.axpy(-1, independent)
        assert residual.norm() < 1e-13*independent.norm()
        stokes._reset()
        assert not bool(second) and stokes._linear_spherical_sheet_rhs is None
        np.testing.assert_array_equal(independent.array, saved)


@pytest.mark.tier_b
def test_physical_sheet_rejects_nonzero_bulk_force_before_setup(shell):
    stokes = _stokes(shell)
    stokes.bodyforce = sp.Matrix([[1, 0, 0]])
    with pytest.raises(ValueError, match="zero bulk force"):
        stokes.install_linear_physical_spherical_sheet_load(**_arguments())
    assert getattr(stokes, "_linear_spherical_sheet_rhs", None) is None


@pytest.mark.tier_b
def test_physical_sheet_restricts_full_field_layout_to_essential_subdm(shell):
    stokes = _stokes(shell)
    for boundary in ("Lower", "Upper"):
        stokes.add_essential_bc((0, 0, 0), boundary)
    rhs = stokes.install_linear_physical_spherical_sheet_load(**_arguments())
    with ExitStack() as cleanup:
        zero = rhs.duplicate(); cleanup.callback(zero.destroy); zero.set(0)
        residual = rhs.duplicate(); cleanup.callback(residual.destroy)
        stokes.snes.computeFunction(zero, residual)
        residual.axpy(-1, rhs)
        assert residual.norm() < 1e-13*rhs.norm()
        pressure_is, _ = stokes._subdict["pressure"]
        pressure_load = rhs.getSubVector(pressure_is)
        try:
            assert pressure_load.norm() == 0
        finally:
            rhs.restoreSubVector(pressure_is, pressure_load)
    stokes._reset()
    assert not bool(rhs)
