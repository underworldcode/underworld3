"""Independent force, gauge and volume identities for true-sphere pressure lifting."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from sympy.vector.scalar import BaseScalar
from underworld3.coordinates import UWCoordinate
from underworld3.function.expressions import extract_meshes

from underworld3.utilities._physical_spherical_sheet import _owned_spherical_facets
from underworld3.utilities._spherical_pressure_lift import _spherical_pressure_lifting
from test_1108_physical_spherical_sheet import _work

pytestmark = pytest.mark.level_2


def _problem():
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=0.55, radiusInternal=0.775, radiusOuter=1,
        cellSize=1/8, degree=1, qdegree=3,
    )
    velocity = uw.discretisation.MeshVariable("SphereLiftU", mesh, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("SphereLiftQ", mesh, 1, degree=1)
    return mesh, uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)


def _lift(stokes, jump, order):
    lift = _spherical_pressure_lifting(stokes, "Lower", "Internal", 0.775, "Upper", 1., jump, order, 8, 4)
    for expression in (lift.pressure_jump, lift.physical_pressure, *lift.regular_force_expression):
        assert not expression.atoms(UWCoordinate)
        assert all(atom is stokes.mesh.N.base_scalars()[int(atom.args[0])]
                   for atom in expression.atoms(BaseScalar))
    assert extract_meshes(lift.physical_pressure) == {stokes.mesh}
    for axis, coordinate in enumerate(stokes.mesh.N.base_scalars()):
        assert sp.simplify(sp.diff(lift.pressure_jump, coordinate)+lift.regular_force_expression[axis]) == 0
    return lift


@pytest.mark.tier_b
def test_constant_jump_true_outer_volume_and_no_regular_force():
    mesh, stokes = _problem()
    original_inner = mesh.cells_labelled("Inner", 101).copy()
    mesh.regions = None  # Persisted meshes need not carry generator region enums.
    jump = sp.Rational(23, 10)
    lift = _lift(stokes, jump, 4)
    try:
        assert lift.owner_stokes() is stokes
        assert lift.volume_force.norm() == 0
        _, triangles = _owned_spherical_facets(mesh, "Upper")
        # Closed oriented outer-triangle divergence volume, independent of
        # tetrahedral and cap quadrature used by the implementation.
        comm_volume = float(np.abs(np.einsum("ed,ed->e", triangles[:, 0],
                                                      np.cross(triangles[:, 1], triangles[:, 2]))).sum()/6)
        volume = uw.mpi.comm.allreduce(comm_volume, op=MPI.SUM)
        expected = float(jump)*(volume-4*np.pi*0.775**3/3)
        np.testing.assert_allclose(lift.pressure_jump_integral, expected, rtol=2e-12)
        np.testing.assert_allclose(lift.checks["physical_lift_integral"],
                                   float(jump)*4*np.pi*(1-0.775**3)/3, rtol=2e-12)
        np.testing.assert_array_equal(mesh.cells_labelled("Inner", 101), original_inner)
        assert lift.checks["packets_before"] == lift.checks["packets_after"] > 0
        assert lift.checks["minimum_barycentric"] >= -1e-10
    finally:
        lift.destroy()


@pytest.mark.tier_b
def test_harmonic_lift_force_work_torque_and_quadrature_change():
    mesh, stokes = _problem()
    x, y, z = mesh.X
    factor = sp.sqrt(sp.Rational(5, 16)/sp.pi)
    jump = -sp.Rational(13, 10)*factor*(2*z*z-x*x-y*y)/(x*x+y*y+z*z)
    coordinates = np.asarray(stokes.u.coords)
    vx, vy, vz = coordinates.T
    gradient = np.sqrt(5/(16*np.pi))*np.column_stack((-2*vx, -2*vy, 4*vz))
    expected = -2*1.3*0.775**3
    work_values = []
    for order in (4, 5):
        lift = _lift(stokes, jump, order)
        combined = lift.volume_force.copy()
        try:
            combined.axpy(1, lift.native_outer_boundary_force)
            work = _work(combined, stokes.u, gradient)
            torques = [_work(combined, stokes.u, np.cross(axis, coordinates)) for axis in np.eye(3)]
            work_values.append(work)
            if uw.mpi.rank == 0:
                print("Spherical pressure lift:", dict(volume_order=order, work=work, expected=expected,
                                                       relative_work_error=abs(work/expected-1), torques=torques,
                                                       checks=lift.checks))
            # Order four is the explicitly retained quadrature audit; order
            # five must satisfy the independent force and torque identities.
            if order == 5:
                assert abs(work/expected-1) < 2e-10
                assert max(map(abs, torques))/abs(expected) < 2e-10
                assert abs(lift.checks["physical_lift_integral"]) < 2e-10
            assert lift.checks["packets_before"] == lift.checks["packets_after"] > 0
        finally:
            combined.destroy()
            lift.destroy()
    # Record the integration-order difference as an audit, not an assumed
    # monotonic error reduction or a mesh-refinement claim.
    if uw.mpi.rank == 0:
        print("Spherical pressure lift q4/q5 relative difference:", abs(work_values[1]-work_values[0])/abs(expected))
