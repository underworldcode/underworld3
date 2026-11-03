"""True-sphere lift adapters: independent hydrostatics, force sign and ownership."""
from contextlib import ExitStack

import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw

from test_1112_spherical_pressure_lift import _problem

pytestmark = pytest.mark.level_2


def _configure(stokes):
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.zeros(1, 3)
    stokes.petsc_use_nullspace = False
    stokes.petsc_use_pressure_nullspace = False


def _lift(stokes, jump):
    return stokes.spherical_pressure_lifting(
        inner_boundary="Lower", interface="Internal", interface_radius=.775,
        outer_boundary="Upper", outer_radius=1., jump=jump,
        volume_order=5, surface_order=8, radial_order=4,
    )


@pytest.mark.tier_b
@pytest.mark.parametrize("zero_slip", [False, True])
def test_true_sphere_hydrostatic_pressure_and_reactions(zero_slip):
    mesh, stokes = _problem()
    _configure(stokes)
    amplitude = 2.3
    lift = _lift(stokes, sp.Rational(23, 10))
    try:
        solve = stokes.solve_spherical_gap_zero_slip if zero_slip else stokes.solve_spherical_gap
        result = solve(
            inner_boundary="Lower", outer_boundary="Upper", radius_inner=.55, radius_outer=1.,
            viscosity=1., pressure_jump=lift.pressure_jump,
            pressure_jump_integral=lift.pressure_jump_integral, volume_force=lift.volume_force,
            gap_bodyforce=lift.regular_force_expression, quadrature=8, batch_size=64,
            linear_solver_options=dict(ksp_type="preonly", pc_type="lu",
                                       pc_factor_mat_solver_type="mumps", mat_mumps_icntl_14=100),
        )
        mean_lift = amplitude*(1-.775**3)/(1-.55**3)
        np.testing.assert_allclose(stokes.p.array[:, 0, 0], -mean_lift, rtol=0, atol=2e-8)
        assert np.max(np.abs(stokes.u.array), initial=0) < 2e-9
        coordinates = np.asarray(stokes.u.coords)
        for radius, expected in ((.55, -mean_lift), (1., amplitude-mean_lift)):
            rows = np.isclose(np.linalg.norm(coordinates, axis=1), radius, atol=1e-10)
            np.testing.assert_allclose(result.normal_reaction.array[rows, 0, 0], expected, rtol=0, atol=2e-8)
            assert np.max(np.abs(result.tangential_reaction.array[rows]), initial=0) < 2e-8
        assert result.checks["relative_residual"] < 1e-8
        assert abs(result.checks["physical_pressure_integral"]) < 1e-10
    finally:
        lift.destroy()


@pytest.mark.tier_b
def test_native_replacement_residual_sign_and_caller_vector_lifetime():
    mesh, stokes = _problem()
    _configure(stokes)
    x, y, z = mesh.X
    factor = sp.sqrt(sp.Rational(5, 16)/sp.pi)
    jump = -sp.Rational(13, 10)*factor*(2*z*z-x*x-y*y)/(x*x+y*y+z*z)
    lift = _lift(stokes, jump)
    # Reject a same-mesh lift belonging to another Stokes instance.
    other = uw.systems.Stokes(mesh, velocityField=stokes.u, pressureField=stokes.p)
    with pytest.raises(ValueError, match="this Stokes"):
        other.install_linear_spherical_pressure_lift(pressure_lift=lift)
    rhs = stokes.install_linear_spherical_pressure_lift(pressure_lift=lift)
    original = rhs.copy()
    lift.destroy()
    with ExitStack() as cleanup:
        cleanup.callback(original.destroy)
        zero = rhs.duplicate(); cleanup.callback(zero.destroy); zero.set(0)
        measured = rhs.duplicate(); cleanup.callback(measured.destroy)
        stokes.snes.computeFunction(zero, measured)
        measured.axpy(-1, original)
        assert measured.norm() < 1e-13*original.norm()
        velocity_is, velocity_dm = stokes._subdict["velocity"]
        load = rhs.getSubVector(velocity_is)
        try:
            local = stokes.u.vec.duplicate(); cleanup.callback(local.destroy)
            probe = velocity_dm.createGlobalVec(); cleanup.callback(probe.destroy)
            vx, vy, vz = np.asarray(stokes.u.coords).T
            local.array.reshape(-1, 3)[:] = np.sqrt(5/(16*np.pi))*np.column_stack((-2*vx, -2*vy, 4*vz))
            velocity_dm.localToGlobal(local, probe, addv=PETSc.InsertMode.INSERT_VALUES)
            np.testing.assert_allclose(load.dot(probe), 2*1.3*.775**3, rtol=2e-10)
        finally:
            rhs.restoreSubVector(velocity_is, load)
    stokes._reset()
    assert not bool(rhs)
