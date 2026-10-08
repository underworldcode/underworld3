"""Extra volume fields solved monolithically with Stokes (``add_coupled_field``).

A coupled field is assembled in the same Newton system as velocity and pressure,
with every Jacobian block touching it derived from its residual. These tests pin

* the residual: a one-way coupled screened-Poisson field equals the same equation
  solved separately by ``Projection`` (the identical weak form);
* the Jacobian: with two-way coupling the assembled operator is the derivative of
  the residual (Taylor remainder of order two);
* the guard: an ordinary Stokes solver compiles and registers only its own terms.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = pytest.mark.level_2

SMOOTHING = 0.05    # l^2 of the screened-Poisson field


def _driven_stokes(viscosity=1):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3)
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    return mesh, stokes


def _exact_linear_solves(solver):
    """Factor the Jacobian itself at every Newton step."""
    solver.saddle_preconditioner = 0
    solver.petsc_options["pc_type"] = "lu"
    solver.petsc_options["pc_factor_mat_solver_type"] = "mumps"
    solver.petsc_options["ksp_type"] = "preonly"
    solver.petsc_options["snes_rtol"] = 1.0e-12
    solver.petsc_options["snes_atol"] = 1.0e-14


def _screened_strain_rate(mesh, stokes, chi):
    """chi - l^2 lap chi = edot_II, with the natural (zero-flux) condition."""
    return dict(F0=chi.sym[0] - stokes.Unknowns.Einv2,
                F1=SMOOTHING * chi.sym.jacobian(mesh.CoordinateSystem.N))


@pytest.mark.tier_b
def test_one_way_coupled_field_equals_projection():
    mesh, stokes = _driven_stokes()
    chi = uw.discretisation.MeshVariable("chi", mesh, 1, degree=1)
    stokes.add_coupled_field(chi, **_screened_strain_rate(mesh, stokes, chi))
    _exact_linear_solves(stokes)
    stokes.solve()
    assert stokes.dm.getNumFields() == 3

    chi_ref = uw.discretisation.MeshVariable("chi_ref", mesh, 1, degree=1)
    projection = uw.systems.Projection(mesh, chi_ref)
    projection.uw_function = stokes.Unknowns.Einv2
    projection.smoothing = SMOOTHING
    projection.petsc_options["pc_type"] = "lu"
    projection.petsc_options["ksp_type"] = "preonly"
    projection.petsc_options["snes_rtol"] = 1.0e-12
    projection.solve()

    coupled = chi.array[:, 0, 0]
    reference = chi_ref.array[:, 0, 0]
    assert np.abs(reference).max() > 0.1
    assert np.abs(coupled - reference).max() < 1.0e-8 * np.abs(reference).max()


@pytest.mark.tier_b
def test_two_way_coupled_jacobian_is_the_residual_derivative():
    mesh, stokes = _driven_stokes()
    chi = uw.discretisation.MeshVariable("chi", mesh, 1, degree=1)
    # the viscosity depends on chi, and chi on the strain rate: both off-diagonal
    # blocks, and the chi row's dependence on grad u, are non-zero
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 + chi.sym[0] ** 2
    stokes.add_coupled_field(chi, **_screened_strain_rate(mesh, stokes, chi))
    stokes.consistent_jacobian = True
    _exact_linear_solves(stokes)
    stokes.solve()
    stokes._set_newton_alpha(1.0)

    snes = stokes.snes
    x = stokes.dm.getGlobalVec()
    stokes._gather_fields_to_global(x)
    rng = np.random.default_rng(1)
    x.array[:] += 0.1 * rng.standard_normal(x.getLocalSize())
    direction = x.duplicate()
    direction.array[:] = rng.standard_normal(x.getLocalSize())

    J, P, _ = snes.getJacobian()
    snes.computeJacobian(x, J, P)
    J_direction = x.duplicate()
    J.mult(direction, J_direction)

    F_x = x.duplicate()
    snes.computeFunction(x, F_x)
    remainders = []
    steps = [1.0e-2, 1.0e-3, 1.0e-4]
    for step in steps:
        x_step = x.duplicate()
        x_step.waxpy(step, direction, x)
        F_step = x.duplicate()
        snes.computeFunction(x_step, F_step)
        remainder = F_step.array - F_x.array - step * J_direction.array
        remainders.append(np.linalg.norm(remainder))

    slopes = np.diff(np.log(remainders)) / np.diff(np.log(steps))
    assert np.all(slopes > 1.9), f"Taylor remainders {remainders}, slopes {slopes}"


@pytest.mark.tier_b
def test_ordinary_stokes_registers_only_its_own_terms():
    mesh, stokes = _driven_stokes()
    stokes.solve()
    assert stokes.dm.getNumFields() == 2
    assert set(stokes.ext_dict.res) == {stokes._u_F0, stokes._u_F1, stokes._p_F0}
    assert stokes._coupled_jacobians == {}
