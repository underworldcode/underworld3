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


# ---- Cosserat shear layer -----------------------------------------------------------
# 2-D Cosserat continuum: micro-rotation rate omega, continuum spin
# W = (du_y/dx - du_x/dy)/2, skew stress 2 eta_c (W - omega) E (E = [[0, -1], [1, 0]])
# added to the momentum flux, and the moment balance -div(B grad omega) +
# 4 eta_c (omega - W) = 0. A layer 0 < y < 1 sheared by u_x(1) = U with omega = 0 on
# both walls has sigma_xy = T constant and
#   omega = -T/(2 eta) [1 - cosh(k(y - 1/2)) / cosh(k/2)],
#   k^2 = 4 eta_c eta / (B (eta + eta_c)),   (eta + eta_c) u_x' + 2 eta_c omega = T.
ETA, ETA_C, B_COUPLE, U_TOP = 1.0, 1.0, 0.02, 1.0


def _shear_layer_exact():
    y = sympy.Symbol("y")
    k = sympy.sqrt(4 * ETA_C * ETA / (B_COUPLE * (ETA + ETA_C)))
    T = sympy.Symbol("T")
    omega = -T / (2 * ETA) * (1 - sympy.cosh(k * (y - sympy.Rational(1, 2)))
                              / sympy.cosh(k / 2))
    du = (T - 2 * ETA_C * omega) / (ETA + ETA_C)
    u = sympy.integrate(du, (y, 0, y))
    T_value = sympy.nsolve(u.subs(y, 1) - U_TOP, T, 1.0)
    return y, omega.subs(T, T_value), u.subs(T, T_value), du.subs(T, T_value)


def _cosserat_shear_layer_error(cell_size):
    """L2 error of omega and of u_x for the linear Cosserat shear layer."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size, qdegree=4)
    x, y = mesh.X
    ys, omega_exact, u_exact, du_exact = _shear_layer_exact()
    omega_e = omega_exact.subs(ys, y)
    u_e = u_exact.subs(ys, y)

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    omega = uw.discretisation.MeshVariable("omega", mesh, 1, degree=2)
    spin = stokes.Unknowns.W[1, 0]
    rotation = sympy.Matrix([[0, -1], [1, 0]])
    stokes.add_coupled_field(
        omega,
        F0=4 * ETA_C * (omega.sym[0] - spin),
        F1=B_COUPLE * omega.sym.jacobian(mesh.CoordinateSystem.N),
        momentum_flux=2 * ETA_C * (spin - omega.sym[0]) * rotation,
    )
    stokes.add_dirichlet_bc((U_TOP, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((u_e, 0.0), "Left")
    # the exact traction F1 . n on the right wall, n = (1, 0): the skew stress makes
    # sigma_yx = (eta - eta_c) u_x' - 2 eta_c omega, not sigma_xy. The natural-BC value
    # enters the boundary residual as is (PETSc's f0), so it is MINUS the traction.
    traction_y = (ETA - ETA_C) * du_exact.subs(ys, y) - 2 * ETA_C * omega_e
    stokes.add_natural_bc((0.0, -traction_y), "Right")
    stokes.add_coupled_dirichlet_bc(0.0, "Top", omega)
    stokes.add_coupled_dirichlet_bc(0.0, "Bottom", omega)
    _exact_linear_solves(stokes)
    stokes.solve()

    def l2(error):
        return float(np.sqrt(uw.maths.Integral(mesh, error ** 2).evaluate()))

    return l2(omega.sym[0] - omega_e), l2(stokes.u.sym[0] - u_e), l2(omega_e)


@pytest.mark.tier_b
def test_cosserat_shear_layer_converges_to_the_closed_form():
    coarse = _cosserat_shear_layer_error(1.0 / 8)
    fine = _cosserat_shear_layer_error(1.0 / 16)
    assert coarse[2] > 0.05      # the micro-rotation is not trivially zero
    omega_order = np.log2(coarse[0] / fine[0])
    u_order = np.log2(coarse[1] / fine[1])
    # P2 omega and P2 velocity: L2 order 3 asymptotically
    assert omega_order > 2.5, f"omega L2 errors {coarse[0]:.3e} -> {fine[0]:.3e}"
    assert u_order > 2.5, f"u_x L2 errors {coarse[1]:.3e} -> {fine[1]:.3e}"
