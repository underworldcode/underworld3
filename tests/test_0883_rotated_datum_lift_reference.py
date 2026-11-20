"""A cold start that lifts a prescribed datum measures its convergence against
the datum's own scale (#883).

``solve_rotated_freeslip`` lifts a prescribed normal velocity (the datum) in its
first increment from u = 0. The residual at u = 0 ignores the datum, so a solve
forced by the datum alone has r0 = 0, and one with a weak body force beside it
has r0 at the body force's scale. The reference then became the residual of the
lifted iterate, which for a linear model is the increment's own Krylov residual,
or stayed at the body force's scale: either way the relative test could not pass
after the lift, and a second increment followed. Measured on development
(annulus, cell size 0.2, datum 0.1 cos 2 theta on the outer wall, serial):
2 increments, reference 7.1e-8 (datum alone) and 1.05e-4 (body force 1e-3).

The reference now includes the norm of the lifted right-hand side on the free
rows, the forcing the datum exerts on the interior (1.72 here), and the linear
solve converges in its one increment. The continuation switch is not tested at
the lift's iteration 0, as the convergence test is not.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.parametrize("bodyforce", [0.0, 1.0e-3], ids=["datum alone", "weak body force"])
def test_a_linear_datum_lift_converges_in_one_increment(bodyforce):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable("V883", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P883", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    th = sympy.atan2(y, x)
    r = sympy.sqrt(x ** 2 + y ** 2)
    stokes.bodyforce = bodyforce * sympy.cos(3 * th) * sympy.Matrix([x / r, y / r]).T
    stokes.add_rotated_freeslip_bc(0.1 * sympy.cos(2 * th), "Upper")
    stokes.add_rotated_freeslip_bc(0, "Lower")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["nonlinear_iterations"] == 1, info["rnorm_history"]
    # the reference is the datum's forcing, not the lifted iterate's residual
    assert info["ref"] > 1.0, info["ref"]
    assert info["rnorm_history"][-1] <= 1.0e-6 * info["ref"], info["rnorm_history"]
