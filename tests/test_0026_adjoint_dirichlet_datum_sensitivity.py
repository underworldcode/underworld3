"""A parameter that enters through a Dirichlet datum (#762).

The wall value of a Poisson problem and the lid velocity of a Stokes
problem are controls. The global vector holds only the unconstrained
degrees of freedom, so neither the assembled operator nor a residual
evaluation carries the block that couples the interior to the prescribed
data; the sensitivity gets that term as the reaction of the adjoint
operator on the constrained rows, dotted with the derivative of the datum.
Checked against central differences, with corners where two boundaries
meet so the order PETSc applies the data in is exercised too. A solver
whose boundary tangent the reaction cannot see refuses.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw


@pytest.mark.level_1
@pytest.mark.tier_a
def test_wall_value_on_poisson_matches_finite_differences():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    x, y = mesh.X
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    m = uw.expression(r"m", 0.7, "amplitude of the wall value")
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1 + x
    poisson.f = sympy.cos(2 * y)
    poisson.add_essential_bc(0.0, "Bottom")
    poisson.add_essential_bc(sympy.Matrix([m * sympy.sin(sympy.pi * x)]), "Top")
    poisson.tolerance = 1e-10
    misfit = u.sym[0] ** 2 / 2

    def J(value):
        m.sym = float(value)
        poisson.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J0 = J(0.7)
    out = poisson.gradient(misfit, parameters=[m])
    assert abs(out["J"] - J0) < 1e-12
    h = 1e-4
    fd = (J(0.7 + h) - J(0.7 - h)) / (2 * h)
    assert out["parameters"][m] != 0.0
    assert abs(fd / out["parameters"][m] - 1) < 1e-4, (fd, out["parameters"][m])


@pytest.mark.level_1
@pytest.mark.tier_a
def test_lid_velocity_on_stokes_matches_finite_differences():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    m = uw.expression(r"m", 1.0, "lid speed")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 + 2 * (x - 0.5) ** 2
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.add_essential_bc((m, 0.0), "Top")          # last: the corners take the lid's value
    stokes.tolerance = 1e-10
    misfit = (v.sym[0] ** 2 + v.sym[1] ** 2) / 2 + (x - 0.5) * v.sym[1]

    def J(value):
        m.sym = float(value)
        stokes.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J0 = J(1.0)
    out = stokes.gradient(misfit, parameters=[m])
    assert abs(out["J"] - J0) < 1e-12
    h = 1e-4
    fd = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    assert out["parameters"][m] != 0.0
    assert abs(fd / out["parameters"][m] - 1) < 1e-4, (fd, out["parameters"][m])


@pytest.mark.level_1
@pytest.mark.tier_a
def test_datum_parameter_refuses_beside_a_boundary_tangent():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 4, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    m = uw.expression(r"m", 1.0, "lid speed")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((m, 0.0), "Top")
    stokes.add_natural_bc(sympy.Matrix([[-v.sym[0], 0.0]]), "Right")    # a traction that reads the unknown
    stokes.solve(zero_init_guess=True)
    with pytest.raises(NotImplementedError, match="reads the unknown"):
        stokes.gradient((v.sym[0] ** 2 + v.sym[1] ** 2) / 2, parameters=[m])
