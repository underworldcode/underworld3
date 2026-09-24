"""Surface observations, and a parameter that lives on a boundary.

Two things an adjoint paper derives by hand and this machinery must get
without derivation. A misfit integrated over a boundary — the uplift rate
along a free surface — whose dual is a facet load. And a parameter that
enters the residual through a natural condition — the traction on a
boundary — whose sensitivity has a facet part. Both against central
differences, on a Stokes problem small enough to run in seconds.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw


def _box():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    v_obs = uw.discretisation.MeshVariable("v_obs", mesh, 2, degree=2)
    return mesh, v, p, v_obs


@pytest.mark.level_1
@pytest.mark.tier_a
def test_misfit_on_the_free_surface_matches_finite_differences():
    mesh, v, p, v_obs = _box()
    x, y = mesh.X
    eta = uw.expression(r"\eta", 1.0, "viscosity")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta * (1 + 2 * (x - 0.5) ** 2)
    stokes.bodyforce = sympy.Matrix([0, -sympy.sin(3 * x)])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.3, None), "Left")
    stokes.add_essential_bc((-0.3, None), "Right")          # free top
    stokes.tolerance = 1e-10                                   # the gradient is small; the difference must be clean

    misfit = (v.sym[1] - v_obs.sym[1]) ** 2 / 2               # uplift rate along the top

    def J(value):
        eta.sym = float(value)
        stokes.solve(zero_init_guess=True)
        return float(uw.maths.BdIntegral(mesh, misfit, "Top").evaluate())

    J(2.5); v_obs.array[...] = np.asarray(v.array)
    J0 = J(1.0)
    out = stokes.gradient(misfit, parameters=[eta], boundary="Top")
    assert abs(out["J"] - J0) < 1e-12
    h = 1e-4
    fd = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    assert abs(fd / out["parameters"][eta] - 1) < 1e-3, (fd, out["parameters"][eta])


@pytest.mark.level_1
@pytest.mark.tier_a
def test_parameter_in_a_traction_condition_matches_finite_differences():
    mesh, v, p, v_obs = _box()
    x, y = mesh.X
    tau = uw.expression(r"\tau", 0.5, "traction on the top")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 + 2 * (x - 0.5) ** 2
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.add_natural_bc(sympy.Matrix([[tau * sympy.sin(sympy.pi * x), 0.0]]), "Top")
    stokes.tolerance = 1e-10

    misfit = (v.sym[0] ** 2 + v.sym[1] ** 2) / 2              # a volume misfit

    def J(value):
        tau.sym = float(value)
        stokes.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J0 = J(0.5)
    out = stokes.gradient(misfit, parameters=[tau])
    assert abs(out["J"] - J0) < 1e-12
    h = 1e-4
    fd = (J(0.5 + h) - J(0.5 - h)) / (2 * h)
    assert out["parameters"][tau] != 0.0
    assert abs(fd / out["parameters"][tau] - 1) < 1e-3, (fd, out["parameters"][tau])
