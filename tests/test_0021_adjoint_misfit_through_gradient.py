"""A misfit on a strain rate: the dual carries a gradient part.

The fault-segments example in miniature — a weak plane in a transversely
isotropic viscosity, two segments of unknown weak-plane viscosity, the misfit
on the surface uplift rate and on the shear stress near a point. The stress
reads the velocity through its gradient, so ``misfit_duals`` must assemble
the gradient part of the load; without it the adjoint gradient is wrong by
the whole stress term.
"""
import math

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.adjoint import misfit_duals


@pytest.mark.level_2
@pytest.mark.tier_a
def test_segment_strength_gradient_matches_finite_differences():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
                                             cellSize=1 / 10, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    v_obs = uw.discretisation.MeshVariable("v_obs", mesh, 2, degree=2)

    theta = math.radians(45.0)
    n_hat = sympy.Matrix([[-math.sin(theta), math.cos(theta)]])
    d = (x - 0.6) * n_hat[0] + y * n_hat[1]
    s = (x - 0.6) * math.cos(theta) + y * math.sin(theta)
    band = sympy.exp(-(d / 0.15) ** 2)
    strengths = [uw.expression(r"\eta_1", 0.1, "segment 1"),
                 uw.expression(r"\eta_2", 0.1, "segment 2")]
    half = 0.5 / math.sin(theta)
    seg = [(1 - sympy.tanh((s - half) / 0.15)) / 2, (1 + sympy.tanh((s - half) / 0.15)) / 2]
    eta_1 = 1 - band * sum((1 - strengths[k]) * seg[k] for k in range(2))

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1
    stokes.constitutive_model.Parameters.director = n_hat
    stokes.tolerance = 1e-9
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.5, None), "Left")
    stokes.add_essential_bc((-0.5, None), "Right")

    def shear(field):
        e = mesh.vector.strain_tensor(field.sym)
        return 2 * e[0, 1]

    w_top = sympy.exp(-((1 - y) / 0.15) ** 2)
    w_pt = sympy.exp(-((x - 1.3) ** 2 + (y - 0.4) ** 2) / 0.3 ** 2)
    misfit = (w_top * (v.sym[1] - v_obs.sym[1]) ** 2
              + w_pt * (shear(v) - shear(v_obs)) ** 2) / 2

    def set_strengths(a, b):
        strengths[0].sym, strengths[1].sym = float(a), float(b)

    def J():
        stokes.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    set_strengths(0.02, 0.2)
    stokes.solve(zero_init_guess=True)
    v_obs.array[...] = np.asarray(v.array)

    set_strengths(0.1, 0.1)
    J()
    dual = misfit_duals(misfit, [v])[v]
    dual.array[...] = -np.asarray(dual.array)
    mu = uw.discretisation.MeshVariable("mu", mesh, 2, degree=2)
    lam = uw.discretisation.MeshVariable("lam", mesh, 1, degree=1)
    _, reason = stokes.adjoint_solve((dual, None), target=(mu, lam))
    assert reason > 0
    adjoint = [stokes.sensitivity(mu, e) for e in strengths]

    h = 1e-4
    for k in range(2):
        plus = [0.1, 0.1]; plus[k] += h
        minus = [0.1, 0.1]; minus[k] -= h
        set_strengths(*plus); Jp = J()
        set_strengths(*minus); Jm = J()
        fd = (Jp - Jm) / (2 * h)
        assert adjoint[k] != 0
        assert abs(fd / adjoint[k] - 1) < 2e-3, (k, fd, adjoint[k])
