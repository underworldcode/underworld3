"""One call for the steady adjoint, and the adjoint problem it assembles, in writing.

``solver.gradient(misfit, parameters=..., fields=...)`` is the three-call
route — dual of the misfit, transposed solve, sensitivities — as one method,
so a steady inversion reads like the time-dependent one. Checked against a
central difference on a non-symmetric SUPG step, in the parameter and in the
initial field. ``adjoint_templates()`` writes the adjoint residual in the
same template language as the forward one; the check here is that it is
the forward Jacobian kernels with trial and test exchanged, by evaluating
both at a point.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw


def _supg(cell=1 / 8):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=cell, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T_obs = uw.discretisation.MeshVariable("T_obs", mesh, 1, degree=2)
    kappa = uw.expression(r"\kappa", 0.05, "diffusivity")
    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=sympy.Matrix([[1.0, 0.3]]), theta=1.0)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = kappa
    adv.add_essential_bc(1.0, "Bottom")
    adv.add_essential_bc(0.0, "Top")
    return mesh, T, T_obs, kappa, adv


@pytest.mark.level_1
@pytest.mark.tier_a
def test_gradient_method_matches_finite_differences_in_parameter_and_initial_field():
    mesh, T, T_obs, kappa, adv = _supg()
    x, y = mesh.X
    dt = 0.05

    def initial(c=0.5):
        X = np.asarray(T.coords)
        return np.exp(-((X[:, 0] - c) ** 2 + (X[:, 1] - 0.5) ** 2) / 0.05)

    def run(k, T0):
        kappa.sym = float(k)
        T.array[:, 0, 0] = T0
        adv.DuDt.initialise_history() if hasattr(adv.DuDt, "initialise_history") else None
        adv.solve(timestep=dt, zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    misfit = (T.sym[0] - T_obs.sym[0]) ** 2 / 2
    run(0.05, initial(0.55))
    T_obs.array[...] = np.asarray(T.array)
    T0 = initial(0.5)
    J0 = run(0.05, T0)
    adv.DuDt.psi_star[0].array[:, 0, 0] = T0          # the step's input, not its output
    out = adv.gradient(misfit, parameters=[kappa], fields=[T])
    assert abs(out["J"] - J0) < 1e-12

    h = 1e-4
    fd = (run(0.05 + h, T0) - run(0.05 - h, T0)) / (2 * h)
    assert abs(fd / out["parameters"][kappa] - 1) < 1e-3, (fd, out["parameters"][kappa])

    # the initial field enters through the history slot the solve reads
    direction = initial(0.45) - initial(0.5)
    dual = out["fields"][T]
    adjoint = uw.adjoint.inner(T, dual, direction)
    fd = (run(0.05, T0 + h * direction) - run(0.05, T0 - h * direction)) / (2 * h)
    assert abs(fd / adjoint - 1) < 1e-3, (fd, adjoint)


@pytest.mark.level_1
@pytest.mark.tier_a
def test_adjoint_templates_are_the_forward_kernels_with_trial_and_test_exchanged():
    mesh, T, T_obs, kappa, adv = _supg(cell=1 / 4)
    adv.solve(timestep=0.05, zero_init_guess=True)
    f0, f1 = adv.adjoint_templates()
    kernels = adv.adjoint_kernels()
    # value part: the coefficient of mu in f0 is H0, of mu_{,d} is H1
    mu = sympy.Symbol(r"\mu")
    assert sympy.simplify(sympy.diff(f0[0], mu) - kernels["F0_u"][0, 0]) == 0
    for d in range(mesh.cdim):
        md = sympy.Symbol(f"\\mu_{{,{d}}}")
        assert sympy.simplify(sympy.diff(f0[0], md) - kernels["F0_grad_u"][0, d]) == 0
        assert sympy.simplify(sympy.diff(f1[0, d], mu) - kernels["F1_u"][0, d]) == 0
    # and the forward kernels' transpose: the SUPG term makes g3 non-symmetric only
    # through the velocity; g1 <-> g2 is where a wrong swap would show
    assert kernels["F0_grad_u"] == sympy.ImmutableMatrix(adv._G2).reshape(1, mesh.cdim)
