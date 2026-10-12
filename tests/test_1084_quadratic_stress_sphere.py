"""The quadratic (normal-stress) benchmark of a corotational Maxwell sheet on
the sphere: a divergence-free flow thins the sheet at second order.

An unfaulted Maxwell sheet (eta = mu = L = 1) on the unit sphere, dragged
(c = 4) by the toroidal force 4 A rhat x grad_s(z^2). At first order the flow is
the zonal toroidal field u_1 = (A/2) rhat x grad_s(z^2), divergence-free: the
sheet neither thins nor thickens. At second order the corotational (Jaumann)
rate turns the shear stress into normal stress differences — the Weissenberg
effect of a Maxwell fluid — whose divergence drives a poloidal flow, and the
sheet thins. The steady second-order perturbation (sympy, derived 2026-10-12;
scripts/sessions/thin_sheet_quadratic_closed_form.py) gives

    -div_s u / A^2 = -3 (65 z^4 - 126 z^2 + 29) / 533
                   = (24/91) P_2(z) - (24/287) P_4(z),

with no P_0 (the sheet's mean thickness is conserved) and no odd degrees (the
forcing is symmetric about the equator). The linear (A^1) problem carries no
thinning at all, so the benchmark isolates the objective rate.

ETD-1 carries the relaxation of the corotational term with an effective time
lambda' = alpha dt / (1 - alpha), alpha = exp(-dt / lambda), in steady state
(lambda' = 0.903 lambda at dt = 0.2), and the second-order thinning is linear in
lambda, so the discrete steady state is the closed form times lambda' / lambda.

Measured (A = 0.1, steady by t = 4), P_2 / A^2 against (24/91) lambda':
h = 0.2, dt = 0.2: -0.65 %; h = 0.12, dt = 0.1: -0.18 %. P_4 / A^2 against
-(24/287) lambda': +14 % at h = 0.2, +5.6 % at h = 0.12 (second order in h).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_3]


@pytest.mark.tier_b
def test_quadratic_thinning_matches_the_second_order_closed_form():
    A, h, dt, steps = 0.1, 0.2, 0.2, 20
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=h, degree=1, qdegree=5)
    x, y, z = sphere.X[0], sphere.X[1], sphere.X[2]
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="jaumann")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    sheet.constitutive_model.Parameters.shear_modulus = 1
    sheet.add_basal_drag(4, "Base")
    sheet.bodyforce = list(4 * sympy.nsimplify(A) * sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
    sheet.tolerance = 1.0e-9
    sheet.consistent_jacobian = True
    for _ in range(steps):
        sheet.solve(timestep=dt, zero_init_guess=False)

    thinning = uw.discretisation.MeshVariable("thinning", sphere, 1, degree=1)
    projection = uw.systems.Projection(sphere, thinning)
    projection.uw_function = -sphere.vector.divergence(sheet.u.sym)
    projection.smoothing = 0.0
    projection.solve()
    coefficients = []
    for l in range(5):
        P_l = sympy.legendre(l, z)
        num = uw.maths.Integral(sphere, thinning.sym[0, 0] * P_l).evaluate()
        den = uw.maths.Integral(sphere, P_l * P_l).evaluate()
        coefficients.append(num / den / A**2)
    c0, c1, c2, c3, c4 = coefficients

    alpha = np.exp(-dt)
    lam = alpha * dt / (1 - alpha)
    p2, p4 = 24 / 91 * lam, -24 / 287 * lam
    assert abs(c2 - p2) < 0.02 * abs(p2), f"P2/A^2 {c2:.5f}, closed form {p2:.5f}"
    assert abs(c4 - p4) < 0.25 * abs(p4), f"P4/A^2 {c4:.5f}, closed form {p4:.5f}"
    for name, c in (("P0", c0), ("P1", c1), ("P3", c3)):
        assert abs(c) < 2.0e-3 * abs(c2), f"{name}/A^2 {c:.3e} against P2 {c2:.3e}"
