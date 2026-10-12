"""The objective step on a curved sheet: a rigid rotation of the sphere carries
a tangent stress rigidly, sigma(X) = R sigma1(R^T X) R^T.

This exercises everything a curved surface adds: the grid transport of the
Cartesian components, the spin of a material element including the turning
of its normal, the Cayley rotation through the step, and the tangent-plane
projection of the committed history (after the rotation, never before it).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_3]


@pytest.mark.tier_b
def test_a_rigid_rotation_of_the_sphere_carries_the_stress_rigidly():
    """A quarter turn about x in 20 steps, h = 0.15. Measured 2026-10-12:
    relative error 9.4e-3 (8.4e-3 at 40 steps, 8.5e-3 at 80: the error is
    spatial; 2.5e-3 at h = 0.075, order 1.77). Before the history was projected
    only after the rotation it was 4.0e-2, first order in dt; without an
    objective rate the stress is not rotated at all (error O(1))."""
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15, degree=1, qdegree=5)
    x, y, z = sphere.X[0], sphere.X[1], sphere.X[2]
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="jaumann")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0e6       # elastic over the run
    sheet.constitutive_model.Parameters.shear_modulus = 1
    sheet.tolerance = 1.0e-10
    # phase 1: an elastic stress from the toroidal drag rhat x grad(z^2)
    sheet.basal_drag = 1
    sheet.bodyforce = list(sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
    for _ in range(5):
        sheet.solve(timestep=0.1, zero_init_guess=False)
    sigma1 = uw.discretisation.MeshVariable("sigma1", sphere, vtype=uw.VarType.SYM_TENSOR, degree=1)
    sigma1.array[...] = np.asarray(sheet.tau.array)
    # phase 2: a rigid rotation about x at unit rate through a stiff drag, the
    # velocity set to it first (the history is advected with the previous step's)
    c = 1.0e4
    sheet.basal_drag = c
    sheet.bodyforce = list(c * sympy.Matrix([[0, -z, y]]))
    sheet.u.array[:, 0, :] = np.cross(np.array([1.0, 0.0, 0.0]), sheet.u.coords)
    angle, steps = np.pi / 2, 20
    for _ in range(steps):
        sheet.solve(timestep=angle / steps, zero_init_guess=False)
    R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    pts = sheet.tau.coords
    s1 = np.asarray(uw.function.evaluate(sigma1.sym, pts @ R)).reshape(-1, 3, 3)
    expected = np.einsum("ij,njk,lk->nil", R, s1, R)
    got = np.asarray(sheet.tau.array)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    assert error < 1.9e-2, f"relative error {error:.3e}"
    rhat = pts / np.linalg.norm(pts, axis=1)[:, None]
    assert np.abs(np.einsum("nij,nj->ni", got, rhat)).max() < 1.0e-12 * np.abs(got).max()
