"""The objective step of the viscoelastic model: how the carried stress is taken
through one step (objective_order), on a flat thin sheet.

A rigid rotation must carry a stress rigidly: sigma(X) = R sigma1(R^T X) R^T for
a quarter turn R. That is exact for any objective rate, so the rigid carry is a
hard baseline for the transport and the rotation together.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _box(cell_size=0.5):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=cell_size, qdegree=3)


def _model(rate, order=None, history_order=1, convected_step=None):
    sheet = uw.systems.ThinSheet(_box())
    return uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=history_order, integrator="etd", objective_rate=rate,
        objective_order=order, convected_step=convected_step)


@pytest.mark.tier_b
def test_the_objective_order_defaults_and_refusals():
    """Cayley for the Jaumann rate, first order otherwise; Cayley needs the
    Jaumann rate; the deformation step takes no order; an order above one needs
    a first-order history."""
    assert _model("jaumann")._objective_order == "cayley"
    assert _model("upper_convected")._objective_order == 1
    assert _model("none")._objective_order == 1
    with pytest.raises(ValueError, match="cayley"):
        _model("upper_convected", order="cayley")
    with pytest.raises(ValueError, match="positive integer"):
        _model("jaumann", order=0)
    with pytest.raises(ValueError, match="deformation"):
        _model("upper_convected", order=2, convected_step="deformation")
    with pytest.raises(NotImplementedError, match="first-order history"):
        _model("jaumann", order=2, history_order=2)


@pytest.mark.tier_b
def test_the_cayley_step_is_the_orthogonal_rotation_of_the_stress():
    """At a spin W = [[0, -w], [w, 0]] the Cayley step carries sigma to
    R sigma R^T with R = (I - dt W/2)^{-1}(I + dt W/2), an exact rotation."""
    from underworld3.function.expressions import unwrap
    model = _model("jaumann")
    model.Parameters.dt_elastic = 0.1
    sigma = sympy.Matrix([[1, 2], [2, -3]])
    carried = model._carried_through_step(sigma)
    L = sympy.Matrix(model.Unknowns.L)
    w = 0.7
    spin = {L[0, 0]: 0, L[0, 1]: -w, L[1, 0]: w, L[1, 1]: 0}
    got = np.array([[float(unwrap(carried[i, j].sym, keep_constants=False).xreplace(spin))
                     for j in range(2)] for i in range(2)])
    A = 0.1 * np.array([[0.0, -w], [w, 0.0]]) / 2
    R = np.linalg.solve(np.eye(2) - A, np.eye(2) + A)
    assert np.abs(R @ R.T - np.eye(2)).max() < 1.0e-14
    assert np.abs(got - R @ np.array([[1.0, 2.0], [2.0, -3.0]]) @ R.T).max() < 1.0e-13


def _rigid_carry(steps, tangent=False, cell_size=0.1):
    """Build an elastic stress on a flat sheet, then turn the sheet rigidly a
    quarter turn about z through a stiff basal drag; return the carried stress
    and the rigidly carried phase-one stress, within r < 0.6."""
    mesh = _box(cell_size)
    x, y = mesh.X
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="jaumann")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0e6       # elastic over the run
    sheet.constitutive_model.Parameters.shear_modulus = 1
    sheet.tolerance = 1.0e-10
    sheet.consistent_jacobian = tangent
    sheet.basal_drag = 1
    sheet.bodyforce = list(sympy.Matrix([[sympy.sin(2 * y) + x * y, sympy.cos(x) * y]]))
    for _ in range(5):
        sheet.solve(timestep=0.1, zero_init_guess=False)
    sigma1 = uw.discretisation.MeshVariable(
        f"sigma1_{steps}_{int(tangent)}", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    sigma1.array[...] = np.asarray(sheet.tau.array)
    c = 1.0e4
    sheet.basal_drag = c
    sheet.bodyforce = list(c * sympy.Matrix([[-y, x]]))
    # the history is advected with the previous step's velocity: start the
    # turn with that velocity already the rotation
    sheet.u.array[:, 0, :] = np.column_stack([-sheet.u.coords[:, 1], sheet.u.coords[:, 0]])
    angle = np.pi / 2
    for _ in range(steps):
        sheet.solve(timestep=angle / steps, zero_init_guess=False)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = sheet.tau.coords
    inside = np.hypot(pts[:, 0], pts[:, 1]) < 0.6
    got = np.asarray(sheet.tau.array)[inside]
    s1 = np.asarray(uw.function.evaluate(sigma1.sym, pts[inside] @ R)).reshape(-1, 2, 2)
    return got, np.einsum("ij,njk,lk->nil", R, s1, R)


@pytest.mark.tier_b
def test_a_rigid_rotation_carries_the_stress_rigidly():
    """The Cayley step and the grid transport together carry a stress through a
    rigid quarter turn: relative error 2.94e-3 measured 2026-10-12 (h = 0.1,
    40 steps; the error is spatial: 3.97e-3 at 80 steps, from the transport's
    upwinding). Without the rotation the components ride unrotated (O(1))."""
    got, expected = _rigid_carry(40)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    assert error < 6.0e-3, f"relative error {error:.3e}"


@pytest.mark.tier_b
def test_the_newton_tangent_converges_to_the_same_stress():
    """The Newton tangent replaces the frozen carried stress by its first-order
    form (the private _tangent_surrogates); the residual is unchanged, so the converged
    stress is the frozen-tangent one to solver tolerance."""
    picard, _ = _rigid_carry(8, tangent=False, cell_size=0.2)
    newton, _ = _rigid_carry(8, tangent=True, cell_size=0.2)
    assert np.abs(newton - picard).max() < 1.0e-7 * np.abs(picard).max()
