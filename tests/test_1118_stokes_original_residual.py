"""No-solve original-equation verification on an independent affine hydrostat."""

import ast
from pathlib import Path

import pytest
import numpy as np
import sympy as sp
import underworld3 as uw

from underworld3.utilities.stokes_residual import evaluate_linear_stokes_residual


pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_no_solve_pressure_gradient_and_rejection():
    # u=0, p=x-1/2, f=(1,0,0) solves -div(2*eps(u))+grad(p)=f.
    # P2/P1 represents this exactly on every affine tetrahedron. All essential
    # data are zero, so no continuation/gap/pressure reconstruction is involved.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0., 0.), maxCoords=(1., 1., 1.),
        cellSize=.5, degree=1, qdegree=3,
    )
    u = uw.discretisation.MeshVariable('SavedResidualU', mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable('SavedResidualP', mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.
    stokes.bodyforce = sp.Matrix([[1., 0., 0.]])
    for name in ('Left', 'Right', 'Bottom', 'Top', 'Front', 'Back'):
        stokes.add_dirichlet_bc((0., 0., 0.), name)

    def set_pressure(scale, shift=0.):
        with mesh.access(p):
            p.data[:, 0] = scale*(p.coords[:, 0]-.5)+shift

    set_pressure(1.)
    before_u, before_p = np.asarray(u.array).copy(), np.asarray(p.array).copy()
    exact = evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    assert exact['passed'] and exact['relative_residual'] < 2e-12
    assert exact['reference_residual_norm'] > 0
    assert exact['scope'] == 'fresh full original constrained native equations'
    np.testing.assert_array_equal(u.array, before_u)
    np.testing.assert_array_equal(p.array, before_p)
    # Constants are true pressure-gauge directions in this zero-Dirichlet box.
    set_pressure(1., 7.)
    shifted = evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    assert shifted['passed'] and shifted['relative_residual'] < 2e-12
    # Linearity supplies an independent residual ratio oracle: doubling the
    # pressure gradient gives exactly the zero-state residual magnitude.
    set_pressure(2.)
    doubled = evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    assert not doubled['passed']
    assert abs(doubled['relative_residual']-1.) < 2e-12
    set_pressure(1.)
    with mesh.access(u):
        x, y, z = u.coords.T
        u.data[:, 0] = .01*x*(1-x)*y*(1-y)*z*(1-z)
    perturbed = evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    assert not perturbed['passed'] and perturbed['relative_residual'] > 1e-5
    assert stokes.snes.getIterationNumber() == 0
    # This helper is observational: no solution-status fabrication or iteration.
    tree = ast.parse(Path(__import__(evaluate_linear_stokes_residual.__module__,
        fromlist=['__file__']).__file__).read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('solve', 'computeJacobian') for n in ast.walk(tree))
    # Exercise the intended post-solve call site, retaining the actual native
    # status and raw fields rather than relying only on manually assigned data.
    stokes.tolerance = 1e-9
    stokes.solve()
    solved_u, solved_p = np.asarray(u.array).copy(), np.asarray(p.array).copy()
    status = (stokes.snes.getConvergedReason(), stokes.snes.getIterationNumber())
    finalized = evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    assert finalized['passed'] and finalized['relative_residual'] <= 1e-9, finalized
    np.testing.assert_array_equal(u.array, solved_u)
    np.testing.assert_array_equal(p.array, solved_p)
    assert (stokes.snes.getConvergedReason(), stokes.snes.getIterationNumber()) == status
    if uw.mpi.rank == 0:
        print('SAVED_RESIDUAL_GATE', exact, shifted, doubled, perturbed, flush=True)

    for tolerance in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rtol"):
            evaluate_linear_stokes_residual(stokes, rtol=tolerance)
    stokes._rotated_freeslip_bcs = [object()]
    with pytest.raises(ValueError, match="Rotated free-slip"):
        evaluate_linear_stokes_residual(stokes, rtol=1e-9)
    stokes._rotated_freeslip_bcs = []
    stokes.add_dirichlet_bc((1., 0., 0.), "Left")
    with pytest.raises(ValueError, match="homogeneous essential"):
        evaluate_linear_stokes_residual(stokes, rtol=1e-9)
