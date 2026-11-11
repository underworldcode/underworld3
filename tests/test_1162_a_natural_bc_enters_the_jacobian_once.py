"""A natural BC that reads the unknown enters the Jacobian once (#1065).

The scalar and vector solvers registered each natural BC's boundary Jacobian
twice, as the Jacobian and as the preconditioner, and their Jacobian and
preconditioner are one matrix: the Robin term entered twice. The residual was
right, so the converged answer was too, but a linear Robin problem took 17 to 23
Newton iterations, or failed, and an adjoint built on the tangent was wrong.

- a linear Poisson problem with a Robin condition is solved by its first Newton
  step (to the linear solve's tolerance), at the exact top value of its
  one-dimensional profile;
- the Robin part of the assembled Jacobian equals the finite-difference one;
- so is a linear vector projection with a Robin condition.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _robin_poisson(k, tag, degree=2, cell_size=0.125):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=cell_size, qdegree=3)
    T = uw.discretisation.MeshVariable(f"T1162{tag}", mesh, 1, degree=degree)
    s = uw.systems.Poisson(mesh, u_Field=T)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1.0
    s.f = 1.0
    s.add_dirichlet_bc(0.0, "Bottom")
    s.add_natural_bc([-k * T.sym[0] + 0.2], "Top")
    s.tolerance = 1.0e-8
    return s, T


@pytest.mark.parametrize("k", [-1.0, -10.0, 2.0])
def test_a_linear_robin_problem_takes_one_newton_step(k):
    s, T = _robin_poisson(k, f"n{int(k)}")
    s.solve()
    # development: 17 (k -1), 23 (k -10), DIVERGED_LINE_SEARCH after 22 (k 2), each
    # step cutting the residual by a fixed factor near 0.3
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    # the problem is linear, so one Newton step cuts the residual to what the linear
    # solve's own tolerance (1e-9 relative here) leaves; a second may finish it
    history = s.solve_report.history
    assert s.snes.getIterationNumber() <= 2, (s.snes.getIterationNumber(), history)
    assert history[1] < 1.0e-6 * history[0], history
    # -T'' = 1, T(0) = 0, flux -k T + 0.2 on the top: T(1) = 0.3 / (1 - k)
    X = np.asarray(T.coords)
    top = np.abs(X[:, 1] - 1.0) < 1.0e-9
    t = np.asarray(T.array).ravel()[top]
    assert np.allclose(t, 0.3 / (1.0 - k), rtol=1.0e-6, atol=1.0e-9), (t.min(), t.max())


def test_the_robin_part_of_the_jacobian_matches_finite_differences():
    def jacobians(k, tag):
        s, _T = _robin_poisson(k, tag, degree=1, cell_size=0.25)
        s.petsc_options["snes_max_it"] = 0
        s.solve()
        snes = s.snes
        x = snes.getSolution().copy()
        x.set(0.3)
        J = snes.getJacobian()[0]
        snes.computeJacobian(x, J, J)
        dense = J.copy()
        dense.convert("dense")
        hand = dense.getDenseArray().copy()
        n = x.getSize()
        F0, xp, Fp = x.duplicate(), x.duplicate(), x.duplicate()
        snes.computeFunction(x, F0)
        fd = np.zeros((n, n))
        h = 1.0e-7
        for j in range(n):
            x.copy(xp)
            xp.setValue(j, xp.getValue(j) + h)
            xp.assemble()
            snes.computeFunction(xp, Fp)
            fd[:, j] = (Fp.getArray() - F0.getArray()) / h
        return hand, fd
    if uw.mpi.size > 1:
        pytest.skip("a dense column-by-column finite difference, serial only")
    hand0, fd0 = jacobians(0.0, "j0")
    hand1, fd1 = jacobians(-1.0, "j1")
    robin_hand, robin_fd = hand1 - hand0, fd1 - fd0
    assert np.linalg.norm(robin_fd) > 0.1                   # the term is there to measure
    # development: the hand-coded Robin part is 2.0 times the finite-difference one
    ratio = np.sum(robin_hand * robin_fd) / np.sum(robin_fd * robin_fd)
    assert abs(ratio - 1.0) < 1.0e-6, ratio
    assert np.linalg.norm(robin_hand - robin_fd) < 1.0e-6 * np.linalg.norm(robin_fd)


def test_a_linear_vector_robin_problem_takes_one_newton_step():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.125, qdegree=3)
    u = uw.discretisation.MeshVariable("U1162", mesh, 2, degree=2)
    projection = uw.systems.Vector_Projection(mesh, u)
    projection.uw_function = sympy.Matrix([[1.0, 0.5]])
    projection.smoothing = 1.0e-2
    projection.add_natural_bc([-10.0 * u.sym[0], -10.0 * u.sym[1]], "Top")
    projection.tolerance = 1.0e-8
    projection.solve()
    # development: 19
    assert projection.snes.getConvergedReason() > 0, projection.snes.getConvergedReason()
    history = projection.solve_report.history
    assert projection.snes.getIterationNumber() <= 2, (projection.snes.getIterationNumber(), history)
    assert history[1] < 1.0e-6 * history[0], history
