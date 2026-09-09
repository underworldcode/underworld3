"""Transporting a stress history on the grid instead of along characteristics.

A viscoelastic solve carries a stress that is not its own unknown. The history
manager owns that transport: the semi-Lagrangian flavour traces it back along
characteristics, and ``EulerianSUPG`` with ``transport_on_update`` assembles it
implicitly with the same SUPG stabilisation the solvers use. These are the
transport tests the existing viscoelastic benchmarks cannot give us, because
those are all spatially uniform and so transport nothing.

Run: pixi run python -m pytest tests/test_1059_stress_transport.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

COMPONENTS = ((0, 0), (0, 1), (1, 1))
AMPLITUDES = (1.0, 0.5, -1.0)


def _blob(x, y, x0, y0=0.0, width=0.04):
    return sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2) / width)


def _plant(var, expression):
    """Set every independent component of a symmetric tensor from one shape."""
    values = uw.function.evaluate(expression, var.coords).reshape(-1)
    for (i, j), amplitude in zip(COMPONENTS, AMPLITUDES):
        var.array[:, i, j] = amplitude * values
        if i != j:
            var.array[:, j, i] = amplitude * values
    return values


def _error(var, expression):
    """Relative L2 error of the tensor against one shape times the amplitudes."""
    exact = uw.function.evaluate(expression, var.coords).reshape(-1)
    numerator = denominator = 0.0
    for (i, j), amplitude in zip(COMPONENTS, AMPLITUDES):
        target = amplitude * exact
        numerator += float(np.sum((np.asarray(var.array[:, i, j]) - target) ** 2))
        denominator += float(np.sum(target ** 2))
    return np.sqrt(numerator / denominator)


def _managers(mesh, tag, velocity, order=1):
    """One Eulerian-SUPG and one semi-Lagrangian history of the same stress."""
    stress = uw.discretisation.MeshVariable(
        f"S_{tag}", mesh, vtype=uw.VarType.SYM_TENSOR, degree=2)
    eulerian = uw.systems.ddt.EulerianSUPG(
        mesh, stress, velocity, vtype=uw.VarType.SYM_TENSOR, degree=2,
        continuous=True, order=order, transport_on_update=True)
    lagrangian = uw.systems.ddt.SemiLagrangian(
        mesh, stress.sym, velocity, vtype=uw.VarType.SYM_TENSOR, degree=2,
        continuous=True, order=order, varsymbol=rf"S^{{{tag}}}")
    return stress, eulerian, lagrangian


def test_a_stress_blob_is_carried_by_a_uniform_flow():
    """Uniform translation has an exact answer: the blob arrives where the flow
    puts it, with its components unchanged (no rotation in a uniform flow)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 24, qdegree=3)
    x, y = mesh.X
    speed, dt, steps, start = 0.5, 0.05, 8, -0.5
    stress, eulerian, lagrangian = _managers(
        mesh, "uniform", sympy.Matrix([[speed, 0.0]]))

    _plant(stress, _blob(x, y, start))
    eulerian.initialise_history()
    lagrangian.initialise_history()
    assert eulerian.transport_on_update and eulerian._advection_mode == "assembled"

    # The step a solver takes: the manager carries the history forward, then the
    # new stress is committed. With no constitutive update the commit is the
    # identity, which is what makes this a pure transport comparison.
    for _ in range(steps):
        for manager in (eulerian, lagrangian):
            manager.update_pre_solve(dt)
            stress.array[...] = manager.psi_star[0].array[...]

    exact = _blob(x, y, start + speed * dt * steps)
    grid = _error(eulerian.psi_star[0], exact)
    traced = _error(lagrangian.psi_star[0], exact)
    assert grid < 0.1, grid
    assert traced < 0.2, traced
    # the grid transport is at least as accurate as the trace-back here
    assert grid <= traced, (grid, traced)
    # negative control: the field really moved
    assert _error(eulerian.psi_star[0], _blob(x, y, start)) > 0.5
    # symmetry is preserved component by component
    carried = np.asarray(eulerian.psi_star[0].array)
    assert np.allclose(carried[:, 0, 1], carried[:, 1, 0])


def test_the_components_ride_round_a_rigid_rotation_unchanged():
    """With no rotation terms in the transport the tensor components are
    advected as scalars: after a quarter turn each component sits where the
    flow carried it, with the same amplitude."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 24, qdegree=3)
    x, y = mesh.X
    stress, eulerian, _sl = _managers(mesh, "rot", sympy.Matrix([[-y, x]]))

    radius, dt, steps = 0.5, np.pi / 2 / 40, 40      # a quarter turn
    _plant(stress, _blob(x, y, radius))
    eulerian.initialise_history()
    for _ in range(steps):
        eulerian.update_pre_solve(dt)
        stress.array[...] = eulerian.psi_star[0].array[...]

    turned = _blob(x, y, 0.0, radius)                 # a quarter turn from (r, 0)
    assert _error(eulerian.psi_star[0], turned) < 0.3
    peak = float(np.abs(np.asarray(eulerian.psi_star[0].array[:, 0, 0])).max())
    assert 0.6 < peak < 1.05, peak                    # amplitude carried, not created


def test_transport_is_off_unless_asked_for():
    """A manager built for a solver's own unknown assembles its advection in
    that solver's residual and must not also move its history."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_off", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(_blob(x, y, 0.3), T.coords).reshape(-1)
    manager = uw.systems.ddt.EulerianSUPG(
        mesh, T, sympy.Matrix([[1.0, 0.0]]), vtype=uw.VarType.SCALAR,
        degree=2, continuous=True)
    manager.initialise_history()
    before = np.array(manager.psi_star[0].array)
    manager.update_pre_solve(0.05)
    assert manager.transport_on_update is False
    assert np.array_equal(np.asarray(manager.psi_star[0].array), before)
    assert manager._transport_solver is None
