"""Legacy diffusion must not change its operator when restore rebuilds it."""

import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _problem(order, theta):
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    temperature.array[:, 0, 0] = temperature.coords[:, 0]
    solver = uw.systems.Diffusion(mesh, temperature, order=order, theta=theta)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 0.05
    solver.tolerance = 1e-12
    return uw.get_default_model(), mesh, solver, temperature


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_compiled_flux_contains_initial_symbolic_history(theta):
    _model, _mesh, solver, _temperature = _problem(2, theta)
    for _ in range(3):
        solver.solve(timestep=0.01, zero_init_guess=False)
    unwrap = uw.function.expressions.unwrap
    compiled = unwrap(solver._f1.sym, keep_constants=False)
    live = unwrap(solver.DFDt.adams_moulton_flux(), keep_constants=False)
    assert all(sympy.simplify(term) == 0 for term in compiled - live)


@pytest.mark.parametrize("order,theta,warm_steps", [
    (1, 1.0, 3), (1, 0.5, 3), (2, 1.0, 0),
    (2, 1.0, 1), (2, 1.0, 3), (3, 1.0, 4),
])
def test_diffusion_snapshot_replays_continuation(order, theta, warm_steps):
    model, mesh, solver, temperature = _problem(order, theta)
    for _ in range(warm_steps):
        solver.solve(timestep=0.01, zero_init_guess=False)
    initial_fields = {name: np.array(var.array) for name, var in mesh.vars.items()}
    snapshot = model.save_state()
    timesteps = [0.01, 0.01, 0.015, 0.0075]
    reference = []
    for dt in timesteps:
        solver.solve(timestep=dt, zero_init_guess=False)
        reference.append(np.array(temperature.array))

    model.load_state(snapshot)
    for name, values in initial_fields.items():
        np.testing.assert_array_equal(mesh.vars[name].array, values)
    max_error = 0.0
    for dt, expected in zip(timesteps, reference):
        solver.solve(timestep=dt, zero_init_guess=False)
        actual = np.asarray(temperature.array)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    max_error = max(uw.mpi.comm.allgather(max_error))
    uw.pprint(f"DIFFUSION_REPLAY order={order} theta={theta} warm={warm_steps} "
              f"ranks={uw.mpi.size} max_error={max_error:.12g}")
