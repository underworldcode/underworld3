"""Temporal convergence validation for implicit SUPG transport."""

import numpy as np
import pytest
import sympy

import underworld3 as uw


pytestmark = pytest.mark.level_3


def _transient_state(timestep, order):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.22,
        regular=True,
        qdegree=3,
    )
    token = str(timestep).replace(".", "p")
    temperature = uw.discretisation.MeshVariable(
        f"T_supg_time_{order}_{token}", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        f"U_supg_time_{order}_{token}", mesh, mesh.dim, degree=1
    )
    x, y = mesh.X
    shape = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)
    diffusivity = 0.05
    advection_speed = 0.4
    with mesh.access(temperature, velocity):
        temperature.data[:, 0] = uw.function.evaluate(
            shape, temperature.coords
        ).reshape(-1)
        velocity.data[:, 0] = advection_speed
        velocity.data[:, 1] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        order=order,
        tau=0.0,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = diffusivity
    for boundary in ("Left", "Right", "Top", "Bottom"):
        thermal.add_dirichlet_bc(0.0, boundary)

    final_time = 0.2
    for step in range(round(final_time / timestep)):
        new_time = (step + 1) * timestep
        amplitude = np.exp(-new_time)
        thermal.f = amplitude * (
            (-1.0 + 2.0 * diffusivity * sympy.pi**2) * shape
            + advection_speed
            * sympy.pi
            * sympy.cos(sympy.pi * x)
            * sympy.sin(sympy.pi * y)
        )
        thermal.solve(timestep=timestep, zero_init_guess=False)

    return temperature.data[:, 0].copy()


@pytest.mark.parametrize(
    ("order", "minimum_rate"),
    ((1, 0.9), (2, 1.8)),
)
def test_bdf_temporal_convergence(order, minimum_rate):
    reference = _transient_state(0.003125, 2)
    timesteps = (0.05, 0.025, 0.0125)
    errors = [
        np.linalg.norm(_transient_state(timestep, order) - reference)
        / np.sqrt(reference.size)
        for timestep in timesteps
    ]
    rates = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0)
        for index in range(2)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert min(rates) > minimum_rate


def _citcoms_decay_error(timestep):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.5,
        regular=True,
    )
    token = str(timestep).replace(".", "p")
    temperature = uw.discretisation.MeshVariable(
        f"T_citcoms_decay_{token}", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        f"U_citcoms_decay_{token}", mesh, mesh.dim, degree=1
    )
    temperature.data[:, 0] = 1.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
        tau=0.0,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.0
    thermal.f = -temperature.sym[0]

    for _ in range(round(1.0 / timestep)):
        thermal.solve(timestep=timestep)

    return abs(float(np.mean(temperature.data[:, 0])) - np.exp(-1.0))


def test_citcoms_predictor_corrector_is_second_order_for_scalar_decay():
    errors = [_citcoms_decay_error(dt) for dt in (0.1, 0.05, 0.025)]
    rates = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0)
        for index in range(2)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert min(rates) > 1.9
