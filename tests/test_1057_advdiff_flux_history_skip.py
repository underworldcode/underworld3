"""Regression tests for inactive Backward-Euler flux history."""

import numpy as np
import pytest
import sympy

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_advdiff(theta=1.0, order=1):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    temperature = uw.discretisation.MeshVariable(
        "T_flux_history", mesh, 1, degree=1
    )
    x, y = mesh.X
    velocity = sympy.Matrix([[-(y - 0.5), x - 0.5]])

    temperature.data[:, 0] = np.asarray(
        temperature.coords[:, 0] + 0.25 * temperature.coords[:, 1]
    )

    thermal = uw.systems.AdvDiffusionSLCN(
        mesh,
        u_Field=temperature,
        V_fn=velocity,
        order=order,
        theta=theta,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    return thermal, temperature


@pytest.mark.parametrize(
    ("theta", "order", "expected_updates"),
    (
        (1.0, 1, 0),
        (0.5, 1, 1),
        (1.0, 2, 1),
    ),
)
def test_only_current_flux_skips_history_lifecycle(
    theta, order, expected_updates
):
    thermal, _ = _make_advdiff(theta=theta, order=order)
    calls = {"pre": 0, "post": 0}
    original_pre = thermal.DFDt.update_pre_solve
    original_post = thermal.DFDt.update_post_solve

    def counted_pre(*args, **kwargs):
        calls["pre"] += 1
        return original_pre(*args, **kwargs)

    def counted_post(*args, **kwargs):
        calls["post"] += 1
        return original_post(*args, **kwargs)

    thermal.DFDt.update_pre_solve = counted_pre
    thermal.DFDt.update_post_solve = counted_post
    thermal.solve(timestep=0.01, zero_init_guess=False)

    assert calls == {"pre": expected_updates, "post": expected_updates}


def test_skipped_backward_euler_flux_matches_history_path():
    optimized, optimized_temperature = _make_advdiff(theta=1.0, order=1)
    reference, reference_temperature = _make_advdiff(theta=1.0, order=1)

    reference._prepare_flux_history = lambda: True

    optimized.solve(timestep=0.01, zero_init_guess=False)
    reference.solve(timestep=0.01, zero_init_guess=False)

    np.testing.assert_allclose(
        optimized_temperature.data,
        reference_temperature.data,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
