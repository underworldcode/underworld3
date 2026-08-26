"""Numerical validation for implicit SUPG scalar transport."""

import numpy as np
import pytest
import sympy

import underworld3 as uw


pytestmark = pytest.mark.level_2


def test_simplex_geometry_is_reused_between_automatic_operations():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
        regular=True,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_geometry_cache", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_geometry_cache", mesh, mesh.dim, degree=1
    )
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 1.0

    first = thermal._simplex_data()
    second = thermal._simplex_data()

    assert all(a is b for a, b in zip(first, second))


def _high_peclet_solution(tau, name):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.22,
        regular=True,
    )
    temperature = uw.discretisation.MeshVariable(
        f"T_layer_{name}", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        f"U_layer_{name}", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
        temperature.data[:, 0] = temperature.coords[:, 0]
        velocity.data[:, 0] = 1.0
        velocity.data[:, 1] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        tau=tau,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    thermal.add_dirichlet_bc(0.0, "Left")
    thermal.add_dirichlet_bc(1.0, "Right")
    thermal.solve(timestep=1.0e6, zero_init_guess=False)

    x = temperature.coords[:, 0]
    exact = np.expm1(100.0 * x) / np.expm1(100.0)
    rms_error = float(np.sqrt(np.mean((temperature.data[:, 0] - exact) ** 2)))
    return temperature.data.copy(), rms_error


def test_supg_reduces_high_peclet_oscillation_and_error():
    galerkin, galerkin_error = _high_peclet_solution(0.0, "galerkin")
    supg, supg_error = _high_peclet_solution(None, "supg")

    galerkin_overshoot = max(0.0, float(galerkin.max() - 1.0))
    galerkin_undershoot = max(0.0, float(-galerkin.min()))
    supg_overshoot = max(0.0, float(supg.max() - 1.0))
    supg_undershoot = max(0.0, float(-supg.min()))

    assert supg_overshoot < galerkin_overshoot
    assert supg_undershoot < galerkin_undershoot
    assert supg_error < 0.2 * galerkin_error


def _manufactured_error(cell_size, degree):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=cell_size,
        regular=True,
        qdegree=4,
    )
    temperature = uw.discretisation.MeshVariable(
        f"T_mms_{degree}_{cell_size}", mesh, 1, degree=degree
    )
    velocity = uw.discretisation.MeshVariable(
        f"U_mms_{degree}_{cell_size}", mesh, mesh.dim, degree=1
    )
    x, y = mesh.X
    exact = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)
    diffusivity = 0.1
    with mesh.access(temperature, velocity):
        temperature.data[:, 0] = uw.function.evaluate(
            exact, temperature.coords
        ).reshape(-1)
        velocity.data[:, 0] = 1.0
        velocity.data[:, 1] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, u_Field=temperature, V_fn=velocity.sym
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = diffusivity
    thermal.f = (
        sympy.pi * sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)
        + 2.0 * diffusivity * sympy.pi**2 * exact
    )
    for boundary in ("Left", "Right", "Top", "Bottom"):
        thermal.add_dirichlet_bc(0.0, boundary)
    thermal.solve(timestep=1.0e8, zero_init_guess=False)

    return float(
        np.sqrt(
            uw.maths.Integral(
                mesh, fn=(temperature.sym[0] - exact) ** 2
            ).evaluate()
        )
    )


@pytest.mark.parametrize("degree", (1, 2))
def test_manufactured_solution_converges_under_refinement(degree):
    cell_sizes = (0.3, 0.2, 0.13)
    errors = [_manufactured_error(cell_size, degree) for cell_size in cell_sizes]
    final_rate = np.log(errors[-2] / errors[-1]) / np.log(
        cell_sizes[-2] / cell_sizes[-1]
    )

    assert errors[0] > errors[1] > errors[2]
    assert final_rate > 1.5


def test_spherical_shell_supg_is_parallel_safe():
    mesh = uw.meshing.SphericalShell(
        radiusInner=0.55,
        radiusOuter=1.0,
        cellSize=0.4,
        qdegree=2,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_supg_spherical", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_supg_spherical", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
        coords = temperature.coords
        radii = np.linalg.norm(coords, axis=1)
        temperature.data[:, 0] = (1.0 - radii) / 0.45 + 0.01 * coords[:, 0]
        velocity.data[:, 0] = -0.02 * coords[:, 1]
        velocity.data[:, 1] = 0.02 * coords[:, 0]
        velocity.data[:, 2] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    thermal.add_dirichlet_bc(0.0, "Upper")
    thermal.add_dirichlet_bc(1.0, "Lower")

    for _ in range(3):
        thermal.solve(timestep=1.0e-3, zero_init_guess=False)

    temperature_l2_squared = float(
        uw.maths.Integral(mesh, fn=temperature.sym[0] ** 2).evaluate()
    )
    assert np.all(np.isfinite(temperature.data))
    assert np.all(np.isfinite(thermal._supg_tau.data))
    assert temperature_l2_squared == pytest.approx(0.833491030982, rel=1.0e-8)


def test_bdf2_snapshot_restore_leaves_no_discarded_step_trace():
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.3,
        regular=True,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_supg_restart", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_supg_restart", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
        temperature.data[:, 0] = np.sin(np.pi * temperature.coords[:, 0])
        velocity.data[:, 0] = 0.1
        velocity.data[:, 1] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        order=2,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.05
    thermal.add_dirichlet_bc(0.0, "Left")
    thermal.add_dirichlet_bc(0.0, "Right")

    for _ in range(3):
        thermal.solve(timestep=0.01, zero_init_guess=False)
    snapshot = model.save_state()

    model.load_state(snapshot)
    for _ in range(3):
        thermal.solve(timestep=0.01, zero_init_guess=False)
    reference = temperature.data.copy()

    model.load_state(snapshot)
    thermal.solve(timestep=0.2, zero_init_guess=False)
    model.load_state(snapshot)
    for _ in range(3):
        thermal.solve(timestep=0.01, zero_init_guess=False)
    resumed = temperature.data.copy()

    np.testing.assert_array_equal(resumed, reference)
    uw.reset_default_model()


def test_repeated_solves_keep_histories_and_transient_state_bounded():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.3,
        regular=True,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_supg_lifecycle", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_supg_lifecycle", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
        temperature.data[:, 0] = temperature.coords[:, 0]
        velocity.data[:, 0] = 0.1
        velocity.data[:, 1] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, u_Field=temperature, V_fn=velocity.sym
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.05
    live_swarms = len(mesh._registered_swarms)

    for _ in range(38):
        thermal.solve(timestep=0.001, zero_init_guess=False)
        assert len(mesh._registered_swarms) == live_swarms

    assert len(thermal.solve_history) == 32
    assert np.all(np.isfinite(temperature.data))


def test_citcoms_spherical_shell_is_parallel_safe():
    mesh = uw.meshing.SphericalShell(
        radiusInner=0.55,
        radiusOuter=1.0,
        cellSize=0.25,
        qdegree=2,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_citcoms_spherical", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_citcoms_spherical", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
        coords = temperature.coords
        radii = np.linalg.norm(coords, axis=1)
        temperature.data[:, 0] = (1.0 - radii) / 0.45 + 0.01 * coords[:, 0]
        velocity.data[:, 0] = -0.02 * coords[:, 1]
        velocity.data[:, 1] = 0.02 * coords[:, 0]
        velocity.data[:, 2] = 0.0

    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    thermal.add_dirichlet_bc(0.0, "Upper")
    thermal.add_dirichlet_bc(1.0, "Lower")
    thermal.solve(timestep=1.0e-3)

    temperature_l2_squared = float(
        uw.maths.Integral(mesh, fn=temperature.sym[0] ** 2).evaluate()
    )
    assert thermal._lumped_mass.getSize() > 0
    assert np.all(np.isfinite(temperature.data))
    assert temperature_l2_squared == pytest.approx(0.814491155536, rel=1.0e-8)


def test_citcoms_snapshot_restores_startup_state_exactly():
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.3,
        regular=True,
    )
    temperature = uw.discretisation.MeshVariable(
        "T_citcoms_restart", mesh, 1, degree=1
    )
    velocity = uw.discretisation.MeshVariable(
        "U_citcoms_restart", mesh, mesh.dim, degree=1
    )
    with mesh.access(temperature, velocity):
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

    initial = model.save_state()
    thermal.solve(timestep=0.05)
    reference_temperature = temperature.data.copy()
    reference_rate = thermal._temperature_rate.data.copy()

    model.load_state(initial)
    assert not thermal._rate_initialised
    thermal.solve(timestep=0.05)

    np.testing.assert_array_equal(temperature.data, reference_temperature)
    np.testing.assert_array_equal(
        thermal._temperature_rate.data, reference_rate
    )
    uw.reset_default_model()
