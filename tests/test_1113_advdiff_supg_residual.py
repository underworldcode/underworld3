"""Focused tests for the implicit SUPG scalar transport residual."""

import numpy as np
import pytest
import sympy

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh_temperature_velocity(prefix, velocity=(1.0, 0.0)):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.3,
        regular=False,
    )
    temperature = uw.discretisation.MeshVariable(f"T_{prefix}", mesh, 1, degree=1)
    flow = uw.discretisation.MeshVariable(f"U_{prefix}", mesh, mesh.dim, degree=1)
    temperature.array[:, 0, 0] = temperature.coords[:, 0]
    flow.array[:, 0, 0] = velocity[0]
    flow.array[:, 0, 1] = velocity[1]
    return mesh, temperature, flow


def _configure_diffusion(solver, diffusivity=0.1):
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = diffusivity


def test_public_api_and_residual_shapes():
    mesh, temperature, velocity = _mesh_temperature_velocity("api")
    thermal = uw.systems.AdvDiffusion(
        mesh, temperature, velocity.sym, theta=0.5, peclet_weight=0.0
    )
    thermal.DuDt.supg_weight = 0.0
    _configure_diffusion(thermal)
    thermal.delta_t = 0.01

    assert thermal.F0.sym.shape == (1, 1)
    assert thermal.F1.sym.shape == (1, mesh.cdim)
    np.testing.assert_array_equal(
        uw.function.evaluate(thermal.DuDt.tau(), mesh._centroids), 0.0
    )


def test_manager_owns_eulerian_advection_once():
    mesh, temperature, velocity = _mesh_temperature_velocity("double")
    manager = uw.systems.ddt.EulerianSUPG(
        mesh,
        temperature,
        velocity.sym,
        vtype=uw.VarType.SCALAR,
        degree=temperature.degree,
        continuous=temperature.continuous,
        theta=1.0,
        peclet_weight=0.0,
    )
    thermal = uw.systems.AdvDiffusion(
        mesh, temperature, velocity.sym, DuDt=manager, theta=1.0
    )
    _configure_diffusion(thermal)
    thermal.delta_t = 0.01
    expected = manager.time_derivative() + manager.advection() - thermal.f
    assert thermal.DuDt is manager
    assert sympy.simplify(thermal.F0.sym - expected) == sympy.zeros(1, 1)


@pytest.mark.parametrize("theta", (0.0, 0.5))
def test_rejects_nonimplicit_flux_history(theta):
    mesh, temperature, velocity = _mesh_temperature_velocity(f"theta_{theta}")
    with pytest.raises(ValueError, match="theta=1.0"):
        uw.systems.AdvDiffusion(
            mesh, temperature, velocity.sym, order=2, theta=theta, peclet_weight=0.0
        )


def test_automatic_tau_is_finite_and_bounded_by_transient_scale():
    mesh, temperature, velocity = _mesh_temperature_velocity("tau")
    thermal = uw.systems.AdvDiffusion(
        mesh, temperature, velocity.sym, theta=0.5, peclet_weight=0.0
    )
    _configure_diffusion(thermal, diffusivity=0.1)
    thermal.delta_t = 0.02
    thermal.DuDt.diffusivity = 0.1

    tau = uw.function.evaluate(thermal.DuDt.tau(), mesh._centroids)
    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0.0)
    assert np.all(tau <= 0.01)


def test_negative_diffusivity_is_rejected():
    mesh, temperature, velocity = _mesh_temperature_velocity("negative_k")
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym, method="citcoms"
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=-0.1)
    thermal.delta_t = 0.01

    with pytest.raises(ValueError, match="non-negative"):
        thermal.DuDt._update_automatic_tau()


def test_zero_velocity_matches_diffusion_solver():
    mesh_a, temperature_a, velocity = _mesh_temperature_velocity(
        "supg_zero", velocity=(0.0, 0.0)
    )
    mesh_b, temperature_b, _ = _mesh_temperature_velocity(
        "diffusion", velocity=(0.0, 0.0)
    )
    temperature_a.array[:, 0, 0] = np.sin(np.pi * temperature_a.coords[:, 0])
    temperature_b.array[:, 0, 0] = np.sin(np.pi * temperature_b.coords[:, 0])

    supg = uw.systems.AdvDiffusion(
        mesh_a, temperature_a, velocity.sym, theta=1.0, peclet_weight=0.0
    )
    diffusion = uw.systems.Diffusion(mesh_b, u_Field=temperature_b, theta=1.0)
    _configure_diffusion(supg, diffusivity=0.1)
    _configure_diffusion(diffusion, diffusivity=0.1)
    # Compare equations at the same solve accuracy, not two preconditioners'
    # different default stopping criteria.
    for solver in (supg, diffusion):
        solver.petsc_options["ksp_rtol"] = 1.0e-13
        solver.petsc_options["snes_rtol"] = 1.0e-12
        solver.petsc_options["snes_atol"] = 1.0e-13

    supg.solve(timestep=0.01, zero_init_guess=False)
    diffusion.solve(timestep=0.01, zero_init_guess=False)

    np.testing.assert_allclose(
        temperature_a.array,
        temperature_b.array,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_citcoms_integrator_requires_continuous_p1_temperature():
    mesh, temperature, velocity = _mesh_temperature_velocity("citcoms_p1")
    temperature_p2 = uw.discretisation.MeshVariable("T_citcoms_p2", mesh, 1, degree=2)

    with pytest.raises(ValueError, match="continuous scalar P1"):
        uw.systems.ddt.EulerianSUPGPC(
            mesh, temperature_p2, velocity.sym,
            method="citcoms",
        )


def test_converged_pc_validates_correction_controls():
    mesh, temperature, velocity = _mesh_temperature_velocity("pc_converged_api")
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="pc_converged",
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    assert manager.method == "pc_converged"
    assert manager.integrator == "pc_converged"
    assert thermal.DuDt.corrector_rtol == pytest.approx(1.0e-10)
    assert thermal.DuDt.corrector_atol == pytest.approx(1.0e-12)
    assert thermal.DuDt.max_corrector_steps == 100

    invalid = (
        ({"corrector_rtol": 0.0}, "corrector_rtol"),
        ({"corrector_atol": -1.0}, "corrector_atol"),
        ({"max_corrector_steps": 0}, "max_corrector_steps"),
        ({"adv_gamma": 0.6}, "adv_gamma=0.5"),
    )
    for kwargs, message in invalid:
        with pytest.raises(ValueError, match=message):
            uw.systems.ddt.EulerianSUPGPC(
                mesh, temperature, velocity.sym,
                method="pc_converged", **kwargs,
            )


def test_converged_pc_fails_when_residual_tolerance_is_not_reached():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "pc_converged_failure", velocity=(0.0, 0.0)
    )
    temperature.array[:, 0, 0] = np.prod(
        np.sin(np.pi * np.asarray(temperature.coords)), axis=1
    )
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="pc_converged",
        corrector_rtol=1.0e-15,
        corrector_atol=0.0,
        max_corrector_steps=1,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.1)

    with pytest.raises(RuntimeError, match="did not reach"):
        thermal.solve(timestep=0.01)


def test_citcoms_lumped_mass_matches_constant_residual():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_mass", velocity=(0.0, 0.0)
    )
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="citcoms", tau=0.0,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.0)
    thermal.delta_t = 0.01
    thermal.DuDt._setup_citcoms_residual()
    mass = thermal.DuDt._assemble_lumped_mass()
    thermal.DuDt.temperature_rate.array[:, 0, 0] = 1.0
    solution, residual = thermal.DuDt._compute_citcoms_residual()

    np.testing.assert_allclose(residual.array / mass.array, 1.0, atol=1.0e-14)
    assert mass.min()[1] > 0.0
    solution.destroy()
    residual.destroy()


def test_citcoms_constant_source_is_exact_from_first_step():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_source", velocity=(0.0, 0.0)
    )
    temperature.array[:, 0, 0] = 0.0
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="citcoms", tau=0.0,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.0)
    thermal.f = 1.0

    thermal.solve(timestep=0.1)

    np.testing.assert_allclose(temperature.array, 0.1, atol=1.0e-14)
    np.testing.assert_allclose(thermal.DuDt.temperature_rate.array, 1.0, atol=1.0e-14)


def test_citcoms_reuses_predictor_corrector_work_vectors():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_workspace", velocity=(0.0, 0.0)
    )
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="citcoms", tau=0.0,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.0)

    thermal.solve(timestep=0.01)
    vector_handles = tuple(vector.handle for vector in thermal.DuDt._citcoms_work_vectors)
    thermal.solve(timestep=0.01)

    assert (
        tuple(vector.handle for vector in thermal.DuDt._citcoms_work_vectors)
        == vector_handles
    )


def test_citcoms_timestep_uses_advection_and_lumped_diffusion_limits():
    mesh, temperature, velocity = _mesh_temperature_velocity("citcoms_dt")
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="citcoms",
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.1)

    timestep = thermal.estimate_dt()

    assert np.isfinite(timestep)
    assert timestep == pytest.approx(0.9 * min(thermal.DuDt.dt_adv, thermal.DuDt.dt_diff))
    assert thermal.DuDt.dt_adv > 0.0
    assert thermal.DuDt.dt_diff > 0.0


def test_timestep_diffusivity_branch_is_collective():
    mesh, temperature, velocity = _mesh_temperature_velocity("collective_diffusivity")
    manager = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity.sym,
        method="citcoms",
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity.sym, DuDt=manager)
    _configure_diffusion(thermal, diffusivity=0.1)
    thermal.delta_t = 0.01
    thermal.DuDt._cell_diffusivity = lambda count: (
        np.ones(count) if uw.mpi.rank == 0 else np.zeros(count)
    )

    timestep = thermal.estimate_dt()

    assert np.isfinite(timestep)
    assert thermal.DuDt.dt_diff > 0.0
