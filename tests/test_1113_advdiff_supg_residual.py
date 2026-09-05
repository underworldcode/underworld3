"""Focused tests for the implicit SUPG scalar transport residual."""

import numpy as np
import pytest

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
    with mesh.access(temperature, flow):
        temperature.data[:, 0] = temperature.coords[:, 0]
        flow.data[:, 0] = velocity[0]
        flow.data[:, 1] = velocity[1]
    return mesh, temperature, flow


def _configure_diffusion(solver, diffusivity=0.1):
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = diffusivity


def test_public_api_and_residual_shapes():
    mesh, temperature, velocity = _mesh_temperature_velocity("api")
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, u_Field=temperature, V_fn=velocity.sym, tau=0.0
    )
    _configure_diffusion(thermal)
    thermal.delta_t = 0.01

    assert thermal.F0.sym.shape == (1, 1)
    assert thermal.F1.sym.shape == (1, mesh.cdim)
    assert float(thermal.tau) == 0.0


def test_rejects_double_counted_eulerian_advection():
    mesh, temperature, velocity = _mesh_temperature_velocity("double")
    history = uw.systems.Eulerian_DDt(
        mesh,
        temperature,
        vtype=uw.VarType.SCALAR,
        degree=temperature.degree,
        continuous=temperature.continuous,
        V_fn=velocity.sym,
    )

    with pytest.raises(ValueError, match="DuDt.V_fn must be None"):
        uw.systems.AdvDiffusionSUPG(
            mesh,
            u_Field=temperature,
            V_fn=velocity.sym,
            DuDt=history,
        )


@pytest.mark.parametrize("theta", (0.0, 0.5))
def test_rejects_nonimplicit_flux_history(theta):
    mesh, temperature, velocity = _mesh_temperature_velocity(f"theta_{theta}")
    with pytest.raises(ValueError, match="theta=1.0"):
        uw.systems.AdvDiffusionSUPG(
            mesh,
            u_Field=temperature,
            V_fn=velocity.sym,
            theta=theta,
            time_integrator="bdf",
        )


def test_automatic_tau_is_finite_and_bounded_by_transient_scale():
    mesh, temperature, velocity = _mesh_temperature_velocity("tau")
    thermal = uw.systems.AdvDiffusionSUPG(mesh, u_Field=temperature, V_fn=velocity.sym)
    _configure_diffusion(thermal, diffusivity=0.1)
    thermal.delta_t = 0.02
    thermal._update_automatic_tau()

    tau = uw.function.evaluate(thermal.tau, mesh._centroids)
    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0.0)
    assert np.all(tau <= 0.01)


def test_negative_diffusivity_is_rejected():
    mesh, temperature, velocity = _mesh_temperature_velocity("negative_k")
    thermal = uw.systems.AdvDiffusionSUPG(mesh, u_Field=temperature, V_fn=velocity.sym)
    _configure_diffusion(thermal, diffusivity=-0.1)
    thermal.delta_t = 0.01

    with pytest.raises(ValueError, match="non-negative"):
        thermal._update_automatic_tau()


def test_zero_velocity_matches_diffusion_solver():
    mesh_a, temperature_a, velocity = _mesh_temperature_velocity(
        "supg_zero", velocity=(0.0, 0.0)
    )
    mesh_b, temperature_b, _ = _mesh_temperature_velocity(
        "diffusion", velocity=(0.0, 0.0)
    )
    with mesh_a.access(temperature_a), mesh_b.access(temperature_b):
        temperature_a.data[:, 0] = np.sin(np.pi * temperature_a.coords[:, 0])
        temperature_b.data[:, 0] = np.sin(np.pi * temperature_b.coords[:, 0])

    supg = uw.systems.AdvDiffusionSUPG(
        mesh_a, u_Field=temperature_a, V_fn=velocity.sym, theta=1.0)
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
        temperature_a.data,
        temperature_b.data,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_citcoms_integrator_requires_continuous_p1_temperature():
    mesh, temperature, velocity = _mesh_temperature_velocity("citcoms_p1")
    temperature_p2 = uw.discretisation.MeshVariable("T_citcoms_p2", mesh, 1, degree=2)

    with pytest.raises(ValueError, match="continuous P1"):
        uw.systems.AdvDiffusionSUPG(
            mesh,
            u_Field=temperature_p2,
            V_fn=velocity.sym,
            time_integrator="citcoms",
        )


def test_citcoms_lumped_mass_matches_constant_residual():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_mass", velocity=(0.0, 0.0)
    )
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
        tau=0.0,
    )
    _configure_diffusion(thermal, diffusivity=0.0)
    thermal.delta_t = 0.01
    thermal._setup_citcoms_residual()
    mass = thermal._assemble_lumped_mass()
    thermal._temperature_rate.data[:, 0] = 1.0
    solution, residual = thermal._compute_citcoms_residual()

    np.testing.assert_allclose(residual.array / mass.array, 1.0, atol=1.0e-14)
    assert mass.min()[1] > 0.0
    solution.destroy()
    residual.destroy()


def test_citcoms_constant_source_is_exact_from_first_step():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_source", velocity=(0.0, 0.0)
    )
    temperature.data[:, 0] = 0.0
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
        tau=0.0,
    )
    _configure_diffusion(thermal, diffusivity=0.0)
    thermal.f = 1.0

    thermal.solve(timestep=0.1)

    np.testing.assert_allclose(temperature.data, 0.1, atol=1.0e-14)
    np.testing.assert_allclose(thermal._temperature_rate.data, 1.0, atol=1.0e-14)


def test_citcoms_reuses_predictor_corrector_work_vectors():
    mesh, temperature, velocity = _mesh_temperature_velocity(
        "citcoms_workspace", velocity=(0.0, 0.0)
    )
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
        tau=0.0,
    )
    _configure_diffusion(thermal, diffusivity=0.0)

    thermal.solve(timestep=0.01)
    vector_handles = tuple(vector.handle for vector in thermal._citcoms_work_vectors)
    thermal.solve(timestep=0.01)

    assert (
        tuple(vector.handle for vector in thermal._citcoms_work_vectors)
        == vector_handles
    )


def test_citcoms_timestep_uses_advection_and_lumped_diffusion_limits():
    mesh, temperature, velocity = _mesh_temperature_velocity("citcoms_dt")
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
    )
    _configure_diffusion(thermal, diffusivity=0.1)

    timestep = thermal.estimate_dt()

    assert np.isfinite(timestep)
    assert timestep == pytest.approx(0.9 * min(thermal.dt_adv, thermal.dt_diff))
    assert thermal.dt_adv > 0.0
    assert thermal.dt_diff > 0.0


def test_timestep_diffusivity_branch_is_collective():
    mesh, temperature, velocity = _mesh_temperature_velocity("collective_diffusivity")
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        time_integrator="citcoms",
    )
    _configure_diffusion(thermal, diffusivity=0.1)
    thermal.delta_t = 0.01
    thermal._cell_diffusivity = lambda count: (
        np.ones(count) if uw.mpi.rank == 0 else np.zeros(count)
    )

    timestep = thermal.estimate_dt()

    assert np.isfinite(timestep)
    assert thermal.dt_diff > 0.0
