"""Predictor-corrector composition with the scalar transport solver."""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def fields():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        qdegree=3,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    return mesh, temperature, sympy.zeros(1, mesh.dim)


def test_manager_rejects_a_different_solver_unknown(fields):
    mesh, temperature, velocity = fields
    other = uw.discretisation.MeshVariable("Other", mesh, 1, degree=1)
    transport = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity)
    with pytest.raises(ValueError, match="unknown|u_Field|field"):
        uw.systems.AdvDiffusion(mesh, other, velocity, DuDt=transport)


@pytest.mark.parametrize("method", ["citcoms", "pc_converged"])
def test_manager_uses_live_boundary_conditions_and_timestep(fields, method):
    mesh, temperature, velocity = fields
    temperature.array[:, 0, 0] = 1.0
    transport = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity, method=method,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=transport)
    thermal.constitutive_model.Parameters.diffusivity = 0.1
    # Conditions are added after the manager is bound to the solver.
    for boundary in ("Left", "Right", "Top", "Bottom"):
        thermal.add_dirichlet_bc(1.0, boundary)
    thermal.solve(timestep=0.001)
    thermal.solve()
    assert thermal.DuDt is transport
    assert float(transport.delta_t.sym) == 0.001
    np.testing.assert_allclose(temperature.array, 1.0, rtol=0, atol=1e-12)


def test_rejected_theta_does_not_change_snapshot_metadata(fields):
    mesh, temperature, velocity = fields
    transport = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity)
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=transport)
    previous = thermal.state
    with pytest.raises(ValueError, match="theta"):
        thermal.theta = 0.5
    assert thermal.state == previous


def test_manager_cannot_execute_against_another_solver(fields):
    mesh, temperature, velocity = fields
    transport = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity)
    first = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=transport)
    other = uw.discretisation.MeshVariable("Other", mesh, 1, degree=1)
    second = uw.systems.AdvDiffusion(mesh, other, velocity)
    with pytest.raises(ValueError, match="field|solver|unknown"):
        second.DuDt = first.DuDt
        second.solve(timestep=0.001)


def test_replacing_velocity_expression_matches_updating_velocity_field(fields):
    mesh, temperature, velocity = fields
    reference = uw.discretisation.MeshVariable("Reference", mesh, 1, degree=1)
    vector = uw.discretisation.MeshVariable("Velocity", mesh, mesh.dim, degree=1)
    temperature.array[:, 0, 0] = temperature.coords[:, 0]
    reference.array[...] = temperature.array
    vector.array[...] = 0.0
    actual_manager = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity)
    reference_manager = uw.systems.ddt.EulerianSUPGPC(mesh, reference, vector.sym)
    actual = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=actual_manager)
    expected = uw.systems.AdvDiffusion(mesh, reference, vector.sym, DuDt=reference_manager)
    actual.solve(timestep=0.001)
    expected.solve(timestep=0.001)
    actual_manager.V_fn = sympy.Matrix([[0.2, 0.0]])
    vector.array[:, 0, 0] = 0.2
    actual.solve(timestep=0.001)
    expected.solve(timestep=0.001)
    np.testing.assert_allclose(temperature.array, reference.array, rtol=0, atol=1e-12)


def test_explicit_tau_accepts_directional_diffusion(fields):
    mesh, temperature, velocity = fields
    temperature.array[:, 0, 0] = 1.0
    transport = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity, tau=0)
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=transport)
    thermal.constitutive_model = uw.constitutive_models.AnisotropicDiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = sympy.Matrix([0.1, 0.2])
    thermal.solve(timestep=0.001)
    np.testing.assert_allclose(temperature.array, 1.0, rtol=0, atol=1e-12)
