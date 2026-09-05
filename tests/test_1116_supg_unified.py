"""Shared SUPG integration, restart, and pre-migration equivalence."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _problem(dim, tag, cellsize=0.25):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=cellsize, qdegree=4, regular=False,
    )
    temperature = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=1)
    velocity = uw.discretisation.MeshVariable(f"U_{tag}", mesh, dim, degree=1)
    temperature.array[:, 0, 0] = temperature.coords[:, 0]
    velocity.array[:, 0, :] = 0.2
    return mesh, temperature, velocity


@pytest.mark.parametrize("dim", [2, 3])
def test_citcoms_matches_pre_migration_implementation(dim):
    """Optional release gate against the frozen source from commit 87b3711d.

    Both assemblers see the same mesh, fields, partition and time sequence.
    The frozen source is an external test artifact, not another installed solver.
    """
    baseline = os.environ.get("UW_SUPG_BASELINE_FILE")
    if baseline is None:
        pytest.skip("Set UW_SUPG_BASELINE_FILE to the frozen pre-migration module.")
    spec = importlib.util.spec_from_file_location("_supg_baseline", baseline)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mesh, temperature, velocity = _problem(dim, f"migration_{dim}")
    reference = uw.discretisation.MeshVariable("T_reference", mesh, 1, degree=1)
    rate = uw.discretisation.MeshVariable("Tdot", mesh, 1, degree=1)
    reference_rate = uw.discretisation.MeshVariable("Tdot_reference", mesh, 1, degree=1)
    shape = sympy.prod(sympy.sin(sympy.pi * x) for x in mesh.X)
    temperature.array[:, 0, 0] = uw.function.evaluate(shape, temperature.coords).reshape(-1)
    reference.array[...] = temperature.array
    current = uw.systems.AdvDiffusionSUPG(
        mesh, temperature, velocity.sym, time_integrator="citcoms",
        temperature_rate_field=rate,
    )
    previous = module.SNES_AdvectionDiffusionSUPG(
        mesh, reference, velocity.sym, time_integrator="citcoms",
        temperature_rate_field=reference_rate,
    )
    for solver in (current, previous):
        solver.constitutive_model = uw.constitutive_models.DiffusionModel
        solver.constitutive_model.Parameters.diffusivity = 0.01
        solver.f = 0.1 * shape
        for boundary in mesh.boundaries:
            if boundary.name not in ("All_Boundaries", "Null_Boundary"):
                solver.add_dirichlet_bc(0.0, boundary.name)

    for step in range(6):
        velocity.array[:, 0, :] = 0.2 * (1.0 + step / 10.0)
        dt = min(0.002, float(current.estimate_dt()))
        np.testing.assert_allclose(
            current.estimate_dt(), previous.estimate_dt(), rtol=1e-12, atol=1e-14)
        current.solve(timestep=dt)
        previous.solve(timestep=dt)
        np.testing.assert_allclose(temperature.array, reference.array, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(rate.array, reference_rate.array, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("settings", [
    {"time_integrator": "citcoms"}, {"order": 1}, {"order": 2},
])
@pytest.mark.parametrize("disk", [False, True])
def test_snapshot_restores_fields_and_timestep_estimator(settings, disk, tmp_path):
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    mesh, temperature, velocity = _problem(2, "snapshot")
    thermal = uw.systems.AdvDiffusionSUPG(mesh, temperature, velocity.sym, **settings)
    thermal.constitutive_model.Parameters.diffusivity = 0.05
    thermal.add_dirichlet_bc(0.0, "Left")
    thermal.add_dirichlet_bc(1.0, "Right")
    for _ in range(3):
        thermal.solve(timestep=0.002)
    if disk:
        path = uw.mpi.comm.bcast(str(tmp_path / "thermal.h5"), root=0)
        snapshot = orchestration_model.save_state(file=path)
    else:
        snapshot = orchestration_model.save_state()
    saved_temperature = np.array(temperature.array)
    estimate = thermal.estimate_dt()
    thermal.solve(timestep=0.003)
    expected = np.array(temperature.array)
    expected_state = thermal.state
    expected_rate = None if thermal.temperature_rate is None else np.array(thermal.temperature_rate.array)
    orchestration_model.load_state(snapshot)
    np.testing.assert_array_equal(temperature.array, saved_temperature)
    assert thermal.estimate_dt() == pytest.approx(estimate, rel=1e-14)
    thermal.solve(timestep=0.01)
    orchestration_model.load_state(snapshot)
    thermal.solve(timestep=0.003)
    # Rebuilding an implicit Krylov solve can change final rounding, but the
    # restored fields above must be exact and replay must agree near machine precision.
    np.testing.assert_allclose(temperature.array, expected, rtol=2e-14, atol=2e-14)
    assert thermal.state.last_timestep == expected_state.last_timestep
    if expected_state.last_change_rate is not None:
        assert thermal.state.last_change_rate == pytest.approx(
            expected_state.last_change_rate, rel=5e-12, abs=1e-12)
    if expected_rate is not None:
        np.testing.assert_array_equal(thermal.temperature_rate.array, expected_rate)
    uw.reset_default_model()


def test_citcoms_does_not_allocate_unused_multistep_history():
    mesh, temperature, velocity = _problem(2, "history")
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, temperature, velocity.sym, time_integrator="citcoms")
    assert thermal.DuDt is None
    assert thermal.temperature_rate is not None
    with pytest.raises(ValueError, match="stability"):
        thermal.estimate_dt(basis="accuracy")
    with pytest.raises(ValueError, match="gamma"):
        thermal.theta = 0.5


def test_empty_partition_is_rejected_on_every_rank():
    mesh, temperature, velocity = _problem(2, "empty_partition", cellsize=0.5)
    counts = uw.mpi.comm.allgather(
        mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0])
    if min(counts) > 0:
        pytest.skip(f"This partition has no empty ranks: {counts}")
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, temperature, velocity.sym, time_integrator="citcoms")
    with pytest.raises(NotImplementedError, match="on every rank"):
        thermal.estimate_dt()
