"""Spherical SLCN lifecycle regression for serial and MPI execution."""

import gc

import numpy as np
import pytest

import underworld3 as uw


pytestmark = pytest.mark.level_2


def test_spherical_slcn_transient_state_is_bounded():
    mesh = uw.meshing.SphericalShell(
        radiusInner=0.55,
        radiusOuter=1.0,
        cellSize=0.4,
        qdegree=2,
    )
    velocity = uw.discretisation.MeshVariable(
        "U_slcn_lifecycle", mesh, mesh.dim, degree=1
    )
    temperature = uw.discretisation.MeshVariable(
        "T_slcn_lifecycle", mesh, 1, degree=1
    )

    with mesh.access(velocity, temperature):
        coords = np.asarray(temperature.coords)
        radii = np.linalg.norm(coords, axis=1)
        temperature.data[:, 0] = (1.0 - radii) / 0.45
        velocity.data[:, 0] = -0.05 * coords[:, 1]
        velocity.data[:, 1] = 0.05 * coords[:, 0]
        velocity.data[:, 2] = 0.0

    thermal = uw.systems.AdvDiffusionSLCN(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        order=1,
        theta=0.5,
    )
    thermal.constitutive_model = uw.constitutive_models.DiffusionModel
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    thermal.add_dirichlet_bc(0.0, "Upper")
    thermal.add_dirichlet_bc(1.0, "Lower")

    live_swarms_before = len(mesh._registered_swarms)
    history_lengths = []

    for step in range(38):
        thermal.solve(timestep=1.0e-3, zero_init_guess=False)
        assert len(mesh._registered_swarms) == live_swarms_before
        assert len(mesh._dminterpolation_cache._cache) <= (
            mesh._dminterpolation_cache.max_entries
        )
        if step in (31, 37):
            gc.collect()
            history_lengths.append(
                (
                    len(thermal.solve_history),
                    len(mesh._eval_work_1x3_projector.solve_history),
                )
            )

    assert history_lengths == [(32, 32), (32, 32)]
    assert np.all(np.isfinite(temperature.data))
