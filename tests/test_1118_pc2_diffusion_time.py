"""Isolate finite-correction PC2 time error without Stokes or a fine mesh.

Analytical P1 element integrals independently supply M, K and D=diag(M*1).
The exact semidiscrete solution is a generalized eigenmode exp(-lambda*t).
Small dense SciPy matrices are an independent test oracle, not solver code.

With consistent M in the residual and two D-preconditioned corrections,
the dt->0 operator is (2I-D^-1 M)D^-1 K, not M^-1 K or D^-1 K. This test
documents that limitation; it does NOT certify second-order PDE accuracy.
"""

import hashlib
import math

import numpy as np
import pytest
from scipy.linalg import eigh, expm

import underworld3 as uw
from underworld3.meshing.smoothing import _owned_cell_mask, _tet_cells, _tri_cells

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _p1_matrices(mesh):
    """Integrate affine basis functions independently of the solver assembly."""
    cells = (_tri_cells if mesh.dim == 2 else _tet_cells)(mesh.dm)
    local_cells = np.asarray(mesh.X.coords)[cells[_owned_cell_mask(mesh.dm)]]
    vertices = np.concatenate(uw.mpi.comm.allgather(local_cells))
    coords = np.unique(vertices.reshape(-1, mesh.dim).round(12), axis=0)
    indices = {tuple(point): index for index, point in enumerate(coords)}
    connectivity = np.array([
        [indices[tuple(point.round(12))] for point in cell] for cell in vertices
    ])
    canonical = np.sort(connectivity, axis=1)
    canonical = canonical[np.lexsort(canonical.T[::-1])]
    fingerprint = hashlib.sha256(coords.tobytes() + canonical.tobytes()).hexdigest()
    mass = np.zeros((len(coords), len(coords)))
    stiffness = np.zeros_like(mass)
    for ids, cell in zip(connectivity, vertices):
        affine = np.column_stack([np.ones(mesh.dim + 1), cell])
        gradients = np.linalg.inv(affine)[1:, :].T
        volume = abs(np.linalg.det(affine)) / math.factorial(mesh.dim)
        mass[np.ix_(ids, ids)] += volume * (
            np.ones((mesh.dim + 1, mesh.dim + 1)) + np.eye(mesh.dim + 1)
        ) / ((mesh.dim + 1) * (mesh.dim + 2))
        stiffness[np.ix_(ids, ids)] += 0.1 * volume * gradients @ gradients.T
    np.testing.assert_allclose(mass.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(stiffness.sum(axis=1), 0.0, atol=1e-12)
    return coords, mass, stiffness, len(vertices), fingerprint


def _norm(values, mass):
    return float(np.sqrt(values @ mass @ values))


def _orders(values):
    return np.log2(np.asarray(values[:-1]) / values[1:])


@pytest.fixture(params=[2, 3])
def diffusion(request):
    dim = request.param
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=0.25, qdegree=4, regular=False,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    velocity = uw.discretisation.MeshVariable("U", mesh, dim, degree=1)
    velocity.array[...] = 0.0
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, temperature, velocity.sym, time_integrator="citcoms",
    )
    thermal.constitutive_model.Parameters.diffusivity = 0.1
    # Natural homogeneous Neumann boundaries remove constrained-DOF effects.
    coords, mass, stiffness, cell_count, fingerprint = _p1_matrices(mesh)
    indices = {tuple(point): index for index, point in enumerate(coords)}
    local_ids = np.array([
        indices[tuple(point.round(12))] for point in np.asarray(temperature.coords)
    ])
    eigenvalues, eigenvectors = eigh(stiffness, mass)
    np.testing.assert_allclose(eigenvalues[0], 0.0, atol=1e-12)
    initial = eigenvectors[:, 1].copy()
    initial *= np.sign(initial[np.argmax(np.abs(initial))])
    initial /= np.max(np.abs(initial))
    uw.pprint(
        f"PC2_ORACLE dim={dim} cells={cell_count} vertices={len(coords)} "
        f"mesh_sha256={fingerprint} lambda={eigenvalues[1]:.12g}")
    return thermal, temperature, local_ids, mass, stiffness, initial, eigenvalues, eigenvectors


def test_two_corrections_match_independent_diffusion_map(diffusion):
    thermal, temperature, ids, mass, stiffness, initial, eigenvalues, _ = diffusion
    lumped = mass.sum(axis=1)
    H = mass / lumped[:, None]
    J = stiffness / lumped[:, None]
    identity = np.eye(len(initial))
    final_time = 0.1
    exact = initial * np.exp(-eigenvalues[1] * final_time)
    effective = expm(-final_time * (2 * identity - H) @ J) @ initial
    initial_state = thermal.state
    dt_limit = thermal.estimate_dt()
    solutions, effective_errors = [], []
    for steps in (16, 32, 64, 128):
        dt = final_time / steps
        assert dt < dt_limit
        temperature.array[:, 0, 0] = initial[ids]
        thermal.temperature_rate.array[...] = 0.0
        thermal.state = initial_state
        expected = initial.copy()
        rate = -J @ initial
        # Algebraically eliminate both corrections; do not call UW3 residuals.
        B = (2 * identity - H - 0.5 * dt * J) @ J
        discrepancy = np.zeros(2)
        for _ in range(steps):
            predictor = expected + 0.5 * dt * rate
            rate = -B @ predictor
            expected = predictor + 0.5 * dt * rate
            thermal.solve(timestep=dt)
            discrepancy = np.maximum(discrepancy, [
                np.max(np.abs(temperature.array[:, 0, 0] - expected[ids])),
                np.max(np.abs(thermal.temperature_rate.array[:, 0, 0] - rate[ids])),
            ])
        discrepancy = np.max(uw.mpi.comm.allgather(discrepancy), axis=0)
        assert discrepancy[0] < 1e-11 and discrepancy[1] < 1e-10, discrepancy
        actual = np.zeros(len(initial))
        for local_ids, values in uw.mpi.comm.allgather(
                (ids, np.array(temperature.array[:, 0, 0]))):
            actual[local_ids] = values
        solutions.append(actual)
        effective_error = _norm(actual - effective, mass) / _norm(effective, mass)
        effective_errors.append(effective_error)
        uw.pprint(
            f"PC2_DIFFUSION dim={thermal.mesh.dim} steps={steps} dt={dt:.12g} "
            f"consistent_error={_norm(actual-exact, mass)/_norm(exact, mass):.12g} "
            f"effective_error={effective_error:.12g} "
            f"T_map_error={discrepancy[0]:.12g} Tdot_map_error={discrepancy[1]:.12g}")
    changes = [_norm(a - b, mass) for a, b in zip(solutions, solutions[1:])]
    rates = _orders(changes)
    # Detect the known finite-correction limit, not a general accuracy guarantee.
    assert np.all((0.9 < rates) & (rates < 1.15)), rates
    assert np.all((0.9 < _orders(effective_errors)) & (_orders(effective_errors) < 1.15))
    uw.pprint(f"PC2_TIME_ORDER dim={thermal.mesh.dim} rates={rates.tolist()}")


def test_exact_matrix_controls_separate_mass_and_startup(diffusion):
    thermal, _, _, mass, stiffness, initial, eigenvalues, eigenvectors = diffusion
    D = mass.sum(axis=1)
    J = stiffness / D[:, None]
    final_time = 0.1
    consistent_exact = initial * np.exp(-eigenvalues[1] * final_time)
    lumped_exact = expm(-final_time * J) @ initial
    errors = {"consistent_cn": [], "cn_lumped_startup": [], "lumped_pc2": []}
    for steps in (16, 32, 64, 128):
        dt = final_time / steps
        factors = (1 - 0.5 * dt * eigenvalues) / (1 + 0.5 * dt * eigenvalues)
        consistent = initial * factors[1]**steps
        errors["consistent_cn"].append(_norm(consistent - consistent_exact, mass))
        # Exactly converged corrections cannot repair an inconsistent first rate.
        predictor = initial - 0.5 * dt * J @ initial
        amplitudes = (eigenvectors.T @ mass @ predictor) / (1 + 0.5 * dt * eigenvalues)
        bad_start = eigenvectors @ (factors**(steps - 1) * amplitudes)
        errors["cn_lumped_startup"].append(_norm(bad_start - consistent_exact, mass))
        # A genuinely lumped residual has H=I; this is a diagnostic alternative,
        # not a replacement of the CitcomS mode installed in UW3.
        values, rate = initial.copy(), -J @ initial
        B = (np.eye(len(initial)) - 0.5 * dt * J) @ J
        for _ in range(steps):
            predictor = values + 0.5 * dt * rate
            rate = -B @ predictor
            values = predictor + 0.5 * dt * rate
        errors["lumped_pc2"].append(_norm(values - lumped_exact, mass))
    for name, values in errors.items():
        rates = _orders(values)
        uw.pprint(f"PC2_CONTROL dim={thermal.mesh.dim} name={name} errors={values} rates={rates.tolist()}")
        if name == "cn_lumped_startup":
            assert np.all((0.9 < rates) & (rates < 1.15)), rates
        else:
            assert np.all((1.9 < rates) & (rates < 2.2)), rates
