"""The Eulerian SUPG solver gives the serial answer on any number of ranks.

The scheme has no rank-local step: history is a mesh variable, the residual
is assembled by PETSc, the timestep is a runtime constant. So the integral
error against the rotating-Gaussian oracle after a few steps must match a
serial reference to solver tolerance, whatever the partition.

Run: mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1077_advdiff_supg_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from serial_reference import emit, mesh_fingerprint, serial_reference

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a, pytest.mark.mpi]

def _run():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8,
        qdegree=3, regular=False)
    x, y = mesh.X
    sol = uw.analytic.RotatingGaussian(mesh, sigma=0.12, centre_radius=0.5, omega=1.0)
    T = uw.discretisation.MeshVariable("T1077", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sol.at(0.0), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusionSUPG(mesh, T, sympy.Matrix([[-y, x]]), order=2)
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    dt = 0.05
    adv.DuDt.set_initial_history(
        [uw.function.evaluate(sol.at(-k * dt), T.coords).reshape(-1, 1, 1) for k in range(2)],
        dt=dt)
    for _ in range(8):
        adv.solve(timestep=dt)
    return sol.error(sol.at(8 * dt), T, norm="integral"), mesh_fingerprint(mesh)


def test_error_is_partition_independent():
    err, fingerprint = _run()
    assert np.isfinite(err) and err < 0.05, err
    gathered = uw.mpi.comm.allgather(err)
    assert max(gathered) - min(gathered) < 1e-12, gathered
    reference = serial_reference(__file__, "gaussian")
    assert int(fingerprint[0]) == int(reference["fingerprint"][0])
    np.testing.assert_allclose(fingerprint[1], reference["fingerprint"][1], rtol=1e-12)
    assert abs(err - reference["values"][0]) < 1e-8, (err, reference)


if __name__ == "__main__":
    error, fingerprint = _run()
    emit([error], fingerprint)
