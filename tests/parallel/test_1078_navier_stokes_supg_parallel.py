"""The Eulerian SUPG Navier-Stokes solver gives the serial answer on any number of ranks.

Kovasznay flow (exact steady Navier-Stokes at Re 40), a few steps from the
exact solution; the integral velocity error must match a serial reference to
solver tolerance, whatever the partition.

Run: mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1078_navier_stokes_supg_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a, pytest.mark.mpi]

# Serial reference: res 1/4 refined once (1/8), geometric multigrid on the velocity
# block, Crank-Nicolson, dt 0.05, 6 steps, peclet_weight 4 (recorded with this file).
# The GAMG fallback without a hierarchy gave a platform-dependent answer (7% on the
# Linux CI), so the mesh carries a refinement hierarchy and the solve is tight.
SERIAL_ERROR = 0.0013227881494769559


def _run(tolerance=1.0e-8):
    Re = 40.0
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-0.5, -0.5), maxCoords=(1.0, 0.5), cellSize=1.0 / 4, qdegree=3, regular=False,
        refinement=1)
    x, y = mesh.X
    lam = Re / 2 - sympy.sqrt(Re ** 2 / 4 + 4 * sympy.pi ** 2)
    U_ex = sympy.Matrix([[1 - sympy.exp(lam * x) * sympy.cos(2 * sympy.pi * y),
                          lam / (2 * sympy.pi) * sympy.exp(lam * x) * sympy.sin(2 * sympy.pi * y)]])
    v = uw.discretisation.MeshVariable("U1078", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1078", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0 / Re
    ns.tolerance = tolerance
    for b in ("Left", "Right", "Top", "Bottom"):
        ns.add_dirichlet_bc(U_ex, b)
    v.array[:, 0, :] = uw.function.evaluate(U_ex, v.coords).reshape(-1, 2)
    for _ in range(6):
        ns.solve(timestep=0.05)
    err2 = uw.maths.Integral(mesh, (v.sym - U_ex).dot(v.sym - U_ex)).evaluate()
    return float(np.sqrt(err2))


def test_error_is_partition_independent():
    err = _run()
    print(f"SERIAL_ERROR_MEASURED {err!r}")
    assert np.isfinite(err) and err < 0.05, err
    gathered = uw.mpi.comm.allgather(err)
    assert max(gathered) - min(gathered) < 1e-12, gathered
    if SERIAL_ERROR is not None:
        # the partition effect this guards against was 5e-4 (#687); platforms differ at 1e-7
        assert abs(err - SERIAL_ERROR) < 1e-6 * SERIAL_ERROR, (err, SERIAL_ERROR)
