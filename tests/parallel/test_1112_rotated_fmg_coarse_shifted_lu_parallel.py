"""The rotated custom-P FMG's shifted-LU coarse solve in parallel (#901).

The same contract as test_1111: the rotated annulus records the redundant
coarse solve, and its answer is the SVD path's. The SVD factored the gathered
coarse matrix on every rank, so its cost rose with the rank count. The
redundant LU, with a zero-pivot shift, gathers it too but factors sparsely.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1112_rotated_fmg_coarse_shifted_lu_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]

# The SVD path's answer on this fixture (development fc67b650, np 2); the
# partition moves it at the solver tolerance only.
VRMS = 4.7900648436
RTOL = 1.0e-8


def test_the_coarse_solve_is_shifted_lu_in_parallel():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=2, qdegree=3)
    v = uw.discretisation.MeshVariable("V1112", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1112", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    stokes.bodyforce = (1.0e3 * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([X[0] / r, X[1] / r]))
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    stokes.solve(zero_init_guess=True)

    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    assert stokes._rotated_velocity_null_modes == 1
    assert info["coarse_solve"] == "redundant", info.get("coarse_solve")
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    vrms = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate() / vol))
    assert abs(vrms - VRMS) <= RTOL * VRMS, vrms
