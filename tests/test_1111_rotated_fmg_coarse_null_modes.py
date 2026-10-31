"""The coarse solve of the rotated custom-P FMG when the problem has null modes (#901).

A closed free-slip wall admits rigid rotations: one on an annulus, three on a
spherical shell with the radial normal. The Galerkin-coarsened rotated velocity
block inherits them, and a plain LU meets a zero pivot there (#306). The coarse
level used to be solved by a dense SVD of the gathered matrix, whose cost grows
with the bottom level and does not fall with the rank count. On a 614-cell
shell bottom level, the factorisation took 64 s at np 1 and 102 s at np 4,
against 7 s for everything else at np 4.

The modes are now handled explicitly. They are restricted to the bottom level,
one unknown per mode is pinned, and the pinned system is solved by PETSc's LU.
Contract: the solve records the null-mode coarse solve, the answer is the SVD
path's, and the iterations are the measured ones.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# The SVD path's answer on each fixture (development fc67b650, np 1), and the
# velocity / Schur iterations of the last outer iteration with the null-mode
# coarse solve.
CASES = {
    "annulus": {"vrms": 4.7900648439, "rtol": 1.0e-9, "modes": 1, "its": (4, 7)},
    "shell": {"vrms": 1.7726841166, "rtol": 1.0e-8, "modes": 3, "its": (18, 18)},
}


def _solve(kind):
    """Rotated free slip on both walls, isoviscous P2/P1, custom-P FMG over the
    mesh's own refinement hierarchy."""
    if kind == "annulus":
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                                  refinement=2, qdegree=3)
    else:
        mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                         refinement=1, qdegree=3)
    dim = mesh.dim
    v = uw.discretisation.MeshVariable("V1111", mesh, dim, degree=2)
    p = uw.discretisation.MeshVariable("P1111", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    r = sympy.sqrt(sum(xi ** 2 for xi in X))
    th = sympy.atan2(X[1], X[0])
    radial = sympy.sin(sympy.pi * (r - 0.5) / 0.5)
    shape = sympy.cos(4 * th) * (1 if dim == 2 else X[2] / r)
    stokes.bodyforce = 1.0e3 * shape * radial * sympy.Matrix([xi / r for xi in X])
    # the shell's three rotations are null modes only under the analytic normal (#906)
    normal = None if dim == 2 else mesh.CoordinateSystem.unit_e_0
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    stokes.solve(zero_init_guess=True)
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    vrms = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate() / vol))
    return stokes, vrms


@pytest.mark.parametrize("kind", ["annulus", "shell"])
def test_the_coarse_solve_handles_the_null_modes(kind):
    case = CASES[kind]
    stokes, vrms = _solve(kind)
    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    assert stokes._rotated_velocity_null_modes == case["modes"]
    assert info["coarse_solve"] == "null_modes", info.get("coarse_solve")
    assert abs(vrms - case["vrms"]) <= case["rtol"] * case["vrms"], vrms
    assert (info["vel_its_last"][-1], info["pres_its_last"][-1]) == case["its"], info
