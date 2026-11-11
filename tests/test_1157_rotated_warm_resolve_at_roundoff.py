"""A warm rotated re-solve whose pressure equations are already satisfied (#910).

Started from ``v = 1.05 v_solved`` with the solved pressure, a warm rotated free-slip
solve has a residual in the velocity rows only. The Schur complement's right-hand
side is then round-off (4e-15 against 7e-2 cold), and the pressure sub-solve was
asked to cut that by its relative tolerance. It ran to its 200-iteration cap, each
iteration a velocity solve; or the noise grew past the divergence tolerance and the
next velocity solve saw a NaN, failing the solve at iteration 0 (PC_FAILED, -11).
Which of the two happened depended on the rounding, so the failure looked random:
16 of 26 serial runs had at least one.

- every warm re-solve converges;
- the pressure sub-solve stops at round-off, nowhere near its cap.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

WARM_SOLVES = 4


def _annulus():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=2, qdegree=3)
    v = uw.discretisation.MeshVariable("V1157", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1157", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    stokes.bodyforce = (1.0e3 * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([X[0] / r, X[1] / r]))
    stokes.add_rotated_freeslip_bc(0, "Lower")
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    return v, stokes


def test_warm_resolves_with_satisfied_pressure_converge_without_thrashing():
    v, stokes = _annulus()
    stokes.tolerate_linear_failure = True      # count failures rather than stop at one
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["converged"]
    v_solved = np.asarray(v.array).copy()
    records = []
    for _ in range(WARM_SOLVES):
        v.array[...] = 1.05 * v_solved
        stokes.solve(zero_init_guess=False)
        info = stokes._rotated_freeslip_info
        records.append((bool(info["converged"]), int(info["ksp_reason"]),
                        max(its for its in info["pres_its_last"] if its is not None)))
    # development: a converged warm solve ran the pressure sub-solve to 200, and one
    # solve in four or so failed with -11
    assert all(converged for converged, _reason, _its in records), records
    assert max(its for _c, _r, its in records) < 50, records
