"""Validate Stokes_Constrained (free-slip via in-saddle multipliers) against the
exact SolCx analytic solution — the canonical free-slip, 1e6-viscosity-jump
benchmark. The constrained solver enforces free-slip on all four walls via four
multipliers; its velocity must match the analytic, and the constraint must hold.

Run: pixi run python -m pytest tests/test_1062_constrained_solcx.py -v
"""

import pytest

pytestmark = [pytest.mark.level_2]

import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import analytic as A

ETA_A, ETA_B, XC, NZ, RES = 1.0, 1.0e6, 0.5, 1, 32


def test_constrained_solcx_matches_analytic():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )
    sol = A.SolCx(mesh, eta_A=ETA_A, eta_B=ETA_B, x_c=XC, n=NZ)

    s = uw.systems.Stokes_Constrained(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.saddle_preconditioner = 1.0 / sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    # free-slip on all four walls via in-saddle multipliers
    s.add_constraint_bc("Left",   g=0.0, normal=sympy.Matrix([[-1.0, 0.0]]))
    s.add_constraint_bc("Right",  g=0.0, normal=sympy.Matrix([[ 1.0, 0.0]]))
    s.add_constraint_bc("Bottom", g=0.0, normal=sympy.Matrix([[ 0.0, -1.0]]))
    s.add_constraint_bc("Top",    g=0.0, normal=sympy.Matrix([[ 0.0,  1.0]]))
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-9
    s.solve()

    # velocity matches the exact analytic
    rel = sol.velocity_error(s.u)
    assert rel < 5.0e-3, f"velocity error vs analytic too large: {rel:.2e}"

    # constraint enforced (u.n -> 0 on the walls)
    c = s.u.coords
    e = 1e-6
    xw = (c[:, 0] < e) | (c[:, 0] > 1 - e)
    yw = (c[:, 1] < e) | (c[:, 1] > 1 - e)
    rms_un = np.sqrt(np.mean(np.r_[s.u.data[xw, 0], s.u.data[yw, 1]] ** 2))
    assert rms_un < 1.0e-6, f"constraint not enforced: RMS(u.n)={rms_un:.2e}"
