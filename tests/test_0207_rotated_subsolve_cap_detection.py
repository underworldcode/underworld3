"""A rotated free-slip solve says when a sub-solve stopped at its cap (#625).

The rotated loop solves its own fieldsplit-Schur KSP, not the solver's SNES, so
the native path's count and warning (test_0205, #585) never saw it. It recorded
the iteration count of the LAST velocity and pressure application only: a
sub-solve that ran out of iterations on any earlier application was invisible,
and the pressure sub-solve's reason was never read at all.

The loop now counts, over every application in the solve, the sub-solves that
stopped at their cap (``_rotated_freeslip_info["sub_capped"]``), and the solve
warns once per capped block, as the native path does.

The fixture asks the pressure sub-solve for a tolerance it cannot reach
(``_rotated_pres_rtol``, the loop's knob for that tolerance), so it stops at its
cap of 200 on every application; at the default tolerance it converges.
"""

import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _annulus_solve(name, pres_rtol=None):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    stokes.bodyforce = (sympy.cos(3 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r]).T)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    if pres_rtol is not None:
        stokes._rotated_pres_rtol = pres_rtol
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    return stokes._rotated_freeslip_info, [str(w.message) for w in record]


def test_a_capped_pressure_sub_solve_is_counted_and_warned():
    info, messages = _annulus_solve("cap", pres_rtol=1.0e-30)
    capped = info["sub_capped"]
    # the pressure sub-solve is applied once per outer iteration, and every
    # application stops at its cap (measured: one outer iteration, 200 pressure
    # iterations). This fixture has one application per increment, so it does not
    # tell the count over every application from a last-application readout; the
    # count rests on the post-solve hook, which PETSc runs after every KSPSolve.
    outer = int(np.sum([k for k in info["ksp_its"] if k is not None]))
    assert outer >= 1 and capped["pressure"] == outer, (capped, info["ksp_its"])
    assert capped["velocity"] == 0, capped
    with uw.selective_ranks(0) as selected:
        if selected:
            assert any("pressure sub-solve stopped at its iteration cap" in m
                       for m in messages), messages


def test_a_converged_solve_counts_no_cap_and_says_nothing():
    info, messages = _annulus_solve("ok")
    assert info["converged"], info
    assert info["sub_capped"] == {"velocity": 0, "pressure": 0}, info["sub_capped"]
    assert not any("stopped at its iteration cap" in m for m in messages), messages
