"""The rotated path's pressure (Schur) sub-solve runs at rtol 1e-2.

A solve with rotated free slip or a split fault solves each Newton increment
with its own fieldsplit-Schur KSP: an outer FGMRES at rtol ``tolerance`` around a
full Schur factorisation, whose pressure sub-KSP ran at 0.1 x ``tolerance``. On
the faulted convection campaign (np 16, ``tolerance`` 1e-3) each of its 41-52
pressure iterations per increment is an inner velocity solve, and that sub-solve
was most of the Stokes time. At rtol 1e-2 the pressure iterations fell to 12-22,
the Stokes time by 28-57 %, and the answer moved by under 7e-6 (vrms, surface
velocity, mobility, slip) with the same nonlinear iteration counts (kaiju,
2026-10-28 and 2026-11-01). The outer KSP still demands ``tolerance``, so the
accuracy target does not move. At tighter tolerances (measured on this box at
``tolerance`` 1e-7, and on test_1106's split fault at 1e-6) the Newton counts are
unchanged and the outer KSP takes at most one more iteration.

The value is fixed, not scaled with ``tolerance``: this test reads the live
pressure sub-KSP's rtol after a solve at two tolerances. The fixture is
test_1018's power-law box under rotated free slip on every wall.
"""
import pytest

import underworld3 as uw

from test_1018_rotated_freeslip import _powerlaw_stokes

pytestmark = [pytest.mark.level_2]

PRESSURE_RTOL = 1.0e-2


@pytest.mark.tier_b
@pytest.mark.parametrize("tolerance", [1.0e-4, 1.0e-7])
def test_the_rotated_pressure_sub_solve_runs_at_1e_2(tolerance):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    s, _v, _p = _powerlaw_stokes(mesh, f"prtol{tolerance:g}", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.tolerance = tolerance
    s.solve()
    assert s._rotated_freeslip_info["converged"]

    ksp = s._rotated_linear_cache["ctx"]["ksp"]
    assert ksp.getTolerances()[0] == pytest.approx(tolerance, rel=1e-12)
    _velocity_ksp, pressure_ksp = ksp.getPC().getFieldSplitSubKSP()
    assert pressure_ksp.getTolerances()[0] == pytest.approx(PRESSURE_RTOL, rel=1e-12), (
        f"the rotated pressure sub-KSP runs at rtol {pressure_ksp.getTolerances()[0]:g} "
        f"under tolerance {tolerance:g}")


@pytest.mark.tier_b
def test_the_newton_count_is_unchanged_at_a_tight_tolerance():
    """At tolerance 1e-7 the box converges in the 8 Newton steps it took at the
    old 0.1 x tolerance, with at most 2 outer iterations per step (macOS)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    s, _v, _p = _powerlaw_stokes(mesh, "prtol_newton", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.tolerance = 1.0e-7
    s.solve()
    info = s._rotated_freeslip_info
    assert info["converged"]
    assert info["nonlinear_iterations"] == 8, info["nonlinear_iterations"]
    assert max(info["ksp_its"]) <= 2, info["ksp_its"]
