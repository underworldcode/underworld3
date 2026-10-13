"""SolCx analytic-solution validation.

SolCx is the canonical free-slip, discontinuous-viscosity Stokes benchmark
(viscosity step eta_A|eta_B at x_c, density forcing (0, cos(pi x) sin(n pi z))).
Here we (a) check the ported Velic kernel binding returns non-trivial values and
(b) confirm a default-solver UW3 Stokes solve CONVERGES to the exact analytic.

This guards a subtle trap: on a free-slip (pressure-nullspace) problem a direct
`ksponly + lu` solve mangles the singular saddle and returns a wrong-but-quiet
answer. Comparing against the analytic catches that immediately.

Run: pixi run python -m pytest tests/test_1015_analytic_solcx.py -v
"""

import pytest

pytestmark = [pytest.mark.level_2]

import numpy as np
import underworld3 as uw
from underworld3 import analytic as A

# The low-level AnalyticSolCx_* bindings are the VENDORED KERNEL, not part of
# the `uw.analytic` public surface — `uw.analytic` exposes solutions, not the
# reference functions they were transcribed from. Import them from where they
# actually live rather than through the deprecated `uw.function.analytic`
# shim, which only forwards here anyway.
from underworld3.analytic._reference import _velic as K


def _solve_solcx(res, eta_B=1.0e6, x_c=0.5, n=1):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=eta_B, x_c=x_c, n=n)

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    stokes.saddle_preconditioner = 1.0 / sol.fn_viscosity
    stokes.bodyforce = sol.fn_bodyforce
    # Free-slip on all four walls (normal component pinned, tangential free).
    stokes.add_dirichlet_bc((0.0, None), "Left")
    stokes.add_dirichlet_bc((0.0, None), "Right")
    stokes.add_dirichlet_bc((None, 0.0), "Bottom")
    stokes.add_dirichlet_bc((None, 0.0), "Top")
    stokes.petsc_use_pressure_nullspace = True  # enclosed -> constant-pressure mode
    stokes.tolerance = 1.0e-9
    stokes.solve()
    return sol.velocity_error(stokes.u)


def test_solcx_analytic_binding():
    """The ported SolCx kernel returns non-trivial values."""
    vy = float(K.AnalyticSolCx_velocity_y(1.0, 1.0e6, 0.5, 1, 0.25, 0.5).evalf())
    assert np.isfinite(vy)
    assert abs(vy) > 1.0e-8


def test_solcx_stokes_converges_to_analytic():
    """A default-solver Stokes solve converges to the exact SolCx solution."""
    e16 = _solve_solcx(16)
    e32 = _solve_solcx(32)
    assert e32 < e16, f"no convergence: e16={e16:.2e} e32={e32:.2e}"
    assert e32 < 1.0e-3, f"poor match to analytic: e32={e32:.2e}"


def test_solcx_stress_binding():
    """The exact total-stress kernel binding returns finite, non-trivial values."""
    pts = [(0.25, 1.0), (0.7, 0.2), (0.9, 0.6), (0.15, 0.4)]
    sxx = [float(K.AnalyticSolCx_stress_xx(1.0, 1.0e6, 0.5, 1, x, y).evalf()) for x, y in pts]
    syy = [float(K.AnalyticSolCx_stress_yy(1.0, 1.0e6, 0.5, 1, x, y).evalf()) for x, y in pts]
    sxy = [float(K.AnalyticSolCx_stress_xy(1.0, 1.0e6, 0.5, 1, x, y).evalf()) for x, y in pts]
    for s in sxx + syy + sxy:
        assert np.isfinite(s)
    # the stress field is non-trivial (the dynamic-topography signal on the top is -syy)
    assert max(abs(s) for s in syy) > 1.0e-3
    assert max(abs(s) for s in sxx) > 1.0e-3


def test_solcx_stress_pressure_consistency():
    """Convention-independent check: in 2D sigma = -p I + tau with tau traceless,
    so the mean normal stress (sigma_xx + sigma_yy)/2 must equal -p at every
    point. This validates the new stress wrapper against the kernel's own
    pressure without assuming any UW3 sign convention."""
    pts = [(0.2, 0.3), (0.7, 0.8), (0.5, 0.5), (0.9, 0.1), (0.35, 0.65)]
    for xi, yi in pts:
        sxx = float(K.AnalyticSolCx_stress_xx(1.0, 1.0e6, 0.5, 1, xi, yi).evalf())
        syy = float(K.AnalyticSolCx_stress_yy(1.0, 1.0e6, 0.5, 1, xi, yi).evalf())
        p = float(K.AnalyticSolCx_pressure(1.0, 1.0e6, 0.5, 1, xi, yi).evalf())
        mean_normal = 0.5 * (sxx + syy)
        assert abs(mean_normal + p) < 1.0e-10, (
            f"(sxx+syy)/2 != -p at ({xi},{yi}): {mean_normal:.3e} vs {-p:.3e}"
        )
