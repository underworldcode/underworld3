"""The multiplier is not the whole traction — the augmented-Lagrangian share is.

`Stokes_Constrained` assembles the momentum row's boundary term as
`(h + r(n.u - g)) n`, so the traction holding the boundary is that sum. `h` alone
is short by `r` times the discrete constraint residual, and the default `r` is
viscosity-weighted (`1e4 * mu`), so a lateral viscosity contrast makes the
omitted share most of the answer.

SolCx is the case that shows it: `mu` steps from 1 to 1e6 at x = 0.5, so
`r = 1e10` on the stiff half. `uw.analytic.SolCx.topography_top` is the exact
surface topography, which is what makes this an oracle rather than a comparison.

Guards `Stokes_Constrained.traction()` and `topography()`, and carries the
NEGATIVE CONTROL in the same test: the bare multiplier fails the same threshold
by a wide margin, so a regression that quietly reverts to `h` cannot pass.

Run: pixi run python -m pytest tests/test_1063_constrained_traction.py -v
"""

import pytest

pytestmark = [pytest.mark.level_2]

import numpy as np
import sympy
import underworld3 as uw
from underworld3 import analytic as A

ETA_B, RES = 1.0e6, 32


def _solve():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=ETA_B, x_c=0.5, n=1)

    s = uw.systems.Stokes_Constrained(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    # Three walls by the ordinary component condition, so the multiplier under
    # test is the only constraint on the wall being measured.
    s.add_dirichlet_bc((0.0, None), "Left")
    s.add_dirichlet_bc((0.0, None), "Right")
    s.add_dirichlet_bc((None, 0.0), "Bottom")
    s.add_constraint_bc(0.0, "Top")
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-9
    s.solve()
    assert s.snes.getConvergedReason() > 0
    return mesh, s, sol


def _relative_error(values, coords, sol):
    """Relative l2 against the exact topography, both mean-removed.

    The box is enclosed, so the level is a gauge and only the deviation is
    determined — which is what topography is anyway.
    """
    exact = np.asarray(sol.topography_top(coords)).reshape(-1)
    got = np.asarray(values).reshape(-1) - np.mean(values)
    exact = exact - exact.mean()
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))


def test_constrained_topography_carries_the_augmentation_share():
    mesh, s, sol = _solve()

    top = np.abs(s.u.coords[:, 1] - 1.0) < 1.0e-9
    coords = s.u.coords[top]

    traction = uw.function.evaluate(s.traction("Top"), coords)
    bare = uw.function.evaluate(s.multiplier("Top").sym[0], coords)

    whole = _relative_error(traction, coords, sol)
    without = _relative_error(bare, coords, sol)

    # The traction is the surface topography, to the accuracy of the discretisation.
    assert whole < 0.15, f"traction wrong by {whole:.3f}"

    # NEGATIVE CONTROL. Without the augmentation share the same read is useless
    # here: measured 1.04 (it is anti-correlated with the right answer). If this
    # ever passes, `traction()` has quietly become `multiplier()` again.
    assert without > 0.5, (
        f"the bare multiplier read {without:.3f}: the negative control no longer "
        "fires, so this test is not guarding anything")

    # topography() is the traction divided by the buoyancy scale, so it inherits
    # the fix; check the public path a user actually calls.
    height = uw.function.evaluate(s.topography("Top", reference="mean"), coords)
    assert _relative_error(height, coords, sol) < 0.15
