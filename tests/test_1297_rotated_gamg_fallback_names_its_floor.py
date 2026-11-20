"""The GAMG fallback's warning names its measured floor when a tight datum solve asks for less.

Without a registered hierarchy the rotated solve's velocity block falls back to
GAMG, and the solver warns. On an annulus with a wall-normal datum that fallback
was measured to stall near 1e-10 relative (4e-10 at cell size 0.2, 1.2e-10 at
0.1), where a registered hierarchy (custom FMG) reaches about 2e-12; asked for
less, it ran to its iteration cap and then failed (UNFILED 36). With a requested
tolerance below 1e-9 and a nonzero datum, the warning says so. Otherwise the
warning is as before.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

FLOOR_TEXT = "measured to stall near 1e-10"


def _annulus_warning(capsys, tag, tolerance, datum):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1297{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1297{tag}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    s.bodyforce = sympy.Matrix([[0, -sympy.cos(sympy.pi * r)]])
    s.add_rotated_freeslip_bc(datum * sympy.cos(2 * sympy.atan2(y, x)) if datum else 0, "Upper")
    s.add_rotated_freeslip_bc(0, "Lower")
    s.tolerance = tolerance
    capsys.readouterr()
    try:
        s.solve()
    except RuntimeError:
        pass                     # below the floor the solve may fail; the warning came first
    out = capsys.readouterr().out
    assert "falls back to GAMG" in out, out
    return out


def test_a_tight_datum_solve_is_told_the_fallbacks_floor(capsys):
    out = _annulus_warning(capsys, "tight", 1.0e-10, 0.1)
    assert FLOOR_TEXT in out and "about 2e-12" in out, out


@pytest.mark.parametrize("tolerance, datum", [(1.0e-6, 0.1), (1.0e-10, 0)],
                         ids=["loose datum solve", "tight solve without a datum"])
def test_other_solves_keep_the_plain_warning(capsys, tolerance, datum):
    out = _annulus_warning(capsys, f"plain{datum}", tolerance, datum)
    assert FLOOR_TEXT not in out, out
