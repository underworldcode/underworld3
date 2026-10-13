#!/usr/bin/env python3
"""Tolerance-key ownership on the saddle-point solvers is honest (#483).

Two defects, both in the middle ground between "owned" and "settable":

1. ``snes_rtol`` / ``ksp_atol`` were re-pushed by every solve
   (``_reassert_outer_tolerances``), silently discarding a user's explicit
   value — documented as settable, actually owned. They are now routed
   through the same latch that made ``snes_max_it`` reachable (ruling D18):
   the framework keeps re-asserting them until the user sets one, after which
   the user's value is honoured.

2. ``Stokes`` and ``Stokes_Constrained`` derived different option keys from
   ``tolerance`` through two unrelated hand-rolled code paths. Both now run
   one mechanism (``_derive_tolerance_margins``) over a per-class table
   (``_TOLERANCE_DERIVED_KEYS``), and the difference is documented: the base
   class derives the inner fieldsplit margins, Constrained derives the outer
   ``ksp_rtol`` and the Eisenstat-Walker pins.

Every arm reads LIVE PETSc objects (or the options DB where EW makes the live
value per-iteration) after TWO solves — the second solve is the one that used
to clobber.
"""
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)


def _stokes(mesh, constrained=False, suffix=""):
    v = uw.discretisation.MeshVariable(f"v{suffix}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{suffix}", mesh, 1, degree=1)
    cls = uw.systems.Stokes_Constrained if constrained else uw.systems.Stokes
    stokes = cls(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.bodyforce = sympy.Matrix([0, -1])
    return stokes


def _solve_twice(stokes):
    stokes.solve()
    stokes.solve()  # the second solve is the one that used to clobber


# --------------------------------------------------------------------------- #
#  Reachability: user overrides survive the solve, on BOTH classes
# --------------------------------------------------------------------------- #
def test_stokes_outer_and_inner_overrides_are_honoured():
    mesh = _mesh()
    stokes = _stokes(mesh, suffix="a")
    stokes.tolerance = 1e-6
    stokes.petsc_options["snes_rtol"] = 3.0e-3
    stokes.petsc_options["ksp_atol"] = 7.0e-9
    stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 0.02
    stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 0.005
    _solve_twice(stokes)
    # outer keys, live off the SNES/KSP
    assert stokes.snes.getTolerances()[0] == pytest.approx(3.0e-3)
    assert stokes.snes.getKSP().getTolerances()[1] == pytest.approx(7.0e-9)
    # inner fieldsplit rtols, live off the sub-KSPs (field 0 = velocity)
    vel_ksp, pres_ksp = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()
    assert vel_ksp.getTolerances()[0] == pytest.approx(0.02)
    assert pres_ksp.getTolerances()[0] == pytest.approx(0.005)


def test_stokes_defaults_still_owned_without_overrides():
    """Negative control: with NO user overrides the ownership must still work —
    the latch must not simply have stopped the framework pushing."""
    mesh = _mesh()
    stokes = _stokes(mesh, suffix="b")
    stokes.tolerance = 1e-6
    _solve_twice(stokes)
    assert stokes.snes.getTolerances()[0] == pytest.approx(1e-6)
    assert stokes.snes.getKSP().getTolerances()[1] == pytest.approx(1e-12)
    vel_ksp, pres_ksp = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()
    assert vel_ksp.getTolerances()[0] == pytest.approx(1e-6 * 0.033)
    assert pres_ksp.getTolerances()[0] == pytest.approx(1e-6 * 0.1)


def test_constrained_outer_overrides_and_ew_pins_are_honoured():
    mesh = _mesh()
    stokes = _stokes(mesh, constrained=True, suffix="c")
    stokes.tolerance = 1e-6
    stokes.petsc_options["snes_rtol"] = 2.0e-3
    stokes.petsc_options["ksp_atol"] = 5.0e-9
    stokes.petsc_options["snes_ksp_ew_rtol0"] = 4.0e-5   # derived-key override
    _solve_twice(stokes)
    assert stokes.snes.getTolerances()[0] == pytest.approx(2.0e-3)
    assert stokes.snes.getKSP().getTolerances()[1] == pytest.approx(5.0e-9)
    # EW re-picks the live ksp rtol per iteration, so the DB is the contract
    # for the pins: the user's value must survive both solves.
    assert float(stokes.petsc_options.getString("snes_ksp_ew_rtol0")) \
        == pytest.approx(4.0e-5)
    # the pin the user left alone keeps its tolerance-derived value
    assert float(stokes.petsc_options.getString("snes_ksp_ew_rtolmax")) \
        == pytest.approx(1e-6 * 0.1)


def test_constrained_defaults_still_owned_without_overrides():
    mesh = _mesh()
    stokes = _stokes(mesh, constrained=True, suffix="d")
    stokes.tolerance = 1e-6
    _solve_twice(stokes)
    assert stokes.snes.getTolerances()[0] == pytest.approx(1e-6)
    assert stokes.snes.getKSP().getTolerances()[1] == pytest.approx(1e-12)
    for key in ("ksp_rtol", "snes_ksp_ew_rtol0", "snes_ksp_ew_rtolmax"):
        assert float(stokes.petsc_options.getString(key)) \
            == pytest.approx(1e-6 * 0.1), key


# --------------------------------------------------------------------------- #
#  Order rule: set-time derivation vs later user override
# --------------------------------------------------------------------------- #
def test_tolerance_wins_when_set_after_the_user_key_and_vice_versa():
    mesh = _mesh()
    # arm 1: user key BEFORE tolerance= -> the set-time derivation wins
    s1 = _stokes(mesh, suffix="e")
    s1.petsc_options["snes_rtol"] = 3.0e-3
    s1.tolerance = 1e-6
    _solve_twice(s1)
    assert s1.snes.getTolerances()[0] == pytest.approx(1e-6)
    # arm 2: user key AFTER tolerance= -> the user wins across two solves
    s2 = _stokes(mesh, suffix="f")
    s2.tolerance = 1e-6
    s2.petsc_options["snes_rtol"] = 3.0e-3
    _solve_twice(s2)
    assert s2.snes.getTolerances()[0] == pytest.approx(3.0e-3)


# --------------------------------------------------------------------------- #
#  snes_max_it: the original D18 latch, now running on the shared mechanism
# --------------------------------------------------------------------------- #
def test_snes_max_it_latch_regression():
    mesh = _mesh()
    s1 = _stokes(mesh, suffix="g")
    s1.petsc_options["snes_max_it"] = 7
    _solve_twice(s1)
    assert s1.snes.getTolerances()[3] == 7
    # negative control: untouched -> the framework default (50) is live
    s2 = _stokes(mesh, suffix="h")
    _solve_twice(s2)
    assert s2.snes.getTolerances()[3] == 50


# --------------------------------------------------------------------------- #
#  Docs/code lockstep: the tables are exactly what the docstrings promise
# --------------------------------------------------------------------------- #
def test_derived_key_tables_match_the_documentation():
    assert uw.systems.Stokes._TOLERANCE_DERIVED_KEYS == {
        "fieldsplit_pressure_ksp_rtol": 0.1,
        "fieldsplit_velocity_ksp_rtol": 0.033,
    }
    assert uw.systems.Stokes_Constrained._TOLERANCE_DERIVED_KEYS == {
        "ksp_rtol": 0.1,
        "snes_ksp_ew_rtol0": 0.1,
        "snes_ksp_ew_rtolmax": 0.1,
    }
