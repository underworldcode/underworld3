"""A solve that diverges says so with a RuntimeWarning, not only a printed notice (#1059).

A Stokes solve at a viscosity of 1e11 or more in PETSc's frame fails at iteration 0
(#1059): the SNES stops with DIVERGED_LINEAR_SOLVE and the velocity stays at zero.
The solver printed a notice and returned, so a script carried on with u = 0. The
fixture here diverges by construction instead (an outer divergence tolerance of 1e-30),
so it does not depend on #1059's cause.

- a diverged solve warns once, on rank 0, naming the reason, attributed to the
  caller's solve() line;
- a converged solve does not, and nor does the stop a bounded difficulty probe
  asks for (DIVERGED_MAX_IT).
"""
import warnings

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(tag):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.tolerance = 1.0e-8
    return stokes


def _divergence_warnings(record):
    return [w for w in record
            if issubclass(w.category, RuntimeWarning) and "the solve diverged" in str(w.message)]


def test_a_diverged_solve_warns():
    stokes = _box("div")
    stokes.petsc_options["ksp_divtol"] = 1.0e-30   # the outer KSP diverges at once
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    assert stokes.snes.getConvergedReason() == -3      # DIVERGED_LINEAR_SOLVE
    # development: printed, and no warning
    shown = _divergence_warnings(record)
    with uw.selective_ranks(0) as selected:
        if selected:
            assert len(shown) == 1, [str(w.message) for w in record]
            assert "DIVERGED_LINEAR_SOLVE" in str(shown[0].message), str(shown[0].message)
            assert "initial guess" in str(shown[0].message), str(shown[0].message)
            assert shown[0].filename == __file__, shown[0].filename
        else:
            assert not shown, [str(w.message) for w in shown]


def test_a_converged_solve_does_not_warn():
    stokes = _box("ok")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    assert not _divergence_warnings(record)


def test_a_bounded_probe_at_its_cap_does_not_warn():
    stokes = _box("probe")
    u = stokes.Unknowns.u.sym
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 + 1.0e4 * u.dot(u)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report = stokes.estimate_difficulty(1, warm=False)
    assert report.reason == -5, report.reason_str      # DIVERGED_MAX_IT, the asked-for stop
    assert not _divergence_warnings(record), [str(w.message) for w in record]
