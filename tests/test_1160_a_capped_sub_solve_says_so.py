"""A Stokes solve whose fieldsplit sub-solve stops at its iteration cap says so (#585, #625).

PETSc does not treat a sub-KSP that stops at its cap (DIVERGED_ITS) as a failure, so
the outer solve can report convergence while the velocity or pressure block never
solved: the Schur operator is then applied through an inexact, varying velocity
solve. The count was recorded (``solve_report.sub[...].capped``) but nothing said
so at solve time.

- a solve whose velocity block runs into its cap warns, on rank 0 only, naming the
  block and the cap, attributed to the caller's solve() line;
- in a loop of solves from one line Python's default filter shows it once;
- a bounded difficulty probe, whose cap is the point, does not warn;
- a solve within the caps does not.
"""
import warnings

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(tag, velocity_cap=None):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8, qdegree=3, refinement=1)
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
    if velocity_cap is not None:
        stokes.petsc_options["fieldsplit_velocity_ksp_max_it"] = velocity_cap
    return stokes


def _velocity_cap_warnings(record):
    # at np 2 the pressure block can cap too under a velocity cap of 2; count the
    # velocity block's own warning
    return [w for w in record
            if "iteration cap" in str(w.message) and "the velocity sub-solve" in str(w.message)]


def _cap_warnings(record):
    return [str(w.message) for w in record if "iteration cap" in str(w.message)]


def test_a_velocity_block_at_its_cap_warns():
    stokes = _box("cap", velocity_cap=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    capped = {name: r.capped for name, r in stokes.solve_report.sub.items()}
    assert capped.get("velocity", 0) > 0, capped
    # development: recorded, and nothing said
    with uw.selective_ranks(0) as selected:
        shown = _velocity_cap_warnings(record)
        if selected:
            assert len(shown) == 1, [str(w.message) for w in record]
            assert "iteration cap (2)" in str(shown[0].message), str(shown[0].message)
            assert shown[0].filename == __file__, shown[0].filename
        else:
            assert not shown, [str(w.message) for w in shown]


def test_a_loop_of_capped_solves_shows_the_warning_once():
    stokes = _box("loop", velocity_cap=2)
    counts = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for tolerance in (1.0e-8, 1.0e-9, 1.0e-10):
            stokes.tolerance = tolerance
            stokes.solve(zero_init_guess=True)
            counts.append(stokes.solve_report.sub["velocity"].capped)
    # the counts differ from solve to solve; the message must not carry them
    assert min(counts) > 0 and len(set(counts)) > 1, counts
    with uw.selective_ranks(0) as selected:
        if selected:
            assert len(_velocity_cap_warnings(record)) == 1, [str(w.message) for w in record]


def test_a_difficulty_probe_does_not_warn():
    stokes = _box("probe", velocity_cap=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report = stokes.estimate_difficulty(1, warm=False)
    assert report.sub["velocity"].capped > 0, report.sub
    assert not _velocity_cap_warnings(record), [str(w.message) for w in record]


def test_a_solve_within_its_caps_is_quiet():
    stokes = _box("quiet")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    assert all(r.capped == 0 for r in stokes.solve_report.sub.values()), stokes.solve_report.sub
    assert not _cap_warnings(record)
