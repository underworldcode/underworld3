"""A Stokes solve whose fieldsplit sub-solve stops at its iteration cap says so (#585, #625).

PETSc does not treat a sub-KSP that stops at its cap (DIVERGED_ITS) as a failure, so
the outer solve can report convergence while the velocity or pressure block never
solved: the Schur operator is then applied through an inexact, varying velocity
solve. The count was recorded (``solve_report.sub[...].capped``) but nothing said
so at solve time.

- a solve whose velocity block runs into its cap warns, naming the block and the cap;
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
        if selected:
            messages = _cap_warnings(record)
            assert len(messages) == 1, messages
            assert "velocity" in messages[0] and "cap" in messages[0], messages


def test_a_solve_within_its_caps_is_quiet():
    stokes = _box("quiet")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    assert all(r.capped == 0 for r in stokes.solve_report.sub.values()), stokes.solve_report.sub
    assert not _cap_warnings(record)
