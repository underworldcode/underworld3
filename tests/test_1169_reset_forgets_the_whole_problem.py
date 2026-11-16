"""``_reset()`` forgets the whole boundary-value problem, not two parts of it (#1169).

``_reset()`` emptied ``essential_bcs`` and ``natural_bcs`` and dropped the DM and the
SNES, and kept three things that belong to the problem it tore down:
- the Stokes nullspace objects, built on the old walls' numbering: a solve after a
  reset with other walls diverged (reason -6) or stopped with PETSc error 75;
- the rotated free-slip conditions: registered again, each wall was held twice;
- the constraint conditions and their multipliers: registered again, four
  multiplier fields for two walls.
The solver's configuration (its constitutive model, tolerances, options, and the
nullspace settings themselves) is kept.

Each case compares a solve after a reset with the same solve on a new solver.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

SLIP_ENDS = {"Top": (None, 0.0), "Bottom": (None, 0.0), "Left": (0.0, 0.0), "Right": (0.0, 0.0)}
SLIP_SIDES = {"Top": (0.0, 0.0), "Bottom": (0.0, 0.0), "Left": (0.0, None), "Right": (0.0, None)}
NO_SLIP = dict.fromkeys(SLIP_ENDS, (0.0, 0.0))
FREE_SLIP = {"Top": (None, 0.0), "Bottom": (None, 0.0), "Left": (0.0, None), "Right": (0.0, None)}
_count = [0]


def _box(walls, solver_class=uw.systems.Stokes):
    _count[0] += 1
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1169_{_count[0]}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1169_{_count[0]}", mesh, 1, degree=1)
    stokes = solver_class(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x) * (1 + y)]])
    stokes.tolerance = 1.0e-10
    stokes.petsc_options.delValue("ksp_monitor")
    for wall, velocity in walls.items():
        stokes.add_essential_bc(velocity, wall)
    return stokes, v


def _speed(v):
    return float(np.linalg.norm(np.asarray(v.array)[:, 0, :], axis=1).max())


@pytest.mark.parametrize("first, second", [(SLIP_ENDS, SLIP_SIDES), (NO_SLIP, FREE_SLIP)],
                         ids=["other walls, same unknowns", "more unknowns"])
def test_the_pressure_nullspace_of_the_old_walls_is_not_kept(first, second):
    fresh, v_fresh = _box(second)
    fresh.petsc_use_pressure_nullspace = True
    fresh.solve()
    stokes, v = _box(first)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    stokes._reset()
    for wall, velocity in second.items():
        stokes.add_essential_bc(velocity, wall)
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getConvergedReason() > 0
    assert _speed(v) == pytest.approx(_speed(v_fresh), rel=1e-8)


def test_rotated_free_slip_conditions_are_forgotten():
    stokes, _ = _box({"Bottom": (0.0, 0.0)})
    stokes.add_rotated_freeslip_bc(0.0, "Top")
    stokes._reset()
    assert stokes._rotated_freeslip_bcs == []


def test_constraint_conditions_and_their_multipliers_are_forgotten():
    def constrain(stokes):
        stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
        stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))

    fresh, v_fresh = _box({"Bottom": (0.0, 0.0)}, uw.systems.Stokes_Constrained)
    constrain(fresh)
    fresh.solve()
    stokes, v = _box({"Bottom": (0.0, 0.0)}, uw.systems.Stokes_Constrained)
    constrain(stokes)
    stokes.solve()
    stokes._reset()
    assert stokes._block_constraint_bcs == [] and stokes._multipliers == []
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    constrain(stokes)
    assert len(stokes._multipliers) == 2
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getConvergedReason() > 0
    assert _speed(v) == pytest.approx(_speed(v_fresh), rel=1e-10)
