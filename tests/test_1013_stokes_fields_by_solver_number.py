"""A Stokes solver reads its fields by the solver's numbers (#1147).

A Stokes solver numbers its own fields: velocity 0, pressure 1. A variable's
position among the mesh's variables is a different number, and the two agree
only when the velocity is the mesh's first variable and the pressure its
second. Three places took one for the other:

- the guard that refuses a pressure nullspace beside a pressure Dirichlet
  condition, and the gauge record of every solve, picked the pressure's
  conditions by the pressure variable's mesh position;
- ``compute_volume_residual_fields`` and ``compute_boundary_residual_fields``,
  called before the first solve, split the residual into fields by the
  variables' mesh positions.

The two residual routines split a vector with the index sets the solve keeps.
Those describe one DM's section, so the build after a ``_reset()`` must make its
own: with the sets of the DM that was dropped, a solver that is used, reset and
then given constraint multipliers copies its solution back wrong, with no error.

Power: every case runs for three creation orders and asserts the positions it
relies on. With the pressure first, the velocity's walls must not be read as
pressure conditions (a refusal of a sound solve, with the nullspace or with a
constraint multiplier), and the two residual fields must not be exchanged.
With a third variable first, a pressure condition must still be seen (the
nullspace attached beside it, and a record of none), and the residual fields
must not stop in PETSc on a field the solver does not have. The numbers are
those of the velocity-then-pressure order, which was never wrong.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2]

WALLS = ("Top", "Bottom", "Left", "Right")
SOLVER_PRESSURE_FIELD = 1

# creation order -> the positions of (velocity, pressure) among the mesh's variables
ORDERS = {("v", "p"): (0, 1), ("p", "v"): (1, 0), ("t", "v", "p"): (1, 2)}

# The largest nodal speed on one rank (macOS, PETSc 3.25.3), fields created
# velocity then pressure: closed walls; closed walls with p = 0 on Top; the
# constrained solve.
SPEED_WALLS = 1.0004843692e-02
SPEED_PRESSURE_ON_TOP = 1.2133379961e-02
SPEED_CONSTRAINED = 6.8786717972e-02

# The 4 x 4 box has 81 velocity nodes and 25 pressure nodes. With every field
# zero the residual is the body force's: its norm over the velocity entries.
VELOCITY_ENTRIES, PRESSURE_ENTRIES = 162, 25
BODY_FORCE_RESIDUAL = 9.3750784657e-02
FIRST_MOMENT = 2.0 / np.pi**2


def _stokes(order, solver_class=uw.systems.Stokes):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=3)
    made = {}
    for name in order:
        components, degree = {"v": (2, 2), "p": (1, 1), "t": (1, 2)}[name]
        made[name] = uw.discretisation.MeshVariable(
            f"{name.upper()}1013g", mesh, components, degree=degree)
    v, p = made["v"], made["p"]
    # the premise: where the mesh holds each variable
    assert (v.field_id, p.field_id) == ORDERS[order]

    stokes = solver_class(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    stokes.tolerance = 1.0e-10
    return stokes, v


def _closed_walls(order, pressure_on_top):
    stokes, v = _stokes(order)
    for wall in WALLS:
        stokes.add_essential_bc((0.0, 0.0), wall)
    if pressure_on_top:
        stokes.add_condition(SOLVER_PRESSURE_FIELD, "dirichlet", 0.0, "Top")
    return stokes, v


def _largest_speed(v):
    return float(np.linalg.norm(np.asarray(v.array)[:, 0, :], axis=1).max())


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_velocity_walls_are_not_pressure_conditions(order):
    stokes, v = _closed_walls(order, pressure_on_top=False)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()

    gauge = stokes.solve_report.config["gauge"]
    assert stokes.snes.getConvergedReason() > 0
    assert gauge["pressure_nullspace_attached"] is True
    assert gauge["pressure_dirichlet_bcs"] == []
    assert abs(_largest_speed(v) - SPEED_WALLS) <= 1.0e-6 * SPEED_WALLS


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_a_pressure_condition_refuses_the_nullspace(order):
    stokes, _v = _closed_walls(order, pressure_on_top=True)
    stokes.petsc_use_pressure_nullspace = True
    with pytest.raises(ValueError, match=r"Found pressure Dirichlet BCs on: Top$"):
        stokes.solve()


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_the_record_names_the_pressure_condition(order):
    stokes, v = _closed_walls(order, pressure_on_top=True)
    stokes.solve()

    gauge = stokes.solve_report.config["gauge"]
    assert stokes.snes.getConvergedReason() > 0
    assert gauge["pressure_nullspace_attached"] is False
    assert gauge["pressure_dirichlet_bcs"] == ["Top"]
    assert abs(_largest_speed(v) - SPEED_PRESSURE_ON_TOP) <= 1.0e-6 * SPEED_PRESSURE_ON_TOP


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_a_constrained_solve_keeps_its_velocity_wall(order):
    """A constraint multiplier brings the guard in with no pressure nullspace asked for."""
    stokes, v = _stokes(order, uw.systems.Stokes_Constrained)
    _no_flow_through_the_sides(stokes)
    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0
    assert stokes.solve_report.config["gauge"]["pressure_dirichlet_bcs"] == []
    assert abs(_largest_speed(v) - SPEED_CONSTRAINED) <= 1.0e-6 * SPEED_CONSTRAINED


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_the_volume_residual_before_a_solve_is_split_by_solver_field(order):
    stokes, v = _closed_walls(order, pressure_on_top=False)
    rows = stokes.compute_volume_residual_fields()

    assert (rows["velocity"].size, rows["pressure"].size) == (VELOCITY_ENTRIES, PRESSURE_ENTRIES)
    momentum = float(np.linalg.norm(rows["velocity"]))
    assert abs(momentum - BODY_FORCE_RESIDUAL) <= 1.0e-6 * BODY_FORCE_RESIDUAL
    # each row sits at a node with its own x: the load is (0, cos(pi x)), the
    # residual is minus the load, and the first moment in x of cos(pi x) over the
    # box is -2/pi^2. Rows exchanged between nodes that share an x would pass.
    at_nodes = rows["velocity"].reshape(-1, 2)
    assert float(np.abs(at_nodes[:, 0]).max()) == 0.0
    assert abs(float(at_nodes[:, 1] @ v.coords[:, 0]) - FIRST_MOMENT) <= 1.0e-6 * FIRST_MOMENT
    # no velocity, so nothing in the continuity rows
    assert float(np.abs(rows["pressure"]).max()) == 0.0


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_the_boundary_residual_before_a_solve_is_split_by_solver_field(order):
    """The continuity-row term of a free-slip wall sums to minus the flow out through it.

    The velocity (0, x(1-x)) is held exactly by the elements; its flow out
    through Top is 1/6.
    """
    stokes, v = _stokes(order)
    for wall in ("Bottom", "Left", "Right"):
        stokes.add_essential_bc((0.0, 0.0), wall)
    stokes.add_nitsche_bc(0.0, "Top")
    x = v.coords[:, 0]
    v.array[:, 0, 0] = 0.0
    v.array[:, 0, 1] = x * (1.0 - x)
    rows = stokes.compute_boundary_residual_fields("Top", residual_field_id=SOLVER_PRESSURE_FIELD)

    assert (rows["velocity"].size, rows["pressure"].size) == (VELOCITY_ENTRIES, PRESSURE_ENTRIES)
    assert float(np.abs(rows["velocity"]).max()) == 0.0
    assert abs(float(rows["pressure"].sum()) + 1.0 / 6.0) < 1.0e-12


def _no_flow_through_the_sides(stokes):
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))


@pytest.mark.tier_b
@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_the_multiplier_rows_before_a_solve_are_split_by_solver_field(order):
    """A multiplier's rows sum to the flow out through its wall.

    The velocity (y, 0) is held exactly by the elements. Its flow out through
    Left is -1/2 and through Right 1/2; the first moments in y are -1/3 and 1/3.
    """
    stokes, v = _stokes(order, uw.systems.Stokes_Constrained)
    _no_flow_through_the_sides(stokes)
    v.array[:, 0, 0] = v.coords[:, 1]
    v.array[:, 0, 1] = 0.0
    rows = stokes.compute_volume_residual_fields()

    assert (rows["velocity"].size, rows["pressure"].size) == (VELOCITY_ENTRIES, PRESSURE_ENTRIES)
    for constraint, outward in zip(stokes._block_constraint_bcs, (-1.0, 1.0)):
        multiplier = constraint.lam
        at_nodes = rows[multiplier._solver_field_name]
        assert at_nodes.size == multiplier.coords.shape[0]
        assert abs(float(at_nodes.sum()) - outward / 2.0) < 1.0e-12
        assert abs(float(at_nodes @ multiplier.coords[:, 1]) - outward / 3.0) < 1.0e-12


@pytest.mark.tier_b
@pytest.mark.parametrize("first_use", ("a residual before any solve", "a solve"))
def test_constraints_added_after_a_reset_get_their_own_index_sets(first_use):
    """``_reset()`` is private, and examples call it between solves."""
    stokes, v = _stokes(("v", "p"), uw.systems.Stokes_Constrained)
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, 0.0), "Top")
    if first_use == "a solve":
        stokes.solve()
        v.array[...] = 0.0
    else:
        stokes.compute_volume_residual_fields()

    stokes._reset()
    _no_flow_through_the_sides(stokes)
    stokes.solve(zero_init_guess=True)

    assert stokes.snes.getConvergedReason() > 0
    assert abs(_largest_speed(v) - SPEED_CONSTRAINED) <= 1.0e-6 * SPEED_CONSTRAINED
