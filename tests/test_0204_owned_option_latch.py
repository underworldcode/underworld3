"""An owned option's user latch dies with the option (#490).

`solve()` re-pushes the options the solver owns, so `_resolve_owned_option` has
to tell its own previous push from a value the user set. It does that by
latching the user's value the first time it sees one it did not push.

The latch outlived the option. Deleting the key made the next solve fall back
correctly, but that solve pushed the default, and the one after read the default
back, found it equal to what it had pushed, and returned the latched value
instead. Measured on `development`:

    user sets 200        -> 200
    user deletes the key -> 50
    next solve           -> 200      <-- resurrected, and permanent
    and the next         -> 200

The sequence is driven through `_resolve_snes_max_it` / `_push_snes_max_it`
rather than through `solve()`: that is the pair `solve()` itself calls, and it
takes no solve to exercise, so the test says what it means and runs in a second.
"""

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

DEFAULT = 50
USER = 200


@pytest.fixture
def solver():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
    v = uw.discretisation.MeshVariable("Vlatch", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Platch", mesh, 1, degree=1)

    return uw.systems.Stokes(mesh, velocityField=v, pressureField=p)


def _solve_cycle(solver):
    """One solve's worth of the resolve-then-push pair."""

    resolved = solver._resolve_snes_max_it(DEFAULT)
    solver._push_snes_max_it(resolved)

    return resolved


def test_a_user_value_is_honoured_and_keeps_being_honoured(solver):
    """The control: the latch does its job while the option is there."""

    solver.petsc_options["snes_max_it"] = USER

    assert _solve_cycle(solver) == USER
    assert _solve_cycle(solver) == USER
    assert _solve_cycle(solver) == USER


def test_deleting_the_option_does_not_leave_the_value_behind(solver):
    """Deleting the key gives the default back, and it stays given back."""

    solver.petsc_options["snes_max_it"] = USER
    assert _solve_cycle(solver) == USER

    del solver.petsc_options["snes_max_it"]

    assert _solve_cycle(solver) == DEFAULT
    assert _solve_cycle(solver) == DEFAULT, (
        "the deleted user value came back on the second solve after deletion — "
        "the latch outlived the option it was latched from")
    assert _solve_cycle(solver) == DEFAULT


def test_a_second_user_value_replaces_the_first(solver):
    """A moved value is picked up rather than shadowed by the latched one."""

    solver.petsc_options["snes_max_it"] = USER
    assert _solve_cycle(solver) == USER

    solver.petsc_options["snes_max_it"] = 7

    assert _solve_cycle(solver) == 7
    assert _solve_cycle(solver) == 7
