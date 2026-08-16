"""A solver takes rotated constraints or block constraints, not both (#464).

The rotated driver builds its own index-set fieldsplit over velocity and
pressure and addresses them by field number. A block constraint registers a
multiplier field, whose DOFs are then in neither index set, so the
preconditioner covers a strict subset of the operator's rows and says nothing
about it. Both mechanisms impose the same wall-normal condition.

The two controls matter as much as the two refusals: a guard that also refused
each mechanism on its own would pass a test that only checked that mixing
raises. Neutering `_reject_mixed_constraint_mechanisms` and repeating the first
case shows what the guard is for — the solver accepts one rotated boundary
condition and one multiplier together, and says nothing.
"""

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def stokes():
    """A constrained Stokes solver, which is the class that has both APIs."""

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5, qdegree=2
    )
    v = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pc", mesh, 1, degree=1)

    return uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)


def test_block_constraint_after_rotated_freeslip_is_refused(stokes):
    stokes.add_rotated_freeslip_bc(0.0, "Left")

    with pytest.raises(RuntimeError, match=r"add_constraint_bc.*#464"):
        stokes.add_constraint_bc(0.0, "Bottom")


def test_rotated_freeslip_after_block_constraint_is_refused(stokes):
    stokes.add_constraint_bc(0.0, "Bottom")

    with pytest.raises(RuntimeError, match=r"add_rotated_freeslip_bc.*#464"):
        stokes.add_rotated_freeslip_bc(0.0, "Left")


def test_several_rotated_boundaries_are_still_allowed(stokes):
    """The control: the guard is about mixing mechanisms, not about counting."""

    stokes.add_rotated_freeslip_bc(0.0, "Left")
    stokes.add_rotated_freeslip_bc(0.0, "Right")

    assert len(stokes._rotated_freeslip_bcs) == 2


def test_several_block_constraints_are_still_allowed(stokes):
    """The other control."""

    stokes.add_constraint_bc(0.0, "Bottom")
    stokes.add_constraint_bc(0.0, "Top")

    assert len(stokes._multipliers) == 2


def test_the_dispatch_refuses_the_pair_even_if_registration_was_bypassed(stokes):
    """The guarantee, as opposed to the message.

    The registration checks only cover what goes through the solver's own
    methods, and `fault_contact` writes `_fault_contact_faults` directly. The
    solve dispatch is where both lists are read together, so it carries the
    check as well. Reaching it here means bypassing registration, which is
    exactly the case the dispatch exists to catch.
    """

    stokes.add_rotated_freeslip_bc(0.0, "Left")
    stokes._multipliers.append(object())

    with pytest.raises(RuntimeError, match=r"solve\(\).*#464"):
        stokes.solve()
