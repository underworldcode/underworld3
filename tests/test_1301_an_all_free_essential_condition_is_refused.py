"""An essential condition that leaves every component free is refused when it
is added (UNFILED 46); so is a natural one.

``add_dirichlet_bc((None, None), "Left")`` names no component to fix. It was
accepted, then failed at the solve inside the discretisation set-up with
``IndexError: Out of bounds on buffer access (axis 0)``, on any boundary, a
plain wall included. It constrains nothing, so it is refused at the call that
made it, with a message saying how to fix one component or drop it. A
condition fixing at least one component is unchanged.
"""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4,
                                             regular=True, qdegree=2)
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    return s


@pytest.mark.parametrize("conds", [(None, None), [None, None]])
def test_an_all_free_condition_is_refused_where_it_is_added(conds):
    s = _stokes()
    with pytest.raises(ValueError, match=r"constrains nothing") as err:
        s.add_dirichlet_bc(conds, "Left")
    assert "'Left'" in str(err.value)


def test_an_all_free_natural_condition_is_refused_too():
    s = _stokes()
    with pytest.raises(ValueError, match=r"natural condition on 'Left'.*constrains nothing"):
        s.add_natural_bc((None, None), "Left")


def test_a_condition_fixing_one_component_still_solves():
    s = _stokes()
    s.add_dirichlet_bc((None, 0.0), "Left")
    s.add_dirichlet_bc((None, 0.0), "Right")
    s.add_dirichlet_bc((1.0, 0.0), "Top")
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    s.solve()
    assert s.snes.getConvergedReason() > 0
