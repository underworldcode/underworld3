"""An essential condition on a fault that also carries a law is refused (#940).

The interface law (``add_fault_bc``) reads the slip and the normal traction at
the fault's pair nodes from pooled local vectors filled by ``globalToLocal``,
which does not write a DOF an essential condition constrains. With a side of
the fault pinned (``add_dirichlet_bc`` on ``<name>Minus``), those reads returned
whatever the pool last held: sigma_nn 1e-10 or 2.1e4 for pool contents 0 or
1e3 at the same node, and the solve then crashed (heap corruption or a SEGV in
8 of 10 serial runs, PETSc error 63 at np 2). The two ways to drive a fault are
exclusive on one fault: a law, or conditions on its sides. The solve now
refuses the mixture, naming the fault and the side, before anything reads the
pool. Conditions on the sides of a fault without a law are unchanged
(test_1174), and so is a natural condition beside a law.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

TRACE = np.array([[0.30, 0.45], [0.70, 0.55]])


def _sheared(law=True):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8,
                                             regular=True, qdegree=2).add_fault([("Flt", TRACE)])
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    s.tolerance = 1e-8
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    if law:
        s.add_fault_bc(1.0, "Flt")            # a viscous interface: the law reads the slip
    return s


@pytest.mark.parametrize("where", ["FltMinus", "FltPlus", "Flt"])
def test_an_essential_condition_on_a_fault_with_a_law_is_refused(where):
    s = _sheared()
    s.add_dirichlet_bc((0.0, 0.0), where)
    with pytest.raises(ValueError, match=r"#940") as err:
        s.solve()
    message = str(err.value)
    assert "'Flt'" in message and repr(where) in message
    assert "add_fault_bc" in message


def test_a_natural_condition_beside_a_law_still_solves():
    s = _sheared()
    s.add_natural_bc((0.0, 0.0), "FltMinus")
    s.solve()
    assert s.snes.getConvergedReason() > 0


def test_conditions_on_both_sides_without_a_law_still_solve():
    s = _sheared(law=False)
    s.add_dirichlet_bc((0.0, 0.0), "FltPlus")
    s.add_dirichlet_bc((0.0, 0.0), "FltMinus")
    s.solve()
    assert s.snes.getConvergedReason() > 0


def _projection_with_a_law():
    """A vector projection with a law and a pinned side: its solve reaches the
    rotated path through _solve_rotated_if_registered, not Stokes' solve."""
    import sympy
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8,
                                             regular=True, qdegree=2).add_fault([("Flt", TRACE)])
    x, y = mesh.X
    u = uw.discretisation.MeshVariable("u0940", mesh, 2, degree=2)
    projection = uw.systems.Vector_Projection(mesh, u)
    projection.uw_function = sympy.Matrix([[x, y]])
    projection.add_fault_bc(1.0, "Flt")
    projection.add_dirichlet_bc((0.0, 0.0), "FltMinus")
    return projection


def test_a_vector_solve_with_a_law_and_a_pinned_side_is_refused():
    with pytest.raises(ValueError, match=r"#940"):
        _projection_with_a_law().solve()


def test_a_frictionless_law_is_refused_with_its_own_reason():
    s = _sheared(law=False)
    s.add_fault_bc(0, "Flt")
    s.add_dirichlet_bc((0.0, 0.0), "FltMinus")
    with pytest.raises(ValueError, match=r"#940") as err:
        s.solve()
    assert "frictionless law is lost" in str(err.value)


def test_a_pressure_condition_on_a_side_is_not_this_refusal():
    """It constrains no velocity pair DOF; the pressure-nullspace check refuses
    it, as on development."""
    s = _sheared()
    s.add_condition(1, "dirichlet", (0.0,), "FltMinus")
    with pytest.raises(ValueError) as err:
        s.solve()
    assert "#940" not in str(err.value) and "pressure" in str(err.value)


def test_an_all_free_condition_on_a_side_is_not_this_refusal():
    """(None, None) constrains no component, so it is not said to. (Such a
    condition fails on any boundary, a wall included, in the discretisation
    set-up; that is not #940's.)"""
    s = _sheared()
    s.add_dirichlet_bc((None, None), "FltMinus")
    try:
        s.solve()
    except Exception as err:            # noqa: BLE001 - only the message is checked
        assert "#940" not in str(err)
