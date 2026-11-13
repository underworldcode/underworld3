"""A rotated free-slip wall meeting an essential wall must still hold its corner.

On a flat, axis-aligned lid the rotated constraint and `add_dirichlet_bc((None,
0.0), "Top")` are the same discrete constraint: the measure-weighted node normal
is exactly (0,1), so striking the rotated normal row is striking u_y.

They were not the same. `build_rotation` skipped any node with a velocity DOF
constrained out of the global vector — which is every node where the lid meets a
wall held by an essential condition — so the wall-normal component was left FREE
at those nodes. The lid leaked at its own end points (max|u_y| = 4.0e-3 against
|u|max 2.5e-2, entirely at the two corners) and the solve differed from the
component-Dirichlet one by 2e-3 globally, with an exact linear solve on both
sides. Issue #616; the corner reaction of #608 is the same node.

Run: pixi run python -m pytest tests/test_1066_rotated_meets_essential.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1]

import numpy as np
import sympy
import underworld3 as uw

RES = 16


def _solve(lid, direct=False):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )
    x, z = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.saddle_preconditioner = 1.0
    s.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * z)]])
    s.tolerance = 1.0e-9
    s.petsc_use_pressure_nullspace = True
    s.add_dirichlet_bc((0.0, None), "Left")
    s.add_dirichlet_bc((0.0, None), "Right")
    s.add_dirichlet_bc((None, 0.0), "Bottom")
    if lid == "dirichlet":
        s.add_dirichlet_bc((None, 0.0), "Top")
        if direct:
            s.petsc_options["ksp_type"] = "preonly"
            s.petsc_options["pc_type"] = "lu"
    else:
        s.add_rotated_freeslip_bc(0.0, "Top")
        if direct:
            s._rotated_use_lu = True
    s.solve()
    assert s.snes.getConvergedReason() > 0
    return np.squeeze(np.asarray(v.array)).copy(), v.coords.copy()


def test_rotated_lid_holds_its_corners():
    """u.n = 0 on the whole lid, corners included, to round-off."""
    u, coords = _solve("rotated")
    top = np.abs(coords[:, 1] - 1.0) < 1.0e-9
    leak = np.abs(u[top, 1]).max() / np.abs(u).max()
    # It was 1.6e-1, at the two corner nodes and nowhere else.
    assert leak < 1.0e-12, f"the lid leaks at u.n = {leak:.2e} of |u|max"


def test_rotated_lid_reproduces_the_component_condition():
    """The two are the same discrete constraint on a flat wall, so the same answer.

    Judged against a monolithic DIRECT solve of the component-Dirichlet problem, so
    a difference cannot be blamed on either iterative solve.
    """
    reference, _c = _solve("dirichlet", direct=True)
    for lid in ("dirichlet", "rotated"):
        u, _c = _solve(lid)
        rel = np.linalg.norm(u - reference) / np.linalg.norm(reference)
        assert rel < 1.0e-7, f"{lid} lid differs from the direct reference by {rel:.2e}"


def _lid_against_a_pinned_wall(left, capsys, lid_normal=(sympy.Rational(3, 5), sympy.Rational(4, 5)),
                               right=None, clock=None, deprecated_component=None):
    """Rotated free slip on Top with the given normal, meeting an essential
    condition ``left`` on Left (and ``right`` on Right, else rotated free slip);
    no slip on Bottom, so the only shared nodes are the lid's two ends.
    Returns the warnings about a pinned value that the solve printed."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    x, z = mesh.X
    v = uw.discretisation.MeshVariable("Upin", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ppin", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    s.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x)]])
    s.tolerance = 1.0e-9
    s.add_rotated_freeslip_bc(0, "Top", normal=sympy.Matrix([list(lid_normal)]))
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    if right is None:
        s.add_rotated_freeslip_bc(0, "Right", normal=sympy.Matrix([[1, 0]]))
    else:
        s.add_dirichlet_bc(right(mesh) if callable(right) else right, "Right")
    if deprecated_component is None:
        s.add_dirichlet_bc(left(mesh) if callable(left) else left, "Left")
    else:
        s.add_dirichlet_bc(left, "Left", deprecated_component)
    if clock is not None:
        uw.get_default_model().tracker.time = clock
    capsys.readouterr()
    s.solve()
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if "essential BC on" in line]


@pytest.mark.tier_b
def test_a_pinned_value_along_the_wall_normal_is_reported(capsys):
    """The corner's reduced constraint holds the pinned components at zero (#939).

    With u_z = 1/20 pinned on Left and the lid normal (3/5, 4/5), the wall-normal
    velocity at the Top-Left corner comes out as 4/5 x 1/20 = 0.04 where the wall
    asks for zero. Until the affine reduction is implemented the solve says so,
    once, naming the wall and the size.
    """
    warnings = _lid_against_a_pinned_wall((None, 0.05), capsys)
    assert len(warnings) == 1
    assert "'Left'" in warnings[0] and "component of 0.04 along the wall normal" in warnings[0]


@pytest.mark.tier_b
def test_a_node_with_every_component_pinned_is_reported_too(capsys):
    """u = (1/20, 1/20) on Left: n.v at the corner is 3/5 x 1/20 + 4/5 x 1/20 = 0.07."""
    warnings = _lid_against_a_pinned_wall((0.05, 0.05), capsys)
    assert len(warnings) == 1 and "component of 0.07 along the wall normal" in warnings[0]


@pytest.mark.tier_b
def test_each_essential_wall_is_named_once(capsys):
    warnings = _lid_against_a_pinned_wall((None, 0.05), capsys, right=(None, 0.07))
    assert len(warnings) == 2
    assert "'Left'" in warnings[0] and "0.04" in warnings[0]
    assert "'Right'" in warnings[1] and "0.056" in warnings[1]


@pytest.mark.tier_b
@pytest.mark.parametrize("case", ["zero", "inflow_beside_a_flat_lid", "profile_zero_at_the_corners",
                                  "clock_at_zero", "expression_holding_zero"])
def test_a_pin_that_leaves_the_wall_condition_intact_is_silent(case, capsys):
    """No warning where n.v at the shared node is what the rotated wall asked for."""
    if case == "zero":
        warnings = _lid_against_a_pinned_wall((None, 0.0), capsys)
    elif case == "inflow_beside_a_flat_lid":
        # the pinned component is tangential to the lid
        warnings = _lid_against_a_pinned_wall((0.05, None), capsys, lid_normal=(0, 1),
                                              right=(0.05, None))
    elif case == "profile_zero_at_the_corners":
        warnings = _lid_against_a_pinned_wall(
            lambda mesh: sympy.Matrix([[sympy.oo, mesh.X[1] * (1 - mesh.X[1]) / 5]]), capsys)
    elif case == "clock_at_zero":
        warnings = _lid_against_a_pinned_wall(
            lambda mesh: sympy.Matrix([[sympy.oo, sympy.Rational(1, 20) * mesh.t]]), capsys,
            clock=0.0)
    else:
        nothing = uw.expression("pinned value", 0.0)
        warnings = _lid_against_a_pinned_wall(
            lambda mesh: sympy.Matrix([[sympy.oo, nothing]]), capsys)
    assert warnings == []


@pytest.mark.tier_b
def test_a_pinned_value_that_follows_the_clock_is_reported_once_it_is_nonzero(capsys):
    try:
        warnings = _lid_against_a_pinned_wall(
            lambda mesh: sympy.Matrix([[sympy.oo, sympy.Rational(1, 20) * mesh.t]]), capsys,
            clock=2.0)
    finally:
        uw.get_default_model().tracker.time = 0.0
    assert len(warnings) == 1 and "component of 0.08 along the wall normal" in warnings[0]


@pytest.mark.tier_b
def test_a_field_valued_pin_is_reported_as_possibly_nonzero(capsys):
    def field(mesh):
        g = uw.discretisation.MeshVariable("Gpin", mesh, 1, degree=1)
        g.array[:, 0, 0] = 0.05
        return sympy.Matrix([[sympy.oo, g.sym[0]]])

    warnings = _lid_against_a_pinned_wall(field, capsys)
    assert len(warnings) == 1 and "may have a component along the wall normal" in warnings[0]


@pytest.mark.tier_b
def test_the_older_components_form_of_an_essential_condition_still_solves(capsys):
    """``add_dirichlet_bc(0.05, "Left", 1)`` stores one value for one component."""
    with pytest.warns(DeprecationWarning):
        warnings = _lid_against_a_pinned_wall(0.05, capsys, deprecated_component=1)
    assert len(warnings) == 1 and "component of 0.04 along the wall normal" in warnings[0]
