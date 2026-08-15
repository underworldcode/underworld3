"""Barr & Houseman (1996) faulted medium — verified as a solution.

These tests do not compare the expressions against a stored answer or against a
UW3 solve. They check that the field IS a Stokes solution satisfying the fault
conditions, symbolically:

    div u = 0
    div sigma = eta lap(u) - grad(p) = 0       (their eq 3, in UW3's sign)
    tau_r_theta = 0 on both faces of the fault      fault condition 3
    u_theta continuous across the fault             fault condition 2
    sigma_theta_theta continuous across the fault   fault condition 1
    slip = 2 U0 sqrt(r/R0)

The Stokes and fault-condition checks run with the parameters left SYMBOLIC.
That is not decoration: with U0 = R0 = eta = 1, a transcription carrying the
wrong power of R0 in the singular pressure term gives a momentum residual of
exactly zero, and the same transcription with the parameters free gives
U0 eta (1 - R0) cos(3 theta / 2) / (2 sqrt(R0) r^(3/2)). Measured.

If those hold simultaneously, the transcription is the solution, whatever any
solver later does with it. That is a stronger statement than a regression test
and it is what makes this usable as a benchmark.

The transcription needed it: the half-integer sine terms of u_theta appear with
one sign in the paper's boundary datum (A8b) and the opposite sign in its
solution (A9b). Incompressibility settles it — for u_r = A sqrt(R) f(theta) and
u_theta = sqrt(R) g(theta), div u = 0 forces g' = -(3/2) A f, which is (A8b)'s
sign. The test below would fail on the other choice.

Two things are new here relative to the fault-frame verification, and both come
from joining the family: the pressure is carried in UW3's sign rather than the
paper's (asserted below against the printed A9c), and the same field is also
exposed in the mesh coordinates, where the family's residual gates reach it.

Run: pixi run python -m pytest tests/test_1029_analytic_faulted_medium.py -v
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.analytic import _validation

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh():
    """A carrier for the coordinate symbols.

    The solution is posed on a disc about the fault tip, which is not this box —
    and does not need to be. `sample_points` returns points on the disc, and the
    gates evaluate symbolic expressions rather than anything the mesh holds.
    """

    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


def _symbolic(mesh):
    """The solution with U0, R0 and eta left free."""

    U0, R0, eta = sympy.symbols("U_0 R_0 eta", positive=True)

    return uw.analytic.FaultedMedium(mesh, U0=U0, R0=R0, eta=eta)


def _polar_operators(u_r, u_t, p, r, t, eta):
    """(div u, momentum_x, momentum_y) for a field given in polar form.

    The momentum operator is `eta lap(u) - grad(p)`: the divergence of
    `sigma = 2 eta edot - p I` at constant viscosity, i.e. UW3's compression-
    positive pressure. The paper writes `+ grad(p)` because its pressure is
    extension-positive; the two statements are the same one.
    """

    div = sympy.diff(r * u_r, r) / r + sympy.diff(u_t, t) / r

    u_x = u_r * sympy.cos(t) - u_t * sympy.sin(t)
    u_y = u_r * sympy.sin(t) + u_t * sympy.cos(t)

    def lap(f):
        return sympy.diff(r * sympy.diff(f, r), r) / r + sympy.diff(f, t, 2) / r**2

    p_r, p_t = sympy.diff(p, r), sympy.diff(p, t)
    grad_p_x = p_r * sympy.cos(t) - p_t * sympy.sin(t) / r
    grad_p_y = p_r * sympy.sin(t) + p_t * sympy.cos(t) / r

    return div, eta * lap(u_x) - grad_p_x, eta * lap(u_y) - grad_p_y


def test_the_field_is_an_incompressible_stokes_solution(mesh):
    """div u = 0 and the momentum balance vanishes identically.

    The parameters are left SYMBOLIC on purpose. With U0 = R0 = eta = 1 a
    transcription carrying the wrong power of R0 in the pressure satisfies the
    identity and still fails the physics, so the unit-parameter version of this
    test is strictly weaker for the same runtime.
    """

    sol = _symbolic(mesh)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar
    p = sol.pressure_polar

    div, mom_x, mom_y = _polar_operators(u_r, u_t, p, r, t, sol.eta)

    assert sympy.simplify(div) == 0
    assert sympy.simplify(mom_x) == 0
    assert sympy.simplify(mom_y) == 0


def test_the_fault_conditions_hold_on_both_faces(mesh):
    """All THREE of the paper's fault conditions, with symbolic parameters.

    Zero shear traction, continuous normal velocity, continuous normal stress.
    """

    sol = _symbolic(mesh)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar

    tau_rt = sol.eta * (r * sympy.diff(u_t / r, r) + sympy.diff(u_r, t) / r)
    assert sympy.simplify(tau_rt.subs(t, 0)) == 0
    assert sympy.simplify(tau_rt.subs(t, 2 * sympy.pi)) == 0

    # u_theta is the fault-NORMAL component and must not jump; u_r is the
    # fault-parallel one and must (that jump is the slip).
    assert sympy.simplify(u_t.subs(t, 0) - u_t.subs(t, 2 * sympy.pi)) == 0

    # Normal STRESS continuity — the third of the paper's three fault
    # conditions. sigma = 2 eta edot - p I with p positive in compression, so
    # the fault-normal component is sigma_tt = 2 eta e_tt - p with
    # e_tt = (1/r) du_theta/dtheta + u_r/r.
    p = sol.pressure_polar
    e_tt = sympy.diff(u_t, t) / r + u_r / r
    sigma_tt = 2 * sol.eta * e_tt - p

    assert sympy.simplify(sigma_tt.subs(t, 0) - sigma_tt.subs(t, 2 * sympy.pi)) == 0


def test_the_pressure_is_the_negative_of_the_published_one(mesh):
    """The one convention this solution does not share with its source.

    Barr & Houseman take extension as positive, so their force balance is
    `d_j tau_ij + d_i p = 0`. UW3 — and so this suite — takes pressure positive
    in compression. The flip is applied once, and asserted here against the
    paper's printed (A9c) rather than left to a comment, because a sign that is
    only documented is a sign that can drift.

    Independent measurement: @gthyagi's UW3 Stokes solve on a Gmsh slit disc
    converges against -p_BH96 (13.1%, 6.9%, 3.3% at h = 0.20, 0.10, 0.05) and
    sits at about 199% at every resolution against the paper's own sign.
    """

    sol = _symbolic(mesh)
    r, t = sol.symbols
    R = r / sol.R0

    published_A9c = (sol.eta * sol.U0 / sol.R0) * (
        -2 * R * sympy.sin(t)
        + 3 * R**2 * sympy.sin(2 * t)
        + sympy.cos(t / 2) / sympy.sqrt(R)
    )

    assert sympy.simplify(sol.pressure_polar + published_A9c) == 0


def test_the_slip_is_the_published_normalisation(mesh):
    """slip = 2 U0 sqrt(r/R0), so 2 U0 at the perimeter — the paper's anchor.

    Negative control: the whole-integer modes alone are continuous, so a
    solution without the half-integer mode would give zero slip and pass a
    weaker test vacuously.
    """

    sol = uw.analytic.FaultedMedium(mesh, U0=1.0, R0=1.0, eta=1.0)
    r, t = sol.symbols
    u_r, _u_t = sol.velocity_polar

    jump = sympy.simplify(u_r.subs(t, 0) - u_r.subs(t, 2 * sympy.pi))
    assert sympy.simplify(jump - 2 * sol.U0 * sympy.sqrt(r / sol.R0)) == 0
    assert float(jump.subs(r, sol.R0)) == pytest.approx(2.0 * sol.U0)
    assert float(sol.slip(sol.R0)) == pytest.approx(2.0 * sol.U0)

    # The slip is carried entirely by the half-integer mode: drop the sqrt term
    # and the fault disappears.
    continuous_only = u_r - (sol.U0 / 4) * sympy.sqrt(r / sol.R0) * (
        sympy.cos(t / 2) + 3 * sympy.cos(3 * t / 2)
    )
    assert sympy.simplify(
        continuous_only.subs(t, 0) - continuous_only.subs(t, 2 * sympy.pi)
    ) == 0


def test_the_numpy_evaluator_agrees_with_the_symbolic_form(mesh):
    """The Cartesian evaluator must reproduce the polar expressions.

    Includes points either side of the fault, which is where the branch cut
    lives and where a bare ``atan2`` would silently return the wrong face.
    """

    sol = uw.analytic.FaultedMedium(mesh, U0=1.3, R0=2.0, eta=0.7)
    r_sym, t_sym = sol.symbols
    u_r_sym, u_t_sym = sol.velocity_polar
    p_sym = sol.pressure_polar

    rng = np.random.default_rng(7)
    radii = rng.uniform(0.2, 1.9, 12)
    angles = np.r_[rng.uniform(0.05, 2 * np.pi - 0.05, 10), 0.02, 2 * np.pi - 0.02]
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    velocity = sol.evaluate_velocity(pts)
    pressure = sol.evaluate_pressure(pts)

    for k, (rr, tt) in enumerate(zip(radii, angles)):
        subs = {r_sym: float(rr), t_sym: float(tt)}
        ur = float(u_r_sym.subs(subs))
        ut = float(u_t_sym.subs(subs))
        expect = np.array(
            [ur * np.cos(tt) - ut * np.sin(tt), ur * np.sin(tt) + ut * np.cos(tt)]
        )

        assert velocity[k] == pytest.approx(expect, rel=1e-10, abs=1e-12)
        assert pressure[k] == pytest.approx(
            float(p_sym.subs(subs)), rel=1e-10, abs=1e-12
        )


def test_the_branch_cut_lies_on_the_fault(mesh):
    """Straddling the fault must show the slip; straddling +x elsewhere must not.

    This is the test that a bare ``atan2`` fails: it would put the cut on the
    negative x axis, reporting a jump where the medium is continuous and none
    where the fault is.
    """

    sol = uw.analytic.FaultedMedium(mesh)
    eps = 1e-7

    above = sol.evaluate_velocity(np.array([[0.5, +eps]]))
    below = sol.evaluate_velocity(np.array([[0.5, -eps]]))
    assert (above[0, 0] - below[0, 0]) == pytest.approx(
        float(sol.slip(0.5)), rel=1e-4
    ), "no slip across the fault"

    left_up = sol.evaluate_velocity(np.array([[-0.5, +eps]]))
    left_dn = sol.evaluate_velocity(np.array([[-0.5, -eps]]))
    assert np.allclose(left_up, left_dn, atol=1e-5), (
        "the medium is continuous on the fault's projection; a jump here means "
        "the branch cut is in the wrong place"
    )


def test_the_boundary_datum_reproduces_the_solution_at_the_perimeter(mesh):
    """The Dirichlet datum a solver would impose is the solution at r = R0."""

    sol = uw.analytic.FaultedMedium(mesh, U0=1.0, R0=1.5, eta=1.0)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar
    U_r, U_t = sol.boundary_velocity()

    assert sympy.simplify(U_r - u_r.subs(r, sol.R0)) == 0
    assert sympy.simplify(U_t - u_t.subs(r, sol.R0)) == 0
    assert not U_r.free_symbols - {t}, "the datum depends on theta only"


def test_the_fault_normal_datum_is_common_to_both_faces(mesh):
    """The datum a split-node model imposes on each face.

    It is `u_theta` on theta = 0, and the SAME expression on theta = 2 pi — that
    equality is fault condition 2, and it is why one datum serves both faces.
    """

    sol = uw.analytic.FaultedMedium(mesh, U0=1.0, R0=1.5, eta=1.0)
    r, t = sol.symbols
    _u_r, u_t = sol.velocity_polar

    datum = sol.fault_normal_velocity()

    assert sympy.simplify(datum - u_t.subs(t, 0)) == 0
    assert sympy.simplify(datum - u_t.subs(t, 2 * sympy.pi)) == 0
    assert not datum.free_symbols - {r}, "the datum depends on r only"


def test_a_degenerate_geometry_is_refused(mesh):
    with pytest.raises(ValueError, match="positive"):
        uw.analytic.FaultedMedium(mesh, R0=0.0)
    with pytest.raises(ValueError, match="positive"):
        uw.analytic.FaultedMedium(mesh, eta=-1.0)

    # The PRESSURE is singular at the tip and refuses; the VELOCITY is defined
    # there — every term carries a positive power of r — and is zero.
    sol = uw.analytic.FaultedMedium(mesh)
    with pytest.raises(ValueError, match="singular at the fault tip"):
        sol.evaluate_pressure(np.array([[0.0, 0.0]]))
    assert np.allclose(sol.evaluate_velocity(np.array([[0.0, 0.0]])), 0.0)

    symbolic = uw.analytic.FaultedMedium(mesh, U0=sympy.Symbol("U_0", positive=True))
    with pytest.raises(ValueError, match="symbolic parameters"):
        symbolic.evaluate_velocity(np.array([[0.5, 0.1]]))
    with pytest.raises(ValueError, match="symbolic parameters"):
        symbolic.slip(0.5)


def test_the_traction_reproduces_the_zero_shear_fault_condition(mesh):
    """The traction machinery must independently give tau_r_theta = 0 on the fault.

    `evaluate_traction` builds the stress by a different route from the symbolic
    fault-condition test — SymPy-derived strain rates lambdified and rotated into
    Cartesian — so agreeing with it is a genuine cross-check rather than a
    restatement.

    On the fault the outward normal of the upper face is -theta_hat, i.e. (0, -1)
    in Cartesian along theta = 0. The SHEAR part of that traction is its
    x-component, and it is the quantity the paper sets to zero.
    """

    sol = uw.analytic.FaultedMedium(mesh)
    x = np.array([0.1, 0.25, 0.4, 0.6])
    on_fault = np.column_stack([x, np.zeros_like(x)])

    traction = sol.evaluate_traction(on_fault, [0.0, -1.0])
    assert np.allclose(traction[:, 0], 0.0, atol=1e-10), (
        f"shear traction on the fault is {traction[:, 0]}, not zero"
    )

    # Negative control: off the fault it is emphatically NOT zero, so the
    # assertion above is not passing for a trivial reason.
    off_fault = np.column_stack([x, np.full_like(x, 0.15)])
    assert np.abs(sol.evaluate_traction(off_fault, [0.0, -1.0])[:, 0]).max() > 0.1

    with pytest.raises(ValueError, match="singular at the fault tip"):
        sol.evaluate_traction(np.array([[0.0, 0.0]]), [1.0, 0.0])


# ---------------------------------------------------------------------------
# Joining the family: the same field in the mesh coordinates
# ---------------------------------------------------------------------------


def test_the_cartesian_fields_are_the_fault_frame_ones(mesh):
    """`fn_velocity` and `fn_pressure` must be the polar solution, rotated.

    The contract's fields are a second representation of the same solution, so
    they need pinning to the first — this is the join the integration adds, and
    the one place a rotation or a branch-cut slip could enter unseen.
    """

    sol = uw.analytic.FaultedMedium(mesh, U0=1.3, R0=2.0, eta=0.7)
    r_sym, t_sym = sol.symbols
    u_r_sym, u_t_sym = sol.velocity_polar
    p_sym = sol.pressure_polar

    points = sol.sample_points(count=8)
    radii = np.hypot(points[:, 0], points[:, 1])
    angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)

    for name, expression, reference in (
        ("velocity_x", sol.fn_velocity[0, 0], None),
        ("velocity_y", sol.fn_velocity[0, 1], None),
        ("pressure", sol.fn_pressure, p_sym),
    ):
        mine = _validation.sample(sol, expression, points)

        for k, (rr, tt) in enumerate(zip(radii, angles)):
            subs = {r_sym: float(rr), t_sym: float(tt)}
            if reference is None:
                ur = float(u_r_sym.subs(subs))
                ut = float(u_t_sym.subs(subs))
                expect = (
                    ur * np.cos(tt) - ut * np.sin(tt)
                    if name == "velocity_x"
                    else ur * np.sin(tt) + ut * np.cos(tt)
                )
            else:
                expect = float(reference.subs(subs))

            assert mine[k] == pytest.approx(expect, rel=1e-9, abs=1e-11), name


def test_the_residual_gates_hold_with_the_tip_off_the_origin(mesh):
    """The oracle-free gates, at a tip the conformance sweep never uses.

    The sweep builds every solution from its defaults, and the default tip is
    the origin — where a dropped offset is invisible. Moving it is this
    solution's equivalent of moving a viscosity off unity.
    """

    sol = uw.analytic.FaultedMedium(mesh, U0=1.3, R0=2.0, eta=0.7, tip=(0.3, -0.2))
    points = sol.sample_points(count=8)

    assert _validation.incompressibility_residual(sol, points) < 1.0e-8
    assert _validation.momentum_residual(sol, points) < 1.0e-8
    assert _validation.strainrate_consistency(sol, points) < 1.0e-8

    # The offset is real: the fields are not the same function of position.
    at_origin = uw.analytic.FaultedMedium(mesh, U0=1.3, R0=2.0, eta=0.7)
    moved = _validation.sample(sol, sol.fn_velocity[0, 0], points)
    unmoved = _validation.sample(at_origin, at_origin.fn_velocity[0, 0], points)
    assert np.abs(moved - unmoved).max() > 0.1 * np.abs(moved).max()


def test_the_sample_points_avoid_the_tip_and_land_on_both_faces(mesh):
    """The override is what keeps the gates off the two places this field is not
    a function of position, and on the one place it is hardest.
    """

    sol = uw.analytic.FaultedMedium(mesh, R0=2.0, tip=(0.3, -0.2))
    points = sol.sample_points(count=8)

    offset = points - sol.tip
    radii = np.hypot(offset[:, 0], offset[:, 1])
    angles = np.mod(np.arctan2(offset[:, 1], offset[:, 0]), 2.0 * np.pi)

    assert radii.min() > 0.0
    assert radii.max() <= sol.R0 * (1.0 + 1.0e-12)

    # One point just above the fault and one just below it.
    assert angles.min() < 1.0e-5
    assert angles.max() > 2.0 * np.pi - 1.0e-5


def test_it_is_registered_in_the_family(mesh):
    """Listed, described, and declaring the same things the others declare."""

    assert "FaultedMedium" in uw.analytic.available()
    assert uw.analytic.is_available("FaultedMedium")
    assert "Barr" in uw.analytic.describe("FaultedMedium")

    assert uw.analytic.FaultedMedium.solves == "stokes"
    assert uw.analytic.FaultedMedium.symbolic is True
    assert uw.analytic.FaultedMedium.expensive_to_validate is False
    assert uw.analytic.FaultedMedium.dim == 2
    assert uw.analytic.FaultedMedium.reference


def test_it_refuses_to_pretend_the_fault_is_a_wall(mesh):
    """The one member of the family that cannot use the mixins says so.

    Applying the perimeter datum and quietly leaving the fault unconstrained
    would solve a different problem and report a plausible error, which is worse
    than refusing.
    """

    sol = uw.analytic.FaultedMedium(mesh)

    assert sol.boundaries == ["Perimeter", "FaultUpper", "FaultLower"]

    velocity = uw.discretisation.MeshVariable("Ubh", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("Pbh", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)

    with pytest.raises(NotImplementedError, match="slit disc"):
        sol.apply_boundary_conditions(stokes)


def test_it_has_no_body_force(mesh):
    """Boundary-driven, like the elliptical inclusion — and for the same reason
    it is excluded from the body-force negative control in the sweeps.
    """

    sol = uw.analytic.FaultedMedium(mesh)

    assert all(component == 0 for component in sol.fn_bodyforce)


def test_flipping_the_pressure_sign_breaks_the_momentum_balance(mesh):
    """The negative control this solution gets INSTEAD of the body-force one.

    The family pins its body-force convention by flipping the force and watching
    the momentum residual go to order unity. There is no force here to flip, so
    the gate would otherwise be an assertion that a small number is small.

    The pressure sign is what it certifies instead, and it certifies it just as
    sharply: with the paper's extension-positive pressure the residual is 1.06,
    with UW3's compression-positive one it is 3.6e-16. Note which gate fires —
    `tr(sigma) + d p` is 5.7e-16 either way, because `set_fields` builds the
    stress from the pressure and tracelessness cannot see the sign of a term it
    cancels by construction.
    """

    sol = uw.analytic.FaultedMedium(mesh)
    points = sol.sample_points(count=8)

    assert _validation.momentum_residual(sol, points) < 1.0e-8

    flipped = uw.analytic.FaultedMedium(mesh)
    flipped.set_fields(
        velocity=list(sol.fn_velocity),
        pressure=-sol.fn_pressure,
        viscosity=sol.fn_viscosity,
        bodyforce=(0, 0),
        strainrate=sol.fn_strainrate.tolist(),
    )

    assert _validation.momentum_residual(flipped, points) > 1.0e-2, (
        "the momentum gate cannot see the pressure sign it is supposed to "
        "certify for this solution"
    )
