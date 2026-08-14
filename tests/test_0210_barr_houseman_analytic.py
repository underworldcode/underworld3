"""Barr & Houseman (1996) faulted-medium solution — verified as a solution.

These tests do not compare the expressions against a stored answer or against
a UW3 solve. They check that the field IS a Stokes solution satisfying the
fault conditions, symbolically:

    div u = 0
    eta * lap(u) + grad(p) = 0        (their eq 3; extension-positive pressure)
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

The transcription needed it: the half-integer sine terms of u_theta appear
with one sign in the paper's boundary datum (A8b) and the opposite sign in its
solution (A9b). Incompressibility settles it — for u_r = A sqrt(R) f(theta)
and u_theta = sqrt(R) g(theta), div u = 0 forces g' = -(3/2) A f, which is
(A8b)'s sign. The test below would fail on the other choice.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.analytic import BarrHouseman

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _polar_operators(u_r, u_t, p, r, t, eta):
    """(div u, momentum_x, momentum_y) for a field given in polar form."""
    div = sympy.diff(r * u_r, r) / r + sympy.diff(u_t, t) / r

    u_x = u_r * sympy.cos(t) - u_t * sympy.sin(t)
    u_y = u_r * sympy.sin(t) + u_t * sympy.cos(t)

    def lap(f):
        return sympy.diff(r * sympy.diff(f, r), r) / r + sympy.diff(f, t, 2) / r**2

    p_r, p_t = sympy.diff(p, r), sympy.diff(p, t)
    grad_p_x = p_r * sympy.cos(t) - p_t * sympy.sin(t) / r
    grad_p_y = p_r * sympy.sin(t) + p_t * sympy.cos(t) / r
    return div, eta * lap(u_x) + grad_p_x, eta * lap(u_y) + grad_p_y


def test_the_field_is_an_incompressible_stokes_solution():
    """div u = 0 and the momentum balance vanishes identically.

    The parameters are left SYMBOLIC on purpose. With U0 = R0 = eta = 1 a
    transcription carrying the wrong power of R0 in the pressure satisfies the
    identity and still fails the physics, so the unit-parameter version of this
    test is strictly weaker for the same runtime.
    """
    U0, R0, eta = sympy.symbols("U_0 R_0 eta", positive=True)
    sol = BarrHouseman(U0=U0, R0=R0, eta=eta)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar
    p = sol.pressure_polar

    div, mom_x, mom_y = _polar_operators(u_r, u_t, p, r, t, sol.eta)
    assert sympy.simplify(div) == 0
    assert sympy.simplify(mom_x) == 0
    assert sympy.simplify(mom_y) == 0


def test_the_fault_conditions_hold_on_both_faces():
    """All THREE of the paper's fault conditions, with symbolic parameters.

    Zero shear traction, continuous normal velocity, continuous normal stress.
    The third was claimed in the original description and not asserted.
    """
    U0, R0, eta = sympy.symbols("U_0 R_0 eta", positive=True)
    sol = BarrHouseman(U0=U0, R0=R0, eta=eta)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar

    tau_rt = sol.eta * (r * sympy.diff(u_t / r, r) + sympy.diff(u_r, t) / r)
    assert sympy.simplify(tau_rt.subs(t, 0)) == 0
    assert sympy.simplify(tau_rt.subs(t, 2 * sympy.pi)) == 0

    # u_theta is the fault-NORMAL component and must not jump; u_r is the
    # fault-parallel one and must (that jump is the slip).
    assert sympy.simplify(u_t.subs(t, 0) - u_t.subs(t, 2 * sympy.pi)) == 0

    # Normal STRESS continuity — the third of the paper's three fault
    # conditions. Extension-positive, so sigma = tau + p I, and the
    # fault-normal component is sigma_tt = 2 eta e_tt + p with
    # e_tt = (1/r) du_theta/dtheta + u_r/r.
    p = sol.pressure_polar
    e_tt = sympy.diff(u_t, t) / r + u_r / r
    sigma_tt = 2 * sol.eta * e_tt + p
    assert sympy.simplify(sigma_tt.subs(t, 0)
                          - sigma_tt.subs(t, 2 * sympy.pi)) == 0


def test_the_slip_is_the_published_normalisation():
    """slip = 2 U0 sqrt(r/R0), so 2 U0 at the perimeter — the paper's anchor.

    Negative control: the whole-integer modes alone are continuous, so a
    solution without the half-integer mode would give zero slip and pass a
    weaker test vacuously.
    """
    sol = BarrHouseman(U0=1.0, R0=1.0, eta=1.0)
    r, t = sol.symbols
    u_r, _u_t = sol.velocity_polar

    jump = sympy.simplify(u_r.subs(t, 0) - u_r.subs(t, 2 * sympy.pi))
    assert sympy.simplify(jump - 2 * sol.U0 * sympy.sqrt(r / sol.R0)) == 0
    assert float(jump.subs(r, sol.R0)) == pytest.approx(2.0 * sol.U0)
    assert float(sol.slip(sol.R0)) == pytest.approx(2.0 * sol.U0)

    # The slip is carried entirely by the half-integer mode: drop the sqrt
    # term and the fault disappears.
    continuous_only = u_r - (sol.U0 / 4) * sympy.sqrt(r / sol.R0) * (
        sympy.cos(t / 2) + 3 * sympy.cos(3 * t / 2))
    assert sympy.simplify(continuous_only.subs(t, 0)
                          - continuous_only.subs(t, 2 * sympy.pi)) == 0


def test_the_numpy_evaluator_agrees_with_the_symbolic_form():
    """The Cartesian evaluator must reproduce the polar expressions.

    Includes points either side of the fault, which is where the branch cut
    lives and where a bare ``atan2`` would silently return the wrong face.
    """
    sol = BarrHouseman(U0=1.3, R0=2.0, eta=0.7)
    r_sym, t_sym = sol.symbols
    u_r_sym, u_t_sym = sol.velocity_polar
    p_sym = sol.pressure_polar

    rng = np.random.default_rng(7)
    radii = rng.uniform(0.2, 1.9, 12)
    angles = np.r_[rng.uniform(0.05, 2 * np.pi - 0.05, 10), 0.02,
                   2 * np.pi - 0.02]
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    velocity, pressure = sol.evaluate(pts)
    for k, (rr, tt) in enumerate(zip(radii, angles)):
        subs = {r_sym: float(rr), t_sym: float(tt)}
        ur = float(u_r_sym.subs(subs))
        ut = float(u_t_sym.subs(subs))
        expect = np.array([ur * np.cos(tt) - ut * np.sin(tt),
                           ur * np.sin(tt) + ut * np.cos(tt)])
        assert velocity[k] == pytest.approx(expect, rel=1e-10, abs=1e-12)
        assert pressure[k] == pytest.approx(float(p_sym.subs(subs)),
                                            rel=1e-10, abs=1e-12)


def test_the_branch_cut_lies_on_the_fault():
    """Straddling the fault must show the slip; straddling +x elsewhere must not.

    This is the test that a bare ``atan2`` fails: it would put the cut on the
    negative x axis, reporting a jump where the medium is continuous and none
    where the fault is.
    """
    sol = BarrHouseman(U0=1.0, R0=1.0, eta=1.0)
    eps = 1e-7

    above, _ = sol.evaluate(np.array([[0.5, +eps]]))
    below, _ = sol.evaluate(np.array([[0.5, -eps]]))
    assert (above[0, 0] - below[0, 0]) == pytest.approx(
        float(sol.slip(0.5)), rel=1e-4), "no slip across the fault"

    left_up, _ = sol.evaluate(np.array([[-0.5, +eps]]))
    left_dn, _ = sol.evaluate(np.array([[-0.5, -eps]]))
    assert np.allclose(left_up, left_dn, atol=1e-5), (
        "the medium is continuous on the fault's projection; a jump here "
        "means the branch cut is in the wrong place")


def test_the_boundary_datum_reproduces_the_solution_at_the_perimeter():
    """The Dirichlet datum a solver would impose is the solution at r = R0."""
    sol = BarrHouseman(U0=1.0, R0=1.5, eta=1.0)
    r, t = sol.symbols
    u_r, u_t = sol.velocity_polar
    U_r, U_t = sol.boundary_velocity()

    assert sympy.simplify(U_r - u_r.subs(r, sol.R0)) == 0
    assert sympy.simplify(U_t - u_t.subs(r, sol.R0)) == 0
    assert not U_r.free_symbols - {t}, "the datum depends on theta only"


def test_a_degenerate_geometry_is_refused():
    with pytest.raises(ValueError, match="positive"):
        BarrHouseman(R0=0.0)
    with pytest.raises(ValueError, match="positive"):
        BarrHouseman(eta=-1.0)
    with pytest.raises(ValueError, match="singular at the fault tip"):
        BarrHouseman().evaluate(np.array([[0.0, 0.0]]))

    symbolic = BarrHouseman(U0=sympy.Symbol("U_0", positive=True))
    with pytest.raises(ValueError, match="symbolic parameters"):
        symbolic.evaluate(np.array([[0.5, 0.1]]))
    with pytest.raises(ValueError, match="symbolic parameters"):
        symbolic.slip(0.5)
