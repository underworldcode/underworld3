"""The discrete adjoint of one solve, checked against finite differences.

``solver.adjoint_solve(b)`` solves :math:`K^T \\mu = b` against the Jacobian
the SNES already assembled; ``solver.sensitivity(mu, m)`` integrates the
symbolic :math:`\\partial R/\\partial m` against it. Together they are the
gradient of a misfit through one implicit solve, with no hand algebra.

The check is the only one that counts: the adjoint gradient against a
central finite difference in the parameter. A symmetric operator (Poisson)
cannot tell a transpose from the operator itself, so the second case is one
SUPG advection–diffusion step, whose Jacobian is not symmetric — a wrong
transpose fails there.
"""

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _fresh():
    import underworld3 as uw

    uw.reset_default_model()
    return uw, uw.get_default_model()


def _mesh(uw):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )


def _misfit(uw, mesh, T):
    return float(uw.maths.Integral(mesh, sympy.Rational(1, 2) * T.sym[0] ** 2).evaluate())


def test_poisson_gradient_in_the_diffusivity_matches_finite_differences():
    """J = 1/2 int T^2 for the Poisson solve with source 1 and diffusivity
    kappa. K^T mu = -dJ/dT, then dJ/dkappa = int (dF1/dkappa) . grad(mu)."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    T = uw.discretisation.MeshVariable("T_adj", mesh, 1, degree=2)
    mu = uw.discretisation.MeshVariable("mu_adj", mesh, 1, degree=2)
    kappa = uw.expression(r"\kappa", 1.0, "diffusivity")

    solver = uw.systems.Poisson(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = kappa
    solver.f = 1.0
    solver.add_dirichlet_bc(0.0, "Top")
    solver.add_dirichlet_bc(0.0, "Bottom")
    solver.petsc_options.delValue("ksp_monitor")
    solver.tolerance = 1.0e-12

    def J_at(value):
        kappa.sym = sympy.Float(value)
        solver.solve(zero_init_guess=True)
        return _misfit(uw, mesh, T)

    J0 = J_at(1.0)
    b = -solver.dual_of(T.sym[0])                 # -dJ/dT as a dual
    _, reason = solver.adjoint_solve(b, target=mu)
    assert reason > 0, reason
    adjoint = solver.sensitivity(mu, kappa)

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    kappa.sym = sympy.Float(1.0)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)
    # the multiplier honours the homogenised Dirichlet conditions
    top = np.abs(np.asarray(mu.coords)[:, 1] - 1.0) < 1.0e-10
    assert np.abs(np.asarray(mu.array)[top, 0, 0]).max() < 1.0e-12


def test_one_supg_step_gradient_matches_finite_differences_where_the_jacobian_is_not_symmetric():
    """One implicit advection–diffusion step from a fixed initial state,
    restored by snapshot before every evaluation so the finite difference
    and the adjoint see the same step. SUPG makes K non-symmetric, so a
    transpose taken the wrong way round fails here and not on Poisson."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_sup", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V_sup", mesh, 2, degree=2)
    mu = uw.discretisation.MeshVariable("mu_sup", mesh, 1, degree=2)
    V.array[:, 0, :] = np.asarray(
        uw.function.evaluate(sympy.Matrix([[-(y - 0.5), (x - 0.5)]]), V.coords)
    ).reshape(-1, 2)
    T.array[:, 0, 0] = np.asarray(
        uw.function.evaluate(sympy.exp(-(((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.02)), T.coords)
    ).ravel()
    kappa = uw.expression(r"\kappa", 1.0e-2, "diffusivity")

    solver = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V.sym)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = kappa
    solver.petsc_options.delValue("ksp_monitor")
    solver.tolerance = 1.0e-12
    dt = 0.05

    start = model.save_state()

    def J_at(value):
        model.load_state(start)
        kappa.sym = sympy.Float(value)
        solver.solve(timestep=dt, zero_init_guess=True)
        return _misfit(uw, mesh, T)

    T_old = np.array(T.array, copy=True)
    J0 = J_at(1.0e-2)
    # The residual of the step is F(T_new; T_old, v, dt). solve() shifted the
    # history forward in its post-hook, so the slot now holds T_new; put the
    # step's INPUT back where the residual reads it before linearising.
    solver.DuDt.psi_star[0].array[...] = T_old
    b = -solver.dual_of(T.sym[0])
    _, reason = solver.adjoint_solve(b, target=mu)
    assert reason > 0, reason
    adjoint = solver.sensitivity(mu, kappa)

    h = 1.0e-5
    fd = (J_at(1.0e-2 + h) - J_at(1.0e-2 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-3), (adjoint, fd)


def test_a_refusing_solve_raises_with_its_reason():
    """A refused verdict must stop ``adjoint_solve`` and carry its own reason,
    rather than let it return a number nobody can trace.

    Driven by a fault contact, the mechanism that still refuses — its rotated
    operator carries an additive interface tangent whose transpose is not routed
    into the adjoint. (Rotated FREE-SLIP used to stand here; it is supported now,
    and ``test_0022_rotated_adjoint.py`` checks its gradient.) The fault list is
    populated directly rather than by splitting a mesh: what is under test is the
    refusal dispatch, and only the length of that list reaches the verdict."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    V = uw.discretisation.MeshVariable("V_ref", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_ref", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes._fault_contact_faults.append("a-fault")
    supported, reason = stokes.adjoint_support()
    assert supported is False
    assert "fault contact" in reason
    with pytest.raises(RuntimeError, match="fault contact"):
        stokes.adjoint_solve(np.zeros(1))


def test_adjoint_before_any_solve_says_to_solve_first():
    uw, model = _fresh()
    mesh = _mesh(uw)
    T = uw.discretisation.MeshVariable("T_none", mesh, 1, degree=2)
    solver = uw.systems.Poisson(mesh, u_Field=T)
    with pytest.raises(RuntimeError, match="solve\\(\\) first"):
        solver.adjoint_solve(np.zeros(1))


def _stokes(uw, mesh, tag, viscosity):
    V = uw.discretisation.MeshVariable(f"V_{tag}", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity(stokes)
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0.0, -sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sympy.oo), b)
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.tolerance = 1.0e-12
    return stokes, V, P


def _kinetic(uw, mesh, V):
    return float(uw.maths.Integral(mesh, sympy.Rational(1, 2) * V.sym.dot(V.sym)).evaluate())


def test_stokes_gradient_in_the_viscosity_matches_finite_differences():
    """Linear viscosity: K is symmetric, and the composite transpose must
    reproduce what the example builds as a second Stokes solver."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    eta0 = uw.expression(r"\eta_0", 1.0, "viscosity")
    stokes, V, P = _stokes(uw, mesh, "lin", lambda s: eta0)
    u_adj = uw.discretisation.MeshVariable("u_adj_lin", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("p_adj_lin", mesh, 1, degree=1)

    def J_at(value):
        eta0.sym = sympy.Float(value)
        stokes.solve(zero_init_guess=True)
        return _kinetic(uw, mesh, V)

    J_at(1.0)
    b = -stokes.dual_of(V.sym)
    _, reason = stokes.adjoint_solve(b, target=(u_adj, p_adj))
    assert reason > 0, reason
    adjoint = stokes.sensitivity(u_adj, eta0)

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)
    # dJ/deta for a viscous flow driven by a fixed body force is negative
    assert adjoint < 0


def test_a_nonlinear_rheology_solved_with_picard_still_gives_the_right_gradient():
    """Picard iterations spoil nothing: the converged state is the same, and
    dR/du is a function of that state alone. What Picard leaves behind is a
    Jacobian KERNEL that is the frozen-viscosity one — so the adjoint
    assembles the consistent tangent itself, and the gradient matches finite
    differences exactly as it does under Newton. The Picard kernel is put
    back for the next forward solve."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    stokes, V, P = _stokes(uw, mesh, "pic", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2))
    stokes.consistent_jacobian = False          # Picard, explicitly
    u_adj = uw.discretisation.MeshVariable("u_adj_pic", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("p_adj_pic", mesh, 1, degree=1)

    def J_at(value):
        eta0.sym = sympy.Float(value)
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged, stokes.solve_report
        return _kinetic(uw, mesh, V)

    J_at(1.0)
    supported, why = stokes.adjoint_support()
    assert supported is True and "Picard" in why, why
    b = -stokes.dual_of(V.sym)
    _, reason = stokes.adjoint_solve(b, target=(u_adj, p_adj))
    assert reason > 0, reason
    adjoint = stokes.sensitivity(u_adj, eta0)
    assert stokes.consistent_jacobian is False   # put back

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-3), (adjoint, fd)


def test_a_nonlinear_rheology_with_the_consistent_tangent_matches_finite_differences():
    """The case the composite transpose exists for: eta(strain rate), where
    the forward operator is not the adjoint operator."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    stokes, V, P = _stokes(uw, mesh, "nl", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2))
    stokes.consistent_jacobian = True
    u_adj = uw.discretisation.MeshVariable("u_adj_nl", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("p_adj_nl", mesh, 1, degree=1)

    def J_at(value):
        eta0.sym = sympy.Float(value)
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged, stokes.solve_report
        return _kinetic(uw, mesh, V)

    J_at(1.0)
    b = -stokes.dual_of(V.sym)
    _, reason = stokes.adjoint_solve(b, target=(u_adj, p_adj))
    assert reason > 0, reason
    adjoint = stokes.sensitivity(u_adj, eta0)

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-3), (adjoint, fd)


def test_a_nonlinear_rheology_under_continuation_matches_finite_differences():
    """"continuation" solves Picard to a loose tolerance, then Newton, then
    puts alpha back to 0. A fresh Jacobian assembly for the adjoint would
    therefore be the PICARD tangent unless alpha is set to 1 for it — the
    solve is right, and the matrix left behind is the wrong one to transpose."""
    uw, model = _fresh()
    mesh = _mesh(uw)
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    stokes, V, P = _stokes(uw, mesh, "cont", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2))
    stokes.consistent_jacobian = "continuation"
    u_adj = uw.discretisation.MeshVariable("u_adj_cont", mesh, 2, degree=2)
    p_adj = uw.discretisation.MeshVariable("p_adj_cont", mesh, 1, degree=1)

    def J_at(value):
        eta0.sym = sympy.Float(value)
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged, stokes.solve_report
        return _kinetic(uw, mesh, V)

    J_at(1.0)
    assert float(stokes._get_newton_alpha().sym) == 0.0     # what the solve leaves behind
    b = -stokes.dual_of(V.sym)
    _, reason = stokes.adjoint_solve(b, target=(u_adj, p_adj))
    assert reason > 0, reason
    adjoint = stokes.sensitivity(u_adj, eta0)
    assert float(stokes._get_newton_alpha().sym) == 0.0     # and is put back

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-3), (adjoint, fd)
