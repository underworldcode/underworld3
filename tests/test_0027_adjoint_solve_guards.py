"""What ``adjoint_solve`` refuses, and the accuracy it holds its linear solve to.

* #769: the adjoint reused the forward SNES's KSP, so under Eisenstat-Walker
  it solved to EW's ``rtol`` (0.3 on its first solve) and reported converged.
  The adjoint has no outer iteration to correct that, so every sensitivity
  came out short by a fixed fraction (10% here). Only an ITERATIVELY solved
  adjoint shows it: on a small box the default Stokes setup solves in one
  outer iteration, so the velocity block is weakened here to make it iterate.
* #781: a target on the wrong space (a P1 pressure adjoint for a P0
  forward pressure) was a bare numpy broadcast error after the solve.
* #804: the adjoint of an unconverged forward solve returned a wrong number
  with no warning.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _stokes(tag, viscosity, cell=1 / 8, p_degree=1, p_continuous=True):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=cell, qdegree=3)
    x, y = mesh.X
    V = uw.discretisation.MeshVariable(f"V_{tag}", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=p_degree,
                                       continuous=p_continuous)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity(stokes)
    stokes.bodyforce = sympy.Matrix([0.0, -sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sympy.oo), b)
    stokes.petsc_options.delValue("ksp_monitor")
    return mesh, stokes, V, P


def test_an_iterating_adjoint_is_not_held_to_the_forward_newtons_inexact_tolerance():
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    mesh, stokes, V, P = _stokes("ew", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2), cell=1 / 12)
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-8
    # a weak velocity block, so the outer Krylov solve has to iterate
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "preonly"
    stokes.petsc_options["fieldsplit_velocity_pc_type"] = "jacobi"
    stokes.petsc_options["ksp_max_it"] = 5000
    misfit = sympy.Rational(1, 2) * V.sym.dot(V.sym)

    def J(value):
        eta0.sym = float(value)
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged, stokes.solve_report
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J(1.0)
    assert stokes.snes.getUseEW()                 # the forward solve is inexact Newton
    ksp = stokes.snes.getKSP()
    before = ksp.getTolerances()
    adjoint = stokes.gradient(misfit, parameters=[eta0])["parameters"][eta0]
    its = ksp.getIterationNumber()
    # put back for the next forward solve
    assert stokes.snes.getUseEW()
    assert ksp.getTolerances() == before
    assert its > 10, f"the adjoint solved in {its} iterations; this test needs one that iterates"

    h = 1.0e-4
    fd = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    # at EW's rtol the ratio was 0.90
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)


def test_a_target_on_the_wrong_space_is_refused_before_the_solve_naming_both_spaces():
    mesh, stokes, V, P = _stokes("p0", lambda s: 1.0, p_degree=0, p_continuous=False)
    stokes.petsc_options["fieldsplit_pressure_pc_type"] = "jacobi"
    stokes.solve(zero_init_guess=True)
    u_adj = uw.discretisation.MeshVariable("u_adj_p0", mesh, 2, degree=2)
    lam = uw.discretisation.MeshVariable("lam_p1", mesh, 1, degree=1)     # the default
    b = -stokes.dual_of(V.sym)
    with pytest.raises(ValueError, match=r"lam_p1.*\(1, True, 1\).*\(0, False, 1\)"):
        stokes.adjoint_solve(b, target=(u_adj, lam))
    # the right space goes through
    lam0 = uw.discretisation.MeshVariable("lam_p0", mesh, 1, degree=0, continuous=False)
    _, reason = stokes.adjoint_solve(b, target=(u_adj, lam0))
    assert reason > 0


def test_a_single_field_target_on_the_wrong_space_is_refused():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    T = uw.discretisation.MeshVariable("T_tgt", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.solve()
    wrong = uw.discretisation.MeshVariable("mu_p1", mesh, 1, degree=1)
    with pytest.raises(ValueError, match=r"mu_p1"):
        poisson.adjoint_solve(-poisson.dual_of(T.sym[0]), target=wrong)


def test_the_adjoint_of_an_unconverged_solve_is_refused():
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    mesh, stokes, V, P = _stokes("unc", lambda s: eta0 / (1 + 40 * s.Unknowns.Einv2))
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-10
    misfit = sympy.Rational(1, 2) * V.sym.dot(V.sym)
    stokes.petsc_options["snes_max_it"] = 1
    stokes.solve(zero_init_guess=True)
    assert not stokes.solve_report.converged
    with pytest.raises(RuntimeError, match="did not converge"):
        stokes.gradient(misfit, parameters=[eta0])
    # and once it converges the adjoint is available again
    stokes.petsc_options["snes_max_it"] = 50
    stokes.solve(zero_init_guess=True)
    assert stokes.solve_report.converged
    assert stokes.gradient(misfit, parameters=[eta0])["parameters"][eta0] < 0


def _iterating_matrix_route():
    """The #769 setup on the MATRIX route: a natural condition sends the adjoint
    through _solve_transposed on the solver's own KSP, where nothing rebuilds the
    KSP afterwards, so the restore of its tolerances and EW is all that stops a
    leak into the next forward solve (base left EW's 0.3 there)."""
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    mesh, stokes, V, P = _stokes("mat", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2), cell=1 / 12)
    stokes.add_natural_bc((0.0, 0.0), "Right")
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "preonly"
    stokes.petsc_options["fieldsplit_velocity_pc_type"] = "jacobi"
    stokes.petsc_options["ksp_max_it"] = 5000
    misfit = sympy.Rational(1, 2) * V.sym.dot(V.sym)
    return mesh, stokes, V, eta0, misfit


def test_on_the_matrix_route_the_adjoint_is_held_tight_and_the_ksp_is_put_back():
    mesh, stokes, V, eta0, misfit = _iterating_matrix_route()
    assert not stokes._adjoint_by_kernels()           # the route this test is about

    def J(value):
        eta0.sym = float(value)
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged, stokes.solve_report
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J(1.0)
    ksp = stokes.snes.getKSP()
    before, ew_before = ksp.getTolerances(), stokes.snes.getUseEW()
    in_force = []
    ksp.setMonitor(lambda k, it, rnorm: in_force.append(k.getTolerances()[0]))
    adjoint = stokes.gradient(misfit, parameters=[eta0])["parameters"][eta0]
    ksp.cancelMonitor()
    assert in_force, "the adjoint did not solve on the solver's KSP"
    assert max(in_force) <= 0.1 * stokes.tolerance * (1 + 1e-12), max(in_force)
    assert ksp.getTolerances() == before
    assert stokes.snes.getUseEW() == ew_before

    h = 1.0e-4
    fd = (J(1.0 + h) - J(1.0 - h)) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)


def test_a_raise_inside_the_adjoint_solve_still_puts_the_ksp_back():
    mesh, stokes, V, eta0, misfit = _iterating_matrix_route()
    stokes.solve(zero_init_guess=True)
    its_cold = stokes.solve_report.nl_its
    ksp = stokes.snes.getKSP()
    before, ew_before = ksp.getTolerances(), stokes.snes.getUseEW()

    def boom(k, it, rnorm):
        if it == 1:
            raise RuntimeError("raised inside the adjoint KSP")

    ksp.setMonitor(boom)
    with pytest.raises(Exception):
        stokes.gradient(misfit, parameters=[eta0])
    ksp.cancelMonitor()
    assert ksp.getTolerances() == before
    assert stokes.snes.getUseEW() == ew_before
    stokes.solve(zero_init_guess=True)
    assert stokes.solve_report.converged
    assert stokes.solve_report.nl_its == its_cold


def test_a_forward_solve_that_raises_leaves_no_adjoint_to_take():
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    mesh, stokes, V, P = _stokes("raise", lambda s: eta0 / (1 + 4 * s.Unknowns.Einv2))
    stokes.consistent_jacobian = True
    misfit = sympy.Rational(1, 2) * V.sym.dot(V.sym)
    stokes.solve(zero_init_guess=True)
    stokes.gradient(misfit, parameters=[eta0])          # converged: accepted

    def boom():
        raise RuntimeError("the forward solve raised")

    eta0.sym = 2.0
    stokes._snapshot_numerics = boom
    try:
        with pytest.raises(RuntimeError, match="forward solve raised"):
            stokes.solve(zero_init_guess=True)
    finally:
        del stokes._snapshot_numerics
    with pytest.raises(RuntimeError, match="did not converge"):
        stokes.gradient(misfit, parameters=[eta0])


def test_a_restored_state_without_a_solution_admits_no_adjoint():
    from underworld3.checkpoint.state import SolverStepState
    eta0 = uw.expression(r"\eta_0", 1.0, "prefactor")
    mesh, stokes, V, P = _stokes("restore", lambda s: eta0)
    misfit = sympy.Rational(1, 2) * V.sym.dot(V.sym)
    stokes.solve(zero_init_guess=True)
    stokes.state = SolverStepState(has_solution=False)
    with pytest.raises(RuntimeError, match="did not converge"):
        stokes.gradient(misfit, parameters=[eta0])
