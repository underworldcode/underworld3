"""The backward pass over a recorded run, against finite differences.

A two-solver, multi-step problem in the shape of the sinking blob: a level
set ``beta`` carried by an SUPG advection–diffusion solve in a velocity
``v``, and a Stokes solve whose body force is ``-Ra * beta``. The misfit is
on the final velocity. ``uw.adjoint.TranscriptAdjoint`` walks the transcript
backwards with no problem-specific wiring: the residuals say what each solve
reads, the transcript says what ran and holds the state each step started
from.

Two controls, two checks: a scalar parameter (the viscosity) against a
central finite difference, and the initial level set as a FIELD control —
the dual on beta_0 dotted with a perturbation direction against the finite
difference of J along that direction.
"""

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


def _build():
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    x, y = mesh.X
    beta = uw.discretisation.MeshVariable("beta", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    eta0 = uw.expression(r"\eta_0", 1.0, "viscosity")
    Ra = uw.expression(r"Ra", 50.0, "buoyancy number")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta0
    stokes.bodyforce = sympy.Matrix([0.0, -Ra * beta.sym[0]])
    for b in ("Top", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    for b in ("Left", "Right"):
        stokes.add_dirichlet_bc((0.0, sympy.oo), b)
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.tolerance = 1.0e-12

    adv = uw.systems.AdvDiffusion(mesh, u_Field=beta, V_fn=v.sym, theta=1.0)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    adv.petsc_options.delValue("ksp_monitor")
    adv.tolerance = 1.0e-12

    def beta0(centre=(0.5, 0.6)):
        X = np.asarray(beta.coords)
        r = np.sqrt((X[:, 0] - centre[0]) ** 2 + (X[:, 1] - centre[1]) ** 2)
        return r - 0.2

    return dict(uw=uw, model=model, mesh=mesh, beta=beta, v=v, p=p, eta0=eta0,
                Ra=Ra, stokes=stokes, adv=adv, beta0=beta0)


DT, NSTEPS = 0.02, 2


def _forward(m, beta_initial):
    """Run from ``beta_initial`` and return (final_state, J)."""
    uw, model = m["uw"], m["model"]
    model.clear_transcript()
    model.tracker.time, model.tracker.step = 0.0, 0
    model.record_every = 1
    m["beta"].array[:, 0, 0] = beta_initial
    m["v"].array[...] = 0.0
    m["p"].array[...] = 0.0
    # The Eulerian history initialises itself only on its FIRST solve; a
    # driver that runs the forward model more than once must reset it each
    # time the initial condition is set, or the second run reads the first
    # run's history.
    m["adv"].DuDt.initialise_history()
    # Every solve inside a step, so every solve is on the tape: a Stokes
    # solve taken before the first step would carry beta_0 -> v_0 with no
    # record, and the walk could not see its dependence on the viscosity.
    for _ in range(NSTEPS):
        with model.step(DT):
            m["stokes"].solve(zero_init_guess=True)     # v_k from beta_k
            m["adv"].solve(timestep=DT, zero_init_guess=False)
    return model.save_state(), _misfit(m)


def _misfit_expr(m):
    v = m["v"]
    return sympy.Rational(1, 2) * v.sym.dot(v.sym)


def _misfit(m):
    return float(m["uw"].maths.Integral(m["mesh"], _misfit_expr(m)).evaluate())


def test_parameter_gradient_matches_finite_differences():
    m = _build()
    uw, eta0 = m["uw"], m["eta0"]
    b0 = m["beta0"]()

    final, J = _forward(m, b0)
    result = uw.adjoint.TranscriptAdjoint(m["model"], final).gradient(
        _misfit_expr(m), parameters=[eta0])
    assert result["J"] == pytest.approx(J, rel=1e-10)
    adjoint = result["parameters"][eta0]

    h = 1.0e-4
    eta0.sym = sympy.Float(1.0 + h)
    _, Jp = _forward(m, b0)
    eta0.sym = sympy.Float(1.0 - h)
    _, Jm = _forward(m, b0)
    eta0.sym = sympy.Float(1.0)
    fd = (Jp - Jm) / (2 * h)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)


def test_initial_field_gradient_matches_a_directional_finite_difference():
    m = _build()
    uw, beta = m["uw"], m["beta"]
    b0 = m["beta0"]()

    final, J = _forward(m, b0)
    result = uw.adjoint.TranscriptAdjoint(m["model"], final).gradient(
        _misfit_expr(m), fields=[beta])
    dual = result["fields"][beta][:, 0, 0]

    # a smooth direction in beta_0, and J along it
    X = np.asarray(beta.coords)
    direction = np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    h = 1.0e-3
    _, Jp = _forward(m, b0 + h * direction)
    _, Jm = _forward(m, b0 - h * direction)
    fd = (Jp - Jm) / (2 * h)
    # over the OWNED degrees of freedom: a NumPy dot on .array counts the
    # ghost nodes of a partition twice (found in review at np=2)
    adjoint = uw.adjoint.inner(beta, dual, direction)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)


def test_a_misfit_that_names_the_parameter_gets_its_explicit_term():
    """dJ/dm = the implicit part through the solves plus dJ/dm at the final
    level for a misfit written in terms of m (found in review: omitted)."""
    m = _build()
    uw, eta0 = m["uw"], m["eta0"]
    b0 = m["beta0"]()
    misfit = eta0 * _misfit_expr(m)

    final, J = _forward(m, b0)
    adjoint = uw.adjoint.TranscriptAdjoint(m["model"], final).gradient(
        misfit, parameters=[eta0])["parameters"][eta0]

    def J_at(value):
        eta0.sym = sympy.Float(value)
        _forward(m, b0)
        return float(uw.maths.Integral(m["mesh"], misfit).evaluate())

    h = 1.0e-4
    fd = (J_at(1.0 + h) - J_at(1.0 - h)) / (2 * h)
    eta0.sym = sympy.Float(1.0)
    assert adjoint == pytest.approx(fd, rel=1.0e-4), (adjoint, fd)


def test_gradient_reuses_its_scratch_fields():
    """Sixteen registered variables leaked per call and the sixth call took
    ten times the first (found in review)."""
    m = _build()
    uw, eta0 = m["uw"], m["eta0"]
    final, _ = _forward(m, m["beta0"]())
    back = uw.adjoint.TranscriptAdjoint(m["model"], final)
    back.gradient(_misfit_expr(m), parameters=[eta0])
    n_after_first = len(m["model"]._variables)
    for _ in range(3):
        back.gradient(_misfit_expr(m), parameters=[eta0])
    assert len(m["model"]._variables) == n_after_first
