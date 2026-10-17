"""stretch_in_momentum: the momentum's share of the viscoelastic stretch, and a switch between steps.

The deformation step stretches the stored conformation by F = I + dt L with the
new velocity gradient. ``"implicit"`` puts the whole stretch in the momentum
residual, which is quadratic in L; ``"linear"`` leaves out its O(dt^2 L^2) terms;
``"tangent"`` replaces them by their tangent about the last step's gradient L^n.
Every form stores the same thing: the full deformation step with the converged
velocity. So the form can change between steps with no state to carry.

Checked here, against exact properties: the momentum flux is quadratic in L under
"implicit" and not under "linear" or "tangent"; "tangent" equals "implicit" at
L = L^n and "linear" does not; every form stores the same stress; a switch between
steps rebuilds the compiled residual (quadratic before, not after) and the solve
records the form it ran; "linear" at a step twice the relaxation time keeps the
stored conformation positive-definite; a form the model cannot honour is refused.
"""

import numpy as np
import pytest

import sympy

import underworld3 as uw
from underworld3.function._function import diff_wrt_field

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

ETA, LAM = 0.01, 0.05
G = 0.5 * ETA / LAM


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _channel(stretch):
    """Oldroyd-B channel flow (parabolic inflow, no-slip walls), Eulerian SUPG
    momentum and stress history, the ETD deformation step."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), cellSize=0.15, qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    x, y = mesh.X
    ns.add_dirichlet_bc((4.0 * y * (1 - y), 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.stress_transport = "eulerian"
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        convected_step="deformation", stretch_in_momentum=stretch)
    cm = ns.constitutive_model
    cm.Parameters.shear_viscosity_0 = 0.5 * ETA
    cm.Parameters.shear_modulus = G
    cm.Parameters.solvent_viscosity = 0.5 * ETA
    ns.devss_viscosity = ETA
    return ns, cm


def _steps(built, schedule):
    """Build the channel with ``built``, then take one step of twice the relaxation
    time per entry of ``schedule``, switching to that form first (None keeps it).
    Returns the velocity, the stored stress and the form each solve recorded."""
    uw.reset_default_model()
    ns, cm = _channel(built)
    dt = 2 * LAM
    forms = []
    for form in schedule:
        if form is not None:
            cm.stretch_in_momentum = form
        cm.Parameters.dt_elastic = dt
        ns.solve(timestep=dt, zero_init_guess=False)
        assert ns.snes.getConvergedReason() > 0
        forms.append(ns.solve_report.config["tangent"].get("stretch_in_momentum", "implicit"))
    velocity = np.asarray(ns.Unknowns.u.array[...]).copy()
    stored = np.asarray(ns.DFDt.psi_star[0].array[...]).copy()
    return velocity, stored, forms


def _quadratic_in_L(expr, L):
    """Whether ``expr`` has a non-zero second derivative in the velocity gradient."""
    M = sympy.Matrix(expr)
    for i in range(2):
        for j in range(2):
            d = sympy.Matrix(diff_wrt_field(M, L[i, j]))
            for k in range(2):
                for l in range(2):
                    if any(sympy.simplify(e) != 0 for e in diff_wrt_field(d, L[k, l])):
                        return True
    return False


def test_the_forms_are_what_they_say():
    ns, cm = _channel("implicit")
    cm.Parameters.dt_elastic = 0.1
    L = ns.Unknowns.L
    implicit = sympy.Matrix(cm.flux)
    assert _quadratic_in_L(implicit, L)
    cm.stretch_in_momentum = "linear"
    linear = sympy.Matrix(cm.flux)
    assert not _quadratic_in_L(linear, L)
    cm.stretch_in_momentum = "tangent"
    tangent = sympy.Matrix(cm.flux)
    assert not _quadratic_in_L(tangent, L)
    # the tangent touches the quadratic terms at the last step's gradient; the
    # linear form does not
    Ln = cm._last_velocity_gradient()
    at_n = {L[i, j]: Ln[i, j] for i in range(2) for j in range(2)}
    assert all(sympy.simplify(e) == 0 for e in (implicit - tangent).xreplace(at_n))
    assert not all(sympy.simplify(e) == 0 for e in (implicit - linear).xreplace(at_n))


def test_every_form_stores_the_same_stress():
    ns, cm = _channel("implicit")
    cm.Parameters.dt_elastic = 0.1
    implicit = cm.history_flux
    cm.stretch_in_momentum = "linear"
    assert cm.history_flux == implicit


def test_linear_converges_at_a_long_step_and_keeps_the_conformation():
    _, stored, forms = _steps("linear", [None] * 4)
    assert forms == ["linear"] * 4
    stored = stored.reshape(-1, 2, 2)
    c = 0.5 * (stored + np.transpose(stored, (0, 2, 1))) / G + np.eye(2)
    assert np.abs(stored).max() > 1.0e-3 * G          # the flow has loaded the memory
    assert np.linalg.eigvalsh(c)[:, 0].min() > 0.0


def test_a_switch_between_steps_rebuilds_the_compiled_residual():
    """An implicit step, then a switch: the residual the solver compiled is
    quadratic in L before and not after, and each solve records its form."""
    uw.reset_default_model()
    ns, cm = _channel("implicit")
    L = ns.Unknowns.L
    dt = 2 * LAM
    cm.Parameters.dt_elastic = dt
    ns.solve(timestep=dt, zero_init_guess=False)
    assert ns.solve_report.config["tangent"].get("stretch_in_momentum", "implicit") == "implicit"
    assert _quadratic_in_L(ns._u_F1, L)
    cm.stretch_in_momentum = "linear"
    ns.solve(timestep=dt, zero_init_guess=False)
    assert ns.snes.getConvergedReason() > 0
    assert ns.solve_report.config["tangent"]["stretch_in_momentum"] == "linear"
    assert not _quadratic_in_L(ns._u_F1, L)


def test_a_form_the_model_cannot_honour_is_refused():
    ns, cm = _channel("implicit")
    with pytest.raises(ValueError):
        cm.stretch_in_momentum = "split"
    assert cm.stretch_in_momentum == "implicit"
    # outside the deformation step there are no second-order terms to drop
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    v = uw.discretisation.MeshVariable("U_r", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_r", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    with pytest.raises(ValueError):
        uw.constitutive_models.ViscoElasticPlasticFlowModel(
            stokes.Unknowns, order=1, objective_rate="upper_convected", convected_step="linear",
            stretch_in_momentum="linear")
    # the semi-Lagrangian Navier-Stokes builds its flux from the stress history,
    # which carries the whole deformation step: a form would change nothing there
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("U_sl", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_sl", mesh, 1, degree=1)
    ns = uw.systems.NavierStokesSLCN(mesh, velocityField=v, pressureField=p, rho=1.0, order=1)
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        convected_step="deformation", stretch_in_momentum="linear")
    ns.constitutive_model.Parameters.dt_elastic = 2 * LAM
    ns.bodyforce = sympy.Matrix([0, 0]).T
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    for wall in ("Bottom", "Left", "Right"):
        ns.add_dirichlet_bc((0.0, 0.0), wall)
    with pytest.raises(NotImplementedError):
        ns.solve(timestep=2 * LAM)
