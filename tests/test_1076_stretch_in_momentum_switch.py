"""stretch_in_momentum: the momentum's share of the viscoelastic stretch, and a switch between steps.

The deformation step stretches the stored conformation by F = I + dt L with the
new velocity gradient. ``"implicit"`` puts the whole stretch in the momentum
residual; ``"linear"`` leaves out its O(dt^2 L^2) terms, so the momentum is linear
in the new velocity. Every form stores the same thing: the full deformation step
with the converged velocity. So the form can change between steps with no state
to carry, and a driver can retake a step that Newton cannot finish under
``"implicit"`` with ``"linear"``.

Checked here: both forms store the same stress expression; ``"linear"`` at a step
twice the relaxation time converges and keeps the stored conformation
positive-definite; a model switched to a form steps exactly as one built with it,
both ways; a switch between steps changes the next solve, which records the form
it ran; an unknown form is refused.
"""

import numpy as np
import pytest

import underworld3 as uw

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


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


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


def test_a_switch_rebuilds_the_momentum_kernels():
    """A model switched to a form steps exactly as one built with it, both ways."""
    v_lin, _, f_lin = _steps("linear", [None, None])
    v_to_lin, _, f_to_lin = _steps("implicit", ["linear", None])
    assert _rel(v_to_lin, v_lin) < 1.0e-8, _rel(v_to_lin, v_lin)
    assert f_lin == f_to_lin == ["linear", "linear"]

    v_imp, _, f_imp = _steps("implicit", [None, None])
    v_to_imp, _, _ = _steps("linear", ["implicit", None])
    assert _rel(v_to_imp, v_imp) < 1.0e-8, _rel(v_to_imp, v_imp)
    assert f_imp == ["implicit", "implicit"]


def test_a_switch_between_steps_changes_the_next_solve():
    v_imp, _, _ = _steps("implicit", [None, None])
    v_mid, _, forms = _steps("implicit", [None, "linear"])
    assert forms == ["implicit", "linear"]
    assert _rel(v_mid, v_imp) > 1.0e-10, _rel(v_mid, v_imp)


def test_an_unknown_form_is_refused():
    ns, cm = _channel("implicit")
    with pytest.raises(ValueError):
        cm.stretch_in_momentum = "split"
    assert cm.stretch_in_momentum == "implicit"
