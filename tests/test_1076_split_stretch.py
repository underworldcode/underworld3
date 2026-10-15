"""stretch_in_momentum="split": the stretch of the stored stress leaves the momentum equation.

The deformation step stretches the stored conformation by F = I + dt L. Done
implicitly, the tangent of that stretch carries the stored stress into the
velocity block of the momentum Jacobian, which is indefinite where
dt |sigma*| / eta ~ 1, and Newton stalls there at a large step. The split takes
the stretch with the last step's velocity gradient L^n instead, so the new
velocity enters the flux only through the relaxed viscous term.

Checked here: the momentum flux's derivative with respect to the velocity
gradient holds no stored stress under "split" (and does under "implicit", the
control); a channel run at a step longer than the relaxation time converges at
every step and its stored conformation stays positive-definite; the option is
refused where the deformation step does not apply.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function._function import diff_wrt_field

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

ETA, LAM = 0.01, 0.05


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _channel(stretch):
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
    cm.Parameters.shear_modulus = 0.5 * ETA / LAM
    cm.Parameters.solvent_viscosity = 0.5 * ETA
    ns.devss_viscosity = ETA
    return ns, cm


def _carries_stored_stress(ns, cm):
    """Whether d flux / d L, for any entry of L, holds the stored stress."""
    cm.Parameters.dt_elastic = 0.1
    # the flux as the model writes it: unwrapping for compilation lowers the
    # history variables, so their symbols would no longer be there to look for
    flux = sympy.Matrix(cm.flux)
    stored = [s for s in np.asarray(ns.DFDt.psi_star[0].sym).ravel()]
    L = ns.Unknowns.L
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            d = diff_wrt_field(flux, L[i, j])
            if any(d.has(s) for s in stored):
                return True
    return False


def test_split_takes_the_stored_stress_out_of_the_velocity_tangent():
    ns, cm = _channel("implicit")
    assert _carries_stored_stress(ns, cm)            # the control: the implicit stretch
    uw.reset_default_model()
    ns, cm = _channel("split")
    assert not _carries_stored_stress(ns, cm)


def test_split_converges_at_a_long_step_and_keeps_the_conformation():
    ns, cm = _channel("split")
    dt = 2 * LAM                                       # twice the relaxation time
    G = 0.5 * ETA / LAM
    for _ in range(4):
        cm.Parameters.dt_elastic = dt
        ns.solve(timestep=dt, zero_init_guess=False)
        assert ns.snes.getConvergedReason() > 0
    stored = np.asarray(ns.DFDt.psi_star[0].array[...]).reshape(-1, 2, 2)
    c = 0.5 * (stored + np.transpose(stored, (0, 2, 1))) / G + np.eye(2)
    assert np.abs(stored).max() > 1.0e-3 * G          # the flow has loaded the memory
    assert np.linalg.eigvalsh(c)[:, 0].min() > 0.0


def test_split_needs_the_deformation_step():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    v = uw.discretisation.MeshVariable("U_s", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_s", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.stress_transport = "eulerian"
    with pytest.raises(ValueError):
        uw.constitutive_models.ViscoElasticPlasticFlowModel(
            ns.Unknowns, order=1, integrator="etd", objective_rate="jaumann",
            stretch_in_momentum="split")


def _steps(built, schedule):
    """Build the channel with ``built``, then take one step per entry of
    ``schedule``, switching stretch_in_momentum to that entry first (None keeps it)."""
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
        forms.append(ns._solve_report.config["tangent"].get("stretch_in_momentum", "implicit"))
    return np.asarray(ns.Unknowns.u.array[...]).copy(), forms


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_a_switch_rebuilds_the_momentum_kernels():
    """A model switched to a form steps exactly as one built with it, both ways,
    and each solve records the form it ran."""
    v_lin, f_lin = _steps("linear", [None, None])
    v_to_lin, f_to_lin = _steps("implicit", ["linear", None])
    assert _rel(v_to_lin, v_lin) < 1.0e-8, _rel(v_to_lin, v_lin)
    assert f_lin == f_to_lin == ["linear", "linear"]

    v_imp, f_imp = _steps("implicit", [None, None])
    v_to_imp, _ = _steps("linear", ["implicit", None])
    assert _rel(v_to_imp, v_imp) < 1.0e-8, _rel(v_to_imp, v_imp)
    assert f_imp == ["implicit", "implicit"]


def test_a_switch_between_steps_takes_effect_and_keeps_the_history():
    """Switching mid-run changes the next step's momentum (not the stored stress's
    meaning: every form stores the full deformation step)."""
    v_imp, _ = _steps("implicit", [None, None])
    v_mid, forms = _steps("implicit", [None, "linear"])
    assert forms == ["implicit", "linear"]
    assert _rel(v_mid, v_imp) > 1.0e-10, _rel(v_mid, v_imp)
