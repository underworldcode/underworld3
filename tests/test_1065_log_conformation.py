"""The positive-definite convected step and the log-conformation stress history.

In the conformation c = sigma/G + I, with F = I + dt L, x = dt/lam, the
deformation step is
    BDF-1:  c (1 + x) = F c* F^T + x I
    ETD-1:  c = a F c* F^T + ((1-a) I + b L)((1-a) I + b L^T)/(1-a),
            a = exp(-x), b = lam (1 - (1 + x) exp(-x))
and the linear step drops the dt^2 L c* L^T of F c* F^T.

One step of uniform planar extension from rest, u = (e x, -e y), UCM with
eta = G = 1, dt e = 1: the linear BDF step leaves c_yy = -0.818, not a
conformation; the deformation step gives the closed forms above, whether the
history stores the stress or its log-conformation.

The start-up of simple shear: every flavour, both integrators, the
log-conformation history against the same recurrence, step by step, in both
the shear stress and the first normal-stress difference (every deformation term
lives in sigma_xx here).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

DT, RATE = 0.1, 10.0            # dt * rate = 1, lam = 1


def _deformation_step(c_star, L, dt, lam, integrator):
    I = np.eye(2)
    F = I + dt * L
    x = dt / lam
    if integrator == "bdf":
        return (F @ c_star @ F.T + x * I) / (1 + x)
    a = np.exp(-x)
    b = lam * (1 - (1 + x) * np.exp(-x))
    return a * F @ c_star @ F.T + ((1 - a) * I + b * L) @ ((1 - a) * I + b * L.T) / (1 - a)


def _stored_conformation(stokes, stress_history):
    """Smallest and largest conformation eigenvalue over the history's stored
    nodes (2-D symmetric storage: xx, yy, xy), decoded from the log if need be."""
    stored = np.asarray(stokes.DFDt.psi_star[0].data).reshape(-1, 3)
    m = np.empty((stored.shape[0], 2, 2))
    m[:, 0, 0], m[:, 1, 1], m[:, 0, 1], m[:, 1, 0] = stored[:, 0], stored[:, 1], stored[:, 2], stored[:, 2]
    if stress_history == "log_conformation":
        w, v = np.linalg.eigh(m)
        c = v @ (np.exp(w)[:, :, None] * np.transpose(v, (0, 2, 1)))
    else:
        c = m + np.eye(2)                          # G = 1
    ev = np.linalg.eigvalsh(c)
    return ev[:, 0].min(), ev[:, 1].max()


def _extension_step(transport, convected_step, stress_history, integrator="bdf"):
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(-1, -1), maxCoords=(1, 1))
    x, y = mesh.X
    tag = f"{transport[:3]}{convected_step[:3]}{stress_history[:3]}{integrator}"
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator=integrator, objective_rate="upper_convected",
        convected_step=convected_step, stress_history=stress_history)
    parameters = stokes.constitutive_model.Parameters
    parameters.shear_viscosity_0 = 1.0
    parameters.shear_modulus = 1.0
    parameters.dt_elastic = DT
    extension = (RATE * x, -RATE * y)
    for boundary in ("Left", "Top", "Bottom"):     # the right side is free: sigma_xx - p = 0 there
        stokes.add_dirichlet_bc(extension, boundary)
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["snes_type"] = "newtonls"
    # The deformation step makes the momentum equation quadratic in grad u; at
    # dt |L| = 1 Newton does not converge from rest, so start from the answer,
    # and from rest in the stress (set_initial_history takes stored values; zero
    # is rest in both representations).
    X = np.asarray(v.coords)
    v.data[:, 0], v.data[:, 1] = RATE * X[:, 0], -RATE * X[:, 1]
    stokes.DFDt.set_initial_history([0.0])
    stokes.solve(timestep=DT, zero_init_guess=False)
    return _stored_conformation(stokes, stress_history)


@pytest.mark.parametrize("integrator", ["bdf", "etd"])
@pytest.mark.parametrize("transport, stress_history", [
    ("semi_lagrangian", "stress"), ("semi_lagrangian", "log_conformation"),
    ("eulerian", "stress"), ("eulerian", "log_conformation")])
def test_deformation_step_is_the_closed_form(transport, stress_history, integrator):
    c = _deformation_step(np.eye(2), np.diag([RATE, -RATE]), DT, 1.0, integrator)
    c_min, c_max = _extension_step(transport, "deformation", stress_history, integrator)
    assert abs(c_min - c[1, 1]) < 1.0e-6, (c_min, c[1, 1])
    assert abs(c_max - c[0, 0]) < 1.0e-6, (c_max, c[0, 0])


def test_linear_step_loses_the_conformation():
    """The defect the deformation step removes, measured against its closed form."""
    c_min, _ = _extension_step("semi_lagrangian", "linear", "stress")
    assert abs(c_min - (1 - 2 * DT * RATE + DT) / (1 + DT)) < 1.0e-6, c_min


def _shear_startup(transport, integrator, steps=10, dt=0.1):
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5))
    tag = f"s{transport[:3]}{integrator}"
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator=integrator, objective_rate="upper_convected",
        stress_history="log_conformation")
    parameters = stokes.constitutive_model.Parameters
    parameters.shear_viscosity_0 = 1.0
    parameters.shear_modulus = 1.0
    parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((0.5, 0.0), "Top")
    stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)
    sigma = stokes.constitutive_model.stress_star.sym
    origin = np.array([[0.0, 0.0]])
    read = lambda e: float(np.asarray(uw.function.evaluate(e, origin)).reshape(-1)[0])
    c = np.eye(2)
    L = np.array([[0.0, 1.0], [0.0, 0.0]])              # shear rate one
    for _ in range(steps):
        c = _deformation_step(c, L, dt, 1.0, integrator)
    return (read(sigma[0, 1]), c[0, 1]), (read(sigma[0, 0] - sigma[1, 1]), c[0, 0] - c[1, 1])


@pytest.mark.parametrize("integrator", ["bdf", "etd"])
@pytest.mark.parametrize("transport", ["semi_lagrangian", "integration_point", "forward", "eulerian"])
def test_log_conformation_history_follows_the_recurrence(transport, integrator):
    """Uniform stress, so transport is a no-op: this checks the encoding, the
    decoding and the step together, against the discrete recurrence."""
    (xy, xy_exact), (n1, n1_exact) = _shear_startup(transport, integrator)
    assert abs(xy - xy_exact) < 1.0e-5 * abs(xy_exact), (transport, integrator, xy, xy_exact)
    assert abs(n1 - n1_exact) < 1.0e-5 * abs(n1_exact), (transport, integrator, n1, n1_exact)
