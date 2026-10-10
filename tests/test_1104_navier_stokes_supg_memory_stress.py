"""The SUPG residual of uw.systems.NavierStokes sees the carried elastic stress.

SUPG weights the strong momentum residual along streamlines; a residual missing
a term that does not vanish at the exact solution injects an O(tau) error. The
divergence of the stress a viscoelastic history carries is such a term, and at
high Weissenberg number the largest one. On a developing Oldroyd-B channel
(Re 250, Wi 0.6, beta 0.59, stress-free fluid entering) it moved the SUPG
solution toward the Galerkin one on the same mesh: velocity 5.0e-5 -> 3.2e-5,
stress 3.4e-6 -> 1.6e-6 (~/+Simulations/supg_memory, 2026-09-28).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

L, H, U = 2.0, 0.5, 1.0
ETA, BETA, WI = 1.0 / 250.0, 0.59, 0.6
POINTS = np.array([[0.4, 0.3], [0.8, -0.2]])
# BASELINES: u_x and sigma_xy at POINTS after ten steps (2026-09-28)
U_X = (0.639601419, 0.840013154)
SIGMA_XY = (-0.00331132215, 0.00220579959)


def developing_channel(transport="forward_integration_points", steps=10, dt=0.05, cell=0.1,
                       velocity_transport="eulerian", order=1, stress_history="stress",
                       modulus_varies=False):
    eta_s, eta_p = BETA * ETA, (1 - BETA) * ETA
    lam = WI * H / U
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, -H), maxCoords=(L, H),
                                             cellSize=cell, qdegree=3, regular=False)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U_ch", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_ch", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=order,
                                 velocity_transport=velocity_transport)
    ns.stress_transport = transport
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        stress_history=stress_history)
    cm = ns.constitutive_model
    cm.Parameters.shear_viscosity_0 = eta_p
    cm.Parameters.shear_modulus = eta_p / lam * ((1 + 0.5 * x) if modulus_varies else 1)
    cm.Parameters.solvent_viscosity = eta_s
    cm.Parameters.dt_elastic = dt
    u_in = U * (1 - y ** 2 / H ** 2)
    ns.add_dirichlet_bc((u_in, 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.bodyforce = sympy.Matrix([[0.0, 0.0]])
    ns.DFDt.inflow_value = sympy.Matrix([[0.0, 0.0], [0.0, 0.0]])
    ns.tolerance = 1.0e-8
    v.array[:, 0, :] = np.column_stack([
        np.asarray(uw.function.evaluate(u_in, v.coords)).reshape(-1), np.zeros(v.coords.shape[0])])
    for _ in range(steps):
        ns.solve(timestep=dt)
    return ns, v


def test_the_supg_residual_carries_the_memory_stress_and_no_velocity_derivative():
    uw.reset_default_model()
    ns, v = developing_channel(steps=1)
    memory = sympy.Matrix(ns._memory_stress())
    X = ns.mesh.X
    own = {ns.u.sym[i].diff(X[j]) for i in range(2) for j in range(2)}
    assert not (memory.atoms(sympy.Function) & own)
    # after one step the entering fluid has built a stress, so the memory is not zero
    probe = np.array([[0.2, 0.3]])
    value = float(np.asarray(uw.function.evaluate(memory[0, 1], probe)).reshape(-1)[0])
    assert np.isfinite(value) and abs(value) > 1.0e-8, value


def test_a_varying_modulus_is_expanded_so_its_gradient_is_seen():
    """A modulus that varies in space is expanded in the memory stress (its
    divergence then has the modulus gradient); the timestep stays a runtime
    parameter."""
    from underworld3.function.expressions import UWexpression
    uw.reset_default_model()
    ns, v = developing_channel(steps=1, modulus_varies=True)
    cm = ns.constitutive_model
    containers = sympy.Matrix(ns._memory_stress()).atoms(UWexpression)
    assert cm.Parameters.shear_modulus not in containers
    assert containers and all(c.is_uw_constant() for c in containers)


def test_a_viscous_fluid_has_no_memory_stress():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    v = uw.discretisation.MeshVariable("U_vf", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_vf", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    assert ns._memory_stress() is None


def test_an_integration_point_store_is_read_through_its_nodal_snapshot():
    """An integration-point store has no derivative: the SUPG residual takes the
    divergence of its nodal snapshot instead."""
    uw.reset_default_model()
    ns, v = developing_channel("backward_integration_points", steps=1)
    atoms = sympy.Matrix(ns._memory_stress()).atoms(sympy.Function)
    ip = set(sympy.Matrix(ns.DFDt.psi_star[0].sym))
    snapshot = set(sympy.Matrix(ns.DFDt.psi_snap[0].sym))
    assert not (atoms & ip) and (atoms & snapshot)


def test_the_developing_channel_keeps_its_recorded_flow():
    uw.reset_default_model()
    ns, v = developing_channel()
    ux = np.asarray(uw.function.global_evaluate(v.sym[0], POINTS)).reshape(-1)
    sxy = np.asarray(uw.function.global_evaluate(
        ns.constitutive_model._carried_stress_sym(0)[0, 1], POINTS)).reshape(-1)
    assert np.allclose(ux, U_X, atol=1.0e-7), ux
    assert np.allclose(sxy, SIGMA_XY, atol=1.0e-9), sxy


# BASELINES: semi-Lagrangian momentum (backward nodes, order 2) against a
# first-order stress history, solvent, log-conformation store: u_x and
# sigma_xy at POINTS (2026-09-29)
SLCN_U_X = (0.639781102, 0.840017737)
SLCN_SIGMA_XY = (-0.00330714249, 0.00221101845)


def test_semi_lagrangian_momentum_keeps_its_recorded_developing_channel():
    """Semi-Lagrangian momentum on the developing channel: a stress that varies
    along the flow, a solvent, the log-conformation store, and momentum order 2
    against a first-order stress history, so every term of the momentum flux is
    live."""
    uw.reset_default_model()
    ns, v = developing_channel(velocity_transport="backward_nodes", order=2,
                               stress_history="log_conformation")
    ux = np.asarray(uw.function.global_evaluate(v.sym[0], POINTS)).reshape(-1)
    sxy = np.asarray(uw.function.global_evaluate(
        ns.constitutive_model._carried_stress_sym(0)[0, 1], POINTS)).reshape(-1)
    assert np.allclose(ux, SLCN_U_X, atol=1.0e-7), ux
    assert np.allclose(sxy, SLCN_SIGMA_XY, atol=1.0e-9), sxy


def test_an_integration_point_velocity_history_carries_a_viscoelastic_flow():
    """The solvent stress of the stored velocity needs its derivative; an
    integration-point velocity store is read through its nodal snapshot."""
    uw.reset_default_model()
    ns, v = developing_channel(velocity_transport="backward_integration_points", steps=2)
    ux = np.asarray(uw.function.global_evaluate(v.sym[0], POINTS)).reshape(-1)
    assert np.all(np.isfinite(ux)) and np.all(ux > 0.3), ux
