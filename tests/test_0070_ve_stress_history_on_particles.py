"""Viscoelastic stress history carried by particles.

The Maxwell shear box (test_1051) with the stress history on a swarm:
``Lagrangian_Swarm`` with the cells proxy supplied as the Stokes solver's
``DFDt``. The constitutive model reads the history through the proxy's
symbol exactly as it reads the nodal one; the particles carry the stress
along the flow, the mesh integrates it at the integration points, and after
each solve the new stress is evaluated at the particles (the Maxwell update
is a local ODE, no projection back to the mesh). Uniform shear has a uniform
stress, so this checks the plumbing and the time integration against the
analytic Maxwell curve, and against the nodal history.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]

ETA, MU, V0, H, W = 1.0, 1.0, 0.5, 1.0, 2.0


def maxwell_stress_xy(t, gamma_dot):
    return ETA * gamma_dot * (1.0 - np.exp(-t / (ETA / MU)))


def _run(history, order, n_steps, dt_over_tr, res=8):
    dt = dt_over_tr * ETA / MU
    gamma_dot = 2.0 * V0 / H
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2 * res, res),
                                       minCoords=(-W / 2, -H / 2), maxCoords=(W / 2, H / 2))
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    swarm = DFDt = None
    if history == "particles":
        swarm = uw.swarm.Swarm(mesh)
        DFDt = uw.systems.ddt.Lagrangian_Swarm(
            swarm=swarm, psi_fn=sympy.Matrix.zeros(2, 2), vtype=uw.VarType.SYM_TENSOR,
            degree=1, continuous=False, order=order, step_averaging=1, proxy_location="cells",
        )
        swarm.populate(fill_param=3)
        swarm.population_control = dict()          # open sides: inflow refill, outflow loss
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, DFDt=DFDt)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=order)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((V0, 0.0), "Top")
    stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    assert stokes.DFDt is DFDt if DFDt is not None else stokes.DFDt is not None
    num, ana = [], []
    time = 0.0
    for _ in range(n_steps):
        if swarm is not None:
            swarm.advection(v.sym, dt, order=2)
        stokes.solve(timestep=dt, zero_init_guess=False, evalf=False)
        time += dt
        num.append(float(np.asarray(uw.function.evaluate(stokes.tau.sym[0, 1], np.array([[0.0, 0.0]]))).flatten()[0]))
        ana.append(maxwell_stress_xy(time, gamma_dot))
    return np.array(num), np.array(ana)


def test_particle_stress_history_tracks_maxwell_order1():
    num, ana = _run("particles", order=1, n_steps=20, dt_over_tr=0.1)
    rel_final = abs(num[-1] - ana[-1]) / abs(ana[-1])
    assert rel_final < 0.05, rel_final
    assert np.all(np.diff(num) > -1e-8)                      # monotone loading


def test_particle_and_nodal_histories_agree():
    """Uniform shear: the two histories carry the same stress, so the answers
    coincide to the transport reconstruction error."""
    num_p, ana = _run("particles", order=1, n_steps=10, dt_over_tr=0.1)
    num_n, _ = _run("nodal", order=1, n_steps=10, dt_over_tr=0.1)
    assert np.abs(num_p - num_n).max() < 2e-3 * abs(ana[-1]), np.abs(num_p - num_n).max()


def test_particle_stress_history_order2():
    num, ana = _run("particles", order=2, n_steps=20, dt_over_tr=0.1)
    rel_final = abs(num[-1] - ana[-1]) / abs(ana[-1])
    assert rel_final < 0.01, rel_final
