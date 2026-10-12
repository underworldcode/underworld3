"""The particle stress history carries the inflow datum into an open channel.

Poiseuille flow of a Maxwell fluid through a channel open at both ends, with
the stress history on the solver's own swarm (``stress_transport =
"lagrangian"``). The particles move downstream, the inlet cells empty and the
population control refills them; what the refilled particles carry is the
question. With no objective rate the fully developed stress is
:math:`\\sigma = 2\\eta\\dot{\\varepsilon}`, so the entering fluid's state is
known exactly and is what ``inflow_value`` says. A refill from the nearest
old particles instead smears the stress across the inlet cell and hands the
smear downstream (#783).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

ETA, MU = 1.0, 1.0
LENGTH, HEIGHT, PEAK_SPEED = 4.0, 1.0, 1.0
CELL = HEIGHT / 8


def _channel(steps, dt, set_inflow):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, -HEIGHT / 2), maxCoords=(LENGTH, HEIGHT / 2), cellSize=CELL)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U_inflow", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_inflow", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "lagrangian"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt

    u_in = PEAK_SPEED * (1.0 - (2.0 * y / HEIGHT) ** 2)
    stokes.add_dirichlet_bc((u_in, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.add_dirichlet_bc((0.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    shear_rate = u_in.diff(y)
    if set_inflow:
        stokes.DFDt.inflow_value = sympy.Matrix([[0.0, ETA * shear_rate], [ETA * shear_rate, 0.0]])

    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)

    # The inlet cell column, away from the walls: what the refilled particles carry.
    probes = np.column_stack([np.full(7, 0.25 * CELL), np.linspace(-0.4, 0.4, 7)])
    carried = np.asarray(uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0, 1], probes)).reshape(-1)
    exact = np.asarray(uw.function.evaluate(ETA * shear_rate, probes)).reshape(-1)
    return float(np.max(np.abs(carried - exact)))


def test_particle_history_applies_the_inflow_value():
    """After more than one transit every inlet particle has been refilled at
    least once. With the inflow datum the carried shear stress matches
    :math:`\\eta\\dot{\\gamma}` to the residual of the particles still loading
    from the cold start; without it the inlet holds the neighbour smear."""
    assert uw.systems.ddt.Lagrangian.applies_inflow_value
    error = _channel(steps=50, dt=0.1, set_inflow=True)
    assert error < 0.01, error                          # measured 2.1e-3 (2026-09-23)
    smear = _channel(steps=50, dt=0.1, set_inflow=False)
    assert smear > 0.1, smear                           # measured 0.17
