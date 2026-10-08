"""Maxwell viscoelastic thin sheet on a flat mesh: closed-form checks.

Uniaxial loading at constant strain rate with free lateral edges relaxes
exactly as sigma_xx = 3 eta edot (1 - exp(-t mu / eta)): the Trouton stress
with the Maxwell time. The exponential integrator reproduces that exactly at
every step (the forcing is constant over a step); backward Euler reproduces
its own recursion exactly. Stress diffusion into an elastic sheet over a
viscous substrate (Elsasser) follows v = V0 erfc(x / sqrt(4 D t)) with
D = mu L / c, approached at first order in the timestep.
"""

import numpy as np
import pytest
import sympy
from scipy.special import erfc

import underworld3 as uw

pytestmark = [pytest.mark.level_2]

ETA, MU, EDOT = 2.0, 4.0, 0.5                 # t_r = eta / mu = 0.5


def _uniaxial(integrator, n_steps, dt_over_tr):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    x, y = mesh.X
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator=integrator)
    sheet.constitutive_model.Parameters.shear_viscosity_0 = ETA
    sheet.constitutive_model.Parameters.shear_modulus = MU
    sheet.tolerance = 1.0e-10
    edot = sympy.Rational(1, 2)
    u_y = -edot * (y - sympy.Rational(1, 2)) / 2
    sheet.add_essential_bc([0, u_y], "Left")
    sheet.add_essential_bc([edot, u_y], "Right")

    t_r = ETA / MU
    dt = dt_over_tr * t_r
    centre = np.array([[0.5, 0.5]])
    times, sigma_xx = [], []
    for step in range(1, n_steps + 1):
        sheet.solve(timestep=dt, zero_init_guess=False)
        tau = uw.function.evaluate(sheet.tau.sym, centre).reshape(2, 2)
        # plane stress: sigma = tau + tr(tau) I
        sigma_xx.append(2.0 * tau[0, 0] + tau[1, 1])
        times.append(step * dt)
    return np.array(times), np.array(sigma_xx), dt


@pytest.mark.tier_b
def test_exponential_integrator_matches_the_maxwell_response_exactly():
    """ETD-1 integrates the relaxation analytically, so at constant strain
    rate it lands on 3 eta edot (1 - exp(-t/t_r)) at every step."""
    times, sigma, _ = _uniaxial("etd", n_steps=8, dt_over_tr=0.5)
    exact = 3.0 * ETA * EDOT * (1.0 - np.exp(-times * MU / ETA))
    assert np.abs(sigma - exact).max() < 1.0e-6 * 3.0 * ETA * EDOT


@pytest.mark.tier_b
def test_backward_euler_follows_its_own_recursion_exactly():
    """BDF-1 on the Maxwell element: sigma_n = (sigma_{n-1} + 3 mu dt edot) / (1 + dt/t_r)."""
    times, sigma, dt = _uniaxial("bdf", n_steps=8, dt_over_tr=0.5)
    recursion, s = [], 0.0
    for _ in times:
        s = (s + 3.0 * MU * dt * EDOT) / (1.0 + dt * MU / ETA)
        recursion.append(s)
    assert np.abs(sigma - np.array(recursion)).max() < 1.0e-6 * 3.0 * ETA * EDOT


def _elsasser(dt, t_end, V0=1.0, length=4.0, cell=0.1):
    """Anti-plane shear v(x, t) in an elastic sheet (t_r >> t_end) over a
    viscous substrate: v_t = D v_xx with D = mu L / c."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(length, 1.0), cellSize=cell, qdegree=3)
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0e6   # t_r = 1e6 >> t_end
    sheet.constitutive_model.Parameters.shear_modulus = 1.0
    sheet.basal_drag = 1                                             # D = mu L / c = 1
    sheet.tolerance = 1.0e-10
    sheet.add_essential_bc([0, V0], "Left")
    sheet.add_essential_bc([0, 0], "Right")
    sheet.add_essential_bc([0, None], "Top")
    sheet.add_essential_bc([0, None], "Bottom")
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        sheet.solve(timestep=dt, zero_init_guess=False)
    xs = np.linspace(0.05, 2.5, 30)
    pts = np.column_stack([xs, np.full_like(xs, 0.5)])
    v = uw.function.evaluate(sheet.u.sym, pts).reshape(-1, 2)[:, 1]
    exact = V0 * erfc(xs / np.sqrt(4.0 * 1.0 * t_end))
    return np.abs(v - exact).max()


@pytest.mark.tier_b
def test_elsasser_stress_diffusion_converges_in_the_timestep():
    """The erfc front is approached at first order in dt (backward Euler in
    time), and is within 5% of V0 at dt = 0.025."""
    err_coarse = _elsasser(dt=0.05, t_end=0.5)
    err_fine = _elsasser(dt=0.025, t_end=0.5)
    order = np.log(err_coarse / err_fine) / np.log(2.0)
    assert err_fine < 0.05, f"error {err_fine:.3e} against the erfc front"
    assert order > 0.8, f"timestep convergence order {order:.2f} ({err_coarse:.3e} -> {err_fine:.3e})"
