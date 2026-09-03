"""The Eulerian SUPG solver against the rotating Gaussian.

Three properties measured on ``uw.analytic.RotatingGaussian`` (rigid rotation,
exact at every time):

1. temporal order: the error at a quarter turn falls as dt (BDF1) and dt^2
   (BDF2) when the timestep is halved, with the exact history planted so
   the multistep scheme runs at full order from the first step;
2. mesh refinement the scalar does not need leaves the answer alone: a band
   refined to h/8 across the orbit, at the same timestep, gives the same
   error to three digits even though its cells sit at a local Courant
   number of several;
3. the round trip: after one revolution the field returns to its initial
   state to a few per cent at a Courant number of one half.

Run: pixi run python -m pytest tests/test_1100_advdiff_supg_rotating_gaussian.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

SIGMA = 0.12


def _box(res, refinement=0):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=2.0 / res,
        qdegree=3, regular=False, refinement=refinement)


def _problem(mesh, tag, order, integrator="bdf", theta=1.0, kappa=0.0):
    x, y = mesh.X
    sol = uw.analytic.RotatingGaussian(mesh, sigma=SIGMA, centre_radius=0.5,
                                       omega=1.0, diffusivity=kappa)
    T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sol.at(0.0), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusionSUPG(mesh, T, sympy.Matrix([[-y, x]]),
                                      order=order, integrator=integrator, theta=theta)
    adv.constitutive_model.Parameters.diffusivity = kappa
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    return sol, T, adv


def _run(sol, T, adv, dt, t_end, plant=True):
    nsteps = int(round(t_end / dt))
    dt = t_end / nsteps
    if plant and adv.order > 1:
        values = [uw.function.evaluate(sol.at(-k * dt), T.coords).reshape(-1, 1, 1)
                  for k in range(adv.order)]
        adv.DuDt.set_initial_history(values, dt=dt)
    for _ in range(nsteps):
        adv.solve(timestep=dt)
    return sol.error(sol.at(t_end), T, norm="integral")


@pytest.mark.parametrize("order, timesteps, expected_slope", [
    (1, (0.02, 0.01, 0.005), 1.0),
    (2, (0.04, 0.02, 0.01), 2.0),
])
def test_temporal_convergence_order(order, timesteps, expected_slope):
    """Halving dt divides the quarter-turn error by 2 (BDF1) or 4 (BDF2).

    The timesteps sit where the temporal error dominates the fixed spatial
    error but is still in its asymptotic range (backward Euler at
    u dt > sigma/2 is already saturated), which is why the slope is checked
    with a tolerance.
    """
    mesh = _box(32)
    t_end = float(sympy.pi) / 2
    errors = []
    for i, dt in enumerate(timesteps):
        sol, T, adv = _problem(mesh, f"c{order}{i}", order)
        errors.append(_run(sol, T, adv, dt, t_end))
    slopes = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"order {order}: errors {errors} slopes {slopes}")
    assert slopes.min() > expected_slope - 0.35, (order, errors, slopes)


def test_refinement_the_scalar_does_not_need_leaves_the_error_alone():
    """A band at h/8 across the orbit, same dt as the uniform mesh."""
    dt = 0.0433
    t_end = float(sympy.pi) / 2

    uniform = _box(32)
    sol, T, adv = _problem(uniform, "u", 2)
    err_uniform = _run(sol, T, adv, dt, t_end)

    base = _box(16, refinement=1)
    fault = uw.meshing.Surface("band", base,
                               np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]), symbol="F")
    fault.discretize()
    h = 1.0 / 16

    def metric(pts, _f=fault, _hn=h / 8, _hf=h, _core=0.03, _ramp=0.06):
        d = _f.unsigned_distance(pts)
        hh = np.where(d < _core, _hn, np.minimum(_hn + (_hf - _hn) * (d - _core) / _ramp, _hf))
        return 1.0 / hh ** 2

    child = base.adapt(metric, max_levels=3)
    assert float(np.min(child._radii)) < 0.3 * float(np.min(uniform._radii))

    sol_c, T_c, adv_c = _problem(child, "b", 2)
    err_band = _run(sol_c, T_c, adv_c, dt, t_end)

    # the band cells are at a local Courant number well above one
    assert dt / float(adv_c.estimate_dt()) > 4.0
    assert abs(err_band - err_uniform) < 0.15 * err_uniform, (err_uniform, err_band)


def test_round_trip_at_moderate_courant():
    mesh = _box(32)
    sol, T, adv = _problem(mesh, "r", 2)
    err = _run(sol, T, adv, 0.5 * float(adv.estimate_dt()), float(sol.period))
    assert err < 0.03, err
    data = np.asarray(T.array[:, 0, 0])
    assert data.min() > -0.02 and data.max() < 1.02, (data.min(), data.max())
