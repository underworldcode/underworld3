"""The swarm (Lagrangian) advection-diffusion solver against the rotating Gaussian.

AdvDiffusionSwarm carries the scalar's history on a user swarm and reads the
mesh solution back onto the particles each step. On the diffusing rotating
Gaussian (exact at every time) on a disc it is as accurate as the
semi-Lagrangian and streamline-upwind solvers over a half revolution, and it
diffuses correctly: the trap of a particle scheme is to keep the particle's old,
sharp value and under-diffuse, which shows as a peak above the exact one.

A DISC is used, not a square box: under rigid rotation the corners of a square
sit at radius > 1 and rotate out of the domain, so a square's corner cells
continuously empty and refill, which destabilises any particle scheme (mesh
schemes are unaffected). The disc has no corners to lose, matching the geometry
of the SLCN and SUPG rotation demonstrators.

The swarm is supplied by the caller and is NOT private to the solver; the solver
declares its history variable, so it is built before the swarm is populated, and
the caller advects the swarm each step.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

SIGMA = 0.12
KAPPA = 0.01
T_END = float(np.pi)          # half revolution
EXACT_PEAK = SIGMA ** 2 / (SIGMA ** 2 + 2 * KAPPA * T_END)


def _disc(res=24):
    return uw.meshing.Annulus(radiusInner=0.0, radiusOuter=1.2, cellSize=2.4 / res, qdegree=3)


def _run(scheme, particle_update="pic", step_averaging=1, res=24):
    uw.reset_default_model()
    mesh = _disc(res)
    x, y = mesh.X
    sol = uw.analytic.RotatingGaussian(mesh, sigma=SIGMA, centre_radius=0.5, omega=1.0, diffusivity=KAPPA)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sol.at(0.0), T.coords).reshape(-1)
    V = sympy.Matrix([[-y, x]])
    swarm = None
    if scheme == "supg":
        adv = uw.systems.AdvDiffusion(mesh, T, V, order=1)
    elif scheme == "slcn":
        adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V, order=1)
    elif scheme == "swarm":
        swarm = uw.swarm.Swarm(mesh)
        adv = uw.systems.AdvDiffusionSwarm(mesh, T, V, swarm=swarm, order=1,
                                           particle_update=particle_update, step_averaging=step_averaging)
        swarm.populate(fill_param=4)
        swarm.population_control = dict()
    if scheme != "supg":
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = KAPPA
    adv.add_dirichlet_bc(0.0, "Upper")
    dt = 0.02
    nsteps = int(round(T_END / dt)); dt = T_END / nsteps
    for _ in range(nsteps):
        if swarm is not None:
            swarm.advection(V, dt, order=2)
        adv.solve(timestep=dt)
    err = float(sol.error(sol.at(T_END), T, norm="integral"))
    peak = float(np.asarray(T.data).max())
    return adv, err, peak


def test_the_swarm_solver_is_as_accurate_as_the_mesh_solvers_over_a_half_revolution():
    adv, err, peak = _run("swarm")
    assert type(adv.DuDt).__name__ == "Lagrangian_Swarm"
    assert adv.swarm is not None
    # BASELINE: half revolution, kappa 0.01, dt 0.02, disc res 24 (see the ledger).
    # SUPG and SLCN give 6.5e-2 and 6.4e-2 here; the swarm solver is at least as
    # accurate, and it is held to a hard number, not a loose particle floor.
    assert abs(err - 0.0173) < 0.004, err
    assert abs(peak - EXACT_PEAK) < 0.01, (peak, EXACT_PEAK)   # it diffuses to the exact peak


def test_the_half_blend_under_diffuses_a_particle_scheme_trap():
    """step_averaging=2 keeps half the particle's old sharp value each step, so
    the field does not diffuse to the exact peak: the trap the default avoids."""
    _adv, err, peak = _run("swarm", step_averaging=2)
    assert peak > EXACT_PEAK + 0.1, peak
    assert err > 0.2, err


def test_it_will_not_run_without_a_swarm():
    uw.reset_default_model()
    mesh = _disc(8)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tn", mesh, 1, degree=2)
    with pytest.raises(ValueError, match="needs a swarm"):
        uw.systems.AdvDiffusionSwarm(mesh, T, sympy.Matrix([[-y, x]]), swarm=None)
