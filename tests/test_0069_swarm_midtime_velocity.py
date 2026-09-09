"""Swarm advection with the mid-point velocity taken at the mid time.

The RK2 step evaluates the mid-point velocity at the mid TIME,
1.5 v^n - 0.5 v^{n-1}, from a CharacteristicTrace (swarm-owned, or a
solver's shared one). On a rotation whose rate ramps linearly in time the
frozen-velocity step is first order in the rate; the mid-time step is exact
for the linear ramp after its first step.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _angle_error(midtime, nsteps=10, dt=0.1):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=2)
    mesh.return_coords_to_bounds = None
    x, y = mesh.X
    c = uw.expression(r"c_{rate}", 1.0, "rotation rate")
    V = c * sympy.Matrix([[-y, x]])
    swarm = uw.swarm.Swarm(mesh)
    swarm.verbose = False
    L = uw.swarm.SwarmVariable("L", swarm, 2)           # launch position, carried by the particle
    swarm.populate(fill_param=1)
    with uw.synchronised_array_update():
        L.data[...] = np.asarray(swarm._particle_coordinates.data)
    t = 0.0
    for _ in range(nsteps):
        c.sym = 1.0 + t                       # the rate at t^n; the mid-time rate is 1 + t + dt/2
        swarm.advection(V, dt, order=2, midtime_velocity=midtime)
        t += dt
    X = np.asarray(swarm._particle_coordinates.data)
    X0 = np.asarray(L.data)
    inner = np.hypot(X0[:, 0], X0[:, 1]) < 0.6
    exact = nsteps * dt + 0.5 * (nsteps * dt) ** 2       # integral of (1 + t)
    ang = np.arctan2(X[inner, 1], X[inner, 0]) - np.arctan2(X0[inner, 1], X0[inner, 0])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(ang - exact).mean())


def test_midtime_velocity_makes_the_swarm_step_second_order_in_time():
    frozen = _angle_error(False)
    midtime = _angle_error(True)
    # Frozen rate: each step misses dt/2 of ramp over dt: 0.5 * N * dt^2 = 0.05 rad.
    assert 0.03 < frozen < 0.07, frozen
    # Mid-time rate: exact for the linear ramp except the first step (no v^{n-1}): 0.005 rad.
    assert midtime < 0.008, midtime
    assert midtime < frozen / 5


def test_shared_trace_from_a_solver_is_accepted():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=2)
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    tr = uw.systems.ddt.CharacteristicTrace(mesh, V)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=1)
    X0 = np.array(swarm._particle_coordinates.data, copy=True)
    swarm.advection(V, 0.05, order=2, characteristics=tr)
    assert not hasattr(swarm, "_characteristics") or swarm._characteristics is not tr
    assert np.abs(np.asarray(swarm._particle_coordinates.data) - X0).max() > 1e-3
