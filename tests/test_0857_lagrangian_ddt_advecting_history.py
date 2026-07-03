"""Advecting-history regression tests for the swarm-based Lagrangian DDt
classes (``Lagrangian`` and ``Lagrangian_Swarm``).

Audit finding SWARM-06 (BF-04, docs/reviews/2026-07): the history-update
blocks in both classes wrote components as ``psi_star_0[i, j].data[:] = ...``.
Indexing a SwarmVariable returns a *symbolic* component (an
UnderworldAppliedFunction via ``MathematicalMixin.__getitem__``) which has no
``.data`` — so every first ``update_pre_solve`` raised AttributeError and the
Lagrangian history machinery was unusable. The fix routes component writes
through the canonical storage: ``psi_star_0.data[:, var._data_layout(i, j)]``.

These tests actually *advect* the history (the pre-existing test_0007
coverage only round-tripped ``.state`` with ``_history_initialised`` set
manually) and fail with AttributeError against the old write path.

All expected values are expressed as functions of the particles' *current*
coordinates (minus the known constant-velocity displacement where relevant)
so the assertions are robust to any particle reordering during migration.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import ddt as ddt_module

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

# Constant velocity: RK advection is exact, displacement is known a priori.
VX, VY = 0.30, 0.20
DT = 0.01


def _make_mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def _linear_scalar_field(mesh, name, a, b):
    """P1 MeshVariable holding the (exactly representable) field a*x + b*y."""
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)
    coords = np.asarray(T.coords)
    T.data[:, 0] = a * coords[:, 0] + b * coords[:, 1]
    return T


def _set_linear(T, a, b):
    coords = np.asarray(T.coords)
    T.data[:, 0] = a * coords[:, 0] + b * coords[:, 1]


def _constant_velocity(mesh):
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1)
    V.data[:, 0] = VX
    V.data[:, 1] = VY
    return V


def test_lagrangian_scalar_advecting_history():
    """Lagrangian (own swarm): initialise, advect, and verify the shifted
    history holds psi at each particle's pre-advection position."""
    mesh = _make_mesh()
    T = _linear_scalar_field(mesh, "T", 1.0, 2.0)  # T1 = x + 2y
    V = _constant_velocity(mesh)

    lag_ddt = ddt_module.Lagrangian(
        mesh=mesh,
        psi_fn=T.sym,
        V_fn=V.sym,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
        order=2,
        fill_param=2,
    )

    # --- first update: auto-initialise (old write path raised here) ---
    lag_ddt.update_pre_solve(dt=DT)
    assert lag_ddt._history_initialised

    coords0 = np.asarray(lag_ddt.swarm._particle_coordinates.data).copy()
    expected_T1 = coords0[:, 0] + 2.0 * coords0[:, 1]
    got0 = np.asarray(lag_ddt.psi_star[0].data)[:, 0]
    got1 = np.asarray(lag_ddt.psi_star[1].data)[:, 0]
    assert np.allclose(got0, expected_T1, atol=1e-10)
    # all history slots initialised identically
    assert np.allclose(got1, expected_T1, atol=1e-10)

    # --- post-solve: field changes, history shifts, swarm advects ---
    _set_linear(T, 3.0, -1.0)  # T2 = 3x - y
    lag_ddt.update_post_solve(dt=DT)

    coords1 = np.asarray(lag_ddt.swarm._particle_coordinates.data)
    # the swarm actually moved (displacement DT * V for interior particles)
    disp = np.linalg.norm(coords1.mean(axis=0) - coords0.mean(axis=0))
    assert disp > 0.5 * DT * np.hypot(VX, VY)

    # Each particle's pre-advection position (constant V => exact rewind).
    x0 = coords1[:, 0] - DT * VX
    y0 = coords1[:, 1] - DT * VY
    # ignore particles clamped back into the domain by the bounds restore
    interior = (
        (x0 > 1e-6) & (x0 < 1 - DT * VX - 1e-6) & (y0 > 1e-6) & (y0 < 1 - DT * VY - 1e-6)
    )
    assert interior.sum() > 0.9 * interior.size

    # slot 0 = T2 sampled at the pre-advection position; slot 1 = old slot 0
    got0 = np.asarray(lag_ddt.psi_star[0].data)[:, 0]
    got1 = np.asarray(lag_ddt.psi_star[1].data)[:, 0]
    assert np.allclose(got0[interior], 3.0 * x0[interior] - y0[interior], atol=1e-8)
    assert np.allclose(got1[interior], x0[interior] + 2.0 * y0[interior], atol=1e-8)


def test_lagrangian_vector_component_layout():
    """Vector psi: each component must land in its own canonical-storage
    column (exercises the _data_layout(i, j) mapping, not just scalars)."""
    mesh = _make_mesh()
    W = uw.discretisation.MeshVariable("W", mesh, mesh.dim, degree=1)
    wc = np.asarray(W.coords)
    W.data[:, 0] = wc[:, 0] + 2.0 * wc[:, 1]  # W_x = x + 2y
    W.data[:, 1] = 3.0 * wc[:, 0] - wc[:, 1]  # W_y = 3x - y
    V = _constant_velocity(mesh)

    lag_ddt = ddt_module.Lagrangian(
        mesh=mesh,
        psi_fn=W.sym,
        V_fn=V.sym,
        vtype=uw.VarType.VECTOR,
        degree=1,
        continuous=True,
        order=1,
        fill_param=2,
    )

    lag_ddt.update_pre_solve(dt=DT)

    coords = np.asarray(lag_ddt.swarm._particle_coordinates.data)
    got = np.asarray(lag_ddt.psi_star[0].data)
    assert got.shape[1] == mesh.dim
    assert np.allclose(got[:, 0], coords[:, 0] + 2.0 * coords[:, 1], atol=1e-10)
    assert np.allclose(got[:, 1], 3.0 * coords[:, 0] - coords[:, 1], atol=1e-10)


def test_lagrangian_swarm_advecting_history_with_step_averaging():
    """Lagrangian_Swarm (user swarm): initialise, blend with step_averaging,
    then advect the user swarm and blend again at the new positions."""
    mesh = _make_mesh()
    T = _linear_scalar_field(mesh, "T", 1.0, 2.0)  # T1 = x + 2y
    V = _constant_velocity(mesh)

    swarm = uw.swarm.Swarm(mesh)
    lag_ddt = ddt_module.Lagrangian_Swarm(
        swarm=swarm,
        psi_fn=T.sym,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
        order=2,
        step_averaging=2,  # phi = 1/2 blending
    )
    swarm.populate(fill_param=2)

    # --- first update: auto-initialise (old write path raised here) ---
    lag_ddt.update_pre_solve(dt=DT)
    coords0 = np.asarray(swarm._particle_coordinates.data).copy()
    T1 = lambda x, y: x + 2.0 * y
    got0 = np.asarray(lag_ddt.psi_star[0].data)[:, 0]
    assert np.allclose(got0, T1(coords0[:, 0], coords0[:, 1]), atol=1e-10)

    # --- post-solve without advection: blended update, shifted chain ---
    _set_linear(T, 3.0, -1.0)  # T2 = 3x - y
    T2 = lambda x, y: 3.0 * x - y
    lag_ddt.update_post_solve(dt=DT)

    # positions unchanged (Lagrangian_Swarm never advects the swarm itself)
    coords_after = np.asarray(swarm._particle_coordinates.data)
    assert np.allclose(coords_after, coords0)

    x, y = coords0[:, 0], coords0[:, 1]
    blend1 = 0.5 * T2(x, y) + 0.5 * T1(x, y)
    got0 = np.asarray(lag_ddt.psi_star[0].data)[:, 0]
    got1 = np.asarray(lag_ddt.psi_star[1].data)[:, 0]
    assert np.allclose(got0, blend1, atol=1e-10)
    assert np.allclose(got1, T1(x, y), atol=1e-10)  # chain copy of old slot 0

    # --- user advects the swarm, then another blended update ---
    swarm.advection(
        V.sym, delta_t=DT, restore_points_to_domain_func=mesh.return_coords_to_bounds
    )
    _set_linear(T, -1.0, 4.0)  # T3 = -x + 4y
    T3 = lambda x, y: -x + 4.0 * y
    lag_ddt.update_post_solve(dt=DT)

    coords1 = np.asarray(swarm._particle_coordinates.data)
    # pre-advection positions (constant V => exact rewind); skip clamped points
    x0 = coords1[:, 0] - DT * VX
    y0 = coords1[:, 1] - DT * VY
    interior = (
        (x0 > 1e-6) & (x0 < 1 - DT * VX - 1e-6) & (y0 > 1e-6) & (y0 < 1 - DT * VY - 1e-6)
    )
    assert interior.sum() > 0.9 * interior.size

    # slot 0 = phi*T3(current pos) + (1-phi)*(previous per-particle blend),
    # where the previous blend is a known function of the pre-advection pos.
    expected0 = 0.5 * T3(coords1[:, 0], coords1[:, 1]) + 0.5 * (
        0.5 * T2(x0, y0) + 0.5 * T1(x0, y0)
    )
    got0 = np.asarray(lag_ddt.psi_star[0].data)[:, 0]
    got1 = np.asarray(lag_ddt.psi_star[1].data)[:, 0]
    assert np.allclose(got0[interior], expected0[interior], atol=1e-8)
    # slot 1 = the previous blend, carried with the particle
    assert np.allclose(
        got1[interior], 0.5 * T2(x0[interior], y0[interior]) + 0.5 * T1(x0[interior], y0[interior]),
        atol=1e-8,
    )
