"""A field the mover remaps must reach the solvers, not only ``var.array``.

Kernels and integrals read the mesh-wide local vector, rebuilt by
``update_lvec`` only when ``_stale_lvec`` is set. The remap wrote through a plain
view of ``var.data``, so that flag stayed clear: ``T.array`` held the remapped
values while every solve saw the old values on the moved nodes. On an annulus with
a sharp thermal layer this put 3 % (np 1) and 4 % (np 4) on the v_rms of the first
solve after a move.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def test_integral_after_a_move_sees_the_remapped_field():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0766", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)), T.coords).reshape(-1)
    mesh.update_lvec()      # the state a solve leaves: the mesh vector is current

    start = np.array(T.array)
    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    assert np.abs(np.array(T.array) - start).max() > 1.0e-3      # the remap did something

    seen_by_a_solve = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    T.array[:, 0, 0] = np.array(T.array[:, 0, 0])       # a write the mesh vector does notice
    remapped = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    assert seen_by_a_solve == pytest.approx(remapped, rel=1.0e-12)


def test_a_swarm_proxy_after_a_move_is_evaluated_at_the_moved_nodes():
    """The proxy's interpolation operator is keyed on the mesh version, which only
    Mesh.deform bumped: after a mover it stayed at the old node positions, even once
    the swarm variable was rewritten."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    swarm = uw.swarm.Swarm(mesh)
    s = uw.swarm.SwarmVariable("s0766", swarm, 1, proxy_degree=1)
    swarm.populate(fill_param=3)
    s.array[:, 0, 0] = swarm._particle_coordinates.data[:, 0]
    mesh.update_lvec()      # the proxy has been evaluated before the move

    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))

    s.array[:, 0, 0] = swarm._particle_coordinates.data[:, 0]
    mesh.update_lvec()      # what a solve does first: refreshes stale proxies
    proxy = s._meshVar
    error = np.abs(np.asarray(proxy.array).reshape(-1) - np.asarray(proxy.coords)[:, 0])
    assert error.max() < 1.0e-6
