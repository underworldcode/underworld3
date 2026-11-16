"""A swarm sees a mover's new node positions (#966).

The swarm's caches that depend on where the mesh nodes are (the proxy's
interpolation operator, the cell projector, and the solve-entry check that
re-bins the particles) were keyed on ``mesh._mesh_version``, which only
``mesh.deform`` and adaptation bump; a mover or a snapshot restore moves the
nodes without it. So after a mover the proxy was evaluated at the pre-move
nodes, and the particles were not re-binned at the next solve. They now key on
``mesh._geometry_version``, which every change of the coordinates bumps.

Each test moves the interior nodes of an annulus by spring smoothing (the
largest move is about 2e-2) and checks a closed form: a proxy of ``s = x`` is
exactly linear, so at every node it must equal the node's own ``x``.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _swarm_of_x(location):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8)
    swarm = uw.swarm.Swarm(mesh)
    s = uw.swarm.SwarmVariable(f"s_{location}", swarm, 1, proxy_degree=1,
                               proxy_location=location)
    swarm.populate(fill_param=3)
    s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, 0]
    return mesh, swarm, s


def _proxy_error(s):
    proxy = s._meshVar
    s._update_proxy_if_stale()
    return np.abs(np.asarray(proxy.array).ravel() - np.asarray(proxy.coords)[:, 0]).max()


# How exactly each proxy reproduces a linear field: the node proxy's moving
# least squares to round-off; the cell proxy's per-cell fit is a least-squares
# solve, measured at 1.1e-10 on this annulus before any move. A proxy left at
# the pre-move nodes is off by the node move, about 2e-2.
EXACT = {"nodes": 1.0e-12, "cells": 1.0e-9}


@pytest.mark.parametrize("location", ["nodes", "cells"])
def test_a_proxy_after_a_mover_is_evaluated_at_the_moved_nodes(location):
    mesh, swarm, s = _swarm_of_x(location)
    assert _proxy_error(s) < EXACT[location]
    before = np.asarray(mesh.X.coords).copy()
    uw.meshing.smoothing.smooth_mesh_interior(mesh, method="spring")
    assert np.linalg.norm(np.asarray(mesh.X.coords) - before, axis=1).max() > 1.0e-2
    s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, 0]     # a rewrite marks it stale
    assert _proxy_error(s) < EXACT[location]


def test_the_particles_are_rebinned_at_the_first_solve_after_a_mover(monkeypatch):
    mesh, swarm, s = _swarm_of_x("nodes")
    calls = []
    original = swarm.migrate
    monkeypatch.setattr(swarm, "migrate", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    swarm._sync_before_assembly()
    assert calls == []                      # nothing moved: no re-binning
    uw.meshing.smoothing.smooth_mesh_interior(mesh, method="spring")
    swarm._sync_before_assembly()
    assert calls == [1]


def test_a_cell_proxy_after_a_snapshot_restore_is_evaluated_at_the_restored_nodes():
    # The projector built on the moved nodes must not survive the restore that
    # puts the nodes back.
    model = uw.get_default_model()
    mesh, swarm, s = _swarm_of_x("cells")
    before = np.asarray(mesh.X.coords).copy()
    token = model.save_state()
    uw.meshing.smoothing.smooth_mesh_interior(mesh, method="spring")
    s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, 0]
    assert _proxy_error(s) < EXACT["cells"]          # a projector on the moved nodes
    model.load_state(token)
    assert np.array_equal(np.asarray(mesh.X.coords), before)
    s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, 0]
    assert _proxy_error(s) < EXACT["cells"]
