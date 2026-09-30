"""Swarm.repopulate: particles added to starved cells and removed from over-full ones.

The trigger is the per-cell census; a starved cell is filled from its own
lattice at the points farthest from the particles present; a new particle
takes the linear-exact RBF reconstruction of every variable from the nearest
existing particles (or a supplied value). ``swarm.population_control`` makes
advection() end with a repopulation, which is what keeps a cells proxy well
posed on a flow that empties cells.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _census(swarm):
    mesh = swarm.mesh
    c0, c1 = mesh.dm.getHeightStratum(0)
    X = np.asarray(swarm._particle_coordinates.data)
    cells = np.asarray(mesh._robust_owning_cells(X))
    return np.bincount(cells[cells >= 0], minlength=c1 - c0)


def _strip_left(swarm, xmax):
    """Test helper: delete every particle with x < xmax (no public removal API)."""
    X = np.asarray(swarm._particle_coordinates.data)
    for index in np.sort(np.nonzero(X[:, 0] < xmax)[0])[::-1]:
        swarm.dm.removePointAtIndex(int(index))
    swarm._invalidate_canonical_data()
    swarm._population_generation += 1


def test_starved_cells_are_refilled_with_linear_exact_values():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2)
    x, y = mesh.X
    swarm = uw.swarm.Swarm(mesh)
    T = uw.swarm.SwarmVariable("T", swarm, 1, proxy_location="cells", proxy_degree=2)
    V = uw.swarm.SwarmVariable("V", swarm, vtype=uw.VarType.VECTOR)
    swarm.populate(fill_param=3)
    n_full = swarm.local_size
    before = _census(swarm)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        T.data[:, 0] = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]
        V.data[:, 0] = X[:, 0]
        V.data[:, 1] = -X[:, 1]
    _strip_left(swarm, 0.35)
    assert (_census(swarm) == 0).sum() > 0             # cells emptied
    added, removed = swarm.repopulate(order=1)      # linear-exact reconstruction
    assert removed == 0 and added > 0
    after = _census(swarm)
    # Every owned cell holds at least the lattice count again (10 for fill 3)
    assert (after >= before.min()).all(), (after.min(), before.min())
    # Values: the linear fields are reconstructed exactly on the new particles
    X = np.asarray(swarm._particle_coordinates.data)
    assert np.allclose(np.asarray(T.data[:, 0]), 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1], atol=1e-8)
    assert np.allclose(np.asarray(V.data), np.c_[X[:, 0], -X[:, 1]], atol=1e-8)
    # ... and the cells proxy reproduces the field through the weak form
    err = uw.maths.Integral(mesh, (T.sym[0] - (1 + 2 * x + 3 * y)) ** 2).evaluate()
    assert err < 1e-14, err
    assert T._cell_projector.n_empty == 0
    # Idempotent: nothing to do on a healthy swarm
    assert swarm.repopulate() == (0, 0)
    assert swarm.local_size <= n_full + 0            # no more than the original lattice


def test_values_override_and_cap():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2)
    swarm = uw.swarm.Swarm(mesh)
    M = uw.swarm.SwarmVariable("M", swarm, 1)
    swarm.populate(fill_param=3)
    with uw.synchronised_array_update():
        M.data[:, 0] = 1.0
    _strip_left(swarm, 0.5)
    added, _ = swarm.repopulate(values={M: 7.0})   # default (Shepard) elsewhere
    assert added > 0
    X = np.asarray(swarm._particle_coordinates.data)
    vals = np.asarray(M.data[:, 0])
    assert np.allclose(vals[X[:, 0] < 0.5], 7.0) and np.allclose(vals[X[:, 0] >= 0.5], 1.0)
    # A callable datum, and a cap that thins over-full cells
    swarm.repopulate(min_per_cell=14, values={"M": lambda C: 2.0 * C[:, 1:2]})
    assert (_census(swarm) >= 14).all()
    added, removed = swarm.repopulate(max_per_cell=8)
    assert added == 0 and removed > 0
    assert (_census(swarm) <= 8).all()


def test_population_control_keeps_cells_filled_under_rotation():
    """Solid-body rotation of a box empties the corner cells; with population
    control the cells proxy never sees an empty cell."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=2)
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    swarm = uw.swarm.Swarm(mesh)
    T = uw.swarm.SwarmVariable("T", swarm, 1, proxy_location="cells", proxy_degree=2)
    swarm.populate(fill_param=3)
    with uw.synchronised_array_update():
        T.data[:, 0] = np.asarray(swarm._particle_coordinates.data)[:, 0]
    swarm.population_control = dict(min_per_cell=6)
    for _ in range(4):
        swarm.advection(V, 0.25, order=2)
        assert (_census(swarm) >= 6).all()
    T._update_proxy_if_stale()
    assert T._cell_projector.n_empty == 0
    # The refilled corners carry the RBF reconstruction of x: bounded, no P2 blow-up
    assert np.abs(np.asarray(T.data[:, 0])).max() < 1.5
