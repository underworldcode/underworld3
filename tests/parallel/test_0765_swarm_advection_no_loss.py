"""
Regression test for swarm particle loss during advection across processor boundaries.

Bug: ``Swarm.advection()`` finished with ``self.migrate(delete_lost_points=True,
max_its=1)``. ``max_its=1`` only tries the *closest* domain centroid for each
unclaimed particle, so any particle whose owning rank happened to be the
2nd-or-3rd closest centroid (typical near a process boundary) failed
``points_in_domain`` on its sole try and got deleted.

Reproducer adapted from @bknight1's report on issue #175.

See: https://github.com/underworldcode/underworld3/issues/175

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0765_swarm_advection_no_loss.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0765_swarm_advection_no_loss.py
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw
from mpi4py import MPI

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(60)]


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_advection_preserves_particle_count_order1():
    """Order-1 advection across rank boundaries must not lose particles."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
    )

    swarm = uw.swarm.Swarm(mesh)
    var = uw.swarm.SwarmVariable("test_var", swarm, 1)
    swarm.populate(fill_param=1)
    var.data[...] = uw.mpi.rank

    comm = uw.mpi.comm
    initial_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)

    # Constant rightward velocity; dt=0.6 sweeps particles ~6 cells across the
    # domain in one step. With max_its=1 in the post-advection migrate this
    # used to drop boundary particles.
    v_fn = sympy.Matrix([1.0, 0.0])
    swarm.advection(v_fn, 0.6, order=1)

    final_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)

    assert final_count == initial_count, (
        f"Lost {initial_count - final_count} particles "
        f"(initial={initial_count}, final={final_count})"
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_advection_preserves_particle_count_order2():
    """Order-2 mid-point advection across rank boundaries must not lose particles.

    Uses solid-body rotation about the domain centre so particles stay inside
    the unit box for the whole rotation. This isolates rank-boundary loss
    from genuine domain-exit, while still forcing every particle across
    multiple processor cuts over many substeps.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
    )

    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=1)

    comm = uw.mpi.comm
    initial_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)

    # Solid-body rotation about (0.5, 0.5) — radius <= sqrt(0.5) so nothing
    # can escape the unit box. omega=2*pi makes one revolution in dt=1.
    x, y = mesh.X
    omega = 2 * sympy.pi
    v_fn = sympy.Matrix([-omega * (y - 0.5), omega * (x - 0.5)])
    swarm.advection(v_fn, 0.5, order=2, step_limit=True)

    final_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)

    assert final_count == initial_count, (
        f"Lost {initial_count - final_count} particles "
        f"(initial={initial_count}, final={final_count})"
    )
