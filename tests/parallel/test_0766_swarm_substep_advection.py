"""
Regression test for substep advection across partition boundaries (SWARM-16 / BF-16).

``Swarm.advection()`` with ``substeps > 1`` evaluates the launch-point velocity
of every substep, but no migration happens inside the substep loop (deferred
migration is deliberately suspended there — see #313). From substep 2 onward a
particle may therefore sit outside its rank's domain when its launch-point
velocity is evaluated, and a rank-local evaluation returns extrapolated
(wrong) values for it. The launch-point evaluation must be a *global*
evaluation, like the midpoint evaluation already is.

The test advects a ring of particles through a solid-body rotation stored in a
degree-1 MeshVariable (linear field, so FE interpolation is exact to roundoff)
and compares each particle's final position against an exact numpy replication
of the substepped midpoint scheme. Any wrong-rank evaluation after a particle
crosses the partition seam shows up as a position error far above roundoff.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0766_swarm_substep_advection.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0766_swarm_substep_advection.py
"""

import pytest
import numpy as np
import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(300)]


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_substep_advection_across_partition_matches_scheme():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))

    # Solid-body rotation about the domain centre, stored in a degree-1
    # variable: a linear field, represented exactly by the element basis, so
    # a correctly-located evaluation reproduces it to machine precision.
    V = uw.discretisation.MeshVariable("V_sub_adv", mesh, mesh.dim, degree=1)
    V.array[:, 0, 0] = -(V.coords[:, 1] - 0.5)
    V.array[:, 0, 1] = V.coords[:, 0] - 0.5

    swarm = uw.swarm.Swarm(mesh)
    pid = uw.swarm.SwarmVariable("pid_sub_adv", swarm, 1, dtype=int, _proxy=False)

    # Ring of particles that the rotation sweeps across the partition seam.
    # The small angular offset keeps every launch point strictly inside one
    # rank's cells (a point exactly on the seam is claimed by both ranks).
    n_pts = 32
    theta = 0.03 + np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    pts = np.column_stack([0.5 + 0.3 * np.cos(theta), 0.5 + 0.3 * np.sin(theta)])

    # Same array on every rank; each rank inserts only the points it owns.
    swarm.add_particles_with_coordinates(pts)

    # Tag each local particle with the index of its (exactly matching)
    # launch coordinate so trajectories can be compared after migration.
    local0 = swarm._particle_coordinates.data
    d2 = ((local0[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    assert d2.size == 0 or d2.min(axis=1).max() < 1.0e-24
    pid.data[:, 0] = d2.argmin(axis=1) if d2.size else np.empty((0,), dtype=int)

    comm = uw.mpi.comm
    n_global = comm.allreduce(local0.shape[0])
    assert n_global == n_pts

    # Force a fixed number of substeps through the step_limit machinery
    # (substeps = round(|dt| / dt_limit) inside advection()).
    n_sub = 6
    dt_limit = swarm.estimate_dt(V.sym)
    delta_t = n_sub * dt_limit

    swarm.advection(V.sym, delta_t, order=2, step_limit=True)

    # Exact replication of the substepped midpoint scheme with the analytic
    # (linear) velocity — what advection() must produce when every velocity
    # evaluation is performed on the rank that owns the point.
    def v_exact(x):
        return np.column_stack([-(x[:, 1] - 0.5), x[:, 0] - 0.5])

    expected = pts.copy()
    dt_sub = delta_t / n_sub
    for _ in range(n_sub):
        mid = expected + 0.5 * dt_sub * v_exact(expected)
        expected = expected + dt_sub * v_exact(mid)

    # Gather (pid, final position) from every rank and compare per particle.
    # np.asarray: the NDArray_With_Callback wrapper cannot be pickled by
    # allgather (closure-defined callback), so gather plain copies.
    final_local = np.asarray(swarm._particle_coordinates.data).copy()
    pid_local = np.asarray(pid.data[:, 0]).copy()

    all_final = np.concatenate(comm.allgather(final_local), axis=0)
    all_pid = np.concatenate(comm.allgather(pid_local), axis=0)

    assert all_final.shape[0] == n_pts, (
        f"Lost particles during substepped advection: {all_final.shape[0]} of {n_pts}"
    )

    order = np.argsort(all_pid)
    np.testing.assert_allclose(
        all_final[order],
        expected,
        atol=1.0e-8,
        err_msg="Substepped advection deviates from the exact midpoint scheme — "
        "launch-point velocities are being evaluated on the wrong rank "
        "after particles cross a partition boundary (SWARM-16).",
    )
