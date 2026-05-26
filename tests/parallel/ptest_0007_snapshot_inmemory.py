"""Parallel (MPI) test of the in-memory snapshot toolkit.

The single open production blocker for the snapshot toolkit was
"works everywhere" — i.e. correct under MPI. The design intent is
that swarm restore is a per-rank *reconstruction* (each rank clears
its local particles and re-adds the per-rank set it captured), not a
redistribution, so the global state is exactly reconstructed
regardless of any intervening cross-rank migration, provided the rank
count is unchanged. This script confirms that.

Run (4 ranks exercises cross-rank migration properly):

    cd tests/parallel
    mpirun -np 4 python ./ptest_0007_snapshot_inmemory.py

Asserts (all collective, checked on rank 0):

  1. Restore recovers the exact global particle count. The disruptive
     step is deliberately allowed to *lose* particles across ranks
     (advect out / clip) — that is exactly the failure stash-and-
     restore exists to undo. The guarantee is that restore brings the
     global count back to its pre-step value regardless.
  2. Exact reconstruction: gather every particle's (global-id, x, y,
     material) across all ranks, sort by global id; the post-restore
     sorted table equals the pre-step sorted table bit-for-bit.
     Order- and rank-independent — this is the real proof that
     per-rank reconstruction yields the correct global state under
     cross-rank migration.
  3. Bit-identical continuation across a stash: a control run and a
     run that took a disruptive step then restored and continued
     produce bit-identical global sorted state and DDt history.
"""

import numpy as np
import sympy
from mpi4py import MPI

import underworld3 as uw

comm = MPI.COMM_WORLD
rank = uw.mpi.rank
size = uw.mpi.size


def build():
    uw.reset_default_model()
    model = uw.get_default_model()
    # Wide box so a strip-partition genuinely splits particles across
    # ranks; rotation field circulates them across the partition.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(4.0, 1.0), cellSize=1.0 / 6.0
    )
    x_sym, y_sym = mesh.X
    # Rotation about the box centre (2.0, 0.5): particles circulate,
    # crossing the vertical rank-partition boundaries.
    V_fn = sympy.Matrix([[-(y_sym - 0.5), 0.25 * (x_sym - 2.0)]]).T

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.array[:, 0, 0] = 0.0

    swarm = uw.swarm.Swarm(mesh)
    gid = swarm.add_variable("gid", 1, dtype=float)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)

    # Globally-unique, migration-stable particle id. Swarm variables
    # travel with their particle through migration and through
    # snapshot/restore, so this is a durable identity tag.
    local_n = swarm.dm.getLocalSize()
    counts = comm.allgather(local_n)
    offset = int(np.sum(counts[:rank]))
    gid.data[:, 0] = offset + np.arange(local_n, dtype=float)
    material.data[:, 0] = swarm._particle_coordinates.data[:, 0]

    ddt = uw.systems.ddt.Symbolic(T.sym, order=2)
    return uw, model, mesh, V_fn, T, swarm, gid, material, ddt


def step(uw, V_fn, T, swarm, ddt, dt):
    ddt.update_pre_solve(dt)
    swarm.advection(V_fn, delta_t=dt, step_limit=False)
    T.array[:, 0, 0] = T.array[:, 0, 0] + dt
    ddt.update_post_solve(dt)


def global_sorted_particles(swarm, gid, material):
    """Gather (gid, x, y, material) from all ranks, sorted by gid.

    Order- and rank-independent canonical view of the whole swarm.
    """
    g = gid.data[:, 0].copy()
    coords = swarm._particle_coordinates.data.copy()
    m = material.data[:, 0].copy()
    local = np.column_stack([g, coords[:, 0], coords[:, 1], m])
    gathered = comm.allgather(local)
    full = np.vstack([a for a in gathered if a.size]) if any(
        a.size for a in gathered
    ) else np.empty((0, 4))
    order = np.argsort(full[:, 0], kind="stable")
    return full[order]


def main():
    uw, model, mesh, V_fn, T, swarm, gid, material, ddt = build()

    # Warm up: a few steps so particles have genuinely migrated across
    # ranks before we snapshot.
    for _ in range(3):
        step(uw, V_fn, T, swarm, ddt, 0.1)

    pre = global_sorted_particles(swarm, gid, material)
    pre_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)
    pre_ddt = (list(ddt.state.dt_history), ddt.state.n_solves_completed)

    snap = model.save_state()

    # --- Property 1 + 2: a migration-inducing step, then restore ---
    step(uw, V_fn, T, swarm, ddt, 0.3)  # bigger dt -> more migration
    mid_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)
    model.load_state(snap)
    post = global_sorted_particles(swarm, gid, material)
    post_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)
    post_ddt = (list(ddt.state.dt_history), ddt.state.n_solves_completed)

    exact = np.array_equal(pre, post)
    ddt_ok = pre_ddt == post_ddt

    # --- Property 3: bit-identical continuation across a stash ---
    snap2 = model.save_state()
    for _ in range(4):
        step(uw, V_fn, T, swarm, ddt, 0.1)
    ctrl = global_sorted_particles(swarm, gid, material)
    ctrl_ddt = (list(ddt.state.dt_history), ddt.state.n_solves_completed)

    model.load_state(snap2)
    step(uw, V_fn, T, swarm, ddt, 0.5)  # the regretted step
    model.load_state(snap2)
    for _ in range(4):
        step(uw, V_fn, T, swarm, ddt, 0.1)
    stash = global_sorted_particles(swarm, gid, material)
    stash_ddt = (list(ddt.state.dt_history), ddt.state.n_solves_completed)

    cont_exact = np.array_equal(ctrl, stash)
    cont_ddt_ok = ctrl_ddt == stash_ddt

    if rank == 0:
        lost = pre_count - mid_count
        print(f"[ranks={size}] particles total = {pre_count}", flush=True)
        print(
            f"  disruptive step global count: {pre_count} -> {mid_count} "
            f"-> {post_count}  ({lost} particle(s) lost by the step, "
            f"recovered by restore)",
            flush=True,
        )
        print(f"  P1 restore recovers exact count:  "
              f"{pre_count == post_count}", flush=True)
        print(f"  P2 exact reconstruction:          {exact}", flush=True)
        print(f"  P2 DDt state restored:            {ddt_ok}", flush=True)
        print(f"  P3 bit-identical continuation:    {cont_exact}", flush=True)
        print(f"  P3 DDt continuation identical:    {cont_ddt_ok}", flush=True)

        # The disruptive step is *allowed* to lose particles across
        # ranks — that is precisely the failure a stash-and-restore
        # exists to undo. The guarantee is that restore brings the
        # global count back exactly and every particle back to the
        # right place (P2), regardless of what the step did.
        assert pre_count == post_count, (
            f"restore did not recover the exact global particle count: "
            f"pre={pre_count} post={post_count} (mid={mid_count})"
        )
        assert exact, "swarm not exactly reconstructed after restore"
        assert ddt_ok, "DDt state not restored"
        assert cont_exact, (
            "continuation after stash is not bit-identical to control"
        )
        assert cont_ddt_ok, "DDt continuation not bit-identical"
        if lost == 0:
            print(
                "  (note: this run's disruptive step happened not to "
                "lose particles; the recovery guarantee still holds)",
                flush=True,
            )
        print(f"[ranks={size}] PASS", flush=True)


if __name__ == "__main__":
    main()
