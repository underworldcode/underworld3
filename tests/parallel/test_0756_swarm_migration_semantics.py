"""Parallel regression tests for swarm migration semantics (Track-0 audit).

Run under MPI, e.g.::

    mpirun -np 2 python -m pytest tests/parallel/test_0756_swarm_migration_semantics.py
    mpirun -np 4 python -m pytest tests/parallel/test_0756_swarm_migration_semantics.py

Covers:

- SWARM-04 / BF-05: writes made inside ``migration_disabled()`` /
  ``migration_control()`` were silently discarded (the PETSc sync callbacks
  early-returned and nothing ever re-packed). Writes must now be flushed to
  the DMSwarm at context exit; only the *migration* is suppressed/deferred.
- SWARM-03 / BF-07: particle coordinate writes through the modern interface
  (``swarm._particle_coordinates.data``) never triggered migration, leaving
  particles on the wrong rank. Migration is now deferred to the next
  collective point (context exit or solve entry) — never per-write, which
  would deadlock when ranks write unevenly.
- SWARM-07 / BF-06: ranks holding <= 1 particles either crashed
  (``KDTree`` on an empty array inside ``IndexSwarmVariable``) or silently
  zeroed their proxy values. Starved ranks must leave the proxy untouched
  and warn.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox


def _box(cell=0.25):
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell
    )


# --------------------------------------------------------------------------
# SWARM-04 / BF-05
# --------------------------------------------------------------------------

@pytest.mark.mpi(min_size=2)
def test_variable_writes_inside_migration_disabled_reach_petsc():
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable("val", 1)
    swarm.populate(fill_param=3)

    with swarm.migration_disabled():
        var.data[:, 0] = 42.0

    raw = var.unpack_raw_data_from_petsc(squeeze=False, sync=False)
    assert raw.shape[0] == swarm.local_size
    assert np.all(raw == 42.0), (
        "variable write inside migration_disabled() was discarded: PETSc "
        f"field max is {raw.max() if raw.size else 'empty'}, expected 42.0"
    )

    del swarm
    del mesh


@pytest.mark.mpi(min_size=2)
def test_coordinate_writes_inside_migration_control_survive_and_migrate():
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable("val", 1)
    swarm.populate(fill_param=3)

    var.data[:, 0] = 1.0
    global_before = uw.mpi.comm.allreduce(int(swarm.local_size), uw.MPI.SUM)

    # contract every particle towards the domain centre: particles owned by
    # the "outer" portions of each rank's partition change owner.
    with swarm.migration_control():
        coords = swarm._particle_coordinates.data
        swarm._particle_coordinates.data[...] = 0.5 + (coords - 0.5) * 0.4

    # deferred migration ran at context exit: every particle must now be
    # inside its owning rank's local domain and none may have been lost.
    # NB: points_in_domain is COLLECTIVE (get_max_radius gathers) — every
    # rank must call it, including ranks whose local swarm is empty.
    local_coords = swarm._particle_coordinates.data
    in_domain = mesh.points_in_domain(np.asarray(local_coords))
    assert np.all(in_domain), (
        f"rank {uw.mpi.rank}: {np.count_nonzero(~in_domain)} particles "
        "left on the wrong rank after migration_control() exit"
    )

    global_after = uw.mpi.comm.allreduce(int(swarm.local_size), uw.MPI.SUM)
    assert global_after == global_before, (
        f"particle count changed across deferred migration: "
        f"{global_before} -> {global_after}"
    )

    # variable payload survives the migration
    vmax = var.global_max()
    vmin = var.global_min()
    assert abs(float(vmax) - 1.0) < 1e-12
    assert abs(float(vmin) - 1.0) < 1e-12

    del swarm
    del mesh


# --------------------------------------------------------------------------
# SWARM-03 / BF-07
# --------------------------------------------------------------------------

@pytest.mark.mpi(min_size=2)
def test_bare_coordinate_write_migrates_at_solve_entry():
    """A bare coordinate write (no context manager) marks the swarm as
    needing migration; the next collective point (solve entry) performs it."""
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=3)

    global_before = uw.mpi.comm.allreduce(int(swarm.local_size), uw.MPI.SUM)

    # push every particle into the left 40% of the box: on >= 2 ranks a
    # whole rank's worth of particles changes owner.
    coords = swarm._particle_coordinates.data
    swarm._particle_coordinates.data[...] = np.column_stack(
        [coords[:, 0] * 0.4 + 0.01, coords[:, 1]]
    )

    assert swarm._needs_migration, (
        "coordinate write through _particle_coordinates.data did not mark "
        "the swarm for deferred migration"
    )

    # a solve is a collective point: its entry must run the deferred migration
    proj_var = uw.discretisation.MeshVariable("pv0756", mesh, 1, degree=1)
    proj = uw.systems.Projection(mesh, proj_var)
    x, y = mesh.X
    proj.uw_function = x
    proj.petsc_options.delValue("ksp_monitor")
    proj.solve()

    assert not swarm._needs_migration

    # NB: points_in_domain is COLLECTIVE (get_max_radius gathers) — every
    # rank must call it; ranks emptied by the migration pass an empty array.
    local_coords = swarm._particle_coordinates.data
    in_domain = mesh.points_in_domain(np.asarray(local_coords))
    assert np.all(in_domain), (
        f"rank {uw.mpi.rank}: particles still on the wrong rank after "
        "solve-entry deferred migration"
    )

    global_after = uw.mpi.comm.allreduce(int(swarm.local_size), uw.MPI.SUM)
    assert global_after == global_before

    del swarm
    del mesh


# --------------------------------------------------------------------------
# SWARM-07 / BF-06
# --------------------------------------------------------------------------

@pytest.mark.mpi(min_size=4)
def test_starved_ranks_leave_proxy_untouched_and_do_not_crash():
    """All particles are seeded into one corner of the box (one rank's
    subdomain). Ranks with <= 1 local particles must neither crash in the
    proxy update (KDTree on an empty array) nor overwrite their proxy nodal
    values with silent zeros."""
    import warnings

    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    scal = swarm.add_variable("s", 1, proxy_degree=1)
    mat = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)

    # cluster near the origin corner — same array passed on every rank;
    # add_particles_with_coordinates keeps only locally-owned points.
    xs = np.linspace(0.02, 0.18, 6)
    xx, yy = np.meshgrid(xs, xs)
    cluster = np.column_stack([xx.ravel(), yy.ravel()])
    swarm.add_particles_with_coordinates(cluster)

    starved = swarm.local_size <= 1
    n_starved = uw.mpi.comm.allreduce(int(starved), uw.MPI.SUM)
    assert n_starved >= 1, "test premise: at least one rank must be starved"
    assert n_starved < uw.mpi.size, "test premise: one rank holds the cluster"

    scal.data[:, 0] = 7.0
    mat.data[...] = 0

    # sentinel the proxy nodal values everywhere (collective writes)
    SENTINEL = 123.0
    scal._meshVar.data[...] = SENTINEL
    for lv in mat._meshLevelSetVars:
        lv.data[...] = SENTINEL

    # force a refresh on every rank (collective; this is what the solve-entry
    # hook does). Starved ranks warn and keep their values.
    scal._proxy_stale = True
    mat._proxy_stale = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scal._update_proxy_if_stale()
        mat._update_proxy_if_stale()  # crashed with IndexError before BF-06

    if starved:
        assert any("particle" in str(w.message).lower() for w in caught), (
            f"rank {uw.mpi.rank}: starved rank did not warn about skipping "
            "the proxy update"
        )
        # owned nodes keep the sentinel (ghost nodes may take values from a
        # populated neighbour, so require a majority rather than all).
        frac = float(np.mean(np.isclose(np.asarray(scal._meshVar.data), SENTINEL)))
        assert frac > 0.5, (
            f"rank {uw.mpi.rank}: starved-rank proxy was overwritten "
            f"(only {frac:.0%} of nodes kept the sentinel — silent zeros?)"
        )
        frac_m = float(
            np.mean(np.isclose(np.asarray(mat._meshLevelSetVars[0].data), SENTINEL))
        )
        assert frac_m > 0.5, (
            f"rank {uw.mpi.rank}: starved-rank IndexSwarmVariable proxy was "
            f"overwritten (only {frac_m:.0%} kept the sentinel)"
        )
    else:
        # the populated rank really did refresh
        assert not np.allclose(np.asarray(scal._meshVar.data), SENTINEL)

    del swarm
    del mesh
