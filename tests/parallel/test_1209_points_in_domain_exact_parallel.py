"""points_in_domain uses the exact locator on simplex meshes, so a swarm's claims
agree with global_evaluate's ownership.

points_in_domain, which a swarm's migration reads, decided a point far from the
local boundary by a boundary-face sign test standing in for containment (Style
Charter section 12). On #1123's sheared box the sign test calls a contained point
exterior: at np 2, development leaves one inside point owned by no rank, so its
particle is on no rank. Now a point is in a rank's domain when a local cell
contains it, or when development's face tolerance near the boundary claims it.

The truth is a brute-force barycentric test over every local cell, in numpy.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1209_points_in_domain_exact_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities._plex_topology import closure_points

import sheared_box_fixture

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _inside_some_local_cell(mesh, P):
    cs, ce = mesh.dm.getHeightStratum(0)
    inside = np.zeros(P.shape[0], dtype=bool)
    if ce == cs:
        return inside
    Xl = np.asarray(mesh._coords).reshape(-1, mesh.cdim)
    V = Xl[mesh._coord_rows_for_points(
        mesh.dm, closure_points(mesh.dm, np.arange(cs, ce), 0).ravel())].reshape(ce - cs, -1, mesh.cdim)
    Ti = np.linalg.inv(np.transpose(V[:, 1:, :] - V[:, :1, :], (0, 2, 1)))
    for start in range(0, P.shape[0], 256):
        Q = P[start:start + 256]
        lam = np.einsum("cij,ncj->nci", Ti, Q[:, None, :] - V[None, :, 0, :])
        inside[start:start + 256] = (np.minimum(lam.min(axis=2), 1.0 - lam.sum(axis=2)) >= 0.0).any(axis=1)
    return inside


def _near_edges(mesh):
    dm = mesh.dm
    es, ee = dm.getDepthStratum(1)
    vs = dm.getDepthStratum(0)[0]
    Xv = np.asarray(mesh._coords).reshape(-1, 3)
    mids = np.array([0.5 * (Xv[dm.getCone(e)[0] - vs] + Xv[dm.getCone(e)[1] - vs])
                     for e in range(es, ee)]).reshape(-1, 3)
    mids = np.unique(np.round(np.vstack(uw.mpi.comm.allgather(mids)), 13), axis=0)
    rng = np.random.default_rng(3)
    unit = lambda n: n / np.linalg.norm(n, axis=1)[:, None]
    return np.vstack([mids + d * unit(rng.normal(size=mids.shape)) for d in (1e-8, 1e-6, 1e-4)])


def test_every_inside_point_is_in_some_ranks_domain_and_its_particle_survives():
    mesh = sheared_box_fixture.sheared_box()
    P = _near_edges(mesh)
    inside = uw.mpi.comm.allreduce(_inside_some_local_cell(mesh, P).astype(np.int64), op=MPI.SUM) > 0
    owned = np.asarray(mesh.points_in_domain(P, strict_validation=True), dtype=np.int64)
    owners = uw.mpi.comm.allreduce(owned, op=MPI.SUM)
    missed = int((inside & (owners == 0)).sum())
    assert missed == 0, f"{missed} inside points in no rank's domain"

    swarm = uw.swarm.Swarm(mesh)
    seed = P[inside] if uw.mpi.rank == 0 else np.zeros((0, 3))
    swarm.add_particles_with_global_coordinates(seed, migrate=True, delete_lost_points=True)
    total = uw.mpi.comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)
    assert total == int(inside.sum()), f"{total} particles kept of {int(inside.sum())} inside"
