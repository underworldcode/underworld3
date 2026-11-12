"""A point inside the mesh reaches the rank whose cell contains it (#1093).

``Swarm.migrate`` offers each point it has not placed to its nearest domain
(by domain centroid), then to the next nearest, and so on. It also stopped
as soon as a round claimed nothing. A point inside the mesh near a partition
seam, whose own rank's centroid is nearest but whose cell is on the next rank,
claims nothing in the first round, because it is offered to its own rank; the
loop stopped there and left it on a rank that does not contain it. A swarm
then lost it (``delete_lost_points``), and ``global_evaluate``, which moves
query points with a swarm, read it from the out-of-domain fallback: a value
extrapolated from the wrong rank, partition-dependent. On the cavity annulus
of #1086 at np 2 one interior point read 5.761 where serial reads 5.058, and
the MMPDE mover's default frozen metric drifted 7e-3 from serial through it.

Checked here, with a P1 field that is linear, so every point FE locates reads
exactly:

* the #1093 point, asked by the last rank;
* interior points that each rank holds but does not contain, and whose
  nearest domain centroid is its own (the points that used to stall the
  loop), through ``global_evaluate``: every one exact, none flagged
  extrapolated;
* such points as a swarm: none lost, each on a rank whose domain holds it.
"""

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]

R_OUTER, R_INNER = 1 / 0.45, 0.55 / 0.45


@pytest.fixture(scope="module")
def annulus():
    """The cavity annulus of #1086 (refinement 2), with f = 2x + 3y on P1."""
    mesh = uw.meshing.Annulus(radiusOuter=R_OUTER, radiusInner=R_INNER, cellSize=0.5,
                              refinement=2, qdegree=3)
    f = uw.discretisation.MeshVariable("f1093", mesh, 1, degree=1)
    X = np.asarray(f.coords)
    f.array[:, 0, 0] = 2 * X[:, 0] + 3 * X[:, 1]
    return mesh, f


def _strays(mesh, seed, n=4000):
    """Points well inside the annulus (0.02 from either wall) that this rank
    does not contain but whose nearest domain centroid is this rank's own: the
    points a first migration round offers back to the rank holding them. Each
    rank asks only for these, so no round claims anything until a point is
    offered to its second-nearest domain."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(R_INNER + 0.02, R_OUTER - 0.02, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    mine = np.asarray(mesh.points_in_domain(points), dtype=bool)      # collective
    _, nearest = mesh._get_domain_kdtree().query(points, k=1, sqr_dists=False)
    strays = points[~mine & (np.ravel(nearest) == uw.mpi.rank)]
    assert uw.mpi.comm.allreduce(len(strays)) > 0, "the partition gave no strays"
    return strays


def _max_error(f, points):
    values, outside = uw.function.global_evaluate(f.sym[0], points, check_extrapolated=True)
    exact = 2 * points[:, 0] + 3 * points[:, 1]
    err = float(np.abs(np.ravel(values) - exact).max()) if len(points) else 0.0
    return (uw.mpi.comm.allreduce(err, op=MPI.MAX),
            uw.mpi.comm.allreduce(int(np.asarray(outside).sum())))


@pytest.mark.tier_b
def test_the_1093_point(annulus):
    mesh, f = annulus
    point = np.array([[1.6355282291554059, -0.0015870618771772818]])
    asked = point if uw.mpi.rank == uw.mpi.size - 1 else np.zeros((0, 2))
    err, outside = _max_error(f, asked)
    assert outside == 0 and err < 1e-12, (err, outside)


@pytest.mark.tier_b
def test_interior_strays_read_exactly_from_every_rank(annulus):
    mesh, f = annulus
    err, outside = _max_error(f, _strays(mesh, 1093 + uw.mpi.rank))
    assert outside == 0 and err < 1e-12, (err, outside)


@pytest.mark.tier_b
def test_a_swarm_keeps_every_interior_stray(annulus):
    mesh, _ = annulus
    points = _strays(mesh, 2093 + uw.mpi.rank)
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_global_coordinates(points, migrate=False)
    total = uw.mpi.comm.allreduce(len(points))
    swarm.migrate(remove_sent_points=True, delete_lost_points=True)
    held = np.asarray(swarm.dm.getField("DMSwarmPIC_coor")).reshape(-1, 2).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")
    assert uw.mpi.comm.allreduce(len(held)) == total
    inside = mesh.points_in_domain(held)          # collective: every rank, even with none
    assert uw.mpi.comm.allreduce(int((~np.asarray(inside, dtype=bool)).sum())) == 0
