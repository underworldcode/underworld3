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

The strays must exist, and whether a platform's partitioner leaves any is its
own business (test_0858 failed its precondition on macOS that way), so the
annulus is cut into vertical slabs by a shell partition, the cuts off centre:
each midpoint between neighbouring domain centroids then misses its cut, and
the strip between them holds points of one rank whose nearest centroid is the
other's, on any platform. The #1093 point itself is asked on the platform's
own partition, where it was found.

A loop that gave up after two rounds would pass those: a stray's domain is its
second-nearest. So the cells are also dealt to the ranks at random (a seeded
shell partition), where a point's domain is often the third-nearest by
centroid or further, and points like that are asked, many at once and one on
its own (from np 3).

Points outside the mesh, just beyond its outer wall and in its hole, are
offered only to the ranks whose padded domain box holds them, and stay on the
last of those (or, in no box, on the nearest domain): never on a rank with no
cells, and in at most two rounds on the slab partition.
"""

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]

R_OUTER, R_INNER = 1 / 0.45, 0.55 / 0.45


def _scatter(dmod, ranks=None):
    """Install a shell partition dealing the loaded DM's cells to the ranks
    at random (seeded, so the same on every platform); to the first
    ``ranks`` only, if given, so the others hold no cells."""
    from petsc4py import PETSc

    original = dmod._from_plexh5

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        sf0, dm = result
        cs, ce = dm.getHeightStratum(0)
        owner = np.random.default_rng(1093).integers(0, ranks or uw.mpi.size, ce - cs)
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(
            uw.mpi.size, sizes=np.bincount(owner, minlength=uw.mpi.size).astype(np.int32),
            points=np.argsort(owner, kind="stable").astype(np.int32))
        return sf0, dm

    dmod._from_plexh5 = load
    return original


def _slabs(dmod):
    """Install a shell partition cutting the loaded DM (every cell on rank 0)
    into vertical slabs, one per rank, at off-centre cuts, before the Mesh
    constructor distributes it."""
    from petsc4py import PETSc

    original = dmod._from_plexh5

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        sf0, dm = result
        cs, ce = dm.getHeightStratum(0)
        x = np.array([dm.computeCellGeometryFVM(c)[1][0] for c in range(cs, ce)])
        cuts = np.linspace(-R_OUTER, R_OUTER, uw.mpi.size + 1)[1:-1] + 0.3
        owner = np.searchsorted(cuts, x).astype(np.int64)
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(
            uw.mpi.size, sizes=np.bincount(owner, minlength=uw.mpi.size).astype(np.int32),
            points=np.argsort(owner, kind="stable").astype(np.int32))
        return sf0, dm

    dmod._from_plexh5 = load
    return original


def _annulus(layout):
    """The cavity annulus of #1086 (refinement 2), with f = 2x + 3y on P1, on
    the platform's partition (None), in slabs, or scattered."""
    dmod = uw.discretisation.discretisation_mesh
    install = {"slabs": _slabs, "scatter": _scatter,
               "last_empty": lambda d: _scatter(d, uw.mpi.size - 1)}.get(layout)
    original = install(dmod) if install is not None and uw.mpi.size > 1 else None
    try:
        mesh = uw.meshing.Annulus(radiusOuter=R_OUTER, radiusInner=R_INNER, cellSize=0.5,
                                  refinement=2, qdegree=3)
    finally:
        if original is not None:
            dmod._from_plexh5 = original
    f = uw.discretisation.MeshVariable(f"f1093{layout or ''}", mesh, 1, degree=1)
    X = np.asarray(f.coords)
    f.array[:, 0, 0] = 2 * X[:, 0] + 3 * X[:, 1]
    return mesh, f


@pytest.fixture(scope="module")
def annulus():
    return _annulus(None)


@pytest.fixture(scope="module")
def slab_annulus():
    return _annulus("slabs")


@pytest.fixture(scope="module")
def scattered_annulus():
    return _annulus("scatter")


@pytest.fixture(scope="module")
def last_empty_annulus():
    return _annulus("last_empty")


def _deep(mesh, seed, n=2000):
    """Points well inside the annulus whose domain is the third-nearest by
    centroid or further (a loop of two rounds misses them). Collective."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(R_INNER + 0.02, R_OUTER - 0.02, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    # whose domain holds each point, from every rank's answer for every point
    every = np.concatenate(uw.mpi.comm.allgather(points))
    mine = np.asarray(mesh.points_in_domain(every), dtype=bool)
    held = np.array(uw.mpi.comm.allgather(mine))               # (ranks, points)
    owner = np.where(held.any(axis=0), held.argmax(axis=0), -1)
    _, order = mesh._get_domain_kdtree().query(every, k=uw.mpi.size, sqr_dists=False)
    order = np.asarray(order).reshape(len(every), -1)
    depth = np.array([int(np.flatnonzero(order[i] == owner[i])[0]) if owner[i] >= 0 else -1
                      for i in range(len(every))])
    start = sum(uw.mpi.comm.allgather(n)[:uw.mpi.rank])
    deep = depth[start:start + n] >= 2
    assert uw.mpi.comm.allreduce(int(deep.sum())) > 0
    return points[deep]


def _needs_three_domains():
    if uw.mpi.size < 3:
        pytest.skip("a third-nearest domain needs three ranks")


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
def test_interior_strays_read_exactly_from_every_rank(slab_annulus):
    mesh, f = slab_annulus
    err, outside = _max_error(f, _strays(mesh, 1093 + uw.mpi.rank))
    assert outside == 0 and err < 1e-12, (err, outside)


@pytest.mark.tier_b
def test_a_swarm_keeps_every_interior_stray(slab_annulus):
    mesh, _ = slab_annulus
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


@pytest.mark.tier_b
def test_points_in_a_third_nearest_domain_read_exactly(scattered_annulus):
    _needs_three_domains()
    mesh, f = scattered_annulus
    err, outside = _max_error(f, _deep(mesh, 3093 + uw.mpi.rank))
    assert outside == 0 and err < 1e-12, (err, outside)


@pytest.mark.tier_b
def test_a_swarm_keeps_points_in_a_third_nearest_domain(scattered_annulus):
    _needs_three_domains()
    mesh, _ = scattered_annulus
    points = _deep(mesh, 4093 + uw.mpi.rank)
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_global_coordinates(points, migrate=False)
    total = uw.mpi.comm.allreduce(len(points))
    swarm.migrate(remove_sent_points=True, delete_lost_points=True)
    held = np.asarray(swarm.dm.getField("DMSwarmPIC_coor")).reshape(-1, 2).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")
    assert uw.mpi.comm.allreduce(len(held)) == total
    inside = mesh.points_in_domain(held)          # collective: every rank, even with none
    assert uw.mpi.comm.allreduce(int((~np.asarray(inside, dtype=bool)).sum())) == 0


@pytest.mark.tier_b
def test_a_single_point_in_a_third_nearest_domain(scattered_annulus):
    # one point, asked by one rank: no other point keeps the loop going
    _needs_three_domains()
    mesh, f = scattered_annulus
    deep = np.concatenate(uw.mpi.comm.allgather(_deep(mesh, 5093 + uw.mpi.rank)))
    asked = deep[:1] if uw.mpi.rank == uw.mpi.size - 1 else np.zeros((0, 2))
    err, outside = _max_error(f, asked)
    assert outside == 0 and err < 1e-12, (err, outside)


def _exterior(seed, n=500):
    """Points just outside the outer wall (up to 0.02 out) and in the hole."""
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.uniform(R_OUTER + 1e-4, R_OUTER + 0.02, n),
                        rng.uniform(R_INNER - 0.2, R_INNER - 1e-4, n)])
    theta = rng.uniform(0, 2 * np.pi, 2 * n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _place_exterior(mesh, seed):
    """Migrate exterior points; returns (held coordinates, rounds taken)."""
    points = _exterior(seed + uw.mpi.rank)
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_global_coordinates(points, migrate=False)
    total = uw.mpi.comm.allreduce(len(points))
    calls = [0]
    classify = mesh.points_in_domain

    def counted(*args, **kwargs):
        calls[0] += 1
        return classify(*args, **kwargs)

    mesh.points_in_domain = counted
    try:
        swarm.migrate(remove_sent_points=True, delete_lost_points=False)
    finally:
        del mesh.points_in_domain
    held = np.asarray(swarm.dm.getField("DMSwarmPIC_coor")).reshape(-1, 2).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")
    assert uw.mpi.comm.allreduce(len(held)) == total       # none lost
    return held, uw.mpi.comm.allreduce(calls[0] - 1, op=MPI.MAX)


def _where_they_belong(mesh, held):
    """Each exterior point on a rank whose padded box holds it, or, in no
    box, on its nearest domain; and none on a rank with no cells."""
    lo, hi = mesh._get_domain_boxes()
    centroids = mesh._get_domain_centroids()
    start, end = mesh.dm.getHeightStratum(0)
    if end == start:
        assert len(held) == 0, "a rank with no cells holds exterior points"
        return
    me = uw.mpi.rank
    in_box = np.all((held >= lo[me]) & (held <= hi[me]), axis=1)
    in_any = np.array([np.all((h >= lo) & (h <= hi), axis=1).any() for h in held], dtype=bool)
    nearest = np.argmin(np.linalg.norm(held[:, None, :] - centroids[None], axis=2), axis=1)
    ok = in_box | (~in_any & (nearest == me))
    assert ok.all(), held[~ok][:5]


@pytest.mark.tier_b
def test_exterior_points_stay_near_where_they_could_be(slab_annulus):
    mesh, _ = slab_annulus
    held, rounds = _place_exterior(mesh, 6093)
    _where_they_belong(mesh, held)
    assert rounds <= 2, rounds


@pytest.mark.tier_b
def test_exterior_points_never_reach_a_rank_with_no_cells(last_empty_annulus):
    mesh, _ = last_empty_annulus
    counts = uw.mpi.comm.allgather(int(np.subtract(*mesh.dm.getHeightStratum(0)[::-1])))
    assert counts[-1] == 0 and all(counts[:-1])
    held, _ = _place_exterior(mesh, 7093)
    _where_they_belong(mesh, held)
