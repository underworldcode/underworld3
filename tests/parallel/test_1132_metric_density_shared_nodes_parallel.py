"""The gradient density metric counts each node once, whatever the partition (#1132).

``metric_density_from_gradient`` evaluates ``|grad field|`` at every local DOF,
gathers the values, and ranks them: the percentile window and, on the
front-following path, every node's rank within the gathered list. A node on a
partition seam is a local DOF on each rank that holds it, so it used to enter
the list once per copy, and the copies are evaluated separately and can differ
in the last bit. With ties in ``|grad field|`` (a plateau, a uniform region) one
bit decides whether a seam node's copy sits at the bottom or the top of a tie
block, and the density there moved by O(1) from launch to launch.

The owner's value is now synced to every copy before the ranking, only owned
DOFs (the field's global section: the point star forest's ownership) enter the
gathered list, and the density is synced owner to ghost afterwards.

The gradient here is replaced by a staircase of exact ties, a function of the
coordinates alone, so the test pins the ranking and not the projection solve:

* ``test_seam_copy_last_bit``: one rank's ghost copies are nudged by one ulp.
  The density at every node is unchanged, bit for bit, and every copy of a node
  agrees with its owner.
* ``test_ties_match_serial``: the density at every node is the np=1 value, bit
  for bit (``serial_reference`` runs this module's ``__main__`` at np=1 on the
  same mesh). Run at np 2 and 4.

Nodes are matched across rank counts by their coordinates, which are the same
bits on every rank and at every rank count on one mesh: this is a test's way to
line up two partitions, not a way the code identifies points.
"""
import contextlib
import sys

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing import metrics

from serial_reference import compare, emit, mesh_fingerprint, serial_reference

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_a,
              pytest.mark.timeout(600, method="thread")]

CASES = {
    "front-following": dict(refinement=4.0),
    "percentile": dict(strategy="med"),
    "raw": dict(mode="raw"),
}


def _mesh():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.07, qdegree=2)
    T = uw.discretisation.MeshVariable("T1132", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    return mesh, T


def _staircase(coords):
    """|grad| on a staircase: 8 x 8 plateaus of exactly equal values."""
    x, y = coords[:, 0], coords[:, 1]
    g = np.zeros_like(coords)
    g[:, 0] = 1.0 + np.floor(8.0 * x) / 8.0 + np.floor(8.0 * y) / 64.0
    return g


@contextlib.contextmanager
def _gradient(nudge_ghosts):
    """Replace the evaluated gradient with the staircase; when ``nudge_ghosts``,
    nudge by one ulp the rows that one rank (the one with most ghosts) does not own."""
    real = uw.function.evaluate

    def fake(expr, coords, *args, **kwargs):
        out = real(expr, coords, *args, **kwargs)   # keep it collective
        if sys._getframe(1).f_code.co_name != "metric_density_from_gradient":
            return out
        out = np.asarray(out)
        g = _staircase(np.asarray(coords))
        if nudge_ghosts and uw.mpi.rank == _RHO["nudged_rank"]:
            ghost = _RHO["ghost"]           # computed on every rank beforehand
            g[ghost, 0] = np.nextafter(g[ghost, 0], np.inf)
        return g.reshape(out.shape)

    uw.function.evaluate = fake
    try:
        yield
    finally:
        uw.function.evaluate = real


_RHO = {}


def _owned_rows(var):
    """Rows of a scalar MeshVariable this rank owns: a non-negative offset in the
    field's global section (written here, not imported, so the test runs on
    development too)."""
    iset, subdm = var.mesh.dm.createSubDM(var.field_id)
    local, glob = subdm.getLocalSection(), subdm.getGlobalSection()
    owned = np.zeros(np.asarray(var.data).shape[0], dtype=bool)
    for point in range(*local.getChart()):
        dof = local.getDof(point)
        if dof and glob.getOffset(point) >= 0:
            owned[local.getOffset(point):local.getOffset(point) + dof] = True
    iset.destroy()
    subdm.destroy()
    return owned


def _density(mesh, T, case, nudge_ghosts=False):
    with _gradient(nudge_ghosts):
        metrics.metric_density_from_gradient(mesh, T, **CASES[case])
    rho = [v[2] for k, v in metrics._MDG_CACHE.items() if k[0] == id(mesh)][0]
    return np.asarray(rho.data[:, 0]).copy(), rho


def _by_coordinate(rho, values):
    """Every node's value, once (from its owner), in coordinate order, on every rank."""
    owned = _owned_rows(rho)
    xy = np.asarray(rho.coords)[owned]
    v = values[owned]
    xy = np.concatenate(uw.mpi.comm.allgather(xy))
    v = np.concatenate(uw.mpi.comm.allgather(v))
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    return v[order]


def _ghosts_agree(rho, values):
    """Does every copy of a node hold its owner's value?"""
    probe = values.copy()
    rho.pack_raw_data_to_petsc(probe.reshape(-1, 1), sync=True)
    synced = np.asarray(rho.data[:, 0]).copy()
    rho.pack_raw_data_to_petsc(values.reshape(-1, 1), sync=True)
    return int(uw.mpi.comm.allreduce(int((synced != values).sum())))


@pytest.mark.parametrize("case", sorted(CASES))
def test_seam_copy_last_bit(case):
    mesh, T = _mesh()
    # The first call builds the cached variables; the nudge needs them to find ghosts.
    clean, rho = _density(mesh, T, case)
    _RHO["ghost"] = ~_owned_rows(rho)           # collective: on every rank
    counts = uw.mpi.comm.allgather(int(_RHO["ghost"].sum()))
    _RHO["nudged_rank"] = int(np.argmax(counts))   # the rank with most ghost rows
    ghosts = max(counts)
    nudged, _ = _density(mesh, T, case, nudge_ghosts=True)
    assert ghosts > 0, "no seam nodes at this rank count; the test measures nothing"
    moved = int(uw.mpi.comm.allreduce(int((nudged != clean).sum())))
    worst = uw.mpi.comm.allreduce(float(np.max(np.abs(nudged - clean), initial=0.0)),
                                  op=MPI.MAX)
    assert moved == 0, (f"{case}: a one-ulp nudge to one rank's seam copies moved the "
                        f"density at {moved} local DOFs, by up to {worst:.3g}")
    assert _ghosts_agree(rho, nudged) == 0, f"{case}: a ghost copy differs from its owner"


@pytest.mark.parametrize("case", sorted(CASES))
def test_ties_match_serial(case):
    mesh, T = _mesh()
    values, rho = _density(mesh, T, case)
    assert _ghosts_agree(rho, values) == 0, f"{case}: a ghost copy differs from its owner"
    ordered = _by_coordinate(rho, values)
    reference = serial_reference(__file__, case)
    labels = [f"node {i}" for i in range(len(ordered))]
    compare(ordered, reference, [0.0] * len(ordered), labels,
            mesh_fingerprint(mesh), f"{case} density, np={uw.mpi.size} vs np=1")


if __name__ == "__main__":
    case = sys.argv[1]
    mesh, T = _mesh()
    values, rho = _density(mesh, T, case)
    emit(_by_coordinate(rho, values), mesh_fingerprint(mesh))
