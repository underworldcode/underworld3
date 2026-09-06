"""The radius ACCESSORS must not depend on how the mesh was partitioned.

``test_1077`` covers the per-cell field cell by cell. This file covers
``get_min_radius()``, ``get_max_radius()`` and ``get_mean_radius()``, which
reduce that field and advertise a global mesh length -- and which were the half
left partition-dependent when only ``cell_size()`` was fixed: ``get_max_radius()``
moved 4.9% at np=8 and ``get_mean_radius()`` at every rank count.

The defect (#569, #687, #694): the per-cell length came from a kd-tree over
THIS RANK's centroids, queried with each cell's vertices. Near a partition
boundary the true nearest centroid can belong to a cell owned by another rank
and so be absent from the tree, and the answer moved with the rank count --
per-cell by 3.3e-03 at np=2, ``get_max_radius()`` by 4.9% at np=8,
``get_mean_radius()`` at every rank count.

It reached users through ``mesh.cell_size()``, which scales the Nitsche penalty
under the default ``local_h=True``, and through the three radius accessors,
whose docstrings advertise a global mesh length.

Two things this file is careful about, both learned the hard way:

* **It compares two rank counts.** Partition independence is a statement about
  two runs agreeing, so nothing measured at a single rank count establishes it.
  The tests that shipped with the first attempt at this fix asserted a
  within-rank oracle -- each cell matching its own vertices -- which is true of
  a partition-dependent field as well.
* **The reference is computed here, not recorded.** ``serial_reference`` runs
  this module's own ``__main__`` at np=1 in this environment and asserts the
  mesh fingerprints match, so a host that triangulates differently is reported
  as that rather than as partition dependence.
"""

import numpy as np
import pytest

import underworld3 as uw
from mpi4py import MPI

from serial_reference import compare, emit, mesh_fingerprint, serial_reference

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_a]

LABELS = ("min radius", "max radius", "mean radius", "sum of cell radii")

# min and max are exact reductions of identical per-cell values, so they must
# agree to the bit. The mean is a distributed sum and reduces in partition
# order, so it is allowed the last couple of bits -- that ordering difference
# is not the defect under test.
RTOLS = (0.0, 0.0, 1.0e-12, 1.0e-12)


def _box():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.12, qdegree=2)
    # mesh_fingerprint integrates over the mesh, and integration needs at
    # least one variable to exist on it.
    uw.discretisation.MeshVariable("cell_size_probe", mesh, 1, degree=1)
    return mesh


def _cell_size_diagnostics():
    """The three accessors, plus a global sum that no single one of them sees.

    The sum is the sharp one: a per-cell change that leaves the extremes alone
    still moves it, and the old field's per-cell values differed while its
    minimum happened not to.
    """
    mesh = _box()
    radii = np.asarray(mesh._cell_radii).reshape(-1)
    local_sum = float(radii.sum())
    global_sum = uw.mpi.comm.allreduce(local_sum, op=MPI.SUM) \
        if uw.mpi.size > 1 else local_sum

    values = (
        mesh.get_min_radius(),
        mesh.get_max_radius(),
        mesh.get_mean_radius(),
        global_sum,
    )
    return values, mesh_fingerprint(mesh)


@pytest.mark.mpi(min_size=2)
def test_cell_size_is_the_same_at_every_rank_count():
    """The accessors and the per-cell sum reproduce their own np=1 answer."""
    values, fingerprint = _cell_size_diagnostics()
    compare(values, serial_reference(__file__, "cell_size"),
            rtols=RTOLS, labels=LABELS, fingerprint=fingerprint,
            what="cell size / radius accessors")


@pytest.mark.mpi(min_size=2)
def test_every_rank_agrees_on_the_accessors():
    """The weaker property, kept because it is the one the allreduce gives.

    An allreduce makes an answer identical on every rank; it does not make it
    identical at every rank count if the values being reduced are themselves
    partition-dependent. Failing this while passing the test above would mean
    the reduction is broken rather than its input, so keeping both separates
    the two.
    """
    mesh = _box()
    for name in ("min", "max", "mean"):
        got = getattr(mesh, f"get_{name}_radius")()
        everyone = uw.mpi.comm.allgather(got)
        assert len(set(everyone)) == 1, (
            f"get_{name}_radius() differs between ranks: {everyone}"
        )


if __name__ == "__main__":
    import sys

    _kind = sys.argv[1] if len(sys.argv) > 1 else "cell_size"
    if _kind == "cell_size":
        _values, _fingerprint = _cell_size_diagnostics()
        emit(_values, _fingerprint)
    else:
        raise SystemExit(f"unknown kind {_kind!r}")
