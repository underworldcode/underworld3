"""An internal boundary has no outward normal, on any partition (#1135).

``Mesh._assemble_boundary_normal`` skips a facet that is not exterior: one with
two cells. In parallel a facet of an internal boundary that lies on a partition
seam has only one LOCAL cell, so it passed as exterior there and the rank
assembled a normal for ``Internal`` that serial does not. The facet-slip normal
(``smoothing._slip_normals``) refuses a vertex that more than one boundary
claims, so on a 3-D ``BoxInternalBoundary`` the vertices where ``Internal``
meets a side were refused their slide on some partitions and slid in serial:
up to 1.2e-2 off serial, against a move of 4.1e-2. A seam facet is now
recognised through the point star forest (``Mesh._seam_points``) and skipped.

* ``test_an_internal_boundary_assembles_no_normal``: the assembled ``Internal``
  normal is zero at every vertex on every rank (serial: zero everywhere).
* ``test_facet_slip_matches_serial_on_a_scattered_partition``: facet slip with
  every label slipping matches serial vertex by vertex to 1e-10, on a random
  cell partition (seed 3).

Measured before the fix (#1135), on a 96-thread Linux workstation, the largest
distance from serial: 5.1e-3 at np 3, 7.0e-3 at np 4, 1.2e-2 at np 6 and 8; after
it, 9e-13 to 1.4e-12. At np 2 the defect does not show (no seed from 1 to 12
gives Internal a normal there), so ``scripts/test.sh`` runs this file at np 3
on every run, as well as in the np 2 batch.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import underworld3 as uw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serial_reference import emit, serial_reference  # noqa: E402

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900, method="thread")]

SEED = 3


def _driver():
    path = Path(__file__).resolve().parents[1] / "test_1103_facet_slip_matches_serial.py"
    spec = importlib.util.spec_from_file_location("_t1103", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mesh():
    """The issue's mesh, on a random cell partition when run in parallel."""
    dmod = uw.discretisation.discretisation_mesh
    original = _driver()._scatter_cells(dmod, seed=SEED) if uw.mpi.size > 1 else None
    try:
        mesh = uw.meshing.BoxInternalBoundary(simplex=True, cellSize=0.25,
                                              minCoords=(0.0, 0.0, 0.0),
                                              maxCoords=(1.0, 1.0, 1.0), zintCoord=0.5)
    finally:
        if original is not None:
            dmod._from_plexh5 = original
    return mesh


def _move():
    """Facet slip with every label slipping (no declared surfaces): ``(before,
    after)`` of every vertex, sorted by starting position, on every rank."""
    import sympy

    mesh = _mesh()
    for label in list(mesh.bounding_surfaces):
        del mesh.bounding_surfaces[label]
    labels = tuple(b.name for b in mesh.boundaries
                   if b.name not in ("All_Boundaries", "Null_Boundary", "Centre"))
    before = np.asarray(mesh.X.coords).copy()
    X = mesh.X
    metric = 1.0 + 4.0 * sympy.exp(-sum((X[i] - c) ** 2 for i, c in
                                        enumerate((0.3, 0.35, 0.4))) / 0.04)
    mesh.redistribute_nodes(metric, pinned_labels=labels, slip_surfaces=True,
                            method_kwargs=dict(n_outer=5, metric_eval="sympy"))
    after = np.asarray(mesh.X.coords).copy()
    before = np.concatenate(uw.mpi.comm.allgather(before))
    after = np.concatenate(uw.mpi.comm.allgather(after))
    _, first = np.unique(before, axis=0, return_index=True)
    before, after = before[first], after[first]
    order = np.lexsort(before.T[::-1])
    return before[order], after[order]


def test_an_internal_boundary_assembles_no_normal():
    mesh = _mesh()
    mesh._update_projected_normals()
    scratch = mesh._projected_normals
    mesh._assemble_boundary_normal(scratch, "Internal")
    worst = float(np.abs(np.asarray(scratch.data)).max()) if scratch.data.size else 0.0
    worst = uw.mpi.comm.allreduce(worst, op=__import__("mpi4py").MPI.MAX)
    assert worst == 0.0, f"np={uw.mpi.size}: Internal assembled a normal of size {worst}"
    # an exterior boundary still assembles its normal
    mesh._assemble_boundary_normal(scratch, "Front")
    front = float(np.abs(np.asarray(scratch.data)).max()) if scratch.data.size else 0.0
    assert uw.mpi.comm.allreduce(front, op=__import__("mpi4py").MPI.MAX) == pytest.approx(1.0)


def test_facet_slip_matches_serial_on_a_scattered_partition():
    before, after = _move()
    ref = np.asarray(serial_reference(__file__, "move", timeout=600)["values"])
    ref = ref.reshape(2, -1, 3)
    np.testing.assert_array_equal(before, ref[0])
    assert np.abs(after - before).max() > 0.01                   # the mesh moved
    np.testing.assert_allclose(after, ref[1], rtol=0, atol=1e-10)


if __name__ == "__main__":
    before, after = _move()
    emit(np.concatenate([before.ravel(), after.ravel()]), [0.0, 0.0])
