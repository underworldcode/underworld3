"""Every coarse fault copy has fine images in the parallel split-level transfer
(#1120).

The parallel half of ``tests/test_1120_split_transfer_fault_copies.py``. The
fault is cut into an adapt child that the add-on partition has moved, so the
transfer between the two finest split levels is ``fault_mg.
cell_located_transfer_parallel``. Its copy columns are summed over every rank's
owned rows, and every rank agrees which copies are weak; the recorded parent's
rows are taken for the rows that reach them.

The test captures the finest split pair's call from a solve, builds the
transfer again, and checks every coarse fault copy's column norm (``|P|^T 1``,
over the global matrix) against a tenth of the median copy column.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from fault_fixtures import TRACE, adapted_2d, shear_fault_solve
from underworld3.utilities import fault_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900, method="thread")]

K = 2
DEGREE = 2


def _finest_split_pair_call():
    calls = []
    original = fault_mg.cell_located_transfer_parallel

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    fault_mg.cell_located_transfer_parallel = spy
    try:
        mesh = adapted_2d(K).add_fault([("Flt", TRACE)])
        _stokes, info, _slip = shear_fault_solve(mesh)
    finally:
        fault_mg.cell_located_transfer_parallel = original
    finest = [c for c in calls if c[0][1] is mesh]
    assert uw.mpi.comm.allreduce(len(finest)) == uw.mpi.size, \
        "the finest split pair did not take the cell-located transfer"
    return finest[-1], info


def _copies_by_point(mesh, degree):
    """The node rows of every split point's two copies, read from the section of
    the paired points (written here, not imported, so the test does not use the
    code it tests). Collective: the basis DM is built on every rank."""
    pairs = getattr(mesh, "_fault_point_pairs", None) or {}
    points = sorted({int(q) for pp in pairs.values() for kv in pp.items() for q in kv})
    dmb = mesh._basis_coordinate_dm(degree, True)
    sec = dmb.getLocalSection()
    rows = [r for q in points if sec.getDof(q)
            for r in range(sec.getOffset(q) // mesh.cdim,
                           (sec.getOffset(q) + sec.getDof(q)) // mesh.cdim)]
    dmb.destroy()
    return np.unique(np.asarray(rows, dtype=np.int64))


def _copy_column_norms(args, kwargs):
    coarse, _fine, degree, continuous = args[:4]
    coarse_layout, ncomp = args[6], args[8]
    P = fault_mg.cell_located_transfer_parallel(*args, **kwargs)
    # |P|, then its column sums: |P|^T 1
    A = P.duplicate(copy=True)
    indptr, indices, values = A.getValuesCSR()
    A.zeroEntries()
    A.setValuesCSR(indptr, indices, np.abs(values))
    A.assemble()
    ones = A.createVecLeft()
    ones.set(1.0)
    sums = A.createVecRight()
    A.multTranspose(ones, sums)
    lo, hi = sums.getOwnershipRange()
    copies = _copies_by_point(coarse, degree)
    l2g = np.asarray(coarse_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    cols = np.unique(l2g[copies].ravel())
    cols = cols[cols >= 0]
    every = np.unique(np.concatenate(uw.mpi.comm.allgather(cols)))
    mine = every[(every >= lo) & (every < hi)]
    values = sums.getArray()[mine - lo]
    gathered = uw.mpi.comm.allgather((mine, values))
    norms = np.zeros(every.size)
    for ids, vals in gathered:
        norms[np.searchsorted(every, ids)] = vals
    return norms


def test_every_coarse_fault_copy_has_fine_images_in_parallel():
    (args, kwargs), info = _finest_split_pair_call()
    assert info["converged"], info
    norms = _copy_column_norms(args, kwargs)
    assert norms.size > 0
    weak = norms < 0.1 * np.median(norms)
    assert not weak.any(), (
        f"np={uw.mpi.size}: {int((norms == 0).sum())} fault-copy column(s) empty and "
        f"{int(weak.sum())} below a tenth of the median copy column "
        f"({np.sort(norms)[:6]})")


def test_the_weak_copies_are_counted_as_nodes_as_in_serial(monkeypatch):
    """``last_parent_rows`` counts coarse fault-copy NODES in parallel, as the
    serial transfer does, not their per-component columns (#1206 review: the
    parallel count was twice the serial one in 2-D). The expected count is
    taken from the located transfer alone: the copy nodes with a column below
    the weak bar."""
    (args, kwargs), _info = _finest_split_pair_call()
    coarse, _fine, degree = args[:3]
    coarse_layout, ncomp = args[6], args[8]
    monkeypatch.setattr(fault_mg, "_WEAK_COPY", 0.0)        # the located rule alone
    norms_located = _copy_column_norms(args, kwargs)
    monkeypatch.undo()
    copies = _copies_by_point(coarse, degree)
    l2g = np.asarray(coarse_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    cols = np.unique(l2g[copies].ravel())
    cols = cols[cols >= 0]
    every = np.unique(np.concatenate(uw.mpi.comm.allgather(cols)))
    weak_cols = every[norms_located < 0.1 * np.median(norms_located)]
    node_cols = l2g[copies]
    node_cols = node_cols[(node_cols >= 0).any(axis=1)]
    table = np.unique(np.vstack(uw.mpi.comm.allgather(node_cols)), axis=0)
    expected = int(np.isin(table, weak_cols).any(axis=1).sum())
    assert expected > 0
    fault_mg.cell_located_transfer_parallel(*args, **kwargs)
    record = fault_mg.last_parent_rows
    assert record["weak_copies"] == expected, (record, expected, weak_cols.size)
