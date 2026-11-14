"""Facet-surface slip gives the serial mesh at the rank count it is run at (#1103).

The in-process twin of ``tests/test_1103_facet_slip_matches_serial.py``, whose
serial driver launches only up to three ranks (it runs inside an xdist worker).
Here the same cases run at whatever rank count the parallel batch uses (2, and 3
and 4 in the full-parallel pass; 5 by hand), and the serial baseline is computed
by a single-rank child in this environment (``serial_reference``). The moved
mesh must match serial vertex by vertex, matched by starting position, to 1e-10.
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

CASES = ["facet_2d", "facet_2d_scattered", "internal_2d", "internal_2d_scattered"]


def _serial_driver():
    path = Path(__file__).resolve().parents[1] / "test_1103_facet_slip_matches_serial.py"
    spec = importlib.util.spec_from_file_location("_t1103", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moved(case):
    """``(before, after)`` of every vertex, sorted by starting position, on
    every rank (the driver's own move, gathered on rank 0)."""
    out = uw.mpi.comm.bcast(tempfile.mkdtemp() if uw.mpi.rank == 0 else None, root=0)
    out = os.path.join(out, "moved.json")
    _serial_driver()._move(case, out)
    data = uw.mpi.comm.bcast(json.loads(Path(out).read_text()) if uw.mpi.rank == 0 else None,
                             root=0)
    before, after = np.asarray(data["before"]), np.asarray(data["after"])
    order = np.lexsort(before.T[::-1])
    return before[order], after[order]


@pytest.mark.parametrize("case", CASES)
def test_facet_slip_matches_serial_at_this_rank_count(case):
    before, after = _moved(case)
    ref = np.asarray(serial_reference(__file__, case, timeout=600)["values"])
    ref = ref.reshape(2, -1, before.shape[1])
    np.testing.assert_array_equal(before, ref[0])
    assert np.abs(after - before).max() > 0.01                   # the mesh moved
    np.testing.assert_allclose(after, ref[1], rtol=0, atol=1e-10)


if __name__ == "__main__":
    # the single-rank child: `python <this file> <case>`
    _before, _after = _moved(sys.argv[1])
    emit(np.concatenate([_before.ravel(), _after.ravel()]), [len(_before), 0.0])
