"""With empty ranks, the MMPDE mover moves the mesh exactly as in serial (#1082).

``UnstructuredSimplexBox(cellSize=1)`` has four cells. At np 4 the partition
is ``[2, 0, 2, 0]``: two ranks hold no cells. Those ranks used to return early
while the others entered the mover's collective reductions, and the job
aborted. Returning on every rank together would avoid the abort but skip the
move. Instead the empty ranks take part in every reduction with nothing to
add, and the mesh moves.

The mover runs once serially and once at np 4, each in a subprocess through
the MPI supervisor (this file is also the worker: ``python -m mpi4py <this
file> <out.json>``). The metric is off the box's centre, so the one interior
vertex moves (by 0.031). Both runs must give the same energy history and the
same final coordinates, vertex by vertex, matched by their starting position,
to rounding. Measured on 2026-11-12: the coordinates agree exactly, and the
energies to the last bit.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _move(out):
    import sympy
    import underworld3 as uw
    from underworld3.meshing.smoothing import _mmpde_mover

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0)
    cs, ce = mesh.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(ce - cs)
    before = np.asarray(mesh.X.coords).copy()
    x, y = mesh.X
    metric = 1.0 + 4.0 * sympy.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.04)
    energies = _mmpde_mover(mesh, metric, ("Top", "Bottom", "Left", "Right"), False,
                            n_outer=2, metric_eval="sympy")
    after = np.asarray(mesh.X.coords).copy()
    before = uw.mpi.comm.gather(before, root=0)
    after = uw.mpi.comm.gather(after, root=0)
    if uw.mpi.rank == 0:
        before, after = np.concatenate(before), np.concatenate(after)
        # Shared vertices appear on more than one rank; keep each once.
        _, first = np.unique(before, axis=0, return_index=True)
        Path(out).write_text(json.dumps({
            "counts": counts, "energies": [float(e) for e in energies],
            "before": before[first].tolist(), "after": after[first].tolist()}))


def _run(ranks, out):
    root = Path(__file__).resolve().parents[1]
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks)]
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "120", "--",
               *launcher, sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()), str(out)]
    log = out.with_suffix(".log")
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=out.parent, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=300)
    assert result.returncode == 0, log.read_text()[-4000:]
    return json.loads(out.read_text())


pytestmark = [pytest.mark.level_2, pytest.mark.timeout(600)]


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
def test_mover_with_empty_ranks_matches_serial(tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and np 4 movers.")
    serial = _run(1, tmp_path / "serial.json")
    parallel = _run(4, tmp_path / "np4.json")
    if min(parallel["counts"]) > 0:
        pytest.skip(f"no rank is empty at np 4 (cells per rank {parallel['counts']})")

    # Equal to rounding: the reductions sum in a different order.
    np.testing.assert_allclose(parallel["energies"], serial["energies"], rtol=1e-12, atol=0)
    before = np.asarray(serial["before"])
    order = [int(np.flatnonzero((before == v).all(axis=1))[0]) for v in parallel["before"]]
    assert len(set(order)) == len(before)
    after = np.asarray(serial["after"])[order]
    assert np.abs(after - before[order]).max() > 0.01           # the mesh moved
    np.testing.assert_allclose(np.asarray(parallel["after"]), after, rtol=0, atol=1e-12)


if __name__ == "__main__":
    _move(sys.argv[1])
