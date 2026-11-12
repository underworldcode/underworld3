"""Node redistribution gives the serial mesh on any partition (#1086).

A volume mesh is distributed without overlap, so a vertex on a partition seam
sees only its own rank's cells. Two per-vertex quantities in the MMPDE
redistribution path were read from those cells alone:

* the step cap, from the vertex's shortest incident edge. A seam vertex
  missed some edges, took a longer first step, and the mesh drifted from
  serial by about 3e-3 on a 42-cell box (up to 1.3e-2 measured);
* which boundaries a vertex lies on, from the labelled facets in its
  closure. A box corner whose two edges are on different ranks counted one
  boundary on each, was classed as a slip vertex instead of pinned, and slid
  off the domain (by up to 0.37 with ``slip_surfaces=True``).

Both are now combined over every rank holding the vertex. Each case runs
``Mesh.redistribute_nodes`` serially and in parallel, in subprocesses
through the MPI supervisor (this file is also the worker: ``python -m mpi4py
<this file> <case> <out.json>``), and the parallel mesh must match the serial
one vertex by vertex, matched by starting position, to rounding. Measured on
2026-11-12: within 4e-12 in every case.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# case -> (dimension, cellSize, slip_surfaces, ranks)
CASES = {
    "pinned_2d": (2, 0.25, False, 2),
    "slip_2d": (2, 0.25, True, 3),
    "slip_corners_2d": (2, 1.0, True, 2),
    "slip_3d": (3, 0.35, True, 2),
}


def _move(case, out):
    import sympy
    import underworld3 as uw

    dim, cell_size, slip, _ = CASES[case]
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
                                             cellSize=cell_size)
    before = np.asarray(mesh.X.coords).copy()
    X = mesh.X
    centre = (0.3, 0.35, 0.4)[:dim]
    metric = 1.0 + 4.0 * sympy.exp(-sum((X[i] - centre[i]) ** 2 for i in range(dim)) / 0.04)
    sides = ("Top", "Bottom", "Left", "Right") + (("Front", "Back") if dim == 3 else ())
    mesh.redistribute_nodes(metric, pinned_labels=sides, slip_surfaces=slip,
                            method_kwargs=dict(n_outer=10 if dim == 2 else 5,
                                               metric_eval="sympy"))
    after = np.asarray(mesh.X.coords).copy()
    before = uw.mpi.comm.gather(before, root=0)
    after = uw.mpi.comm.gather(after, root=0)
    if uw.mpi.rank == 0:
        before, after = np.concatenate(before), np.concatenate(after)
        # Shared vertices appear on more than one rank; keep each once.
        _, first = np.unique(before, axis=0, return_index=True)
        Path(out).write_text(json.dumps({"before": before[first].tolist(),
                                         "after": after[first].tolist()}))


def _run(case, ranks, out):
    root = Path(__file__).resolve().parents[1]
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks)]
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "120", "--",
               *launcher, sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()),
               case, str(out)]
    log = out.with_suffix(".log")
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=out.parent, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=600)
    assert result.returncode == 0, log.read_text()[-4000:]
    return json.loads(out.read_text())


pytestmark = [pytest.mark.level_2, pytest.mark.timeout(1200)]


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
@pytest.mark.parametrize("case", list(CASES))
def test_redistribution_matches_serial(case, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    serial = _run(case, 1, tmp_path / "serial.json")
    parallel = _run(case, CASES[case][3], tmp_path / "parallel.json")

    before = np.asarray(serial["before"])
    order = [int(np.flatnonzero((before == v).all(axis=1))[0]) for v in parallel["before"]]
    assert len(set(order)) == len(before)
    after = np.asarray(serial["after"])[order]
    assert np.abs(after - before[order]).max() > 0.01           # the mesh moved
    moved = np.asarray(parallel["after"])
    assert ((moved > -1e-12) & (moved < 1 + 1e-12)).all(), "a vertex left the box"
    np.testing.assert_allclose(moved, after, rtol=0, atol=1e-10)


if __name__ == "__main__":
    _move(sys.argv[1], sys.argv[2])
