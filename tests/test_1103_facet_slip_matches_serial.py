"""Facet-surface slip gives the serial mesh on any partition (#1103).

A slip boundary with no registered analytic surface slides its vertices back
onto a ``facet`` surface: the nearest point on the label's boundary facets.
Each rank built that surface from its own facets of the label only. A vertex
that slid past the end of its rank's facets was clamped to their end, so the
moved mesh differed from serial by 3e-2 to 5e-2 at np 2 to 4. A rank owning
slip vertices of a label without any of its facets raised
(``argmin of an empty sequence``) while the others waited: at the natural
np 5 partition of this box the job hung. The facets of each label are now
gathered over the ranks, so every rank restores onto the serial set.

The box's registered plane surfaces are removed, so its four sides slip on
facet surfaces. Each case runs ``Mesh.redistribute_nodes`` serially and in
parallel, in subprocesses through the MPI supervisor (this file is also the
worker: ``python -m mpi4py <this file> <case> <out.json>``), and the parallel
mesh must match the serial one vertex by vertex, matched by starting position.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# case -> (dimension, cellSize, rank counts, scatter the cells?)
CASES = {
    "facet_2d": (2, 0.25, (2, 3, 4, 5), False),
    "facet_2d_scattered": (2, 0.25, (3, 4), True),
    "facet_3d": (3, 0.35, (2, 3), False),
}


def _scatter_cells(dmod, seed=1103):
    """A shell partition dealing the cells to ranks at random, installed on
    the loaded DM (every cell on rank 0) before the Mesh distributes it."""
    from petsc4py import PETSc
    import underworld3 as uw

    original = dmod._from_plexh5

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        sf0, dm = result
        cs, ce = dm.getHeightStratum(0)
        owner = np.random.default_rng(seed).integers(0, uw.mpi.size, ce - cs)
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(
            uw.mpi.size, sizes=np.bincount(owner, minlength=uw.mpi.size).astype(np.int32),
            points=np.argsort(owner, kind="stable").astype(np.int32))
        return sf0, dm

    dmod._from_plexh5 = load
    return original


def _move(case, out):
    import sympy
    import underworld3 as uw

    dim, cell_size, _, scatter = CASES[case]
    dmod = uw.discretisation.discretisation_mesh
    original = _scatter_cells(dmod) if scatter and uw.mpi.size > 1 else None
    try:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
                                                 cellSize=cell_size)
    finally:
        if original is not None:
            dmod._from_plexh5 = original
    for label in list(mesh.bounding_surfaces):
        del mesh.bounding_surfaces[label]
    before = np.asarray(mesh.X.coords).copy()
    X = mesh.X
    centre = (0.3, 0.35, 0.4)[:dim]
    metric = 1.0 + 4.0 * sympy.exp(-sum((X[i] - centre[i]) ** 2 for i in range(dim)) / 0.04)
    sides = ("Top", "Bottom", "Left", "Right") + (("Front", "Back") if dim == 3 else ())
    mesh.redistribute_nodes(metric, pinned_labels=sides, slip_surfaces=True,
                            method_kwargs=dict(n_outer=10 if dim == 2 else 5,
                                               metric_eval="sympy"))
    before = uw.mpi.comm.gather(before, root=0)
    after = uw.mpi.comm.gather(np.asarray(mesh.X.coords).copy(), root=0)
    if uw.mpi.rank == 0:
        before, after = np.concatenate(before), np.concatenate(after)
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

_serial = {}


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
@pytest.mark.parametrize("case,ranks", [(case, n) for case, spec in CASES.items()
                                         for n in spec[2]])
def test_facet_slip_matches_serial(case, ranks, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    if case not in _serial:
        _serial[case] = _run(case, 1, tmp_path / "serial.json")
    serial = _serial[case]
    parallel = _run(case, ranks, tmp_path / "parallel.json")

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
