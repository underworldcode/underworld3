"""With empty ranks, node redistribution moves the mesh exactly as in serial (#1082).

Every step of the MMPDE mover is collective. A rank holding no cells used to
leave early, at four places, while the others went on into collectives, and
the job aborted or hung:

* ``_mmpde_mover`` returned ``[]`` when its cell list was empty;
* ``Mesh.redistribute_nodes``, behind ``node_redistribution`` and ``Mesh.relax``,
  asked the DM, rank by rank, whether the mesh is simplicial, and an empty
  rank answered no;
* ``Mesh.boundary_slip`` returned early on a rank with no slip vertices,
  skipping the collective normal projection of a ``facet`` slip surface.

Two quieter effects made the result depend on the partition:

* with a MeshVariable metric and the default ``metric_eval="frozen"``, the
  frozen copy is repartitioned, and its field was filled by a rank-local
  evaluate;
* the stop test's scale ``h0`` was a mean of per-rank means, to which an
  empty rank contributed 1.0.

Each case runs serially and then with empty ranks, in subprocesses through
the MPI supervisor (this file is also the worker: ``python -m mpi4py <this
file> <case> <out.json>``). The empty ranks come from a shell partitioner
installed on the DM before the mesh distributes it: either every cell on
rank 0, or, for the direct mover, the four-cell box's natural
``[2, 0, 2, 0]`` at np 4. The metric is off the box's centre, so the mesh
moves. Every case must give the serial run's final coordinates, vertex by
vertex, matched by starting position, to rounding. The redistribution cases
run 40 outer iterations with ``stol=0``, so the dmax stop test (h0) decides
when they end.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BOUNDARIES = ("Top", "Bottom", "Left", "Right")

# case -> (rank count with empty ranks, every cell on rank 0?)
CASES = {
    "mover": (4, False),
    "redistribute_nodes": (2, True),
    "node_redistribution": (2, True),
    "relax": (2, True),
    "frozen_meshvariable_metric": (2, True),
    "facet_slip": (2, True),
}


def _all_cells_on_rank_0(dmod):
    """Install a shell partition with every cell on rank 0 on each DM the
    mesh loader returns, before the Mesh constructor distributes it."""
    from petsc4py import PETSc
    import underworld3 as uw

    original = dmod._from_plexh5

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        sf0, dm = result
        cs, ce = dm.getHeightStratum(0)
        sizes = np.zeros(uw.mpi.size, dtype=np.int32)
        sizes[0] = ce - cs if uw.mpi.rank == 0 else 0
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(uw.mpi.size, sizes=sizes,
                                      points=np.arange(ce - cs, dtype=np.int32))
        return sf0, dm

    dmod._from_plexh5 = load
    return original


def _move(case, out):
    import sympy
    import underworld3 as uw
    from underworld3.meshing.smoothing import _mmpde_mover

    dmod = uw.discretisation.discretisation_mesh
    gather = CASES[case][1] and uw.mpi.size > 1
    original = _all_cells_on_rank_0(dmod) if gather else None
    try:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 if case == "mover" else 0.25)
    finally:
        if original is not None:
            dmod._from_plexh5 = original
    cs, ce = mesh.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(ce - cs)
    before = np.asarray(mesh.X.coords).copy()

    x, y = mesh.X
    metric = 1.0 + 4.0 * sympy.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.04)
    options = dict(n_outer=40, stol=0.0)
    if case == "mover":
        _mmpde_mover(mesh, metric, BOUNDARIES, False, n_outer=2, metric_eval="sympy")
    elif case == "redistribute_nodes":
        mesh.redistribute_nodes(metric, pinned_labels=BOUNDARIES, method_kwargs=options)
    elif case == "node_redistribution":
        uw.meshing.node_redistribution(mesh, metric, pinned_labels=BOUNDARIES,
                                       method_kwargs=options)
    elif case == "relax":
        mesh.relax()
    elif case == "frozen_meshvariable_metric":
        M = uw.discretisation.MeshVariable("M", mesh, vtype=uw.VarType.TENSOR, degree=1)
        value = np.asarray(uw.function.evaluate(metric, np.asarray(M.coords_nd))).reshape(-1)
        tensor = np.zeros((len(value), 2, 2))
        tensor[:, 0, 0] = tensor[:, 1, 1] = value
        M.array[...] = tensor.reshape(M.array.shape)
        mesh.redistribute_nodes(M, pinned_labels=BOUNDARIES, method_kwargs=options)
    elif case == "facet_slip":
        # Without the box's registered plane surfaces, its sides slip on
        # facet surfaces, whose normals are a collective projection.
        for label in list(mesh.bounding_surfaces):
            del mesh.bounding_surfaces[label]
        mesh.redistribute_nodes(metric, pinned_labels=BOUNDARIES, slip_surfaces=True,
                                method_kwargs=options)

    after = np.asarray(mesh.X.coords).copy()
    before = uw.mpi.comm.gather(before, root=0)
    after = uw.mpi.comm.gather(after, root=0)
    if uw.mpi.rank == 0:
        before, after = np.concatenate(before), np.concatenate(after)
        # Shared vertices appear on more than one rank; keep each once.
        _, first = np.unique(before, axis=0, return_index=True)
        Path(out).write_text(json.dumps({
            "counts": counts, "before": before[first].tolist(), "after": after[first].tolist()}))


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
def test_with_empty_ranks_matches_serial(case, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    ranks = CASES[case][0]
    serial = _run(case, 1, tmp_path / "serial.json")
    parallel = _run(case, ranks, tmp_path / f"np{ranks}.json")
    assert min(parallel["counts"]) == 0, f"no rank is empty: {parallel['counts']}"

    before = np.asarray(serial["before"])
    order = [int(np.flatnonzero((before == v).all(axis=1))[0]) for v in parallel["before"]]
    assert len(set(order)) == len(before)
    after = np.asarray(serial["after"])[order]
    assert np.abs(after - before[order]).max() > 0.01           # the mesh moved
    np.testing.assert_allclose(np.asarray(parallel["after"]), after, rtol=0, atol=1e-12)


if __name__ == "__main__":
    _move(sys.argv[1], sys.argv[2])
