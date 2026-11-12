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

Both are now combined over every rank holding the vertex. The cases are
boxes in 2D and 3D, pinned and slipping, and H1's cavity annulus (a
refinement-2 annulus moved towards a fault trace at metric amplitude 20) at
np 2, 3, 4 and 5. Each case runs
``Mesh.redistribute_nodes`` serially and in parallel, in subprocesses
through the MPI supervisor (this file is also the worker: ``python -m mpi4py
<this file> <case> <out.json>``), and the parallel mesh must match the serial
one vertex by vertex, matched by starting position, to rounding. Measured on
2026-11-12: within 4e-12 on the boxes, and within 7e-8 on the annulus, whose
longer redistribution accumulates more rounding (it was 7e-3 to 2.4e-2
before the fix).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# case -> (dimension, cellSize, slip_surfaces, rank counts, partition)
CASES = {
    "pinned_2d": (2, 0.25, False, (2,), None),
    "slip_2d": (2, 0.25, True, (2, 3, 4), None),
    "slip_corners_2d": (2, 1.0, True, (2, 3, 4), None),
    "slip_3d": (3, 0.35, True, (2, 3, 4), None),
    # cells dealt to ranks at random, so seams cross the walls everywhere
    "slip_2d_scattered": (2, 0.25, True, (3, 4), "scatter"),
}
ANNULUS_RANKS = (2, 3, 4, 5)


def _scatter_cells(dmod, seed=1086):
    """Install a shell partition dealing the cells to ranks at random on each
    DM the mesh loader returns, before the Mesh constructor distributes it
    (the loader holds every cell on rank 0)."""
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
        sizes = np.bincount(owner, minlength=uw.mpi.size).astype(np.int32)
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(uw.mpi.size, sizes=sizes,
                                      points=np.argsort(owner, kind="stable").astype(np.int32))
        return sf0, dm

    dmod._from_plexh5 = load
    return original


def _move_annulus(out, slip=False):
    """H1's cavity case (#1086): a refinement-2 annulus moved by node
    redistribution towards a dipping fault trace, metric amplitude 20. The
    metric is read exactly (``metric_eval="sympy"``): the default frozen read
    goes through ``global_evaluate``, which is partition-dependent at seams
    on its own (#1093)."""
    import sympy
    import underworld3 as uw

    r_o, r_i, refinement = 1 / 0.45, 0.55 / 0.45, 2
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=0.5,
                              refinement=refinement, qdegree=3)
    h, dip = 0.5 / 2**refinement, np.radians(45.0)
    length = 0.4 / np.sin(dip)
    s = np.linspace(0, length, max(8, int(np.ceil(length / (h / 2)))) + 1)
    trace = np.array([r_o - 0.3, 0.0]) + s[:, None] * np.array([-np.sin(dip), -np.cos(dip)])
    x, y = mesh.X
    p0, p1 = trace[0], trace[-1]
    t = sympy.Max(0, sympy.Min(1, ((x - p0[0]) * (p1[0] - p0[0]) + (y - p0[1]) * (p1[1] - p0[1]))
                               / float(((p1 - p0) ** 2).sum())))
    d2 = (x - p0[0] - t * (p1[0] - p0[0])) ** 2 + (y - p0[1] - t * (p1[1] - p0[1])) ** 2
    before = np.asarray(mesh.X.coords).copy()
    uw.meshing.node_redistribution(mesh, 1 + 20.0 * sympy.exp(-d2 / 0.15**2),
                                   slip_surfaces=slip,
                                   method_kwargs=dict(metric_eval="sympy"))
    _write(out, before, np.asarray(mesh.X.coords).copy())


def _write(out, before, after):
    import underworld3 as uw

    before = uw.mpi.comm.gather(before, root=0)
    after = uw.mpi.comm.gather(after, root=0)
    if uw.mpi.rank == 0:
        before, after = np.concatenate(before), np.concatenate(after)
        # Shared vertices appear on more than one rank; keep each once.
        _, first = np.unique(before, axis=0, return_index=True)
        Path(out).write_text(json.dumps({"before": before[first].tolist(),
                                         "after": after[first].tolist()}))


def _move(case, out):
    import sympy
    import underworld3 as uw

    if case.startswith("cavity_annulus"):
        return _move_annulus(out, slip=case.endswith("_slip"))
    dim, cell_size, slip, _, partition = CASES[case]
    dmod = uw.discretisation.discretisation_mesh
    original = _scatter_cells(dmod) if partition == "scatter" and uw.mpi.size > 1 else None
    try:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
                                                 cellSize=cell_size)
    finally:
        if original is not None:
            dmod._from_plexh5 = original
    before = np.asarray(mesh.X.coords).copy()
    X = mesh.X
    centre = (0.3, 0.35, 0.4)[:dim]
    metric = 1.0 + 4.0 * sympy.exp(-sum((X[i] - centre[i]) ** 2 for i in range(dim)) / 0.04)
    sides = ("Top", "Bottom", "Left", "Right") + (("Front", "Back") if dim == 3 else ())
    mesh.redistribute_nodes(metric, pinned_labels=sides, slip_surfaces=slip,
                            method_kwargs=dict(n_outer=10 if dim == 2 else 5,
                                               metric_eval="sympy"))
    _write(out, before, np.asarray(mesh.X.coords).copy())


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


def _assert_matches(serial, parallel, inside_unit_box=True, atol=1e-10):
    before = np.asarray(serial["before"])
    order = [int(np.flatnonzero((before == v).all(axis=1))[0]) for v in parallel["before"]]
    assert len(set(order)) == len(before)
    after = np.asarray(serial["after"])[order]
    assert np.abs(after - before[order]).max() > 0.01           # the mesh moved
    moved = np.asarray(parallel["after"])
    if inside_unit_box:
        assert ((moved > -1e-12) & (moved < 1 + 1e-12)).all(), "a vertex left the box"
    np.testing.assert_allclose(moved, after, rtol=0, atol=atol)


needs_mpirun = pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                                  reason="needs mpirun beside the Python interpreter")


@pytest.mark.tier_b
@needs_mpirun
@pytest.mark.parametrize("case,ranks", [(case, n) for case, spec in CASES.items()
                                         for n in spec[3]])
def test_redistribution_matches_serial(case, ranks, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    serial = _run(case, 1, tmp_path / "serial.json")
    parallel = _run(case, ranks, tmp_path / "parallel.json")
    _assert_matches(serial, parallel)


@pytest.fixture(scope="module")
def cavity_serial(tmp_path_factory):
    return _run("cavity_annulus", 1, tmp_path_factory.mktemp("cavity") / "serial.json")


@pytest.mark.tier_b
@needs_mpirun
@pytest.mark.parametrize("ranks", ANNULUS_RANKS)
def test_cavity_annulus_matches_serial(ranks, cavity_serial, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    # The full redistribution runs many outer iterations, and the reductions
    # sum in a different order on each rank count, so rounding accumulates:
    # measured 3e-8 to 7e-8 at np 2-5 on a move of 0.139 (2026-11-12). The
    # seam effects it guards against were 7e-3 to 2.4e-2.
    _assert_matches(cavity_serial, _run("cavity_annulus", ranks, tmp_path / "parallel.json"),
                    inside_unit_box=False, atol=1e-6)



@pytest.fixture(scope="module")
def cavity_slip_serial(tmp_path_factory):
    return _run("cavity_annulus_slip", 1, tmp_path_factory.mktemp("cavity_slip") / "serial.json")


@pytest.mark.tier_b
@needs_mpirun
@pytest.mark.parametrize("ranks", ANNULUS_RANKS)
def test_cavity_annulus_with_slip_matches_serial(ranks, cavity_slip_serial, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own serial and parallel runs.")
    # The same move, with the walls slipping on their registered radial
    # surfaces. Measured 1.0e-6 to 7.7e-6 off serial at np 2-5 on a move of
    # 0.148 (2026-11-12), against 3e-8 to 7e-8 with the walls pinned. That is
    # NOT rounding: serial runs whose metric amplitude differs by 1e-14 differ
    # by only 1.2e-7 to 1.4e-7 with slip (3.5e-7 pinned), so a small partition
    # dependence remains in the slip path, unexplained. This bounds it; the
    # seam effects fixed here were 1e-3 and more.
    _assert_matches(cavity_slip_serial,
                    _run("cavity_annulus_slip", ranks, tmp_path / "parallel.json"),
                    inside_unit_box=False, atol=2e-5)


if __name__ == "__main__":
    _move(sys.argv[1], sys.argv[2])
