"""Prolongation from a submesh reaches every parent DOF it covers, at odd rank counts (#1087, #858).

``Mesh.prolongate`` maps parent DOFs to submesh DOFs by coordinate, rank by
rank. A parent DOF on the region's edge can be owned by a rank whose cells
around it all lie outside the region; that rank has no submesh DOF there.
Each rank used to write its matched rows and sync, which keeps the owner's
value, so those DOFs kept their old value: on ``AnnulusInternalBoundary``
with ``extract_region("Inner")`` at np 3, two DOFs on r = 1 came back as 0
(error 1.0). This test launches np 3 and np 5 itself, in subprocesses through
the MPI supervisor (this file is also the worker: ``python -m mpi4py <this
file> <out.json>``).

The partition is built, not left to the partitioner, whose choice differs
between platforms (on macOS the default one left no such DOF, so the test
could only fail its precondition): a shell partition puts the region's cells
on rank 0 and deals the rest to ranks 1 to n-1 by angle. PETSc gives a point
shared between ranks to the highest of them, so the region's edge at r = 1
is owned by ranks with no child cell there, on any platform.

A P2 vector field ``(r, -r)`` on the submesh is prolongated into a parent
field of sevens, in ``replace`` and ``add`` mode. Every parent DOF in the
region must take the submesh value (plus seven in ``add`` mode) to 1e-12
(shared copies agree, and their mean is within rounding of them), and every
DOF outside must keep its seven. The worker also asserts that the case
arises: some parent DOFs are owned by a rank with no child copy of them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _prolongate(out):
    import underworld3 as uw
    from mpi4py import MPI

    full = _region_on_rank_0(uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.5, radiusInternal=1.0, radiusInner=0.5, cellSize=1.0 / 8.0))
    rock = full.extract_region("Inner")
    v_full = uw.discretisation.MeshVariable("Vf", full, full.dim, degree=2)
    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)
    r_rock = np.hypot(*np.asarray(v_rock.coords).T)
    v_rock.data[:, 0] = r_rock
    v_rock.data[:, 1] = -r_rock

    r = np.hypot(*np.asarray(v_full.coords).T)
    inside = r < 1.0 + 1.0e-6
    errors = {}
    for mode in ("replace", "add"):
        # Seven, not zero: a transfer that dropped the old values of DOFs it
        # does not cover would leave zeros, and a zero start would hide that.
        v_full.data[:] = 7.0
        rock.prolongate(v_rock, v_full, mode=mode)
        data = np.asarray(v_full.data)
        start = 7.0 if mode == "add" else 0.0
        expect = np.where(inside[:, None], start + np.column_stack([r, -r]), 7.0)
        err = float(np.abs(data - expect).max()) if len(data) else 0.0
        errors[mode] = uw.mpi.comm.allreduce(err, op=MPI.MAX)
    start, end = rock.dm.getHeightStratum(0)
    errors["child_cells"] = uw.mpi.comm.allgather(end - start)
    errors["orphans"] = uw.mpi.comm.allreduce(
        _orphans_module()._orphans(rock, v_rock, v_full, inside))
    if uw.mpi.rank == 0:
        Path(out).write_text(json.dumps(errors))


def _region_on_rank_0(parent):
    """``parent`` repartitioned: the cells inside r = 1 on rank 0, the others
    on ranks 1 to n-1 by angle (a shell partition, so no partitioner choice
    enters)."""
    import underworld3 as uw

    dm = parent.dm.clone()
    dm.setCoordinatesLocal(parent.dm.getCoordinatesLocal())
    start, end = dm.getHeightStratum(0)
    dest = np.empty(end - start, dtype=np.int64)
    for cell in range(start, end):
        _, centre, _ = dm.computeCellGeometryFVM(cell)
        if np.hypot(centre[0], centre[1]) < 1.0:
            dest[cell - start] = 0
        else:
            turn = (np.arctan2(centre[1], centre[0]) / (2 * np.pi)) % 1.0
            dest[cell - start] = 1 + min(int(turn * (uw.mpi.size - 1)), uw.mpi.size - 2)
    partitioner = dm.getPartitioner()
    partitioner.setType("shell")
    from petsc4py import PETSc

    partitioner.setShellPartition(
        uw.mpi.size, np.bincount(dest, minlength=uw.mpi.size).astype(PETSc.IntType),
        np.argsort(dest, kind="stable").astype(PETSc.IntType))
    migration = dm.distribute()
    mesh = uw.discretisation.Mesh(
        dm, boundaries=parent.boundaries, qdegree=parent.qdegree,
        coordinate_system_type=parent.CoordinateSystem.coordinate_type)
    mesh.regions = parent.regions
    if migration is not None:
        migration.destroy()
    return mesh


def _orphans_module():
    """The orphan-DOF count lives with the in-process parallel test."""
    import importlib.util

    path = Path(__file__).resolve().parent / "parallel" / "test_1087_submesh_prolongation_ownership.py"
    spec = importlib.util.spec_from_file_location("_submesh_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = [pytest.mark.level_2, pytest.mark.timeout(600)]


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
@pytest.mark.parametrize("ranks", [3, 5])
def test_prolongate_reaches_dofs_owned_outside_the_region(ranks, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own parallel run.")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "errors.json"
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "120", "--",
               str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks),
               sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()), str(out)]
    log = tmp_path / "run.log"
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=tmp_path, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=300)
    assert result.returncode == 0, log.read_text()[-4000:]
    errors = json.loads(out.read_text())
    # the layout built above, and the case it exists for
    assert errors["child_cells"][0] > 0 and not any(errors["child_cells"][1:]), errors
    assert errors["orphans"] > 0, errors
    assert errors["replace"] < 1e-12, errors
    assert errors["add"] < 1e-12, errors


if __name__ == "__main__":
    _prolongate(sys.argv[1])
