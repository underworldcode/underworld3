"""A refined mesh's checkpoint reloads correctly on another rank count (#690).

A mesh built with ``refinement`` is checkpointed with its coarsest level in a
sidecar, and ``Mesh(file)`` rebuilds the multigrid hierarchy by refining that
level. The rebuilt fine mesh is the saved one, numbered and distributed
differently. Until #690 was fixed the reload paired the two by array index,
which holds only for a serial write: a mesh written on two or more ranks came
back with cells joined to the wrong vertices (the integral of 1 was negative)
on every rank count, and a mesh written in serial could not have its fields
read back by section on more ranks than it was written on.

This test writes in one process and reloads in another, with the rank counts
the bug depends on. Both run as subprocesses of a serial pytest, through the
MPI supervisor; this file is also the worker (``python -m mpi4py <this file>
write|read <dir>``).

The meshes: a unit box and an annulus at refinement 2, and the box moved
further than half its finest spacing before writing, so a reload that paired
vertices by their saved positions instead of their reference ones would fail.
Each carries a P2 field ``T = x``, written with both ``write_timestep`` and
``write_checkpoint``. Every reload must reproduce the integral of 1 of the
written mesh, read ``T`` back exactly by both readers, and restore the
hierarchy.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CASES = ("box", "moved_box", "annulus")
REFINEMENT = 2


def _build(case):
    import underworld3 as uw

    if case == "annulus":
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25,
                                  refinement=REFINEMENT)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=0.25, refinement=REFINEMENT)
    if case == "moved_box":
        # Finest spacing is 0.25 / 4; the largest move is 0.06, nearly all of it.
        x = np.asarray(mesh.X.coords).copy()
        x[:, 0] += 0.06 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        mesh.deform(x)
    return mesh


def _write(out):
    import underworld3 as uw

    expected = {}
    for case in CASES:
        mesh = _build(case)
        T = uw.discretisation.MeshVariable(f"T_{case}", mesh, 1, degree=2)
        T.data[:, 0] = np.asarray(T.coords)[:, 0]
        expected[case] = {
            "area": float(uw.maths.Integral(mesh, 1.0).evaluate()),
            "levels": len(mesh.dm_hierarchy),
        }
        mesh.write_timestep(f"ts_{case}", 0, outputPath=out, meshVars=[T], meshUpdates=True)
        mesh.write_checkpoint(f"ck_{case}", outputPath=out, meshVars=[T])
    if uw.mpi.rank == 0:
        Path(out, "expected.json").write_text(json.dumps(expected))


def _read(out):
    import underworld3 as uw
    from mpi4py import MPI

    results = {}
    for case in CASES:
        for writer in ("ts", "ck"):
            mesh = uw.discretisation.Mesh(os.path.join(out, f"{writer}_{case}.mesh.00000.h5"))
            T = uw.discretisation.MeshVariable(f"T_{case}", mesh, 1, degree=2)
            area = float(uw.maths.Integral(mesh, 1.0).evaluate())
            if writer == "ts":
                T.read_timestep(f"ts_{case}", f"T_{case}", 0, outputPath=out)
            else:
                T.read_checkpoint(os.path.join(out, f"ck_{case}.T_{case}.00000.h5"),
                                  data_name=f"T_{case}")
            d = np.asarray(T.data)[:, 0] - np.asarray(T.coords)[:, 0]
            err = uw.mpi.comm.allreduce(float(np.abs(d).max()) if d.size else 0.0, op=MPI.MAX)
            results[f"{writer}_{case}"] = {"area": area, "T_error": err,
                                           "levels": len(mesh.dm_hierarchy)}
    if uw.mpi.rank == 0:
        Path(out, f"read_np{uw.mpi.size}.json").write_text(json.dumps(results))


def _run(ranks, phase, out):
    root = Path(__file__).resolve().parents[1]
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks)]
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "120", "--",
               *launcher, sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()), phase, str(out)]
    log = Path(out, f"{phase}_np{ranks}.log")
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=out, stdout=fh, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, log.read_text()[-4000:]


pytestmark = [pytest.mark.level_2, pytest.mark.timeout(900)]


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
@pytest.mark.parametrize("write_np, read_np", [(2, 1), (1, 2), (2, 2)])
def test_refined_checkpoint_reloads_on_any_rank_count(tmp_path, write_np, read_np):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own MPI writer and reader.")
    _run(write_np, "write", tmp_path)
    _run(read_np, "read", tmp_path)
    expected = json.loads((tmp_path / "expected.json").read_text())
    results = json.loads((tmp_path / f"read_np{read_np}.json").read_text())

    for key, got in results.items():
        want = expected[key.split("_", 1)[1]]
        assert got["area"] == pytest.approx(want["area"], rel=1e-12), (key, got, want)
        assert got["T_error"] < 1e-12, (key, got)
        assert got["levels"] == want["levels"] == REFINEMENT + 1, (key, got, want)


if __name__ == "__main__":
    {"write": _write, "read": _read}[sys.argv[1]](sys.argv[2])
