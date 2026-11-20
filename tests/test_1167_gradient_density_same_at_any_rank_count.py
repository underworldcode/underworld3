"""The front-following gradient density is the same at any rank count (#1166).

``metric_density_from_gradient(..., refinement=...)`` ranks the projected
``|grad field|`` and builds the density from each node's rank. Where many nodes
have nearly equal gradients (the flat region of a front, 57 % of the nodes here)
the rank is decided by differences far below anything physical. At the
projection's default tolerance the projected gradient differed between
partitions by about 1e-3 relative, so the density differed from np 1 at about
200 of 303 nodes, by up to 0.41, and a mover step's interpolation error by 40 %
at np 4. The gradient is now projected to a relative tolerance of 1e-12, and
the density is then the same at every rank count.

Three cases on #1166's mesh: its smooth front; the same front pre-smoothed with
``smoothing_length``, which projects the field first; and a field exactly constant
below a curve (the projection spreads a small, converged gradient into the flat
part, which then ranks the same everywhere). Each rank count writes the density at every node, sorted by
coordinate; np 2 and np 4 must equal np 1.

This file is also the worker (``python -m mpi4py <this file> <dir>``), run
through the MPI supervisor by a serial pytest.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(900)]


def _fields(x, y):
    import sympy
    front = 0.5 * (1 - sympy.tanh((y - 0.5 - 0.1 * sympy.cos(6 * x)) / 0.05))
    # exactly zero below y = 0.4 + 0.1 sin(4x), a smooth rise above it
    rise = sympy.Max(0, y - 0.4 - 0.1 * sympy.sin(4 * x)) ** 2
    return {"front": front, "flat": rise}


def _densities(out):
    import underworld3 as uw
    from underworld3.meshing.smoothing.metrics import _owned_dof_rows

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.07, qdegree=3)
    x, y = mesh.X
    cases = dict(_fields(x, y))
    cases["smoothed"] = cases["front"]
    for case, expression in cases.items():
        T = uw.discretisation.MeshVariable(f"T1166_{case}", mesh, 1, degree=1)
        T.array[:, 0, 0] = np.asarray(uw.function.evaluate(expression, T.coords)).reshape(-1)
        # "smoothed" takes the field-side pre-smoothing path, a second projection
        smoothing = {"smoothing_length": 0.05} if case == "smoothed" else {}
        uw.meshing.metric_density_from_gradient(mesh, T, refinement=4, name=case,
                                                **smoothing)
        rho = [v for v in mesh.vars.values() if v.clean_name == f"mdg_rho_{case}1"][0]
        rows = _owned_dof_rows(rho)
        mine = np.column_stack([np.asarray(rho.coords)[rows], np.asarray(rho.data[:, 0])[rows]])
        every = np.vstack(uw.mpi.comm.allgather(mine))
        if uw.mpi.rank == 0:
            every = every[np.lexsort((every[:, 1], every[:, 0]))]
            np.save(Path(out, f"{case}_np{uw.mpi.size}.npy"), every)


def _run(ranks, out):
    root = Path(__file__).resolve().parents[1]
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np",
                                      str(ranks)]
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "300", "--",
               *launcher, sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()), str(out)]
    log = Path(out, f"np{ranks}.log")
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=out, stdout=fh, stderr=subprocess.STDOUT, timeout=800)
    assert result.returncode == 0, log.read_text()[-4000:]


@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
def test_the_density_is_the_same_at_np_1_2_and_4(tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own MPI runs.")
    for ranks in (1, 2, 4):
        _run(ranks, tmp_path)
    for case in ("front", "flat", "smoothed"):
        reference = np.load(tmp_path / f"{case}_np1.npy")
        assert np.ptp(reference[:, 2]) > 1.0, f"{case}: the density does not grade at all"
        for ranks in (2, 4):
            got = np.load(tmp_path / f"{case}_np{ranks}.npy")
            assert got.shape == reference.shape
            assert np.abs(got[:, :2] - reference[:, :2]).max() < 1e-12
            gap = np.abs(got[:, 2] - reference[:, 2])
            assert gap.max() < 1.0e-8, (
                f"{case} at np {ranks}: {int((gap > 1e-8).sum())} of {gap.size} nodes "
                f"differ from np 1, by up to {gap.max():.2e}")


if __name__ == "__main__":
    _densities(sys.argv[1])
