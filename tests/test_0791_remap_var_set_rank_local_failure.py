"""A mesh move's field transfer does not let one rank's failure pair the other
ranks' evaluations with the wrong variables (#960).

``remesh.remap_var_set`` evaluates each REMAP variable at its new DOF
coordinates with ``global_evaluate``, which is collective. It used to skip a
variable on the rank where something went wrong (its new coordinates could not
be read, or the evaluation raised) while the other ranks went into that
variable's collectives. The ranks then evaluated different variables against
each other and hung. Measured at np 2: T skipped on rank 1, S's collectives
matched against rank 0's T, and the job silent until the supervisor stopped it.

- If a variable's new coordinates cannot be read on one rank, the variable is
  skipped on every rank: every rank returns, the variable keeps its values, and
  the other variable is still transferred.
- If the evaluation raises on one rank, the error is no longer swallowed into a
  skip on that rank: its traceback reaches the output, and the ranks never
  evaluate different variables against each other. A rank-local raise inside
  ``global_evaluate`` still leaves the other ranks waiting in it, so the job
  ends only by the supervisor's silence limit; agreeing on such a failure is
  ``global_evaluate``'s to do.

Both cases run at np 2 in subprocesses through the MPI supervisor (this file is
also the worker: ``python -m mpi4py <this file> <case>``), so a regression ends
in the supervisor's silence limit instead of hanging the suite.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SILENCE = 40      # seconds without output before the supervisor ends a child
RANKS = 2


def _worker(case):
    import numpy as np
    import underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 16, qdegree=3, regular=False)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    S = uw.discretisation.MeshVariable("S", mesh, 1, degree=1)
    ct = np.asarray(T.coords_nd)
    T.data[:, 0] = np.tanh((ct[:, 0] + 0.3 * ct[:, 1] - 0.6) / 0.03)
    cs = np.asarray(S.coords_nd)
    S.data[:, 0] = 2.0 * cs[:, 0] - cs[:, 1]
    t_before = np.asarray(T.data).copy()

    c = np.asarray(mesh.X.coords).copy()
    x, y = c[:, 0], c[:, 1]
    inside = (x > 1e-9) & (x < 1 - 1e-9) & (y > 1e-9) & (y < 1 - 1e-9)
    c[inside, 0] += 0.02 * np.sin(np.pi * x[inside]) * np.sin(np.pi * y[inside])

    def called_from_remap():
        return sys._getframe(2).f_code.co_name == "remap_var_set"

    if case == "coords":
        # T's new coordinates cannot be read on rank 1, inside remap_var_set only
        # remap_var_set holds the base variables, not the wrapper T
        base = getattr(T, "_base_var", T)
        cls = next(k for k in type(base).__mro__ if "coords_nd" in vars(k))
        original = vars(cls)["coords_nd"]

        def coords_nd(self):
            if self is base and uw.mpi.rank == 1 and called_from_remap():
                raise RuntimeError("forced: T's coordinates unavailable on rank 1")
            return original.fget(self)

        setattr(cls, "coords_nd", property(coords_nd))
    elif case == "evaluate":
        original = uw.function.global_evaluate
        fired = []

        def global_evaluate(*args, **kwargs):
            if not fired and uw.mpi.rank == 1 and sys._getframe(1).f_code.co_name == "remap_var_set":
                fired.append(1)
                print("FORCED RAISE on rank 1", flush=True)
                raise RuntimeError("forced rank-local failure in the remap evaluation")
            return original(*args, **kwargs)

        uw.function.global_evaluate = global_evaluate

    mesh.deform(c)

    s_err = float(np.abs(np.asarray(S.data)[:, 0]
                         - (2.0 * np.asarray(S.coords_nd)[:, 0] - np.asarray(S.coords_nd)[:, 1])).max(initial=0.0))
    t_kept = bool(np.array_equal(np.asarray(T.data), t_before))
    results = uw.mpi.comm.gather((s_err, t_kept), root=0)
    if uw.mpi.rank == 0:
        print(f"REMAP DONE s_err {max(r[0] for r in results):.3e} "
              f"t_kept {[r[1] for r in results]}", flush=True)


def _run(case, tmp_path):
    root = Path(__file__).resolve().parents[1]
    mpirun = str(Path(sys.executable).with_name("mpirun"))
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", str(SILENCE), "--",
               mpirun, "-np", str(RANKS), sys.executable, "-m", "mpi4py",
               str(Path(__file__).resolve()), case]
    log = tmp_path / f"{case}.log"
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=tmp_path, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=10 * SILENCE)
    return result.returncode, log.read_text()


pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(1200),
              pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                                 reason="needs mpirun beside the Python interpreter")]


def _serial_only():
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own parallel runs.")


def test_unreadable_coordinates_on_one_rank_skip_the_variable_on_every_rank(tmp_path):
    _serial_only()
    code, log = _run("coords", tmp_path)
    assert code == 0, log[-4000:]
    line = next(ln for ln in log.splitlines() if "REMAP DONE" in ln)
    # T is skipped on BOTH ranks and keeps its values; S is still transferred.
    assert "t_kept [True, True]" in line, line
    s_err = float(line.split("s_err ")[1].split()[0])
    assert s_err < 1.0e-4, line           # a linear P1 field (1e-5 face slab: #1115)


def test_a_raise_in_the_evaluation_on_one_rank_is_not_swallowed(tmp_path):
    _serial_only()
    code, log = _run("evaluate", tmp_path)
    assert "FORCED RAISE on rank 1" in log, log[-4000:]
    # The error reaches the output: it is no longer swallowed into a skip that
    # lets the ranks pair different variables. The job does not complete. The
    # other rank is still waiting inside global_evaluate, so the supervisor's
    # silence limit is what ends it (agreeing on the failure there is
    # global_evaluate's to do).
    assert "forced rank-local failure in the remap evaluation" in log, log[-4000:]
    assert "REMAP DONE" not in log, log[-4000:]
    assert code != 0, log[-4000:]


if __name__ == "__main__":
    _worker(sys.argv[1])
