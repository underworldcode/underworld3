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
- If the evaluation fails on one rank after its collectives (a rank-local
  limiter can, #994), the variable fails on every rank: it keeps its values,
  the move and the other variables complete, and every rank raises the same
  error.
- If the evaluation raises on one rank before its collectives, the error is no
  longer swallowed into a skip on that rank: its traceback reaches the output, and the ranks never
  evaluate different variables against each other. A rank-local raise inside
  ``global_evaluate`` still leaves the other ranks waiting in it, so the job
  ends only by the supervisor's silence limit; agreeing on such a failure is
  ``global_evaluate``'s to do.

Both cases run at np 2 in subprocesses through the MPI supervisor (this file is
also the worker: ``python -m mpi4py <this file> <case>``), so a regression ends
in the supervisor's silence limit instead of hanging the suite.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SILENCE = 40      # seconds without output before the supervisor ends a child
RANKS = 2


def _hook_worker():
    """A failed transfer inside the semi-Lagrangian DDt's on_remesh hook (its
    ALE opt-out REMAP path) reaches the caller on every rank, after the move. Rank 1
    fails once, after the evaluation's collectives, for a variable the hook owns.
    (The probe of the second review of #960, as a case.)"""
    import numpy as np
    import underworld3 as uw
    from underworld3.discretisation import remesh

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 12, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V", mesh, 2, degree=1)
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    tc = np.asarray(T.coords)
    T.data[:, 0] = tc[:, 0] + 2 * tc[:, 1]
    owned = [v for v in mesh.vars.values() if getattr(v, "_remesh_managed_by", None) is not None]
    assert owned, "the DDt manages no variable"
    for v in owned:
        vc = np.asarray(v.coords)
        d = np.array(v.data)
        d[:] = (vc[:, 0] - vc[:, 1])[:, None]
        v.data[...] = d
    before = [np.asarray(v.data).copy() for v in owned]

    original = uw.function.global_evaluate
    fired = []

    def global_evaluate(*args, **kwargs):
        value = original(*args, **kwargs)
        if (not fired and uw.mpi.rank == 1
                and sys._getframe(1).f_code.co_name == "remap_var_set"
                and sys._getframe(2).f_code.co_name == "on_remesh"):
            fired.append(1)
            raise RuntimeError("forced failure on rank 1 inside the DDt hook's remap")
        return value

    uw.function.global_evaluate = global_evaluate
    c0 = np.asarray(mesh.X.coords).copy()
    x, y = c0[:, 0], c0[:, 1]
    inside = (x > 1e-9) & (x < 1 - 1e-9) & (y > 1e-9) & (y < 1 - 1e-9)
    c1 = c0.copy()
    c1[inside, 0] += 0.02 * np.sin(np.pi * x[inside]) * np.sin(np.pi * y[inside])

    def do_move():
        mesh._remesh_pending_scratch["ale_opt_out"] = True     # a discrete jump: the hook REMAPs
        mesh._deform_mesh(c1)

    try:
        remesh.remesh_with_field_transfer(mesh, do_move, dt=0.1)
        raised = False
    except RuntimeError as error:
        raised = "#960" in str(error)
    moved = bool(np.abs(np.asarray(mesh.X.coords) - c1).max(initial=0.0) < 1e-12)
    kept = tuple(bool(np.array_equal(np.asarray(v.data), b)) for v, b in zip(owned, before))
    results = uw.mpi.comm.gather((raised, moved, kept, len(fired)), root=0)
    if uw.mpi.rank == 0:
        print(f"REMAP DONE raised {[r[0] for r in results]} at_new_coords {[r[1] for r in results]} "
              f"history_kept {[r[2] for r in results]} fired {[r[3] for r in results]}", flush=True)


def _worker(case):
    import numpy as np
    import underworld3 as uw

    if case == "hook":
        return _hook_worker()

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
    elif case == "after":
        # rank 1 fails AFTER the evaluation's collectives (as a rank-local
        # limiter can, #994): every rank took part, then rank 1 raises
        original = uw.function.global_evaluate
        fired = []

        def global_evaluate(*args, **kwargs):
            value = original(*args, **kwargs)
            if not fired and uw.mpi.rank == 1 and sys._getframe(1).f_code.co_name == "remap_var_set":
                fired.append(1)
                raise RuntimeError("forced failure on rank 1 after the evaluation")
            return value

        uw.function.global_evaluate = global_evaluate

    elif case == "limiter":
        # REMESH_MONOTONE=clamp (set for this child): the limiter raises on rank 1
        # once, after the evaluation's collectives. Before, the ValueError sent
        # rank 1 alone into a second global_evaluate.
        from underworld3.function import functions_unit_system as fus
        original = fus._apply_monotone_limit
        fired = []

        def limiter(*args, **kwargs):
            if not fired and uw.mpi.rank == 1:
                fired.append(1)
                raise ValueError("forced limiter failure on rank 1")
            return original(*args, **kwargs)

        fus._apply_monotone_limit = limiter

    try:
        mesh.deform(c)
        raised = False
    except RuntimeError as error:
        raised = "#960" in str(error)

    s_err = float(np.abs(np.asarray(S.data)[:, 0]
                         - (2.0 * np.asarray(S.coords_nd)[:, 0] - np.asarray(S.coords_nd)[:, 1])).max(initial=0.0))
    t_kept = bool(np.array_equal(np.asarray(T.data), t_before))
    moved = bool(np.abs(np.asarray(mesh.X.coords) - c).max(initial=0.0) < 1e-12)
    results = uw.mpi.comm.gather((s_err, t_kept, raised, moved), root=0)
    if uw.mpi.rank == 0:
        print(f"REMAP DONE s_err {max(r[0] for r in results):.3e} "
              f"t_kept {[r[1] for r in results]} raised {[r[2] for r in results]} "
              f"at_new_coords {[r[3] for r in results]}", flush=True)


def _run(case, tmp_path, env=None):
    root = Path(__file__).resolve().parents[1]
    mpirun = str(Path(sys.executable).with_name("mpirun"))
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", str(SILENCE), "--",
               mpirun, "-np", str(RANKS), sys.executable, "-m", "mpi4py",
               str(Path(__file__).resolve()), case]
    log = tmp_path / f"{case}.log"
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=tmp_path, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=10 * SILENCE,
                                env=None if env is None else {**os.environ, **env})
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
    assert "raised [False, False]" in line, line
    s_err = float(line.split("s_err ")[1].split()[0])
    assert s_err < 1.0e-4, line           # a linear P1 field (1e-5 face slab: #1115)


def test_a_failure_after_the_evaluation_on_one_rank_fails_the_variable_on_every_rank(tmp_path):
    _serial_only()
    code, log = _run("after", tmp_path)
    assert code == 0, log[-4000:]
    line = next(ln for ln in log.splitlines() if "REMAP DONE" in ln)
    # every rank raises the same error, after the move is complete on every
    # rank; T keeps its values on both ranks, S is transferred
    assert "raised [True, True]" in line, line
    assert "at_new_coords [True, True]" in line, line
    assert "t_kept [True, True]" in line, line
    s_err = float(line.split("s_err ")[1].split()[0])
    assert s_err < 1.0e-4, line


def test_a_limiter_failure_on_one_rank_fails_the_variable_on_every_rank(tmp_path):
    _serial_only()
    code, log = _run("limiter", tmp_path, env={"REMESH_MONOTONE": "clamp"})
    assert code == 0, log[-4000:]
    line = next(ln for ln in log.splitlines() if "REMAP DONE" in ln)
    # with the limiter opted in, a ValueError on one rank is a failure agreed by
    # every rank, not a reason for that rank alone to evaluate again
    assert "raised [True, True]" in line, line
    assert "at_new_coords [True, True]" in line, line


def test_a_failure_in_a_ddt_hook_on_one_rank_reaches_the_caller_on_every_rank(tmp_path):
    _serial_only()
    code, log = _run("hook", tmp_path)
    assert code == 0, log[-4000:]
    line = next(ln for ln in log.splitlines() if "REMAP DONE" in ln)
    # the hook dispatcher swallows a hook's own exception, so the failure is
    # handed to remesh_with_field_transfer, which raises on every rank once the
    # move is complete; the failed history keeps its pre-move values
    assert "fired [0, 1]" in line, line
    assert "raised [True, True]" in line, line
    assert "at_new_coords [True, True]" in line, line
    # per variable the hook owns: the failed one kept on BOTH ranks, the others
    # transferred, the same pattern on every rank
    per_rank = ast.literal_eval(line.split("history_kept ")[1].split(" fired")[0])
    assert len(set(per_rank)) == 1 and any(per_rank[0]) and not all(per_rank[0]), line


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
