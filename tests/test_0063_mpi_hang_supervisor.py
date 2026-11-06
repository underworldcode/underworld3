"""The MPI supervisor must bound a planted collective hang and diagnose it.

A harness that catches hangs is worth nothing until it has caught one. These
tests plant a hang whose answer is known — rank 0 sits in a function no other
rank can be inside, while the rest wait in a barrier — and require the
supervisor to end the job and name rank 0 from the stacks it collects.

The second requirement matters as much as the first: a supervisor that kills
healthy jobs, or that reports a diagnosis it cannot support, is worse than no
supervisor. So a clean run must pass through with its own exit status, and a
two-rank hang, where there is no majority to appeal to, must say so rather than
pick a rank arbitrarily.
"""

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

REPO = Path(__file__).resolve().parent.parent
SUPERVISOR = REPO / "scripts" / "mpi_supervisor.py"
PLANTED_HANG = REPO / "tests" / "parallel" / "hang_controls" / "rank_zero_misses_the_collective.py"

# Short enough to keep the test quick, long enough that a slow import cannot
# be mistaken for a hang. The supervisor adds its two sampling pauses on top.
SILENCE = 15.0

# The supervisor's own budget: silence, two dumps, the gap between them, and
# the kill ladder. Generous, because this asserts "it ends", not "it is fast".
BUDGET = 180.0

needs_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="no mpirun on this machine")


def _run_supervised(job, artifacts, ranks=2):
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(SUPERVISOR), "--silence", str(SILENCE),
         "--artifacts", str(artifacts),
         "--", "mpirun", "-n", str(ranks), sys.executable, str(job)],
        capture_output=True, text=True, timeout=BUDGET)
    return completed, time.monotonic() - started


@needs_mpirun
def test_a_planted_hang_is_bounded_and_the_guilty_rank_named(tmp_path):
    """Four ranks, one of which skips the barrier. Rank 0 must be named."""
    completed, elapsed = _run_supervised(PLANTED_HANG, tmp_path, ranks=4)

    assert completed.returncode == 124, (
        f"the supervisor did not report a kill (got {completed.returncode}):\n"
        f"{completed.stdout}\n{completed.stderr}")
    assert elapsed < BUDGET, "the supervisor did not end the job within its budget"

    report = completed.stdout
    assert "stuck, not slow" in report, (
        f"two identical samples should have read as stuck:\n{report}")
    assert "rank(s) 0 are somewhere the other 3 are not" in report, (
        f"the rank that missed the collective was not named:\n{report}")
    assert "_rank_zero_misses_the_collective" in report, (
        f"the diagnosis did not reach the frame that proves it:\n{report}")

    stacks = sorted(tmp_path.glob("rank_*_pid_*.stack"))
    assert len(stacks) == 4, f"expected a dump per rank, got {[p.name for p in stacks]}"


@needs_mpirun
def test_two_ranks_have_no_majority_and_the_report_says_so(tmp_path):
    """At np=2 neither rank outvotes the other.

    The supervisor must report both positions and decline to nominate one.
    Claiming a culprit from a one-all split would be a confident guess, which
    is the failure mode this whole harness exists to avoid.
    """
    completed, _elapsed = _run_supervised(PLANTED_HANG, tmp_path, ranks=2)

    assert completed.returncode == 124
    report = completed.stdout
    assert "no majority" in report, f"the tie was not reported as a tie:\n{report}"
    assert "rank 0 is in _rank_zero_misses_the_collective()" in report, report
    assert "rank 1 is in main()" in report, report


def _needs_gdb():
    """CI installs gdb for this test, so there a missing gdb is a failure, not a skip."""
    if shutil.which("gdb") is None:
        if os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("no gdb on the runner: the install step did not work")
        pytest.skip("no gdb on this machine")


@needs_mpirun
def test_a_hang_gets_native_stacks_that_reach_the_mpi_library(tmp_path):
    """A Python stack stops at the C boundary, where a rank in an MPI wait and a
    rank in a long KSP solve look the same. The native stack tells them apart.

    Rank 1 waits in the barrier, so its native stack must name it, which a
    Python frame in ``main()`` cannot.
    """
    _needs_gdb()
    completed, _elapsed = _run_supervised(PLANTED_HANG, tmp_path, ranks=2)
    assert completed.returncode == 124, f"{completed.stdout}\n{completed.stderr}"

    native = {p.name: p.read_text() for p in tmp_path.glob("rank_*_pid_*.native")}
    assert len(native) == 2, f"expected a native stack per rank, got {sorted(native)}"
    waiting = next(text for name, text in native.items() if name.startswith("rank_1_"))
    assert "#0" in waiting and "barrier" in waiting.lower(), (
        f"rank 1's native stack does not reach the barrier it waits in:\n{waiting}")
    logged = next((line for line in completed.stdout.splitlines()
                   if "native: rank 1:" in line), "")
    assert "barrier" in logged.lower(), (
        f"the log does not carry the main thread's native frames, and a log is all "
        f"a reader may have:\n{completed.stdout}")


@needs_mpirun
def test_a_healthy_job_is_left_alone(tmp_path):
    """The negative control: no false positives, and the exit status passes through.

    Without this, every assertion above could be satisfied by a supervisor that
    simply kills everything it is given.
    """
    healthy = tmp_path / "healthy_job.py"
    healthy.write_text(
        "from mpi4py import MPI\n"
        "comm = MPI.COMM_WORLD\n"
        "print(f'rank {comm.rank} reporting', flush=True)\n"
        "comm.barrier()\n"
        "print('all ranks through the barrier', flush=True)\n")

    completed, elapsed = _run_supervised(healthy, tmp_path / "stacks", ranks=2)

    assert completed.returncode == 0, (
        f"a healthy job was not left alone:\n{completed.stdout}\n{completed.stderr}")
    assert "MPI SUPERVISOR" not in completed.stdout, (
        f"the supervisor intervened in a healthy run:\n{completed.stdout}")
    assert elapsed < SILENCE + 30.0, "a healthy job should finish well inside the silence budget"
    assert "all ranks through the barrier" in completed.stdout, (
        "the job's own output was not forwarded")


# The native path, serially: a script stands in for gdb.
TRANSCRIPT = """Thread 2 (Thread 0x7f (LWP 2) "async"):
#0  __GI___poll (fds=0x1) at poll.c:29
Thread 1 (Thread 0x7e (LWP 1) "python"):
#0  sched_yield () at syscall.S:78
#1  0x00007f in MPIR_Wait_state () from libmpi.so
#2  0x00007f in MPIR_Barrier_impl () from libmpi.so
"""


@pytest.fixture
def supervisor():
    spec = importlib.util.spec_from_file_location("mpi_supervisor", SUPERVISOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SECONDS_BETWEEN_NATIVE_SAMPLES = 0.0
    return module


def _fake_gdb(tmp_path, monkeypatch, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gdb = bin_dir / "gdb"
    gdb.write_text("#!/bin/sh\n" + body + "\n")
    gdb.chmod(gdb.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


DUMPS = {"0": (101, []), "1": (102, [])}


def test_the_main_thread_is_read_and_quoted_from_the_mpi_routine(supervisor):
    frames = supervisor._native_frames(TRANSCRIPT)
    assert frames == ["sched_yield", "MPIR_Wait_state", "MPIR_Barrier_impl"]
    assert supervisor._quoted_frames(frames)[0] == "MPIR_Wait_state"
    assert supervisor._quoted_frames(["KSPSolve", "main"]) == ["KSPSolve", "main"]


def test_a_refused_attach_says_why_in_the_log(supervisor, tmp_path, monkeypatch):
    _fake_gdb(tmp_path, monkeypatch, 'echo "ptrace: Operation not permitted." >&2; exit 1')
    lines = supervisor._collect_native_stacks(DUMPS, tmp_path)
    assert sum("Operation not permitted" in line for line in lines) == 2, lines


def test_a_hung_gdb_is_bounded_and_keeps_what_it_printed(supervisor, tmp_path, monkeypatch):
    supervisor.SECONDS_FOR_NATIVE_STACK = 1.0
    _fake_gdb(tmp_path, monkeypatch, 'echo "#0  partial_frame ()"; exec sleep 30')
    started = time.monotonic()
    lines = supervisor._collect_native_stacks(DUMPS, tmp_path)
    assert time.monotonic() - started < 10.0, "gdb held up the kill"
    assert sum("gdb took over" in line for line in lines) == 2, lines
    assert "partial_frame" in (tmp_path / "rank_0_pid_101.native").read_text()


def test_an_undecodable_byte_costs_no_rank_its_stack(supervisor, tmp_path, monkeypatch):
    _fake_gdb(tmp_path, monkeypatch, "printf '#0  frame_one ()\\n\\377\\n'")
    lines = supervisor._collect_native_stacks(DUMPS, tmp_path)
    assert sum("frame_one" in line for line in lines) == 2, lines
    assert sorted(p.name for p in tmp_path.glob("*.native")) == [
        "rank_0_pid_101.native", "rank_1_pid_102.native"]


def test_two_samples_tell_a_rank_still_working_from_one_that_is_not(supervisor, tmp_path,
                                                                    monkeypatch):
    counter = tmp_path / "count"
    _fake_gdb(tmp_path, monkeypatch,
              f'echo x >> {counter}; echo "#0  frame_$(wc -l < {counter}) ()"')
    moved = supervisor._collect_native_stacks({"0": (101, [])}, tmp_path)
    assert any("moved between samples on rank(s) 0" in line for line in moved), moved

    (tmp_path / "bin" / "gdb").write_text('#!/bin/sh\necho "#0  same_frame ()"\n')
    still = supervisor._collect_native_stacks({"0": (101, [])}, tmp_path)
    assert any("did not move" in line for line in still), still


def test_no_gdb_is_one_line_and_no_stack(supervisor, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: None)
    assert supervisor._collect_native_stacks(DUMPS, tmp_path) == [
        "    no gdb on PATH: Python stacks only"]
