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

import shutil
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
