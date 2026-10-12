"""The watchdog on a rank genuinely blocked in a collective (#793).

``tests/test_0053_hang_watchdog.py`` covers the serial behaviour, and its
docstring has always named this file as the parallel half. It did not exist.
The case it describes is the one the watchdog was built for and the only one
the serial tests cannot reach: a rank sitting inside MPI, holding the GIL,
with no Python code of its own able to run.

That is why the report goes through :mod:`faulthandler` and a kernel interval
timer rather than a Python thread. A Python watcher needs the GIL the blocked
thread is holding; measured on a blocked rank it filed 1 report in 5, while
faulthandler filed 5 in 5.

It is also why the registration asks for the signalled thread only
(``all_threads=False``). Walking every thread state races any thread that is
starting or exiting and takes the process down inside
``_Py_DumpTracebackThreads`` — the #793 segfault. The thread we give up is
never the interesting one: the signal lands on the thread in the collective,
which is the thread the report exists to name.

Run under MPI::

    mpirun -np 2 python -m pytest --with-mpi \
        tests/parallel/test_0778_hang_watchdog_mpi.py
    mpirun -np 4 python -m pytest --with-mpi \
        tests/parallel/test_0778_hang_watchdog_mpi.py

The block is bounded: rank 0 is late by a fixed interval and then joins, so
the collective always completes and the test cannot become the hang it is
testing for.
"""

import time

import pytest
from mpi4py import MPI

import underworld3 as uw

# The timeout runs on a THREAD, not on SIGALRM. pytest-timeout's default
# method takes ITIMER_REAL, which is the timer the watchdog takes over, so the
# default would leave this file -- the one file that arms a watchdog on every
# test -- with no hang guard of its own.
pytestmark = [
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120, method="thread"),
    pytest.mark.level_1,
    pytest.mark.tier_a,
]

# faulthandler's single-thread header. The all-threads walk writes
# "Current thread 0x..." instead, once per thread state it reads.
STACK = "Stack (most recent call first):"

# The watchdog must fire several times while rank 0 is away, so that a repeat
# is being read rather than a single lucky dump. Six windows inside the block
# leaves room for a slow runner to lose most of them and still show two.
TIMEOUT = 0.5
LATENESS = 3.0


@pytest.fixture
def report(tmp_path_factory):
    """A per-rank file, and a reader for whatever landed in it.

    The ranks interleave on a shared stream, so each one writes its own file.
    ``tmp_path_factory`` is per-rank already, but the rank goes in the name as
    well so that a shared temporary directory cannot alias two ranks onto one
    file and hide a missing report.

    Disarms BEFORE closing, for the reason test_0053 gives: faulthandler holds
    the descriptor, not the Python object.
    """
    rank = MPI.COMM_WORLD.rank
    path = tmp_path_factory.mktemp("watchdog") / f"rank{rank}.log"
    handle = path.open("w")

    def read():
        handle.flush()
        return path.read_text()

    try:
        yield handle, read
    finally:
        uw.mpi.unwatch()
        handle.close()


def _rank_zero_is_late_to_the_allreduce(comm, handle):
    """Rank 0 spends LATENESS seconds here; everyone else waits in MPI.

    Named so that it is unmistakable in a stack dump, the way
    ``hang_controls/rank_zero_misses_the_collective.py`` is.
    """
    if comm.rank == 0:
        time.sleep(LATENESS)
    return comm.allreduce(comm.rank, op=MPI.SUM)


def test_a_blocked_rank_still_files_a_report(report):
    """Every rank reports, and repeats, on the path that needs no lock.

    The repeat is asserted on the faulthandler dumps, not on the labelled
    report, because the labelled report is the one with no guarantee here. It
    is formatted in Python and needs the interpreter lock, which is why
    :meth:`_Watchdog.report` says it is silent on a rank inside MPI. Measured
    at np=4 against a 4 s block: the pre-#793 ``threading.Timer`` filed zero
    labelled reports on the blocked ranks, the persistent reporter thread
    files one, and neither number is a contract. faulthandler filed every
    dump in both cases, and that is.
    """
    comm = MPI.COMM_WORLD
    handle, read = report

    uw.mpi.watch(seconds=TIMEOUT, stream=handle)
    total = _rank_zero_is_late_to_the_allreduce(comm, handle)
    uw.mpi.unwatch()

    assert total == sum(range(comm.size))        # the collective really completed

    text = read()
    dumps = text.count(STACK)
    assert dumps >= 2, (
        f"rank {comm.rank} spent {LATENESS} s blocked with a {TIMEOUT} s "
        f"watchdog and filed {dumps} dump(s). One reads as slow; the watchdog "
        f"has to repeat for a run to read as stuck:\n{text}"
    )


def test_the_report_names_the_frame_the_rank_is_stuck_in(report):
    """The stack is the blocked thread's, so it carries our own call site.

    This is the whole point of the C path: the signal lands on the thread
    sitting in the collective, so the frame it names is the one the rank is
    stuck in rather than the watchdog's own.
    """
    comm = MPI.COMM_WORLD
    handle, read = report

    uw.mpi.watch(seconds=TIMEOUT, stream=handle)
    _rank_zero_is_late_to_the_allreduce(comm, handle)
    uw.mpi.unwatch()

    text = read()
    assert "_rank_zero_is_late_to_the_allreduce" in text, (
        f"rank {comm.rank}'s report does not name the function it was in:\n{text}"
    )


def test_one_thread_is_dumped_not_every_thread(report):
    """``all_threads=False``: the report carries the signalled thread alone (#793).

    Dumping every thread state is what crashed inside CPython during full
    tier_a runs, and it is the C path that did it -- the labelled report from
    ``_stack_dump()`` goes through ``sys._current_frames()``, which holds the
    interpreter lock and walks every thread safely.

    The two faulthandler paths label themselves, and that is the
    discriminator: ``dump_traceback(all_threads=False)`` writes ``Stack (most
    recent call first):``, while the all-threads walk writes ``Current thread
    0x...`` once per thread state it reads.
    """
    comm = MPI.COMM_WORLD
    handle, read = report

    uw.mpi.watch(seconds=TIMEOUT, stream=handle)
    _rank_zero_is_late_to_the_allreduce(comm, handle)
    uw.mpi.unwatch()

    text = read()
    assert STACK in text, (
        f"rank {comm.rank}: no faulthandler dump at all, so this test cannot "
        f"see which path ran:\n{text}"
    )
    assert "Current thread" not in text, (
        f"rank {comm.rank}: the C dump walked thread states, so all_threads "
        f"came back on -- that is the #793 segfault:\n{text}"
    )
