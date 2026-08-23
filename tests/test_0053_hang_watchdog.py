"""Serial behaviour of the hang watchdog and the rank-agreement audit.

The parallel half --- a rank genuinely blocked in a collective still filing a
report --- is in ``tests/parallel/test_0778_hang_watchdog_mpi.py``, because it
needs more than one rank to block against.

Every test here writes to a real file rather than a buffer, and that is not
incidental. The dump goes through :mod:`faulthandler`, which writes to a file
descriptor. Measured at np=4 against a 4 s block in ``comm.allreduce``: a
re-arming ``threading.Timer`` at 0.5 s fired zero times on the blocked ranks,
while faulthandler produced all 7 expected dumps. A buffer-backed watchdog
would pass every test in this file and still report nothing in the case it
exists for, which is what :func:`test_a_buffer_is_refused` guards.
"""

import io
import time

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

FIRED = "UW HANG WATCHDOG"


@pytest.fixture(autouse=True)
def always_disarm():
    """A leaked timer would fire in the middle of some later test."""
    yield
    uw.mpi.unwatch()


@pytest.fixture
def report(tmp_path):
    """A real file to watch into, and a reader for whatever landed in it.

    Disarms BEFORE closing. faulthandler holds the descriptor, not the Python
    object, so a watchdog left armed over a closed file writes into whatever
    that descriptor is reused for -- which in a captured pytest run is the
    capture pipe, and the session then wedges. The autouse fixture above tears
    down after this one, which is too late.
    """
    path = tmp_path / "watchdog.log"
    handle = path.open("w")
    try:
        yield handle, (lambda: path.read_text())
    finally:
        uw.mpi.unwatch()
        handle.close()


def test_a_buffer_is_refused():
    """An in-memory stream must be rejected, not quietly half-work.

    faulthandler needs a descriptor. Accepting a buffer would leave the
    watchdog silent exactly when a rank is blocked inside MPI.
    """
    with pytest.raises(ValueError, match="file descriptor"):
        uw.mpi.watch(seconds=5, stream=io.StringIO())


def test_watchdog_reports_when_progress_stops(report):
    """The whole point: no checkpoint for `seconds`, and it says so."""
    stream, read_back = report
    uw.mpi.watch(seconds=0.3, stream=stream)
    uw.mpi.checkpoint("the last thing this rank did")
    time.sleep(0.9)
    uw.mpi.unwatch()

    text = read_back()
    assert FIRED in text
    assert f"rank {uw.mpi.rank} of {uw.mpi.size}" in text
    assert "the last thing this rank did" in text


def test_report_carries_the_main_thread_stack(report):
    """A dump of the watchdog's own frames would diagnose nothing.

    The stack must name this test function, which runs on the main thread.
    """
    stream, read_back = report
    uw.mpi.watch(seconds=0.3, stream=stream)
    time.sleep(0.9)
    uw.mpi.unwatch()

    text = read_back()
    assert "test_report_carries_the_main_thread_stack" in text, (
        f"the dump did not include the main thread:\n{text}"
    )


def test_progress_never_fires(report):
    """Negative control. A checkpoint inside the window keeps it quiet.

    Without this, the first test proves only that a timer can print something.
    """
    stream, read_back = report
    uw.mpi.watch(seconds=0.6, stream=stream)
    for step in range(8):
        time.sleep(0.05)
        uw.mpi.checkpoint(f"step {step}")
    uw.mpi.unwatch()

    assert FIRED not in read_back(), (
        f"fired despite steady progress:\n{read_back()}"
    )


def test_repeats_rather_than_reporting_once(report):
    """One dump reads as "slow"; the same dump again reads as "stuck"."""
    stream, read_back = report
    uw.mpi.watch(seconds=0.3, stream=stream)
    time.sleep(1.4)
    uw.mpi.unwatch()

    assert read_back().count(FIRED) >= 2


def test_unwatch_disarms(report):
    stream, read_back = report
    uw.mpi.watch(seconds=0.3, stream=stream)
    uw.mpi.unwatch()
    time.sleep(0.8)
    assert FIRED not in read_back()


def test_checkpoint_without_a_watchdog_is_a_no_op():
    """It is meant to be left in production code, so it must cost nothing."""
    uw.mpi.unwatch()
    uw.mpi.checkpoint("nobody is listening")


def test_watching_restores_the_previous_watchdog(tmp_path):
    """A nested block must not silently disarm the outer one on exit."""
    outer_path = tmp_path / "outer.log"
    inner_path = tmp_path / "inner.log"

    with outer_path.open("w") as outer, inner_path.open("w") as inner:
        try:
            uw.mpi.watch(seconds=0.4, stream=outer)
            with uw.mpi.watching(seconds=0.3, stream=inner):
                time.sleep(0.9)
            assert FIRED in inner_path.read_text()

            # The outer watchdog is armed again, and still works.
            time.sleep(1.1)
        finally:
            # Disarm before the files close -- see the `report` fixture.
            uw.mpi.unwatch()

    assert FIRED in outer_path.read_text(), (
        "the outer watchdog was left disarmed by the nested block"
    )


def test_declared_collectives_check_in_automatically(report):
    """`@collective_operation` re-arms the timer and labels the report.

    This is what makes the watchdog usable without seeding `checkpoint` calls
    through the library by hand.
    """
    @uw.mpi.collective_operation
    def a_collective_thing():
        return 42

    stream, read_back = report
    uw.mpi.watch(seconds=0.3, stream=stream)
    assert a_collective_thing() == 42
    time.sleep(0.9)
    uw.mpi.unwatch()

    assert "a_collective_thing() [collective]" in read_back()


def test_ranks_agree_passes_when_they_do():
    assert uw.mpi.ranks_agree("same label") == "same label"


def test_ranks_agree_reports_the_split(monkeypatch):
    """Serially every rank is this one, so the disagreement is injected.

    The failure has to be provoked somehow, or the test asserts only that
    agreement is possible --- which is the case the check is not for. The
    parallel suite exercises the real thing.
    """
    class ThreeRanksThatSplit:
        # mpi4py's Intracomm attributes are read-only, so the communicator
        # itself is replaced rather than one of its methods.
        size = 3

        def allgather(self, _value):
            return ["took branch A", "took branch B", "took branch A"]

    monkeypatch.setattr(uw.mpi, "comm", ThreeRanksThatSplit())
    monkeypatch.setattr(uw.mpi, "size", 3)

    with pytest.raises(uw.mpi.CollectiveOperationError) as excinfo:
        uw.mpi.ranks_agree("took branch A")

    message = str(excinfo.value)
    assert "RANKS DISAGREE" in message
    assert "took branch A" in message and "took branch B" in message
    # The table must name WHICH ranks, or it does not localise anything.
    assert "[0, 2]" in message and "[1]" in message
