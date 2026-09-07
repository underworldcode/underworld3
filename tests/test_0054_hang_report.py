"""The hang analyser: parsing, the roll call, and one end-to-end run.

pytest never arms the watchdog here. It is process-global state --- faulthandler
has one ``dump_traceback_later`` slot and pytest's own plugin already owns it ---
and a rank inside a test framework is one process looking at itself, which is
the one thing the analysis cannot be done from. So the end-to-end test launches
a real ``mpirun`` job as a subprocess and checks the verdict it produces.

Everything above that is ordinary text processing and is tested directly.
"""

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from underworld3.utilities import hang_report

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def dump(*thread_stacks):
    """One faulthandler dump, in the exact layout it writes."""
    out = ["Timeout (0:00:01.000000)!"]
    for index, frames in enumerate(thread_stacks):
        out.append(f"Thread 0x{index:016x} (most recent call first):")
        out.extend(f'  File "{f}", line {n} in {fn}' for f, n, fn in frames)
        out.append("")
    return "\n".join(out)


def stack(*frames):
    return [hang_report.Frame(*f) for f in frames]


WATCHDOG_THREAD = stack(
    ("/usr/lib/python3.12/threading.py", 359, "wait"),
    ("/usr/lib/python3.12/threading.py", 1431, "run"),
    ("/usr/lib/python3.12/threading.py", 1032, "_bootstrap"),
)
BLOCKED_IN_COLLECTIVE = stack(
    ("/uw/discretisation/discretisation_mesh.py", 6954, "get_max_radius"),
    ("/uw/discretisation/discretisation_mesh.py", 6140, "_classify_points_in_domain"),
    ("/uw/swarm.py", 3786, "migrate"),
    ("/home/run/model.py", 41, "<module>"),
)
RAN_AHEAD = stack(
    ("/uw/swarm.py", 3801, "migrate"),
    ("/home/run/model.py", 41, "<module>"),
)


def test_parses_every_dump_in_a_file():
    """`repeat` means many dumps per file; the last one is the live state."""
    text = dump(WATCHDOG_THREAD, BLOCKED_IN_COLLECTIVE) * 3
    dumps = hang_report.parse_dump_file(text)
    assert len(dumps) == 3
    assert all(len(threads) == 2 for threads in dumps)


def test_main_thread_is_chosen_by_its_root_not_its_depth():
    """Regression: a deep helper thread must not be mistaken for the model.

    Measured on a real run --- rank 0 sat two frames into `allreduce` while a
    telemetry thread was twelve frames deep inside `requests`, and picking the
    longest stack reported the rank as being in an HTTP call.
    """
    deep_helper = stack(
        ("/usr/lib/python3.12/http/client.py", 1390, "getresponse"),
        ("/site-packages/requests/sessions.py", 579, "request"),
        ("/site-packages/urllib3/connectionpool.py", 715, "urlopen"),
        ("/usr/lib/python3.12/threading.py", 1032, "_bootstrap"),
    )
    assert len(deep_helper) > len(RAN_AHEAD)

    chosen = hang_report.main_thread_stack([deep_helper, RAN_AHEAD])
    assert chosen == RAN_AHEAD, (
        "the deeper helper thread was reported as the main thread"
    )


def test_roll_call_separates_the_majority_from_the_odd_one_out():
    states = {
        0: [BLOCKED_IN_COLLECTIVE],
        1: [RAN_AHEAD],
        2: [BLOCKED_IN_COLLECTIVE],
        3: [BLOCKED_IN_COLLECTIVE],
    }
    groups, moving = hang_report.roll_call(states)

    assert not moving
    assert [ranks for _where, ranks, _stack in groups] == [[0, 2, 3], [1]]


def test_a_rank_with_no_dump_counts_as_still_moving():
    """Silence is the commonest signature of the rank that caused it.

    A rank that branched away and kept working never stops checking in, so it
    never dumps. The report has to say so rather than ignore it.
    """
    states = {0: [BLOCKED_IN_COLLECTIVE], 1: [], 2: [BLOCKED_IN_COLLECTIVE]}
    groups, moving = hang_report.roll_call(states)

    assert moving == [1]
    assert [ranks for _where, ranks, _stack in groups] == [[0, 2]]

    text = hang_report.format_report(states)
    assert "never stopped" in text
    assert "ranks [1] are where the bug is" in text.replace("\n", " ")


def test_all_ranks_together_is_not_reported_as_a_missed_collective():
    """Everyone in the same place is a slow phase, not divergence.

    Saying "missed collective" here would send the reader hunting a bug that
    is not there.
    """
    states = {r: [BLOCKED_IN_COLLECTIVE] for r in range(4)}
    text = hang_report.format_report(states)
    assert "together" in text
    assert "Not a missed collective" in text


def test_empty_directory_says_so_rather_than_passing_quietly(tmp_path):
    assert "No rank*.log files found" in hang_report.format_report(
        hang_report.read_directory(tmp_path)
    )


def _launcher(ranks):
    """An `mpirun` command line that works on both MPI families.

    The launch flags are NOT portable. OpenMPI refuses to start more ranks than
    cores without `--oversubscribe`; MPICH oversubscribes by default and rejects
    the flag outright, taking the whole command down with it. `--timeout` is
    likewise OpenMPI-only. So the family is detected, and the time limit is
    enforced from Python instead of by the launcher.
    """
    command = ["mpirun", "-n", str(ranks)]
    try:
        banner = subprocess.run(["mpirun", "--version"], capture_output=True,
                                text=True, timeout=30)
        flavour = (banner.stdout + banner.stderr)
    except (OSError, subprocess.SubprocessError):
        flavour = ""
    if "Open MPI" in flavour or "OpenRTE" in flavour:
        # The ranks are blocked or asleep throughout, so cores are not the
        # constraint -- but OpenMPI counts slots, not activity.
        command.insert(1, "--oversubscribe")
    return command


def _wait_for(condition, what, cap=600.0, poll=0.25):
    """Block until `condition()` is true. Returns how long it took.

    The cap is a backstop against a genuinely broken run, not a measurement --
    it is set far above any plausible duration precisely so that reaching it
    means something is wrong rather than something is slow.
    """
    deadline = time.monotonic() + cap
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll)
    raise AssertionError(f"gave up after {cap:g} s waiting for {what}")


def _dump_count(dumps, rank):
    """How many times this rank has dumped so far.

    Counted by the "Current thread" line rather than the "Timeout (" header:
    the watchdog dumps through faulthandler's signal handler (#661), which
    writes no header, and there is exactly one such line per dump either way.
    """
    path = dumps / f"rank{rank:04d}.log"
    if not path.exists():
        return 0
    return path.read_text(errors="replace").count("Current thread ")


def _run_until_the_evidence_exists(argv, ranks, environment, dumps, ready,
                                   blocked_ranks):
    """Launch under MPI, wait for the evidence, then kill the job.

    Nothing here is timed against the clock, and that is the point. Two earlier
    versions of this test raced a fixed window against machine speed and failed
    on CI twice for reasons that were true reports of a slower runner rather
    than defects: first the waiting group came back short because a rank had not
    been scheduled enough to dump, then the whole file filled with
    `importlib._bootstrap` frames because four oversubscribed ranks took longer
    to import than the watchdog allowed.

    So the script signals when it has armed its watchdog -- after import, at a
    known program point -- and this waits for that, then waits for the blocked
    ranks to have dumped twice, then kills. Import may take as long as it likes.
    The blocked ranks are blocked until killed, so the second wait always
    completes; only a real failure reaches the cap.
    """
    process = subprocess.Popen(
        _launcher(ranks) + argv, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        _wait_for(lambda: len(list(ready.glob("ready.*"))) == ranks
                  or process.poll() is not None,
                  f"all {ranks} ranks to import and arm the watchdog")
        assert process.poll() is None, "the job exited before arming"

        # Two dumps, not one: one says "slow", two says "stuck", and the roll
        # call places a rank by the mode of its recent dumps.
        _wait_for(lambda: all(_dump_count(dumps, r) >= 2 for r in blocked_ranks),
                  f"ranks {sorted(blocked_ranks)} to dump twice while blocked")
    finally:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        _out, err = process.communicate()
    return err or ""


DIVERGENT = """
import pathlib
import time
import underworld3 as uw

def classify():
    '''Stands in for library code deciding how much work this rank has.'''
    return [] if uw.mpi.rank == 1 else [1, 2, 3]

def reduce_the_count(undecided):
    '''The un-noticed collective, behind a rank-local guard.'''
    return uw.mpi.comm.allreduce(len(undecided))

# Armed HERE, not from the environment: at a known point in the program rather
# than at import, so how long the import took cannot decide what the dumps
# contain.
dumps = pathlib.Path(DUMPS)
dumps.mkdir(parents=True, exist_ok=True)
log = open(dumps / ("rank%04d.log" % uw.mpi.rank), "w", buffering=1)
uw.mpi.watch(seconds=1.0, stream=log)

# Tell the test the watchdog is live. It waits for this instead of guessing.
(pathlib.Path(READY) / ("ready.%d" % uw.mpi.rank)).write_text("armed")

undecided = classify()
if undecided:
    reduce_the_count(undecided)
else:
    time.sleep(3600)
"""


@pytest.mark.timeout(900)
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
def test_end_to_end_names_the_divergent_rank(tmp_path):
    """A real four-rank job that really hangs, killed, then analysed.

    This is the workflow the tool is for: the job does NOT recover, it is killed,
    and the dumps are all that is left.

    Nothing is timed against the clock. The script arms its own watchdog after
    import and writes a marker; the test waits for those markers, then waits for
    the blocked ranks to have dumped twice, then kills. Two earlier versions
    raced a fixed window against machine speed and failed on CI twice — both
    times reporting a slower runner rather than a defect.
    """
    ready = tmp_path / "ready"
    dumps = tmp_path / "uw-hang-dumps"
    ready.mkdir()

    script = tmp_path / "divergent.py"
    script.write_text(
        f"DUMPS = {str(dumps)!r}\nREADY = {str(ready)!r}\n"
        + textwrap.dedent(DIVERGENT)
    )

    stderr = _run_until_the_evidence_exists(
        [sys.executable, "-u", str(script)], ranks=4,
        environment=dict(os.environ, UW_NO_USAGE_METRICS="1"),
        dumps=dumps, ready=ready, blocked_ranks=(0, 2, 3),
    )

    if not dumps.is_dir():
        pytest.fail(f"the job wrote no dumps at all:\n{stderr[-2000:]}")

    states = hang_report.read_directory(dumps)
    assert len(states) == 4, f"expected four dump files, got {sorted(states)}"

    groups, moving = hang_report.roll_call(states)
    biggest_where, biggest_ranks, _stack = groups[0]
    report = hang_report.format_report(states)

    assert biggest_where[2] == "reduce_the_count", (
        f"the majority was located at {biggest_where[0]}:{biggest_where[1]} in "
        f"{biggest_where[2]}, not at the collective.\n{report}"
    )
    assert sorted(biggest_ranks) == [0, 2, 3], (
        f"the waiting group was {biggest_ranks}; every blocked rank was waited "
        f"for, so all three must be present.\n{report}"
    )
    odd_ones_out = sorted(r for _w, ranks, _s in groups[1:] for r in ranks) + moving
    assert odd_ones_out == [1], (
        f"rank 1 took the branch; the report blamed {odd_ones_out}.\n{report}"
    )
    # The verdict has to point at the branch, not merely list stacks.
    assert "where the bug is" in report


ARMED_BY_ENVIRONMENT = """
import time
import underworld3 as uw
time.sleep(3600)
"""


@pytest.mark.timeout(300)
def test_the_environment_variable_arms_the_watchdog_at_import():
    """`UW_HANG_WATCHDOG=...` must arm without the script asking.

    The end-to-end test above arms from the script, deliberately — that is what
    makes it independent of how long the import takes. So this covers the other
    entry point, which is the documented one and the reason arming happens at
    import at all: a rank that diverges or dies before reaching a `watch()` call
    reports nothing.

    One rank, no MPI launcher: this is about the environment being read, not
    about ranks.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        root = pathlib.Path(workspace)
        script = root / "sleepy.py"
        script.write_text(textwrap.dedent(ARMED_BY_ENVIRONMENT))
        dumps = root / "dumps"

        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            env=dict(os.environ, UW_HANG_WATCHDOG="1.0",
                     UW_HANG_WATCHDOG_DIR=str(dumps), UW_NO_USAGE_METRICS="1"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            _wait_for(
                lambda: _dump_count(dumps, 0) >= 2 or process.poll() is not None,
                "the environment-armed watchdog to dump twice",
            )
            assert process.poll() is None, "the script exited instead of hanging"
            assert _dump_count(dumps, 0) >= 2
            assert "sleepy.py" in (dumps / "rank0000.log").read_text(), (
                "the dump did not carry the stack of the script that hung"
            )
        finally:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.communicate()
