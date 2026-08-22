"""The hang analyser: parsing, the roll call, and one end-to-end run.

pytest never arms the watchdog here. It is process-global state --- faulthandler
has one ``dump_traceback_later`` slot and pytest's own plugin already owns it ---
and a rank inside a test framework is one process looking at itself, which is
the one thing the analysis cannot be done from. So the end-to-end test launches
a real ``mpirun`` job as a subprocess and checks the verdict it produces.

Everything above that is ordinary text processing and is tested directly.
"""

import os
import shutil
import subprocess
import sys
import textwrap

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


DIVERGENT = """
import time
import underworld3 as uw

def classify():
    '''Stands in for library code deciding how much work this rank has.'''
    return [] if uw.mpi.rank == 1 else [1, 2, 3]

def reduce_the_count(undecided):
    '''The un-noticed collective, behind a rank-local guard.'''
    return uw.mpi.comm.allreduce(len(undecided))

undecided = classify()
if undecided:
    reduce_the_count(undecided)
else:
    time.sleep(3600)
"""


@pytest.mark.timeout(300)
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
def test_end_to_end_names_the_divergent_rank(tmp_path):
    """A real four-rank job that really hangs, killed, then analysed.

    This is the workflow the tool is for: the job does NOT recover, mpirun
    times it out, and the dumps are all that is left. An earlier version had
    the divergent rank join late so the job exited cleanly, which tested a
    situation nobody is ever in and made termination the flaky part.

    Ranks 0, 2 and 3 block in the collective; rank 1 branches around it and
    sleeps. Both groups dump, so the roll call has to separate them.
    """
    script = tmp_path / "divergent.py"
    script.write_text(textwrap.dedent(DIVERGENT))
    dumps = tmp_path / "uw-hang-dumps"

    environment = dict(
        os.environ,
        UW_HANG_WATCHDOG="1.0",
        UW_HANG_WATCHDOG_DIR=str(dumps),
        UW_NO_USAGE_METRICS="1",
    )
    # `--oversubscribe` because a CI runner has two cores and this wants four
    # ranks. They spend the whole test blocked or asleep, so the cores are not
    # the constraint -- but without it OpenMPI declines to start and the job
    # produces no dumps at all.
    #
    # A non-zero exit is EXPECTED: mpirun kills a job that never finishes.
    finished = subprocess.run(
        ["mpirun", "--oversubscribe", "--timeout", "25", "-n", "4",
         sys.executable, "-u", str(script)],
        env=environment, capture_output=True, text=True, timeout=120,
    )

    if not dumps.is_dir():
        pytest.fail(
            "the job wrote no dumps at all, so it never reached "
            f"`import underworld3`:\n{finished.stderr[-2000:]}"
        )

    states = hang_report.read_directory(dumps)
    assert len(states) == 4, f"expected four dump files, got {sorted(states)}"

    groups, moving = hang_report.roll_call(states)
    biggest_where, biggest_ranks, _stack = groups[0]
    report = hang_report.format_report(states)

    assert biggest_ranks == [0, 2, 3], (
        f"the waiting ranks were {biggest_ranks}, not [0, 2, 3]:\n{report}"
    )
    assert biggest_where[2] == "reduce_the_count", (
        f"the majority was located at {biggest_where}, not at the collective"
    )
    odd_ones_out = sorted(r for _w, ranks, _s in groups[1:] for r in ranks) + moving
    assert odd_ones_out == [1], (
        f"rank 1 took the branch; the report blamed {odd_ones_out}:\n{report}"
    )
    # The verdict has to point at the branch, not merely list stacks.
    assert "where the bug is" in report
