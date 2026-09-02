"""Read a directory of per-rank hang dumps and say which rank went its own way.

A collective entered by only some ranks leaves a distinctive trace: N-1 ranks
stopped in the same frame, and one somewhere else. No single rank can see that
--- it exists only across the set --- so the watchdog writes one dump per rank
and the comparison happens here, afterwards.

That is also why this is a script rather than a test. The analysis needs every
rank's file at once, and a rank inside a test framework is one process looking
at itself.

Usage::

    UW_HANG_WATCHDOG=120 mpirun -n 4 python myrun.py
    python -m underworld3.utilities.hang_report uw-hang-dumps

The dumps come from :mod:`faulthandler`, so a frame is ``File "x.py", line N in
func``. It names the CALLING line, not the C function it called: a rank blocked
in ``comm.allreduce`` shows the Python line that made the call, which is the
localisation you want anyway.
"""

import argparse
import collections
import os
import re
import sys

#: Start of one dump. faulthandler writes this before each set of stacks.
_TIMEOUT = re.compile(r"^Timeout \(")
#: Start of one thread's stack within a dump.
_THREAD = re.compile(r"^(?:Current thread|Thread) 0x[0-9a-fA-F]+")
#: A single frame.
_FRAME = re.compile(r'^\s+File "(?P<file>.*)", line (?P<line>\d+) in (?P<func>.*)$')

#: Frames from these belong to the machinery, not to the model. A thread whose
#: own top frame is one of these is a helper --- the watchdog's timer, an I/O
#: thread --- and is not where the rank is stuck.
_MACHINERY = ("threading.py", "concurrent/futures", "selectors.py")


class Frame(collections.namedtuple("Frame", "file line func")):
    """One stack frame, rendered short enough to compare by eye."""

    def __str__(self):
        return f"{os.path.basename(self.file)}:{self.line} in {self.func}"

    @property
    def where(self):
        """Identity for grouping: the same source line on any rank."""
        return (os.path.basename(self.file), self.line, self.func)


def parse_dump_file(text):
    """Every dump in one rank's file, newest last, as lists of thread stacks."""
    dumps, threads, frames = [], [], []

    def close_thread():
        if frames:
            threads.append(list(frames))
        frames.clear()

    def close_dump():
        close_thread()
        if threads:
            dumps.append(list(threads))
        threads.clear()

    for raw in text.splitlines():
        if _TIMEOUT.match(raw):
            close_dump()
            continue
        if _THREAD.match(raw):
            close_thread()
            continue
        found = _FRAME.match(raw)
        if found:
            frames.append(
                Frame(found["file"], int(found["line"]), found["func"].strip())
            )
    close_dump()
    return dumps


def main_thread_stack(threads):
    """The stack that is the model, not a helper thread.

    faulthandler does not label the main thread when it dumps from a helper, so
    it is picked by its OUTERMOST frame --- the last line, since frames print
    most-recent-first. A spawned thread bottoms out in ``threading.py``'s
    ``_bootstrap``; the main thread bottoms out in the script.

    Depth is the wrong test, and measurably so: a rank blocked two frames into
    ``allreduce`` while a telemetry thread sat twelve frames deep inside
    ``requests`` was reported as being in the HTTP call.
    """
    candidates = [
        stack for stack in threads
        if stack and not any(part in stack[-1].file for part in _MACHINERY)
    ]
    if not candidates:
        return None
    # A script's main thread ends in `<module>`; under runpy or pytest it does
    # not, so that is a preference and not a requirement.
    rooted = [stack for stack in candidates if stack[-1].func == "<module>"]
    return (rooted or candidates)[0]


def read_directory(directory):
    """``{rank: [stack, ...]}`` for every ``rank*.log`` found, oldest first."""
    states = {}
    for name in sorted(os.listdir(directory)):
        found = re.fullmatch(r"rank(\d+)\.log", name)
        if not found:
            continue
        with open(os.path.join(directory, name), errors="replace") as handle:
            text = handle.read()
        stacks = [
            stack for stack in
            (main_thread_stack(threads) for threads in parse_dump_file(text))
            if stack
        ]
        states[int(found[1])] = stacks
    return states


#: Dumps to consider when deciding where a rank has settled.
RECENT = 5


def settled_position(stacks, recent=RECENT):
    """Where a rank has come to rest, from its last few dumps.

    Not simply the final dump. A rank stuck in a collective shows the same
    position every period, so the mode over recent samples is the stable
    signal; the last sample alone can catch a rank mid-transition and put it in
    a group of its own. Measured: one rank of four fell out of the waiting
    group that way, and the report named two culprits instead of one.

    Ties go to the most recent, so a rank that really did move on is not held
    at an older position.
    """
    window = stacks[-recent:]
    counts = collections.Counter(stack[0].where for stack in window)
    best = max(counts.items(), key=lambda kv: kv[1])[1]
    for stack in reversed(window):
        if counts[stack[0].where] == best:
            return stack
    return window[-1]


def roll_call(states):
    """Group ranks by where they last were. Returns (groups, ranks_never_stuck).

    ``groups`` maps a position to the ranks stopped there, largest first. A
    rank with no dump at all never stopped, which is itself informative: it is
    the one that kept going.
    """
    stuck, moving = {}, []
    for who, stacks in sorted(states.items()):
        if not stacks:
            moving.append(who)
            continue
        stuck[who] = settled_position(stacks)

    groups = collections.defaultdict(list)
    for who, stack in stuck.items():
        groups[stack[0].where].append(who)

    ordered = sorted(
        ((where, sorted(ranks), stuck[ranks[0]])
         for where, ranks in groups.items()),
        key=lambda item: (-len(item[1]), item[1][0]),
    )
    return ordered, sorted(moving)


def format_report(states, depth=6):
    """The verdict, as text."""
    if not states:
        return ("No rank*.log files found. Was UW_HANG_WATCHDOG set, and is "
                "this the directory it wrote to?")

    groups, moving = roll_call(states)
    total = len(states)
    out = [f"Hang report: {total} rank(s)", ""]

    if not groups:
        out.append("No rank ever stopped long enough to dump. Either nothing "
                   "hung, or the timeout is longer than the run.")
        return "\n".join(out)

    for where, ranks, stack in groups:
        head = f"ranks {ranks}" if len(ranks) > 1 else f"rank  {ranks}"
        out.append(f"{head}  stopped at  {stack[0]}")
        for frame in stack[1:depth]:
            out.append(f"        via  {frame}")
        if len(stack) > depth:
            out.append(f"        ... {len(stack) - depth} more frame(s)")
        out.append("")

    if moving:
        out.append(f"ranks {moving}  never stopped -- still making progress.")
        out.append("")

    out.append(_verdict(groups, moving, total))
    return "\n".join(out)


def _verdict(groups, moving, total):
    """The one sentence worth reading."""
    stopped = sum(len(ranks) for _where, ranks, _stack in groups)

    if moving and stopped:
        where, ranks, _stack = groups[0]
        return (
            f"=> {stopped} of {total} ranks are waiting at "
            f"{where[0]}:{where[1]}, and ranks {moving} went past it.\n"
            f"   That frame is where the collective is; ranks {moving} are "
            f"where the bug is.\n"
            f"   Look for a branch on rank-local data before it, and reduce "
            f"the predicate\n   before branching on it."
        )

    if len(groups) > 1:
        where, ranks, _stack = groups[0]
        others = sorted(r for _w, group, _s in groups[1:] for r in group)
        return (
            f"=> {len(ranks)} of {total} ranks are waiting together at "
            f"{where[0]}:{where[1]}, and ranks {others}\n"
            f"   are somewhere else. That frame is where the collective is; "
            f"ranks {others} are\n   where the bug is. Look for a branch on "
            f"rank-local data before it, and reduce\n   the predicate before "
            f"branching on it."
        )

    if stopped == total:
        where = groups[0][0]
        return (
            f"=> all {total} ranks are stopped at {where[0]}:{where[1]}, "
            f"together.\n"
            f"   Not a missed collective -- they agree. Something outside the "
            f"job is not\n   completing, or this phase is simply slower than "
            f"the timeout."
        )

    return f"=> {stopped} of {total} ranks stopped; no rank ran on."


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m underworld3.utilities.hang_report",
        description="Compare per-rank hang dumps and name the divergent rank.",
    )
    parser.add_argument(
        "directory", nargs="?", default="uw-hang-dumps",
        help="directory of rankNNNN.log files (default: uw-hang-dumps)",
    )
    parser.add_argument(
        "--depth", type=int, default=6,
        help="frames of context to show per group (default: 6)",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.directory):
        parser.error(f"no such directory: {args.directory}")

    print(format_report(read_directory(args.directory), depth=args.depth))
    return 0


if __name__ == "__main__":
    sys.exit(main())
