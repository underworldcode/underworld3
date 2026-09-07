#!/usr/bin/env python3
"""Run an MPI job under a supervisor that bounds a collective hang and diagnoses it.

A rank blocked inside a collective cannot help you. It does not reach a
``threading.Timer`` (measured: a re-arming timer at 0.5 s fired zero times on
blocked ranks), it cannot be reasoned with by ``MPI_Abort`` from another
thread, and ``faulthandler.dump_traceback_later`` — a C thread walking other
threads' live frames — can wedge or crash the process it is meant to diagnose
(underworld3#661). Every one of those mechanisms asks the stuck process to
cooperate, which is the one thing it has stopped being able to do.

This supervisor asks nothing of it. It is a parent process holding a clock and
a signal, so it works the same under Open MPI and MPICH, on macOS and Linux,
because it is POSIX process management and knows nothing about MPI.

What it does
------------
1. Launches the job in a new session, so the whole tree can be signalled as one
   process group.
2. Watches the job's OUTPUT, not its total runtime. A batch still printing is
   alive; silence is the signal. This avoids inventing a wall-clock budget for a
   test matrix nobody has timed, and it does not punish a legitimately slow run.
3. On silence: signals every rank to dump its Python stack (a signal handler
   runs on the blocked thread itself and never runs while things are healthy),
   samples twice to separate "stuck" from "slow", then compares the ranks and
   names the one that is somewhere the others are not.
4. Kills the process group and verifies nothing survived. Killing the launcher
   is not killing the job: an unsupervised kill leaves orphaned ``mpirun``,
   ``hydra_pmi_proxy`` and pytest children spinning at 100% CPU.

Usage
-----
::

    scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest ...

Exit status is the job's own, or 124 if the supervisor had to kill it.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

# The job outlived its silence budget and was killed. Matches coreutils
# `timeout`, which is what a reader will assume this number means.
EXIT_KILLED = 124

# Between the two stack samples. Two identical stacks say "stuck"; one says
# only "slow", and that distinction is the whole point of sampling twice.
SECONDS_BETWEEN_SAMPLES = 20.0

# After the dump request, before escalating. Enough for a signal handler to
# write a few frames to a file, not enough to matter to a CI budget.
SECONDS_TO_DUMP = 5.0

# Between SIGTERM and SIGKILL.
SECONDS_TO_TERMINATE = 5.0


def _dump_handler_module(dump_dir):
    """Source for the module that arms each rank to dump on request.

    Installed as ``usercustomize``, which the ``site`` machinery imports at
    interpreter start-up, so every rank is armed without the job's own code
    knowing anything about it.

    This registers a SIGNAL handler, which is the important difference from
    ``faulthandler.dump_traceback_later``: it runs on the thread that receives
    the signal, only when we ask for it, and never walks live frames from a
    concurrent thread. That is why arming it costs a healthy run nothing and
    cannot reproduce underworld3#661.
    """
    return f'''"""Arm this rank to dump its stack on SIGUSR1. Written by mpi_supervisor."""
import faulthandler
import os
import signal

# Rank as the launchers advertise it, so a dump can be attributed without
# asking mpi4py (importing it here would run before the job's own import).
_rank = (os.environ.get("OMPI_COMM_WORLD_RANK")
         or os.environ.get("PMI_RANK")
         or os.environ.get("MV2_COMM_WORLD_RANK")
         or "unknown")

# Held open for the life of the process on purpose: faulthandler keeps the
# descriptor, not this object, so letting it close would point the handler at
# whatever reuses the fd.
_path = os.path.join({str(dump_dir)!r}, "rank_%s_pid_%d.stack" % (_rank, os.getpid()))
_stream = open(_path, "w")
faulthandler.register(signal.SIGUSR1, file=_stream, all_threads=True, chain=False)
'''


def _install_dump_handler(dump_dir, env):
    """Put the handler module on the job's import path.

    ``usercustomize`` rather than ``sitecustomize`` because it is imported
    later and is far less likely to shadow one the environment already
    provides.
    """
    handler_dir = Path(tempfile.mkdtemp(prefix="uw-supervisor-"))
    (handler_dir / "usercustomize.py").write_text(_dump_handler_module(dump_dir))

    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{handler_dir}{os.pathsep}{existing}" if existing else str(handler_dir)
    return handler_dir


def _descendants(pid):
    """Every live descendant of *pid*, deepest last.

    ``ps`` rather than psutil: this must run on a bare CI image, and the
    supervisor has to work when the thing it supervises is the broken part.
    """
    listing = subprocess.run(
        ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True
    ).stdout

    children = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) == 2:
            child, parent = int(fields[0]), int(fields[1])
            children.setdefault(parent, []).append(child)

    found, frontier = [], [pid]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            found.append(child)
            frontier.append(child)
    return found


def _cpu_percent(pids):
    """Per-PID CPU, which separates a spinning collective from a still one.

    A rank inside a busy-waiting MPI progress engine burns a whole core; one
    blocked on a lock sleeps. That is a cheap and surprisingly sharp signal
    about which kind of stuck we are looking at.
    """
    if not pids:
        return {}
    listing = subprocess.run(
        ["ps", "-Ao", "pid=,pcpu="], capture_output=True, text=True
    ).stdout

    wanted = set(pids)
    usage = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) == 2 and int(fields[0]) in wanted:
            usage[int(fields[0])] = float(fields[1])
    return usage


def _request_dumps(pids):
    """Ask every rank for a stack. Ranks that are gone are simply not asked."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            # The rank exited between listing and signalling. Nothing to dump.
            pass


def _dump_blocks(text):
    """The individual dumps in one rank's file.

    ``faulthandler`` appends, so a file accumulates every sample we have asked
    for. Splitting them apart is what lets one read answer both "where is this
    rank" and "has it moved since last time".
    """
    blocks = re.split(r"^Current thread ", text, flags=re.MULTILINE)
    return [block for block in blocks if block.strip()]


def _stack_signature(block):
    """The innermost frames of one dump, as the identity of where a rank is.

    Function names only: paths and line numbers differ between ranks running
    from different working directories and would make two ranks in the same
    place look different. ``faulthandler`` prints most-recent-call-first, so
    the innermost frame is the FIRST one.
    """
    frames = re.findall(r'File "[^"]*", line \d+ in (\S+)', block)
    return tuple(frames[:4])


def _read_dumps(dump_dir):
    """Every rank's samples, as ``{rank: (pid, [block, ...])}``."""
    dumps = {}
    for path in sorted(Path(dump_dir).glob("rank_*_pid_*.stack")):
        rank, _, pid = path.stem.removeprefix("rank_").partition("_pid_")
        blocks = _dump_blocks(path.read_text())
        if blocks:
            dumps[rank] = (int(pid), blocks)
    return dumps


def _diagnose(dumps, cpu_by_rank):
    """Say which rank missed the collective, and whether it is stuck or slow.

    This is the reading a human would make from a pile of stack dumps, made
    here instead: the ranks that agree are waiting for the one that does not.
    Naming that rank is the difference between a diagnosis and more log volume.

    *dumps* is ``{rank: (pid, [block, ...])}`` with the samples in the order
    they were taken.
    """
    lines = []

    moved = {rank: _stack_signature(blocks[-2]) != _stack_signature(blocks[-1])
             for rank, (_pid, blocks) in dumps.items() if len(blocks) >= 2}
    if moved and not any(moved.values()):
        lines.append(f"    no rank moved in {SECONDS_BETWEEN_SAMPLES:.0f} s: "
                     "this is stuck, not slow")
    elif any(moved.values()):
        still = sorted(rank for rank, m in moved.items() if not m)
        busy = sorted(rank for rank, m in moved.items() if m)
        lines.append(f"    ranks {', '.join(busy)} are still moving; "
                     f"ranks {', '.join(still)} are not — this may be slow, not stuck")

    signatures = {rank: _stack_signature(blocks[-1]) for rank, (_pid, blocks) in dumps.items()}
    tally = Counter(signatures.values())
    majority, majority_count = tally.most_common(1)[0]

    def _where(rank):
        return signatures[rank][0] if signatures[rank] else "an unknown frame"

    if len(tally) == 1:
        lines.append(f"    every rank is in {majority[0] if majority else 'the same place'}()"
                     " — no rank is the odd one out, so suspect a collective they "
                     "have all entered")
    elif any(count < majority_count for count in tally.values()):
        odd = sorted(rank for rank, sig in signatures.items() if sig != majority)
        lines.append(f"    rank(s) {', '.join(odd)} are somewhere the other "
                     f"{majority_count} are not — that is where to look first")
        for rank in odd:
            lines.append(f"      rank {rank} is in {_where(rank)}()")
        lines.append(f"      the other {majority_count} are in "
                     f"{majority[0] if majority else 'an unknown frame'}()")
    else:
        # No majority to appeal to — the usual case at np=2. Still worth
        # reporting where each rank is; a reader can see the mismatch even
        # when the supervisor cannot vote on it.
        lines.append("    the ranks are in different places, and with no majority "
                     "none can be called the odd one out:")
        for rank in sorted(signatures):
            lines.append(f"      rank {rank} is in {_where(rank)}()")

    for rank in sorted(cpu_by_rank):
        percent = cpu_by_rank[rank]
        state = ("spinning — a busy-wait, which is what an MPI progress engine does"
                 if percent > 50 else "idle — blocked rather than polling")
        lines.append(f"    rank {rank}: {percent:.0f}% CPU, {state}")

    return lines


def _kill_the_job(pid):
    """End the whole tree and confirm it is gone. Returns any survivors.

    Descendants are killed individually before the group, because the group
    call is the one that fails: ``mpirun`` puts its children in a group of
    their own, and ``killpg`` on macOS has been seen to refuse with EPERM.
    Enumerating the tree does not depend on either.

    The confirmation is not ceremony. Killing only the launcher leaves the
    children orphaned to init and spinning at 100% CPU, which is how a
    "finished" run keeps eating cores.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for victim in reversed(_descendants(pid)):
            try:
                os.kill(victim, sig)
            except (ProcessLookupError, PermissionError):
                # Already gone, or not ours to kill: either way the group
                # sweep below is the remaining chance, and survivors are
                # reported rather than hidden.
                pass
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

        deadline = time.monotonic() + SECONDS_TO_TERMINATE
        while time.monotonic() < deadline and _descendants(pid):
            time.sleep(0.5)
        if not _descendants(pid):
            break

    return _descendants(pid)


def main():
    parser = argparse.ArgumentParser(
        description="Run an MPI job, bound any collective hang, and diagnose it.")
    parser.add_argument(
        "--silence", type=float, default=300.0, metavar="SECONDS",
        help="kill the job after this long with no output (default: 300)")
    parser.add_argument(
        "--hard-cap", type=float, default=0.0, metavar="SECONDS",
        help="also kill it after this much total runtime (default: no cap)")
    parser.add_argument(
        "--artifacts", type=Path, default=None, metavar="DIR",
        help="where to keep the stack dumps (default: a temporary directory)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the job, after a bare --")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; put it after a bare --")

    dump_dir = args.artifacts or Path(tempfile.mkdtemp(prefix="uw-stacks-"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    _install_dump_handler(dump_dir, env)

    job = subprocess.Popen(
        command, env=env, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # The job's output is both the thing we forward and the heartbeat we judge
    # it by, so it is read on a thread rather than polled.
    last_output = [time.monotonic()]

    def forward_output():
        for line in job.stdout:
            last_output[0] = time.monotonic()
            sys.stdout.write(line)
            sys.stdout.flush()

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()

    started = time.monotonic()
    while job.poll() is None:
        time.sleep(1.0)
        silent_for = time.monotonic() - last_output[0]
        overran = args.hard_cap and (time.monotonic() - started) > args.hard_cap
        if silent_for <= args.silence and not overran:
            continue

        reason = (f"no output for {silent_for:.0f} s"
                  if not overran else f"exceeded the {args.hard_cap:.0f} s cap")
        print(f"\n=== MPI SUPERVISOR: {reason} ===", flush=True)

        ranks = _descendants(job.pid)
        _request_dumps(ranks)
        time.sleep(SECONDS_TO_DUMP)

        time.sleep(SECONDS_BETWEEN_SAMPLES)
        _request_dumps(ranks)
        time.sleep(SECONDS_TO_DUMP)

        # One read: faulthandler appends, so each rank's file already holds
        # both samples in order.
        dumps = _read_dumps(dump_dir)

        # Attribute CPU through the pid the handler recorded. Guessing which
        # rank owned which figure would produce a confident, wrong label.
        usage = _cpu_percent([pid for pid, _blocks in dumps.values()])
        cpu_by_rank = {rank: usage[pid] for rank, (pid, _blocks) in dumps.items()
                       if pid in usage}

        if dumps:
            for line in _diagnose(dumps, cpu_by_rank):
                print(line, flush=True)
            print(f"    stacks written to {dump_dir}", flush=True)
        else:
            print("    no rank answered the dump request — no stacks available; "
                  "killing the job regardless", flush=True)

        survivors = _kill_the_job(job.pid)
        if survivors:
            print(f"    WARNING: {len(survivors)} process(es) survived the kill: "
                  f"{survivors}", flush=True)
        return EXIT_KILLED

    reader.join(timeout=5.0)
    return job.returncode


if __name__ == "__main__":
    sys.exit(main())
