##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Underworld geophysics modelling application.         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module contains routines related to parallel operation via
the Message Passing Interface (MPI).

Attributes
----------
comm :: mpi4py.MPI.Intracomm
    The MPI communicator.
rank :: int
    The rank of the current process.
size :: int
    The size of the pool of processes.

"""

from mpi4py import MPI as _MPI
import atexit as _atexit
import faulthandler as _faulthandler
import os as _os
import secrets as _secrets
import sys as _sys
import io as _io
import threading as _threading
import time as _time
from contextlib import contextmanager as _contextmanager

# Pre-import EVERYTHING the watchdog reporter thread can touch. The
# reporter (`_Watchdog.report` -> `_stack_dump` -> traceback/linecache)
# runs on a daemon thread, and with `UW_HANG_WATCHDOG` armed at import
# it can fire WHILE the main thread is still inside `import underworld3`.
# `traceback.format_stack` lazily imports through `linecache` (which
# reaches for `tokenize` on first use); a lazy import on a secondary
# thread while the main thread holds per-module import locks is the
# classic cross-thread import deadlock — measured: a 0.2 s watchdog
# livelocks `import underworld3` locally (the main thread pinned
# mid-import, reporter cycling), and CI's 1.0 s watchdog dies at -11 in
# the same window (test_0054). With these loaded before any reporter
# can run, the reporter never enters the import system.
import traceback as _traceback_preload            # noqa: F401
import linecache as _linecache_preload            # noqa: F401
import tokenize as _tokenize_preload              # noqa: F401


comm = _MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# State tracking for selective execution
_in_selective_ranks = False
_selective_executing_ranks = None
_this_rank_executes = True

# get the pid of the root process
pid0 = _os.getpid()
pid0 = comm.bcast(pid0, root=0)

# get a common unique (random) id across all processes
unique = _secrets.token_urlsafe(nbytes=6)
unique = comm.bcast(unique, root=0)


def barrier():
    """
    Creates an MPI barrier. All processes wait here for others to catch up.

    """
    comm.Barrier()


def _should_rank_execute(current_rank, rank_selector, total_size):
    """
    Determine if a rank should execute based on rank selector.

    Args:
        current_rank: The rank to check
        rank_selector: int, slice, list, tuple, callable, str, or numpy array
        total_size: Total number of ranks

    Returns:
        bool: True if rank should execute
    """
    import numpy as np

    if rank_selector is None or rank_selector == "all":
        return True

    if isinstance(rank_selector, int):
        return current_rank == rank_selector

    if isinstance(rank_selector, slice):
        return current_rank in range(*rank_selector.indices(total_size))

    if isinstance(rank_selector, (list, tuple)):
        return current_rank in rank_selector

    if isinstance(rank_selector, str):
        if rank_selector == "first":
            return current_rank == 0
        elif rank_selector == "last":
            return current_rank == total_size - 1
        elif rank_selector == "even":
            return current_rank % 2 == 0
        elif rank_selector == "odd":
            return current_rank % 2 == 1
        elif rank_selector.endswith("%"):
            pct = float(rank_selector[:-1]) / 100
            return current_rank < int(total_size * pct)

    if callable(rank_selector):
        return rank_selector(current_rank)

    if isinstance(rank_selector, np.ndarray):
        if rank_selector.dtype == bool and len(rank_selector) > current_rank:
            return bool(rank_selector[current_rank])
        elif current_rank in rank_selector:
            return True

    return False


def _get_executing_ranks(rank_selector, total_size):
    """
    Get set of ranks that will execute for a given selector.

    Args:
        rank_selector: Rank selection specification
        total_size: Total number of ranks

    Returns:
        set: Set of rank numbers that will execute
    """
    import numpy as np

    if rank_selector is None or rank_selector == "all":
        return set(range(total_size))

    if isinstance(rank_selector, int):
        return {rank_selector}

    if isinstance(rank_selector, slice):
        return set(range(*rank_selector.indices(total_size)))

    if isinstance(rank_selector, (list, tuple)):
        return set(rank_selector)

    if isinstance(rank_selector, str):
        if rank_selector == "first":
            return {0}
        elif rank_selector == "last":
            return {total_size - 1}
        elif rank_selector == "even":
            return set(range(0, total_size, 2))
        elif rank_selector == "odd":
            return set(range(1, total_size, 2))
        elif rank_selector.endswith("%"):
            pct = float(rank_selector[:-1]) / 100
            return set(range(int(total_size * pct)))

    if callable(rank_selector):
        return {r for r in range(total_size) if rank_selector(r)}

    if isinstance(rank_selector, np.ndarray):
        if rank_selector.dtype == bool:
            return {r for r in range(min(len(rank_selector), total_size)) if rank_selector[r]}
        else:
            return set(rank_selector[rank_selector < total_size])

    return set()


@_contextmanager
def selective_ranks(ranks):
    """
    Execute code only on selected ranks, with collective operation detection.

    This context manager allows you to selectively execute code on specific MPI ranks
    while protecting against deadlocks from collective operations.

    Args:
        ranks: Which ranks should execute the code block. Can be:
            - int: Single rank (e.g., 0)
            - slice: Range of ranks (e.g., slice(0, 4))
            - list/tuple: Specific ranks (e.g., [0, 3, 7])
            - str: Named patterns ('all', 'first', 'last', 'even', 'odd', '10%')
            - callable: Function taking rank and returning bool
            - numpy array: Boolean mask or integer indices

    Raises:
        CollectiveOperationError: If a collective operation is detected within
            the selective execution block (would cause deadlock)

    Example:
        >>> with uw.mpi.selective_ranks(0):
        ...     import matplotlib.pyplot as plt
        ...     plt.plot(x, y)
        ...     plt.savefig("output.png")
    """
    global _in_selective_ranks, _selective_executing_ranks, _this_rank_executes

    should_execute = _should_rank_execute(rank, ranks, size)

    old_selective = _in_selective_ranks
    old_executing_ranks = _selective_executing_ranks
    old_this_executes = _this_rank_executes

    _in_selective_ranks = True
    _selective_executing_ranks = _get_executing_ranks(ranks, size)
    _this_rank_executes = should_execute

    try:
        if should_execute:
            yield True
        else:
            yield False
    finally:
        _in_selective_ranks = old_selective
        _selective_executing_ranks = old_executing_ranks
        _this_rank_executes = old_this_executes


class CollectiveOperationError(RuntimeError):
    """Raised when a collective operation is called inside selective_ranks()"""

    pass


def collective_operation(func):
    """
    Decorator to mark a function as a collective operation.

    Collective operations must be called on ALL MPI ranks. If called inside
    a selective_ranks() context where not all ranks execute, raises CollectiveOperationError.

    Example:
        >>> @collective_operation
        ... def compute_global_stats(self):
        ...     # This requires all ranks to participate
        ...     return self.vec.norm()
    """

    def wrapper(*args, **kwargs):
        if _in_selective_ranks:
            # Check if all ranks are executing
            if _selective_executing_ranks is not None and len(_selective_executing_ranks) != size:
                # Not all ranks will execute - this is a collective operation error
                func_name = func.__name__
                executing_ranks = list(_selective_executing_ranks)
                all_ranks = list(range(size))
                excluded_ranks = [r for r in all_ranks if r not in executing_ranks]

                error_msg = (
                    f"\n{'='*70}\n"
                    f"COLLECTIVE OPERATION DEADLOCK DETECTED\n"
                    f"{'='*70}\n\n"
                    f"Function '{func_name}' is a collective operation that requires ALL ranks.\n"
                    f"Currently executing on ranks {executing_ranks}\n"
                    f"but NOT executing on ranks {excluded_ranks}.\n\n"
                    f"This will cause a DEADLOCK because not all ranks participate.\n\n"
                    f"SOLUTION:\n"
                    f"  Execute on all ranks, print on selected ranks:\n"
                    f'    uw.pprint(f"Result: {{obj.{func_name}()}}", proc={executing_ranks[0] if executing_ranks else 0})\n\n'
                    f"Or use the return value pattern:\n"
                    f"    result = obj.{func_name}()  # All ranks execute\n"
                    f'    uw.pprint(f"Result: {{result}}", proc={executing_ranks[0] if executing_ranks else 0})\n'
                    f"{'='*70}\n"
                )
                raise CollectiveOperationError(error_msg)

        # Entering a declared collective is exactly the progress the watchdog
        # wants to hear about, and it gives the report a label without the
        # caller having to place one. Costs a single test when disarmed.
        if _watchdog is not None:
            _watchdog.arm(f"entering {func.__name__}() [collective]")

        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper._is_collective = True
    return wrapper


# ---------------------------------------------------------------------------
# Hang watchdog
# ---------------------------------------------------------------------------
#
# When ranks diverge over a collective, the ones that arrived block inside MPI
# and the job sits there until the queue kills it. Nothing in the output says
# which rank went the other way, or from where.
#
# The detector for that must NOT be collective. By the time anything is stuck
# the ranks have already split, so a probe that needs all of them can only make
# it worse -- it either blocks alongside the others or it becomes one more
# collective for the divergent rank to miss. So this is a plain per-rank timer.
# Every rank arms one, every rank reports on its own, and the diagnosis is the
# comparison between reports: three ranks in `allreduce`, one somewhere else.
#
# The reporting has to survive the main thread being inside MPI, and a Python
# thread does not: measured at np=4 against a 4 s block in `comm.allreduce`,
# a re-arming `threading.Timer` set to 0.5 s fired ZERO times on the blocked
# ranks -- the interpreter lock is held for the duration -- while
# `faulthandler.dump_traceback_later` produced all 7 expected dumps. It is
# written in C for exactly this case and does not need the lock.
#
# So faulthandler is the mechanism, and the price is that it writes to a file
# DESCRIPTOR: the destination must be a real file or stream, never a buffer.
# The Python timer is kept alongside it, because it is the only one that can
# print the checkpoint LABEL, and it does run for the many hangs that are not
# inside MPI -- a spin in Python, a stuck read, a solve that releases the lock.

_watchdog = None
_watchdog_lock = _threading.Lock()


def _stack_dump():
    """Every thread's Python stack, main thread first.

    Formatted here rather than through :mod:`faulthandler` because that writes
    to a file descriptor, so it cannot report into a pipe, a string buffer or
    anything a test can read back. ``sys._current_frames`` needs the
    interpreter lock, which mpi4py releases around blocking calls -- so a rank
    sitting in ``allreduce`` does still report. If some extension ever holds
    the lock through a block, nothing running in Python can report on it.

    ``lookup_lines=False`` is load-bearing, not cosmetic: the reporter runs
    on a daemon thread, and with the environment-armed watchdog it can fire
    while the MAIN thread is still importing. ``format_stack`` reads every
    frame's SOURCE through ``linecache`` — file IO and loader calls against
    modules mid-import, from a second thread — and that interleaving was
    measured to freeze the import outright (a 0.2 s watchdog livelocked
    `import underworld3` 12 times in 15; CI's 1 s watchdog died at -11 in
    the same window, test_0054). File names, line numbers and function
    names carry the hang report; the source text was the deadlock.
    """
    import traceback

    main = _threading.main_thread().ident
    frames = _sys._current_frames()
    names = {t.ident: t.name for t in _threading.enumerate()}

    out = []
    for ident in sorted(frames, key=lambda i: (i != main, i)):
        who = "MainThread (this is the one that is stuck)" if ident == main \
            else names.get(ident, "unknown")
        out.append(f"\n  --- thread {ident}: {who} ---")
        summary = traceback.StackSummary.extract(
            traceback.walk_stack(frames[ident]), lookup_lines=False)
        summary.reverse()                 # walk_stack yields innermost first
        out.extend("  " + line.rstrip()
                   for line in summary.format())
    return "\n".join(out)


class _Watchdog:
    """A per-rank timer that reports where this rank is when it stops moving."""

    def __init__(self, seconds, stream, abort):
        self.seconds = float(seconds)
        self.stream = stream
        self.abort = bool(abort)
        self.timer = None
        # Set by cancel(). A report already running when the watchdog is
        # disarmed must not re-arm behind it: the caller is about to close the
        # stream, and faulthandler holds the DESCRIPTOR, so the next dump
        # lands in whatever inherits it.
        self.cancelled = False
        self.label = "watch() -- no checkpoint reached yet"
        self.since = _time.monotonic()
        self.reports = 0

        try:
            stream.fileno()
        except Exception as not_a_file:
            raise ValueError(
                "the watchdog writes through faulthandler, which needs a real "
                "file descriptor, so `stream` cannot be a StringIO or other "
                "in-memory buffer. Pass sys.stderr, or a file you opened. "
                "This is not a detail to work around: a Python-level fallback "
                "cannot report a rank blocked inside MPI, which is the case "
                "the watchdog exists for."
            ) from not_a_file

        print(
            f"=== UW watchdog armed: rank {rank} of {size}, "
            f"pid {_os.getpid()}, limit {self.seconds:g} s ===",
            file=stream,
            flush=True,
        )

    def arm(self, label=None, resume=False):
        # `resume` is the deliberate re-arm of a watchdog that was cancelled on
        # purpose -- restoring an outer one after a nested `watching` block.
        # Without it the `cancelled` latch, which exists to stop an in-flight
        # report re-arming behind unwatch(), would also silence the restore.
        if resume:
            self.cancelled = False
        if self.cancelled:
            return
        if label is not None:
            self.label = label
        self.since = _time.monotonic()

        # The mechanism. Re-arming resets the countdown, so a job that keeps
        # checking in never reaches it. `repeat` keeps it dumping once stuck:
        # two identical stacks a minute apart say "stuck", one says "slow".
        _faulthandler.dump_traceback_later(
            self.seconds, repeat=True, file=self.stream, exit=self.abort
        )
        self._rearm_timer()

    def _rearm_timer(self):
        # Secondary, and only for hangs that leave the interpreter lock free.
        # It adds the checkpoint label, which faulthandler cannot know about.
        #
        # The REPORTER re-arms through this method ALONE, never through
        # arm(): dump_traceback_later() internally cancels the running C
        # watchdog thread and waits on its lock, and when that thread is
        # mid-dump — walking frames the main thread is churning (an
        # import in progress) — the wait never returns. Measured as the
        # test_0054 deadlock triangle (native `sample`): the C thread
        # pinned in dump_traceback, the reporter cond-waiting inside
        # cancel_dump_traceback_later, the main thread starved in the
        # import machinery — 12 of 15 runs frozen at a 0.2 s watchdog.
        # faulthandler was armed with repeat=True; it needs no re-arm
        # from the reporter. Checkpoints (watch()) still go through
        # arm(), where resetting the countdown is the point and the main
        # thread is in ordinary running state.
        if self.timer is not None:
            self.timer.cancel()
        self.timer = _threading.Timer(self.seconds, self.report)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        self.cancelled = True
        _faulthandler.cancel_dump_traceback_later()
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def report(self):
        """The labelled report, when this thread can get the interpreter lock.

        Silent on a rank blocked inside MPI -- faulthandler covers that one.
        """
        self.reports += 1
        stalled = _time.monotonic() - self.since
        print(
            f"\n=== UW HANG WATCHDOG: rank {rank} of {size}, pid {_os.getpid()}"
            f" ===\n"
            f"    no progress for {stalled:.1f} s (limit {self.seconds:g} s)\n"
            f"    last checkpoint: {self.label}\n"
            f"    Compare this stack with the other ranks': the one in a "
            f"different\n"
            f"    place is the rank that missed the collective.\n"
            f"{_stack_dump()}",
            file=self.stream,
            flush=True,
        )
        if not self.abort and not self.cancelled:
            self._rearm_timer()     # timer only — see _rearm_timer


def watch(seconds=300, stream=None, abort=False):
    """Arm a per-rank watchdog that reports where this rank is when it hangs.

    Call it once, high up -- at the top of a script, or in a test fixture.
    Thereafter :func:`checkpoint` re-arms it, and every function carrying
    :func:`collective_operation` re-arms it automatically, so a normally
    progressing job never fires.

    Nothing here is collective, which is the point: a rank that has missed a
    collective is exactly the rank that cannot take part in a probe. Each rank
    times itself and reports alone.

    Parameters
    ----------
    seconds : float
        How long without a checkpoint counts as stuck. Set it well above the
        slowest legitimate phase -- a coarse solve or a mesh generation can
        take minutes and is not a hang.
    stream : file, optional
        Where the report goes. Defaults to ``sys.stderr``. It must be a real
        file or stream --- an in-memory buffer is refused, because the dump
        goes through :mod:`faulthandler` and that writes to a file descriptor.
        Under ``mpirun`` the ranks interleave, so for anything but a small job
        give each rank its own file, or use ``mpirun --output-filename``.
    abort : bool
        Exit the process straight after dumping. Useful in CI, where the
        alternative is the job sitting until the wall clock kills it with no
        output at all. Off by default: it kills the job.

    Returns
    -------
    float
        The timeout in force.

    Warnings
    --------
    Call :func:`unwatch` before closing *stream*. faulthandler holds the file
    DESCRIPTOR, not the Python object, so an armed watchdog over a closed file
    writes into whatever that descriptor is reused for next. It is the default
    ``sys.stderr`` that makes this harmless most of the time; a log file you
    open yourself needs the disarm.

    Example
    -------
    >>> uw.mpi.watch(seconds=120)                        # doctest: +SKIP
    >>> for step in range(100):                          # doctest: +SKIP
    ...     uw.mpi.checkpoint(f"step {step}")
    ...     stokes.solve()
    """
    global _watchdog
    if stream is None:
        stream = _sys.stderr
    with _watchdog_lock:
        if _watchdog is not None:
            _watchdog.cancel()
        _watchdog = _Watchdog(seconds, stream, abort)
        _watchdog.arm()
    return float(seconds)


def unwatch():
    """Disarm the watchdog. Safe to call when it was never armed."""
    global _watchdog
    with _watchdog_lock:
        if _watchdog is not None:
            _watchdog.cancel()
            _watchdog = None


def checkpoint(label=None):
    """Tell the watchdog this rank is still moving, and where it is.

    A local timer reset -- no communication, and a single test when the
    watchdog is not armed, so it is safe to leave in production code.

    The *label* is what makes the eventual report readable: it is printed
    alongside the stack, so "rank 3 last saw `step 12 / before adapt`" while
    the others saw `step 13` localises the divergence to one iteration.
    """
    if _watchdog is not None:
        _watchdog.arm(label)


@_contextmanager
def watching(seconds=300, stream=None, abort=False):
    """:func:`watch` for the duration of a block, then restore what was there.

    >>> with uw.mpi.watching(seconds=60):        # doctest: +SKIP
    ...     mesh.adapt(metric, max_levels=2)
    """
    global _watchdog
    with _watchdog_lock:
        previous = _watchdog
        if previous is not None:
            previous.cancel()
        _watchdog = None
    watch(seconds=seconds, stream=stream, abort=abort)
    try:
        yield
    finally:
        unwatch()
        with _watchdog_lock:
            _watchdog = previous
        if previous is not None:
            previous.arm(resume=True)


@collective_operation
def ranks_agree(label, verbose=False):
    """Check that every rank reached this point by the same route. COLLECTIVE.

    The watchdog is a post-mortem: it tells you where a job stopped. This is
    the positive audit -- put it after a phase you suspect and it either passes
    or names the ranks that took a different path.

    It compares *label* across ranks, so it catches divergence the barrier
    alone cannot: ranks that all arrive, but from different branches. Give it a
    label that varies with the path taken, not a constant.

    Being collective, it can itself hang if a rank never arrives -- that case
    is the watchdog's, and the two are meant to be used together.

    Parameters
    ----------
    label : str
        What this rank believes it just did. Compared verbatim.
    verbose : bool
        Print the agreed label from rank 0 on success.

    Raises
    ------
    CollectiveOperationError
        If the labels differ, with the rank-to-label table.

    Example
    -------
    >>> uw.mpi.ranks_agree(f"after adapt, {mesh.dm.getNumCells()} cells")
    ... # doctest: +SKIP
    """
    label = str(label)
    checkpoint(f"ranks_agree({label!r})")
    seen = comm.allgather(label)

    if len(set(seen)) > 1:
        groups = {}
        for r, entry in enumerate(seen):
            groups.setdefault(entry, []).append(r)
        table = "\n".join(
            f"    ranks {ranks_here}: {entry!r}"
            for entry, ranks_here in sorted(groups.items(),
                                            key=lambda kv: kv[1][0])
        )
        raise CollectiveOperationError(
            f"\n{'=' * 70}\n"
            f"RANKS DISAGREE AT A CHECKPOINT\n"
            f"{'=' * 70}\n\n"
            f"{len(groups)} different labels across {size} ranks:\n\n"
            f"{table}\n\n"
            f"The ranks took different paths to get here. Whatever collective "
            f"comes\nnext will be reached by some of them and not the others.\n"
            f"Look for a branch on rank-local data -- an array's size, an "
            f"emptiness\ntest, a `None` -- between the last agreed point and "
            f"this one, and\nreduce the predicate before branching on it.\n"
            f"{'=' * 70}\n"
        )

    if verbose and rank == 0:
        print(f"ranks_agree: all {size} ranks at {label!r}", flush=True)

    return label


def pprint(*args, proc=0, prefix=None, clean_display=True, flush=False, **kwargs):
    """
    Parallel-safe print that works as a drop-in replacement for print().

    This function ensures all ranks execute any collective operations in the arguments,
    but only selected ranks actually print output. This prevents deadlocks from
    collective operations inside rank conditionals.

    Args:
        *args: Arguments to print (same as standard print())
        proc: Which ranks should print. Can be:
            - int: Single rank (e.g., 0) [default: 0]
            - slice: Range of ranks (e.g., slice(0, 4))
            - list/tuple: Specific ranks (e.g., [0, 3, 7])
            - str: Named patterns ('all', 'first', 'last', 'even', 'odd', '10%')
            - callable: Function taking rank and returning bool
            - numpy array: Boolean mask or integer indices
        prefix: If True, prefix output with rank number. If None (default),
            automatically enables in parallel (size > 1) and disables in serial.
        clean_display: If True, filter out SymPy uniqueness strings for cleaner display (default: True)
        flush: If True, forcibly flush the stream (default: False, same as print())
        **kwargs: Additional keyword arguments passed to print() (sep, end, file)

    Example:
        >>> uw.pprint(f"Global max: {var.stats()['max']}")  # Only rank 0 prints
        Global max: 42.5

        >>> # In parallel, automatic prefix
        >>> uw.pprint(f"Local max: {var.data.max()}", proc=slice(0, 4))
        [0] Local max: 12.3
        [1] Local max: 15.7
        [2] Local max: 9.8
        [3] Local max: 11.2

        >>> uw.pprint(f"Expression: {expr}")  # Automatically cleans symbols
        Expression: T(x,y)
    """
    # Auto-detect prefix: True in parallel, False in serial
    if prefix is None:
        prefix = size > 1

    if _should_rank_execute(rank, proc, size):
        if clean_display:
            # Clean up display strings by filtering out SymPy uniqueness patterns
            import re

            cleaned_args = []
            for arg in args:
                if hasattr(arg, "__str__"):
                    # Filter out \hspace{XXpt} patterns used for SymPy symbol uniqueness
                    cleaned_str = re.sub(r"\\hspace\{\s*[\d\.]+pt\s*\}\s*", "", str(arg))
                    # Clean up nested braces like { {T} } → T (apply multiple times for nested cases)
                    for _ in range(3):  # Apply up to 3 times for deep nesting
                        cleaned_str = re.sub(r"\{\s*([^{}]*)\s*\}", r"\1", cleaned_str)
                    # Clean up latex commands like {\mathbf{v}} → v and \mathbfv → v
                    cleaned_str = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", cleaned_str)
                    cleaned_str = re.sub(r"\\mathbf([a-zA-Z])", r"\1", cleaned_str)
                    # Clean up extra spaces and underscores
                    cleaned_str = re.sub(
                        r"_\s*(\d+)", r"_\1", cleaned_str
                    )  # Fix spacing around subscripts
                    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
                    cleaned_args.append(cleaned_str)
                else:
                    cleaned_args.append(arg)
            args = tuple(cleaned_args)

        if prefix:
            print(f"[{rank}]", *args, flush=flush, **kwargs)
        else:
            print(*args, flush=flush, **kwargs)
    elif flush:
        # Even if this rank doesn't print, handle flush if requested
        _sys.stdout.flush()


def pprint_old(ranks, *args, prefix=True, clean_display=True, **kwargs):
    """
    Legacy pprint interface (deprecated). Use pprint() with proc= parameter instead.

    This function maintains backward compatibility for existing code.
    """
    import warnings

    warnings.warn(
        "pprint_old() is deprecated. Use pprint() with proc= parameter instead:\n"
        "  Old: uw.pprint('message')\n"
        "  New: uw.pprint('message', proc=0)",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert to new interface
    return pprint(*args, proc=ranks, prefix=prefix, clean_display=clean_display, **kwargs)


class call_pattern:
    """
    This context manager calls the code within its block using the
    specified calling pattern.

    Parameters
    ----------
    pattern: str
        'collective', each process calls the block of code simultaneously.
        'sequential', processes call block of code in order of rank.

    Example
    -------
    This example is redundant as it will only run with a single process.
    However, where run in parallel, you should expect the outputs to be
    ordered according to process rank. Note also that for deterministic
    printing in parallel, and you may need to run Python unbuffered
    (`mpirun -np 4 python -u yourscript.py`, for example).

    >>> import underworld as uw
    >>> with uw.mpi.call_pattern(pattern="sequential"):
    ...     print("My rank is {}".format(uw.mpi.rank))
    My rank is 0

    """

    def __init__(self, pattern="collective", returnobj=None):
        if not isinstance(pattern, str):
            raise TypeError("`pattern` parameter must be of type `str`")
        pattern = pattern.lower()
        if pattern not in ("collective", "sequential"):
            raise ValueError("`pattern` must take values `collective` or `sequential`.")
        self.pattern = pattern
        self.returnobj = returnobj

    def __enter__(self):
        if self.pattern == "sequential":
            if rank != 0:
                comm.recv(source=rank - 1, tag=333)
        return self.returnobj

    def __exit__(self, *args):
        if self.pattern == "sequential":
            dest = rank + 1
            if dest < comm.size:
                comm.send(None, dest=rank + 1, tag=333)


def _watch_from_environment():
    """Arm the watchdog from ``UW_HANG_WATCHDOG``.

    Arming from the environment rather than the user's script is the point: a
    rank that dies, or diverges, before reaching a ``watch()`` call reports
    nothing, and arming at import time covers mesh construction and
    everything after. It is invoked from the END of ``import underworld3``
    (the bottom of ``underworld3/__init__``), NOT from this module's import —
    deliberately: ``faulthandler.dump_traceback_later``'s repeating C dump
    walks live frames without synchronisation, and against an interpreter
    that is still importing (frames churning, bytecode compiling) that walk
    was measured to loop forever or die at SIGSEGV (test_0054 on CI; locally
    a 0.2 s watchdog froze ``import underworld3`` on the first piped run).
    The price is that a hang INSIDE the import graph itself goes unreported;
    everything the tool exists for — meshing, solves, collectives — runs
    after import and is covered.

    ``UW_HANG_WATCHDOG``
        Seconds of silence that count as stuck. Unset or 0 disables.
    ``UW_HANG_WATCHDOG_DIR``
        Directory for the per-rank dumps. Default ``uw-hang-dumps``.
    ``UW_HANG_WATCHDOG_ABORT``
        Anything but empty or ``0``: exit after dumping, rather than dumping
        repeatedly. For batch queues, where the alternative is the wall clock.

    Usage::

        UW_HANG_WATCHDOG=120 mpirun -n 4 python myrun.py
        python -m underworld3.utilities.hang_report uw-hang-dumps
    """
    setting = _os.environ.get("UW_HANG_WATCHDOG", "").strip()
    if not setting:
        return None
    try:
        seconds = float(setting)
    except ValueError:
        print(
            f"UW_HANG_WATCHDOG={setting!r} is not a number of seconds; "
            "the hang watchdog is NOT armed.",
            file=_sys.stderr, flush=True,
        )
        return None
    if seconds <= 0:
        return None

    directory = _os.environ.get("UW_HANG_WATCHDOG_DIR", "uw-hang-dumps")
    # Idempotent on every rank, so no collective is needed to make it -- and a
    # collective here would be the very thing the watchdog exists to diagnose.
    _os.makedirs(directory, exist_ok=True)
    path = _os.path.join(directory, f"rank{rank:04d}.log")

    # Left open for the life of the process on purpose: faulthandler holds the
    # descriptor, so closing it would point the dump at whatever reuses it.
    stream = open(path, "w", buffering=1)
    # Disarm on the way out. Interpreter shutdown waits on the faulthandler
    # thread, and a watchdog still counting down while the process tears itself
    # down keeps the job alive long after the model has finished.
    _atexit.register(unwatch)
    watch(
        seconds=seconds,
        stream=stream,
        abort=_os.environ.get("UW_HANG_WATCHDOG_ABORT", "").strip() not in ("", "0"),
    )
    return path


#: Path this rank will dump to, or None when the environment did not ask.
#: Set by _arm_environment_watchdog(), called at the END of
#: ``import underworld3`` — see _watch_from_environment for why not here.
environment_dump_path = None


def _arm_environment_watchdog():
    global environment_dump_path
    environment_dump_path = _watch_from_environment()
