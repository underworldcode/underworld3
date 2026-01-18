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
import os as _os
import secrets as _secrets
import sys as _sys
import io as _io
from contextlib import contextmanager as _contextmanager


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

        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper._is_collective = True
    return wrapper


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
