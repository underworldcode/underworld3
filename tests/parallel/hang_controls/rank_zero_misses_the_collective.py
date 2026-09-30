"""A planted collective hang, for validating the MPI supervisor.

Every rank but 0 enters a barrier. Rank 0 does not, and instead sits in a
function with a name no other rank can be inside. The job therefore hangs
forever, and the correct diagnosis is unambiguous: rank 0 is the one that
missed the collective.

This is the negative control for ``scripts/mpi_supervisor.py``. A supervisor
that cannot bound this run, and cannot name rank 0 from the stacks it collects,
has no business gating a test suite.

It lives outside the ``test_*.py`` pattern deliberately: pytest must never
collect it, because running it under any harness that does not kill it is the
exact failure this whole exercise exists to prevent. It is launched by name,
only by ``tests/test_0056_mpi_hang_supervisor.py``.
"""

import sys
import time

from mpi4py import MPI


def _rank_zero_misses_the_collective():
    """Rank 0 spends forever here while the others wait in the barrier.

    Named so that it is visible in a stack dump and cannot be confused with
    the MPI frames the other ranks are in.
    """
    while True:
        time.sleep(3600)


def main():
    comm = MPI.COMM_WORLD
    # Unbuffered, and flushed, so the supervisor sees output stop at a known
    # point: the silence after this line is the hang it must detect.
    print(f"rank {comm.rank} of {comm.size} ready", flush=True)

    if comm.rank == 0:
        _rank_zero_misses_the_collective()
    else:
        comm.barrier()

    print("this line is unreachable while rank 0 is missing", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
