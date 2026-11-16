"""Collective agreement on failure for a parallel build (#874).

A parallel builder interleaves rank-local work with collectives. A rank-local
exception on one rank used to leave the others waiting in the next collective,
and the cure was a guard around every rank-local step: hundreds of sites, and
every new line a new one. This module turns that around. Inside
:class:`agreed_build`, an exception on one rank is caught at the context's exit,
where that rank makes ONE agreement call; every other rank meets it at its next
agreement point, and every rank raises the same
:class:`~underworld3.utilities.custom_mg.CollectiveBuildError`, carrying the
failing rank's message.

The agreement point is :func:`fault_contact._collective_raise`, which goes
through :func:`underworld3.mpi.first_verdict`: a one-integer flag allreduce on
the verdict communicator (so it cannot pair with the build's own collectives),
and the messages gathered only on a failure. Every rank raises the class and
the message of the first failing rank. Builds are WORLD-collective: the verdict
communicator duplicates the world, as every solver's communicator in the
library is.

:func:`agree` makes one: it is placed before every collective of the build that
cannot itself agree (a PETSc collective, a mesh method that is collective
inside). The wrappers :func:`allreduce`, :func:`allgather` and :func:`bcast`
agree and then communicate, and are preferred over a separate ``agree()``
line before an mpi4py collective. Outside an
:class:`agreed_build`, :func:`agree` does nothing, so a helper can carry it
whoever calls it.

The rule a converted builder keeps: inside the context, between any statement
that can raise and the next collective, there is an agreement point. What it
covers: every rank-local exception, now and in code added later, with no guard
of its own. What it does not cover: an exception inside a PETSc collective
itself, and a collective added later without an agreement point in front.

The context also agrees once on a normal exit, so a rank that finished cannot
leave a failed one waiting. An error that is already agreed (raised by
``_collective_raise``) passes through without a second agreement, so every rank
makes the same number of agreement calls.
"""

_active = []


def _raise_agreed(problem):
    from underworld3.utilities.custom_mg import CollectiveBuildError
    from underworld3.utilities.fault_contact import _collective_raise

    _collective_raise(problem, error=CollectiveBuildError)


class agreed_build:
    """Context for a collective build: a rank-local exception inside it is
    raised on every rank, as the same ``CollectiveBuildError``. ``what`` names
    the build in the message. COLLECTIVE: every rank enters and leaves it."""

    def __init__(self, what):
        self.what = what

    def __enter__(self):
        _active.append(self)
        return self

    def __exit__(self, kind, err, tb):
        _active.remove(self)
        if kind is None:
            _raise_agreed(None)        # the final agreement: a failed rank meets it
            return False
        if getattr(err, "agreed", False) or not issubclass(kind, Exception):
            return False               # already agreed on every rank, or an interrupt
        try:
            _raise_agreed(f"{self.what}: {kind.__name__}: {err}")
        except Exception as agreed:
            raise agreed from err
        return False


def agree():
    """An agreement point: raise on every rank if any rank of the active build
    has failed. Place it before a collective that cannot agree by itself. A
    no-op outside an :class:`agreed_build`. COLLECTIVE inside one."""
    if _active:
        _raise_agreed(None)


def allreduce(comm, value, **kwargs):
    """``comm.allreduce`` with an agreement point in front (mpi4py communicator)."""
    agree()
    return comm.allreduce(value, **kwargs)


def allgather(comm, value):
    """``comm.allgather`` with an agreement point in front."""
    agree()
    return comm.allgather(value)


def bcast(comm, value, root=0):
    """``comm.bcast`` with an agreement point in front."""
    agree()
    return comm.bcast(value, root=root)
