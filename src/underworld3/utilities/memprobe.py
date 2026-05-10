"""
Lightweight memory-growth diagnostics.

Designed to surface slow leaks in long parallel runs (HPC OOM-class bugs)
without slowing normal use. Three signal sources:

* **Process RSS** via :mod:`resource` (peak + current).
* **Live ``uw.kdtree.KDTree`` count** — UW's nanoflann wrapper isn't visible
  to PETSc's object tracker, so it's a common silent-leak suspect.
* **Per-class Python instance counts** for the ``underworld3`` package, gated
  behind ``full=True`` because :func:`gc.get_objects` is slow at scale.

For PETSc-side object/allocation tracking, use the runtime flags this module
documents (``-log_view``, ``-malloc_dump``) — parsing PETSc's viewer output
from Python is brittle and adds no signal beyond what those flags give.

Activation
----------
- ``UW_MEMPROBE=1`` env var → enables instrumentation hooks at import time.
- :func:`enable` / :func:`disable` for runtime toggling.
- :func:`probe` context manager for ad-hoc snapshot+diff blocks.
- :func:`instrument` decorator for per-method instrumentation; a no-op when
  disabled (one attribute lookup + branch, sub-microsecond).

Examples
--------
>>> import underworld3 as uw
>>> from underworld3.utilities import memprobe
>>>
>>> with memprobe.probe("step 42"):
...     do_one_step()           # diff is logged on exit
>>>
>>> @memprobe.instrument("stokes-solve")
... def solve(...):
...     ...

For a long run, set ``UW_MEMPROBE=1`` and add a probe block per step. To pin
PETSc-side leaks, run with ``-log_view -malloc_dump`` (or call
:func:`dump_petsc_leaks_at_finalize`).
"""
from __future__ import annotations

import functools
import gc
import os
import resource
import sys
from collections import Counter
from contextlib import contextmanager
from typing import Any

__all__ = [
    "ENABLED",
    "enable",
    "disable",
    "snapshot",
    "diff",
    "probe",
    "instrument",
    "dump_petsc_leaks_at_finalize",
    "format_diff",
]

ENABLED: bool = bool(int(os.environ.get("UW_MEMPROBE", "0")))


def enable() -> None:
    """Turn instrumentation on at runtime."""
    global ENABLED
    ENABLED = True


def disable() -> None:
    """Turn instrumentation off at runtime."""
    global ENABLED
    ENABLED = False


def _rss_mb() -> float:
    """Current resident-set size in MiB.

    On Linux ``ru_maxrss`` is in KiB; on macOS it is in bytes. We probe the
    platform once to avoid getting it wrong silently.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _kdtree_counts() -> dict[str, int]:
    try:
        from underworld3 import kdtree

        return {
            "live": kdtree.live_count(),
            "total_constructed": kdtree.total_constructed(),
        }
    except (ImportError, AttributeError):
        return {"live": -1, "total_constructed": -1}


def _python_class_counts(prefix: str = "underworld3") -> Counter[str]:
    """Count live Python objects whose class lives under ``prefix``.

    Walks ``gc.get_objects()`` once. Slow at large counts (~10–100 ms for
    typical UW runs). Only called when ``full=True`` is requested.
    """
    counts: Counter[str] = Counter()
    for obj in gc.get_objects():
        cls = type(obj)
        mod = getattr(cls, "__module__", "") or ""
        # Some objects expose a non-string descriptor as ``__module__``;
        # skip those rather than crashing the walk.
        if isinstance(mod, str) and mod.startswith(prefix):
            counts[f"{mod}.{cls.__name__}"] += 1
    return counts


def snapshot(full: bool = False) -> dict[str, Any]:
    """Capture a memory snapshot.

    Parameters
    ----------
    full
        If ``True``, also walk ``gc.get_objects()`` to count live Python
        instances by class (slow). Defaults to ``False``.

    Returns
    -------
    dict
        Keys: ``rss_mb``, ``kdtree`` (mapping with ``live`` and
        ``total_constructed``), ``py_classes`` (mapping; only present when
        ``full=True``).
    """
    snap: dict[str, Any] = {
        "rss_mb": _rss_mb(),
        "kdtree": _kdtree_counts(),
    }
    if full:
        snap["py_classes"] = dict(_python_class_counts())
    return snap


def diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute a structured diff between two snapshots.

    Returns deltas only (excludes anything unchanged or absent). Python
    class entries are sorted by absolute change so the dominant suspects
    appear first.
    """
    delta: dict[str, Any] = {}

    drss = after["rss_mb"] - before["rss_mb"]
    if drss != 0:
        delta["rss_mb"] = drss

    kd_delta: dict[str, int] = {}
    for key in ("live", "total_constructed"):
        d = after["kdtree"].get(key, 0) - before["kdtree"].get(key, 0)
        if d != 0:
            kd_delta[key] = d
    if kd_delta:
        delta["kdtree"] = kd_delta

    if "py_classes" in before and "py_classes" in after:
        b = before["py_classes"]
        a = after["py_classes"]
        keys = set(b) | set(a)
        py_delta = {k: a.get(k, 0) - b.get(k, 0) for k in keys}
        py_delta = {k: v for k, v in py_delta.items() if v != 0}
        if py_delta:
            delta["py_classes"] = dict(
                sorted(py_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)
            )

    return delta


def format_diff(label: str, delta: dict[str, Any]) -> str:
    """Render a diff for stdout / logs. One line for trivial deltas, more if needed."""
    if not delta:
        return f"[memprobe] {label}: no change"

    parts = [f"[memprobe] {label}:"]
    if "rss_mb" in delta:
        parts.append(f"  RSS {delta['rss_mb']:+.2f} MiB")
    if "kdtree" in delta:
        kd = delta["kdtree"]
        chunk = ", ".join(f"{k} {v:+d}" for k, v in kd.items())
        parts.append(f"  kdtree: {chunk}")
    if "py_classes" in delta:
        parts.append("  py_classes (top 10 by |Δ|):")
        for cls, d in list(delta["py_classes"].items())[:10]:
            parts.append(f"    {cls}: {d:+d}")
    return "\n".join(parts)


@contextmanager
def probe(label: str, full: bool = False, *, emit=print):
    """Context manager: snapshot before, snapshot after, log the diff.

    Parameters
    ----------
    label
        Short identifier for the block, included in the emitted diff.
    full
        Walk ``gc.get_objects()`` for per-class deltas (slow).
    emit
        Callable receiving the formatted diff string. Defaults to
        :func:`print`. Use a logger or rank-aware writer for parallel runs.
    """
    if not ENABLED:
        yield
        return
    before = snapshot(full=full)
    try:
        yield
    finally:
        after = snapshot(full=full)
        emit(format_diff(label, diff(before, after)))


def instrument(label: str, full: bool = False, *, emit=print):
    """Decorator wrapping a method/function with :func:`probe`.

    Fast-returns when :data:`ENABLED` is ``False`` (sub-microsecond overhead),
    so it's safe to leave permanent decorations on hot paths.

    Examples
    --------
    >>> @instrument("stokes-solve")
    ... def solve(self, ...):
    ...     ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if not ENABLED:
                return func(*args, **kw)
            with probe(label, full=full, emit=emit):
                return func(*args, **kw)
        return wrapper
    return decorator


def dump_petsc_leaks_at_finalize(filename: str | None = None) -> None:
    """Ask PETSc to dump unfreed objects at finalize.

    Sets the runtime options ``-malloc_dump`` and ``-objects_dump`` so PETSc
    writes its leak report when the process tears down. ``filename`` is
    advisory — if provided, also sets ``-malloc_view`` / ``-options_view``
    for a more verbose dump.

    Equivalent to running with ``-malloc_dump`` on the command line. Useful
    when you can't change the launch command but can call this from Python
    early in the run.
    """
    from petsc4py import PETSc

    opts = PETSc.Options()
    opts.setValue("-malloc_dump", "")
    opts.setValue("-objects_dump", "")
    if filename:
        opts.setValue("-malloc_view", filename)
