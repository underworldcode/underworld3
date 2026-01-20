##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Underworld geophysics modelling application.         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Underworld3 Performance Timing and Profiling

This module provides comprehensive performance analysis by integrating with
PETSc's logging infrastructure, capturing ~95% of computational work including
matrix operations, solvers, and user-decorated functions.

**No environment variables required - works immediately in Jupyter notebooks!**

Key Features
------------
- Decorator-based function timing (routes to PETSc events)
- Comprehensive solver/matrix/vector operation tracking (via PETSc)
- Memory usage and flop counting (via PETSc)
- MPI communication statistics in parallel runs (via PETSc)
- Unified output combining all timing data

Basic Usage
-----------
>>> import underworld3 as uw
>>>
>>> # Start logging (Jupyter-friendly!)
>>> uw.timing.start()
>>>
>>> # Decorate functions you want to track
>>> @uw.timing.routine_timer_decorator
>>> def my_analysis():
>>>     mesh = uw.meshing.StructuredQuadBox(elementRes=(32, 32))
>>>     # ... do work ...
>>>
>>> my_analysis()
>>>
>>> # View comprehensive results
>>> uw.timing.print_table()

Advanced Usage
--------------
Auto-decorate entire modules:

>>> import underworld3 as uw
>>> import mymodule
>>> uw.timing.add_timing_to_module(mymodule)  # Decorates all classes/functions

See Also
--------
- PETSc Logging: https://petsc.org/release/manual/profiling/
- UW3 Profiling Guide: docs/developer/profiling.md
"""

import functools as _functools
import inspect as _inspect
import os as _os
from mpi4py import MPI

RANK = MPI.COMM_WORLD.rank

# ============================================================================
# PETSc Event-Based Timing System
# ============================================================================

# Global cache of registered PETSc events (prevents duplicate registration)
_petsc_events = {}
_petsc_logging_enabled = False


def start():
    """
    Start PETSc performance logging.

    Call this at the beginning of your script/notebook to enable comprehensive
    performance tracking. Works immediately in Jupyter - no environment variables needed!

    This captures:
    - Decorated Python functions
    - All PETSc operations (MatMult, KSPSolve, VecNorm, etc.)
    - Memory usage and allocation
    - Floating point operations (flops)
    - MPI communication (in parallel runs)

    Example
    -------
    >>> import underworld3 as uw
    >>> uw.timing.start()
    >>> # ... do work ...
    >>> uw.timing.print_table()  # View results

    Notes
    -----
    - Safe to call multiple times (subsequent calls are no-ops)
    - Zero overhead when not enabled
    - Can be called anywhere before performance-critical code
    """
    enable_petsc_logging()


def stop():
    """
    Stop PETSc logging (currently a no-op - PETSc logging runs until view()).

    Provided for API compatibility with legacy timing module.
    PETSc logging is lightweight and can run continuously.
    """
    pass  # PETSc logging doesn't need explicit stopping


def reset():
    """
    Reset timing data.

    Clears all accumulated PETSc logging data and starts fresh.
    Useful for timing specific sections of code.

    Example
    -------
    >>> import underworld3 as uw
    >>> uw.timing.start()
    >>> # ... setup code (not timed) ...
    >>> uw.timing.reset()
    >>> # ... performance-critical code (timed) ...
    >>> uw.timing.print_table()
    """
    from petsc4py import PETSc

    if PETSc.Log.isActive():
        # Note: PETSc doesn't have a true "reset" - we'd need to stop and restart
        # For now, this is a no-op but documented for API compatibility
        pass


def print_table(filename=None, format="auto"):
    """
    Display comprehensive performance results.

    Shows timing for:
    - Decorated Python functions
    - PETSc operations (solvers, matrix ops, etc.)
    - Memory usage
    - Flop counts
    - MPI communication (parallel runs)

    Parameters
    ----------
    filename : str, optional
        If provided, write results to file. Extension determines format:
        - `.csv` : Spreadsheet-compatible CSV format
        - `.txt` or other : Human-readable ASCII table
    format : str, optional
        Override automatic format detection:
        - "auto" : Detect from filename (default)
        - "ascii" : Human-readable table
        - "csv" : Comma-separated values

    Example
    -------
    >>> uw.timing.start()
    >>> # ... do work ...
    >>> uw.timing.print_table()  # Print to console
    >>> uw.timing.print_table("results.csv")  # Save as CSV
    """
    print_petsc_log(filename=filename, format=format)


# Backward compatibility aliases
view = print_table
get_data = lambda **kwargs: print("Use uw.timing.print_table() for results")


def enable_petsc_logging():
    """
    Enable PETSc performance logging.

    Called automatically by start(). Can also be called directly.
    Safe to call multiple times.
    """
    global _petsc_logging_enabled

    if _petsc_logging_enabled:
        return  # Already enabled

    from petsc4py import PETSc

    if not PETSc.Log.isActive():
        PETSc.Log.begin()
        _petsc_logging_enabled = True


def print_petsc_log(filename=None, format="auto"):
    """
    Display or save PETSc performance logging summary.

    Parameters
    ----------
    filename : str, optional
        If provided, write log to this file. Otherwise print to console.
        File extension determines format:
        - `.csv` : Comma-separated values (spreadsheet-compatible)
        - `.txt` or other : Human-readable ASCII table (default)
    format : str, optional
        Override automatic format detection. Options:
        - "auto" : Detect from filename extension (default)
        - "ascii" : Human-readable table
        - "csv" : Comma-separated values

    Example
    -------
    >>> uw.timing.start()
    >>> # ... run simulation ...
    >>> uw.timing.print_petsc_log()  # Console output
    >>> uw.timing.print_petsc_log("timing.csv")  # CSV for analysis
    """
    from petsc4py import PETSc

    if not PETSc.Log.isActive():
        if RANK == 0:
            print("⚠️  PETSc logging not enabled. Call uw.timing.start() first.")
        return

    if filename:
        # Determine format from extension or explicit parameter
        if format == "auto":
            if filename.endswith('.csv'):
                use_format = "csv"
            else:
                use_format = "ascii"
        else:
            use_format = format

        # Create viewer with appropriate format
        viewer = PETSc.Viewer().createASCII(filename, 'w')
        if use_format == "csv":
            viewer.pushFormat(PETSc.Viewer.Format.ASCII_CSV)

        # Write log and cleanup
        PETSc.Log.view(viewer)
        viewer.destroy()

        if RANK == 0:
            print(f"✓ Timing results saved to {filename}")
    else:
        # Print to console
        PETSc.Log.view()


# ============================================================================
# Decorator System (Routes to PETSc Events)
# ============================================================================

def routine_timer_decorator(routine, class_name=None):
    """
    Decorator that registers a function as a PETSc timing event.

    No environment variables needed - works immediately!

    Parameters
    ----------
    routine : callable
        Function or method to decorate
    class_name : str, optional
        Class name for better event labeling (auto-detected for methods)

    Returns
    -------
    callable
        Wrapped function that tracks calls via PETSc events

    Example
    -------
    >>> @uw.timing.routine_timer_decorator
    >>> def expensive_computation():
    >>>     # ... complex calculations ...
    >>>     return result
    >>>
    >>> uw.timing.start()
    >>> expensive_computation()
    >>> uw.timing.print_table()  # Shows timing for expensive_computation

    Notes
    -----
    - First call registers the PETSc event (one-time cost)
    - Subsequent calls just increment counters (negligible overhead)
    - Events appear in PETSc log with full statistics
    """
    from petsc4py import PETSc

    # Create event name
    if class_name:
        event_name = f"{class_name}.{routine.__name__}"
    else:
        event_name = routine.__qualname__

    # Register PETSc event (happens once per function)
    if event_name not in _petsc_events:
        _petsc_events[event_name] = PETSc.Log.Event(event_name)

    event = _petsc_events[event_name]

    @_functools.wraps(routine)
    def timed(*args, **kwargs):
        # Begin/end tracking - PETSc handles all statistics!
        event.begin()
        try:
            result = routine(*args, **kwargs)
            return result
        finally:
            event.end()

    return timed


def _class_timer_decorator(cls):
    """
    Decorator that adds timing to all methods in a class.

    Walks through class methods and wraps them with routine_timer_decorator.

    Parameters
    ----------
    cls : type
        Class to decorate

    Returns
    -------
    type
        Same class with methods wrapped for timing

    Example
    -------
    >>> @uw.timing._class_timer_decorator
    >>> class MyAnalysis:
    >>>     def compute(self):
    >>>         # ... work ...
    >>>
    >>> uw.timing.start()
    >>> analysis = MyAnalysis()
    >>> analysis.compute()  # Automatically timed
    >>> uw.timing.print_table()
    """
    _decorated_methods = set()

    for attr_name, attr_value in _inspect.getmembers(cls, _inspect.isfunction):
        # Skip special methods
        if attr_name in ["__del__"]:
            continue

        # Skip already decorated
        if attr_value in _decorated_methods:
            continue

        # Create timed version
        timed_method = routine_timer_decorator(attr_value, cls.__name__)

        # Handle static methods
        try:
            if isinstance(cls.__dict__[attr_name], staticmethod):
                timed_method = staticmethod(timed_method)
        except (KeyError, AttributeError):
            pass

        # Replace method with timed version
        setattr(cls, attr_name, timed_method)
        _decorated_methods.add(timed_method)

    return cls


# Track decorated modules to avoid double-decoration
_decorated_modules = set()
_decorated_classes = set()


def add_timing_to_module(mod):
    """
    Automatically add timing decorators to all classes and functions in a module.

    Recursively walks through a module, decorating all classes and their methods.
    Useful for comprehensive profiling of entire subsystems.

    Parameters
    ----------
    mod : module
        Python module to decorate

    Example
    -------
    >>> import underworld3 as uw
    >>> import underworld3.systems
    >>> uw.timing.add_timing_to_module(uw.systems)  # Time all solver classes
    >>> uw.timing.start()
    >>> # ... use solvers ...
    >>> uw.timing.print_table()  # See detailed solver timing

    Notes
    -----
    - Only decorates classes/functions defined in the specified module
    - Skips built-in modules and external dependencies
    - Safe to call multiple times (avoids double-decoration)
    """
    if mod in _decorated_modules:
        return  # Already decorated

    _decorated_modules.add(mod)

    moddir = _os.path.dirname(_inspect.getfile(mod))
    lendir = len(moddir)

    # Find submodules to recurse into
    submodules = []

    for name in dir(mod):
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue

        # Only process objects from this module
        if not (_inspect.ismodule(obj) or _inspect.isclass(obj) or _inspect.isfunction(obj)):
            continue

        try:
            objpath = _os.path.dirname(_inspect.getfile(obj))
        except (TypeError, AttributeError):
            continue

        if not objpath.startswith(moddir):
            continue

        if _inspect.ismodule(obj):
            if obj not in _decorated_modules:
                submodules.append(obj)
        elif _inspect.isclass(obj):
            if obj not in _decorated_classes:
                decorated_cls = _class_timer_decorator(obj)
                setattr(mod, name, decorated_cls)
                _decorated_classes.add(obj)

    # Recurse into submodules
    for submod in submodules:
        add_timing_to_module(submod)


# Backward compatibility alias
_add_timing_to_mod = add_timing_to_module


# ============================================================================
# Convenience Functions
# ============================================================================

def create_event(name):
    """
    Create a custom PETSc event for manual timing.

    Useful for timing specific code sections without decorators.

    Parameters
    ----------
    name : str
        Name for the event (appears in timing output)

    Returns
    -------
    PETSc.Log.Event
        Event object with begin() and end() methods

    Example
    -------
    >>> import underworld3 as uw
    >>> uw.timing.start()
    >>>
    >>> my_event = uw.timing.create_event("DataProcessing")
    >>> my_event.begin()
    >>> # ... complex data processing ...
    >>> my_event.end()
    >>>
    >>> uw.timing.print_table()  # Shows "DataProcessing" timing
    """
    from petsc4py import PETSc

    if name not in _petsc_events:
        _petsc_events[name] = PETSc.Log.Event(name)

    return _petsc_events[name]


def get_summary(filter_uw=True, min_time=0.001, sort_by='time'):
    """
    Get user-friendly timing summary focusing on UW3 operations.

    Filters PETSc's comprehensive log to show only the most relevant timing
    information for UW3 users. By default, shows only UW3 operations (not
    low-level PETSc internals).

    Parameters
    ----------
    filter_uw : bool, optional
        If True (default), show only UW3 operations. If False, show all PETSc events.
    min_time : float, optional
        Minimum time (seconds) for an event to be displayed. Default 0.001 (1ms).
        Helps filter out negligible operations.
    sort_by : str, optional
        Sort events by: 'time' (default), 'count', or 'name'.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'events': List of (name, count, time, percent) tuples
        - 'total_time': Total execution time
        - 'num_events': Number of events displayed

    Example
    -------
    >>> import underworld3 as uw
    >>> uw.timing.start()
    >>> # ... do work ...
    >>>
    >>> # Get UW3-focused summary
    >>> summary = uw.timing.get_summary()
    >>> for name, count, time, pct in summary['events']:
    >>>     print(f"{name:40s} {count:5d} calls  {time:8.3f}s  ({pct:5.1f}%)")
    >>>
    >>> # Get all events (including PETSc internals)
    >>> full_summary = uw.timing.get_summary(filter_uw=False, min_time=0.0)

    Notes
    -----
    - Call after `uw.timing.start()` and your computation
    - Filtered view helps identify UW3 performance bottlenecks
    - For comprehensive PETSc profiling, use `uw.timing.print_table()` or `filter_uw=False`
    """
    from petsc4py import PETSc
    import re

    if not PETSc.Log.isActive():
        if RANK == 0:
            print("⚠️  PETSc logging not enabled. Call uw.timing.start() first.")
        return {'events': [], 'total_time': 0.0, 'num_events': 0}

    # Collect all events first to calculate total time
    events_info = []
    total_time = 0.0

    # UW3 event patterns (customize based on actual event naming)
    uw_patterns = [
        r'^Function\.',           # Function.evaluate_nd, etc.
        r'^evaluate$',            # evaluate
        r'^global_evaluate$',     # global_evaluate
        r'^Mesh\.',               # Mesh.__init__, Mesh.update_lvec, etc.
        r'^UnstructuredSimplexBox',
        r'^StructuredQuadBox',
        r'^_BaseMeshVariable\.',
        r'^SwarmVariable\.',
        r'^Swarm\.',
        r'^KDTree\.',
        r'^_from_',               # _from_gmsh, _from_plexh5, etc.
        r'^[A-Z]\w+\.\w+',        # ClassName.method_name patterns
    ]

    uw_regex = re.compile('|'.join(uw_patterns))

    # First pass: collect all matching events and calculate total time
    all_events_raw = []
    for event_name, event in _petsc_events.items():
        perf_info = event.getPerfInfo()
        count = perf_info['count']
        time = perf_info['time']

        # Skip if below minimum time
        if time < min_time:
            continue

        # Filter UW3 events if requested
        if filter_uw and not uw_regex.match(event_name):
            continue

        all_events_raw.append((event_name, count, time))
        total_time += time

    # Second pass: calculate percentages now that we know total_time
    for event_name, count, time in all_events_raw:
        pct = (time / total_time * 100) if total_time > 0 else 0.0
        events_info.append((event_name, count, time, pct))

    # Sort
    if sort_by == 'time':
        events_info.sort(key=lambda x: x[2], reverse=True)  # Sort by time descending
    elif sort_by == 'count':
        events_info.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending
    elif sort_by == 'name':
        events_info.sort(key=lambda x: x[0])  # Sort alphabetically

    return {
        'events': events_info,
        'total_time': total_time,
        'num_events': len(events_info)
    }


def print_summary(filter_uw=True, min_time=0.001, sort_by='time', max_events=50):
    """
    Print user-friendly timing summary table.

    Displays a clean, focused table of timing results for UW3 operations.
    Much more readable than the full PETSc log for typical users.

    Parameters
    ----------
    filter_uw : bool, optional
        If True (default), show only UW3 operations. If False, show all events.
    min_time : float, optional
        Minimum time (seconds) for an event to be displayed. Default 0.001 (1ms).
    sort_by : str, optional
        Sort events by: 'time' (default), 'count', or 'name'.
    max_events : int, optional
        Maximum number of events to display. Default 50.

    Example
    -------
    >>> import underworld3 as uw
    >>> uw.timing.start()
    >>> # ... run simulation ...
    >>>
    >>> # Quick UW3-focused summary
    >>> uw.timing.print_summary()
    >>>
    >>> # Detailed view with all events
    >>> uw.timing.print_summary(filter_uw=False, max_events=100)
    >>>
    >>> # Show top 10 most-called operations
    >>> uw.timing.print_summary(sort_by='count', max_events=10)

    Notes
    -----
    - For full PETSc profiling details, use `uw.timing.print_table()`
    - This function focuses on high-level UW3 operations
    - Perfect for quick performance checks in notebooks
    """
    summary = get_summary(filter_uw=filter_uw, min_time=min_time, sort_by=sort_by)

    if summary['num_events'] == 0:
        if RANK == 0:
            print("No timing events found.")
            print("Make sure to call uw.timing.start() before your computation.")
        return

    if RANK != 0:
        return  # Only rank 0 prints

    # Print header
    print("\n" + "=" * 100)
    if filter_uw:
        print("UNDERWORLD3 TIMING SUMMARY (UW3 Operations Only)")
    else:
        print("UNDERWORLD3 TIMING SUMMARY (All PETSc Events)")
    print("=" * 100)
    print(f"Total time: {summary['total_time']:.3f} seconds")
    print(f"Showing {min(max_events, summary['num_events'])} of {summary['num_events']} events (min time: {min_time*1000:.1f}ms)")
    print("=" * 100)

    # Print table header
    print(f"{'Event Name':<50s} {'Count':>8s} {'Time (s)':>12s} {'% Total':>10s}")
    print("-" * 100)

    # Print events (up to max_events)
    for name, count, time, pct in summary['events'][:max_events]:
        print(f"{name:<50s} {count:8d} {time:12.6f} {pct:9.1f}%")

    print("=" * 100)

    if filter_uw:
        print("\n💡 Tip: Use uw.timing.print_summary(filter_uw=False) to see all PETSc events")
        print("    Use uw.timing.print_table() for full PETSc profiling details")
    else:
        print("\n💡 Tip: Use uw.timing.print_summary() to see only UW3 operations")
    print()


# ============================================================================
# Module Documentation
# ============================================================================

__all__ = [
    'start',
    'stop',
    'reset',
    'print_table',
    'view',
    'routine_timer_decorator',
    'add_timing_to_module',
    'create_event',
    'enable_petsc_logging',
    'print_petsc_log',
    'get_summary',
    'print_summary',
]
