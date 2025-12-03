# Parallel Printing and Selective Execution Design

## Philosophy

Remove "mpi" from user-facing code. Users shouldn't need to know about MPI - they just want to print/debug.

## 1. Unified Printing Function

### Basic Design

```python
def uw_print(*args, 
             ranks=0,           # Which rank(s) to print from
             prefix=True,       # Prepend rank number
             style='default',   # 'default', 'debug', 'warning', 'error'
             **kwargs):
    """
    Parallel-safe printing with rank control and formatting.
    
    Args:
        *args: Items to print
        ranks: Which ranks print:
            - int: single rank (e.g., 0)
            - slice: range of ranks (e.g., slice(0,4) or 0:4)
            - list: specific ranks (e.g., [0, 2, 5])
            - 'all': all ranks
            - 'first': rank 0 only (default)
            - 'last': highest rank only
        prefix: If True, prepend "[Rank N]" to output
        style: Output style:
            - 'default': normal output
            - 'debug': colored/styled for debugging
            - 'warning': warning style
            - 'error': error style
        **kwargs: Passed to print()
    
    Examples:
        uw.print("Hello")                    # Rank 0 only, with prefix
        uw.print("Value:", x, ranks='all')   # All ranks
        uw.print("Debug", ranks=0:4)         # Ranks 0-3
        uw.print("Error!", style='error')    # Styled error message
    """
    # Normalize ranks specification
    should_print = _should_rank_print(uw.mpi.rank, ranks)
    
    if should_print:
        # Build prefix
        prefix_str = ""
        if prefix:
            if style == 'debug':
                prefix_str = f"[Rank {uw.mpi.rank:3d}] "
            elif style == 'warning':
                prefix_str = f"[Rank {uw.mpi.rank} WARNING] "
            elif style == 'error':
                prefix_str = f"[Rank {uw.mpi.rank} ERROR] "
            else:
                prefix_str = f"[Rank {uw.mpi.rank}] "
        
        # Print with optional styling
        if prefix_str:
            print(prefix_str, *args, **kwargs)
        else:
            print(*args, **kwargs)

def _should_rank_print(rank, ranks):
    """Determine if this rank should print."""
    if ranks == 'all':
        return True
    elif ranks == 'first':
        return rank == 0
    elif ranks == 'last':
        return rank == uw.mpi.size - 1
    elif isinstance(ranks, int):
        return rank == ranks
    elif isinstance(ranks, slice):
        # Handle slice like ranks=0:4
        start = ranks.start or 0
        stop = ranks.stop or uw.mpi.size
        step = ranks.step or 1
        return rank in range(start, stop, step)
    elif isinstance(ranks, (list, tuple)):
        return rank in ranks
    else:
        return rank == 0  # Default to rank 0
```

### Convenience Aliases

```python
# Short aliases for common patterns
def print0(*args, **kwargs):
    """Print on rank 0 only (no prefix by default)."""
    uw_print(*args, ranks=0, prefix=False, **kwargs)

def printall(*args, **kwargs):
    """Print on all ranks with rank prefix."""
    uw_print(*args, ranks='all', prefix=True, **kwargs)

def debug(*args, **kwargs):
    """Debug print on all ranks with highlighting."""
    uw_print(*args, ranks='all', style='debug', prefix=True, **kwargs)

def warning(*args, **kwargs):
    """Warning message on rank 0."""
    uw_print(*args, ranks=0, style='warning', prefix=True, **kwargs)

def error(*args, **kwargs):
    """Error message on rank 0."""
    uw_print(*args, ranks=0, style='error', prefix=True, **kwargs)
```

### Slice Syntax Support

```python
# Enable nice slice syntax
class PrintSlice:
    """Enable uw.print[0:4]("message") syntax."""
    
    def __getitem__(self, ranks):
        def printer(*args, **kwargs):
            return uw_print(*args, ranks=ranks, **kwargs)
        return printer

# Usage:
uw.print[0]("Rank 0")           # Single rank
uw.print[0:4]("First 4 ranks")  # Slice
uw.print['all']("All ranks")    # All
```

## 2. Selective Ranks Context Manager

### Safe Execution with Validation

```python
@contextmanager
def selective_ranks(ranks, 
                   allow_collective=False,
                   suppress_stdout=True):
    """
    Execute code on selected ranks only, with collective operation protection.
    
    Unlike rank_conditional (which runs on all ranks), this ONLY executes
    on specified ranks. It can detect and prevent collective operation deadlocks.
    
    Args:
        ranks: Which ranks execute (int, slice, list, 'all', 'first', 'last')
        allow_collective: If False (default), raise error if collective op called
        suppress_stdout: Suppress output on non-executing ranks
    
    Raises:
        CollectiveOperationError: If collective operation called when disallowed
    
    Examples:
        # Safe - no collective ops
        with uw.selective_ranks(0):
            print("Only rank 0 executes this")
            import matplotlib.pyplot as plt
            plt.plot(x, y)
        
        # Dangerous - will raise error
        with uw.selective_ranks(0):
            stats = var.stats()  # CollectiveOperationError!
        
        # Explicitly allow (for advanced users)
        with uw.selective_ranks(0, allow_collective=True):
            # I know what I'm doing...
            result = some_collective_op()
    """
    # Determine if this rank should execute
    should_execute = _should_rank_print(uw.mpi.rank, ranks)
    
    # Track state
    old_selective = uw.mpi._in_selective_ranks
    old_allow = uw.mpi._selective_allows_collective
    uw.mpi._in_selective_ranks = True
    uw.mpi._selective_allows_collective = allow_collective
    uw.mpi._selective_active = should_execute
    
    # Optionally suppress stdout on non-executing ranks
    if suppress_stdout and not should_execute:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    
    try:
        if should_execute:
            yield True
        else:
            yield False
    finally:
        # Restore state
        if suppress_stdout and not should_execute:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        uw.mpi._in_selective_ranks = old_selective
        uw.mpi._selective_allows_collective = old_allow
        uw.mpi._selective_active = False
```

### Collective Operation Detection

```python
class CollectiveOperationError(RuntimeError):
    """Raised when collective operation called in unsafe context."""
    pass

def collective_operation(func):
    """
    Decorator for collective operations with safety checks.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if we're in selective_ranks context
        if uw.mpi._in_selective_ranks:
            if not uw.mpi._selective_allows_collective:
                # Check if all ranks would execute
                if not uw.mpi._selective_active:
                    # This rank won't execute - would cause deadlock!
                    raise CollectiveOperationError(
                        f"{func.__name__} is a collective operation but called "
                        f"inside selective_ranks() where rank {uw.mpi.rank} doesn't execute. "
                        f"This would cause a deadlock. Use rank_conditional() instead, "
                        f"or set allow_collective=True if you know what you're doing."
                    )
        
        # Execute the collective operation
        result = func(self, *args, **kwargs)
        
        # Handle return suppression for rank_conditional
        if uw.mpi.is_conditional and not uw.mpi.is_active_rank:
            return None
        
        return result
    
    wrapper._is_collective = True
    return wrapper
```

## 3. MPI Context Extensions

```python
class MPIContext:
    def __init__(self):
        # ... existing ...
        
        # Selective ranks tracking
        self._in_selective_ranks = False
        self._selective_allows_collective = False
        self._selective_active = False
        
        # rank_conditional tracking (existing)
        self._conditional_rank = None
    
    @property
    def in_selective(self):
        """Check if inside selective_ranks context."""
        return self._in_selective_ranks
    
    @property
    def in_safe_context(self):
        """Check if in any safe execution context."""
        return self.is_conditional or self._in_selective_ranks
```

## 4. Usage Examples

### Example 1: Simple Printing

```python
# Old way (explicit MPI)
if uw.mpi.rank == 0:
    print(f"Stats: {var.stats()}")

# New way (no MPI knowledge needed)
uw.print0(f"Stats: {var.stats()}")

# Debug on all ranks
uw.debug(f"Local max: {var.data.max()}")
# Output:
# [Rank 0] Local max: 1.234
# [Rank 1] Local max: 0.987
# [Rank 2] Local max: 1.456
```

### Example 2: Rank Slicing

```python
# Print on first 4 ranks
uw.print[0:4](f"Processing partition {partition_id}")

# Print on specific ranks
uw.print[[0, 5, 10]](f"Checkpoint: {var.stats()}")

# Print on last rank
uw.print['last'](f"Final rank reporting in")
```

### Example 3: Safe Serial Operations

```python
# Visualization - only rank 0, collective ops blocked
with uw.selective_ranks(0):
    import matplotlib.pyplot as plt
    plt.figure()
    # This is safe - only uses local data
    plt.plot(coords, temperature.data[:, 0])  
    plt.savefig("output.png")

# This would raise CollectiveOperationError
with uw.selective_ranks(0):
    stats = var.stats()  # ERROR: collective op in selective context!
```

### Example 4: Collective-Safe Printing

```python
# This works - stats() runs on all ranks, only rank 0 prints
with uw.rank_conditional(0):
    stats = var.stats()  # All ranks execute
    print(f"Stats: {stats}")  # Only rank 0 prints

# Equivalent using uw_print
uw.print0(f"Stats: {var.stats()}")  # Implicit rank_conditional
```

### Example 5: Debugging

```python
# Debug print to see what each rank has
uw.debug(f"My data shape: {var.data.shape}")
# [Rank 0] My data shape: (100, 1, 3)
# [Rank 1] My data shape: (95, 1, 3)
# [Rank 2] My data shape: (105, 1, 3)

# Warning on rank 0
if convergence_issues:
    uw.warning(f"Convergence poor: {iterations} iterations")
```

## 5. Enhanced print0() Implementation

```python
def print0(*args, **kwargs):
    """
    Print only on rank 0, safe for collective arguments.
    
    This wraps arguments in rank_conditional to ensure collective
    operations execute on all ranks.
    """
    # Detect if any args might be collective
    # (function calls, property access, etc.)
    with uw.mpi.rank_conditional(0, suppress_stdout=True):
        print(*args, **kwargs)

# Alternative implementation with explicit collective detection
def print0_smart(*args, **kwargs):
    """
    Smart print that detects and handles collective operations.
    """
    # Check if we're already in a safe context
    if uw.mpi.in_safe_context:
        # Already safe, just print
        if uw.mpi.is_active_rank or uw.mpi.rank == 0:
            print(*args, **kwargs)
    else:
        # Not in safe context - use rank_conditional
        with uw.mpi.rank_conditional(0, suppress_stdout=True):
            print(*args, **kwargs)
```

## 6. API Summary

```python
# Printing functions (no MPI knowledge needed!)
uw.print0("message")              # Rank 0 only
uw.printall("message")            # All ranks with prefix
uw.debug("message")               # Debug style, all ranks
uw.warning("message")             # Warning style, rank 0
uw.error("message")               # Error style, rank 0

# Flexible printing
uw.print("msg", ranks=0)          # Specify ranks
uw.print("msg", ranks='all')      # All ranks
uw.print("msg", ranks=0:4)        # Slice
uw.print("msg", ranks=[0,2,5])    # List
uw.print("msg", prefix=True)      # With rank prefix

# Slice syntax
uw.print[0]("rank 0")             # Single rank
uw.print[0:4]("first 4")          # Range
uw.print['all']("everyone")       # All

# Safe execution contexts
with uw.rank_conditional(0):      # Collective-safe
    print(var.stats())             # All execute, rank 0 prints

with uw.selective_ranks(0):       # Serial-only, protected
    plt.plot(x, y)                 # Only rank 0 executes
    # var.stats()                  # Would raise error!

with uw.selective_ranks(0, allow_collective=True):
    # Advanced: explicit override

# Context queries
uw.mpi.in_selective               # In selective_ranks?
uw.mpi.in_safe_context            # In any safe context?
uw.mpi.is_active_rank             # Active in current context?
```

## 7. Error Messages

```python
# Clear, helpful errors
CollectiveOperationError: stats() is a collective operation but called 
inside selective_ranks() where rank 2 doesn't execute. This would cause 
a deadlock.

Solution: Use rank_conditional() instead:
    with uw.rank_conditional(0):
        stats = var.stats()

Or set allow_collective=True if you understand the consequences.
```

## 8. Implementation Priority

1. **Phase 1**: Basic `uw_print()` with rank selection
2. **Phase 2**: Convenience aliases (`print0`, `debug`, etc.)
3. **Phase 3**: `selective_ranks()` context with collective detection
4. **Phase 4**: Enhanced `@collective_operation` with error handling
5. **Phase 5**: Slice syntax support

## Benefits

1. **No MPI in user code** - users just call `uw.print0()` or `uw.debug()`
2. **Flexible rank selection** - print on any rank(s) easily
3. **Safety** - `selective_ranks()` prevents collective operation deadlocks
4. **Clear errors** - helpful messages when something would hang
5. **Backward compatible** - old `if uw.mpi.rank == 0:` still works
6. **Natural syntax** - `uw.print[0:4]("msg")` is intuitive