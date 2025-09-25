# Simplified Parallel Print Design

## Core Concept

A simple `uw.pprint()` function that executes and prints on specified ranks, with `selective_ranks()` context manager as the underlying mechanism.

## 1. The `pprint()` Function

```python
def pprint(ranks, *args, prefix=True, **kwargs):
    """
    Parallel print - execute and print on specified ranks.
    
    Args:
        ranks: Which ranks to execute/print on:
            - int: single rank (e.g., 0)
            - slice: range of ranks (e.g., slice(0,4))
            - list/tuple: specific ranks (e.g., [0, 2, 5])
            - None or 'all': all ranks
        *args: Items to print
        prefix: If True, prepend "[Rank N]" to output
        **kwargs: Passed to print()
    
    Examples:
        uw.pprint(0, "Hello")                    # Only rank 0
        uw.pprint(slice(0,4), "First 4 ranks")   # Ranks 0-3
        uw.pprint([0, 2], "Ranks 0 and 2")       # Specific ranks
        uw.pprint(None, "All ranks")             # Everyone
        uw.pprint(slice(None), "Also all ranks") # slice(None) = all
    """
    with selective_ranks(ranks):
        if prefix and ranks != 0:  # No prefix needed for single rank 0
            print(f"[Rank {uw.mpi.rank}]", *args, **kwargs)
        else:
            print(*args, **kwargs)
```

## 2. The `selective_ranks()` Context Manager

```python
from contextlib import contextmanager
import sys
import io

@contextmanager
def selective_ranks(ranks):
    """
    Execute code only on selected ranks, with collective operation detection.
    
    Args:
        ranks: Which ranks execute:
            - int: single rank (e.g., 0)
            - slice: range of ranks (e.g., slice(0,4))
            - list/tuple: specific ranks (e.g., [0, 2, 5])
            - None or 'all': all ranks
    
    Raises:
        CollectiveOperationError: If collective operation detected (would hang)
    
    Examples:
        with uw.selective_ranks(0):
            # Only rank 0 executes
            import matplotlib.pyplot as plt
            plt.plot(x, y)
        
        with uw.selective_ranks(slice(0, 4)):
            # Ranks 0-3 execute
            process_local_data()
    """
    # Determine if this rank should execute
    should_execute = _should_rank_execute(uw.mpi.rank, ranks)
    
    # Track state for collective detection
    old_selective = getattr(uw.mpi, '_in_selective_ranks', False)
    old_executing_ranks = getattr(uw.mpi, '_selective_executing_ranks', None)
    
    uw.mpi._in_selective_ranks = True
    uw.mpi._selective_executing_ranks = _get_executing_ranks(ranks)
    uw.mpi._this_rank_executes = should_execute
    
    try:
        if should_execute:
            yield True
        else:
            # Non-executing ranks skip the block entirely
            yield False
    finally:
        # Restore state
        uw.mpi._in_selective_ranks = old_selective
        uw.mpi._selective_executing_ranks = old_executing_ranks
        uw.mpi._this_rank_executes = True

def _should_rank_execute(rank, ranks):
    """Check if this rank should execute."""
    if ranks is None or ranks == 'all':
        return True
    elif isinstance(ranks, int):
        return rank == ranks
    elif isinstance(ranks, slice):
        # Handle slice(start, stop, step)
        start = ranks.start if ranks.start is not None else 0
        stop = ranks.stop if ranks.stop is not None else uw.mpi.size
        step = ranks.step if ranks.step is not None else 1
        return rank in range(start, stop, step)
    elif isinstance(ranks, (list, tuple)):
        return rank in ranks
    else:
        raise ValueError(f"Invalid ranks specification: {ranks}")

def _get_executing_ranks(ranks):
    """Get list of all ranks that will execute."""
    if ranks is None or ranks == 'all':
        return list(range(uw.mpi.size))
    elif isinstance(ranks, int):
        return [ranks]
    elif isinstance(ranks, slice):
        start = ranks.start if ranks.start is not None else 0
        stop = ranks.stop if ranks.stop is not None else uw.mpi.size
        step = ranks.step if ranks.step is not None else 1
        return list(range(start, stop, step))
    elif isinstance(ranks, (list, tuple)):
        return list(ranks)
    else:
        return []
```

## 3. Collective Operation Detection

```python
class CollectiveOperationError(RuntimeError):
    """Raised when collective operation called in selective context."""
    pass

def collective_operation(func):
    """
    Decorator to mark and protect collective operations.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if we're in selective_ranks context
        if getattr(uw.mpi, '_in_selective_ranks', False):
            executing = uw.mpi._selective_executing_ranks
            total = uw.mpi.size
            
            # If not all ranks execute, this will hang
            if len(executing) < total:
                if uw.mpi._this_rank_executes:
                    # This rank would execute but others won't - deadlock!
                    non_executing = set(range(total)) - set(executing)
                    raise CollectiveOperationError(
                        f"\n{func.__name__}() is a collective operation that requires ALL ranks.\n"
                        f"Currently executing on ranks {executing} but not on {list(non_executing)}.\n"
                        f"This would cause a deadlock.\n\n"
                        f"Solutions:\n"
                        f"1. Execute on all ranks: with uw.selective_ranks(None):\n"
                        f"2. Move collective operation outside selective context\n"
                        f"3. Use a non-collective alternative if available"
                    )
                else:
                    # This rank won't execute - just skip
                    return None
        
        # Execute the operation
        return func(self, *args, **kwargs)
    
    wrapper._is_collective = True
    return wrapper
```

## 4. Usage Examples

### Basic Printing

```python
# Print on rank 0 only
uw.pprint(0, "Starting simulation")

# Print on first 4 ranks with rank prefix
uw.pprint(slice(0, 4), "Processing partition")
# Output:
# [Rank 0] Processing partition
# [Rank 1] Processing partition
# [Rank 2] Processing partition
# [Rank 3] Processing partition

# Print on specific ranks
uw.pprint([0, 5, 10], "Checkpoint reached")

# Print on all ranks
uw.pprint(None, "Local data size:", data.shape)
```

### With Selective Ranks Context

```python
# Serial visualization - only rank 0
with uw.selective_ranks(0):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.savefig("output.png")
    print("Plot saved")  # Only rank 0 prints

# Multiple ranks for debugging
with uw.selective_ranks(slice(0, 4)):
    # Ranks 0-3 execute this
    local_max = data.max()
    print(f"[Rank {uw.mpi.rank}] Local max: {local_max}")

# This would raise an error
with uw.selective_ranks(0):
    stats = var.stats()  # ERROR: Collective operation!
```

### Convenience Functions

```python
# Common patterns as shortcuts
def print0(*args, **kwargs):
    """Print on rank 0 only (no prefix)."""
    pprint(0, *args, prefix=False, **kwargs)

def printall(*args, **kwargs):
    """Print on all ranks with prefix."""
    pprint(None, *args, prefix=True, **kwargs)

def debug(*args, **kwargs):
    """Debug print on all ranks."""
    pprint(None, "[DEBUG]", *args, prefix=True, **kwargs)

# Usage
uw.print0("Simulation complete")
uw.printall("Local elements:", mesh.nElements)
uw.debug("Solver iterations:", iterations)
```

### Error Detection Example

```python
# This code would hang in MPI:
if uw.mpi.rank == 0:
    stats = var.stats()  # Other ranks wait forever

# With selective_ranks, you get a clear error:
with uw.selective_ranks(0):
    stats = var.stats()
    
# Error message:
# CollectiveOperationError:
# stats() is a collective operation that requires ALL ranks.
# Currently executing on ranks [0] but not on [1, 2, 3].
# This would cause a deadlock.
#
# Solutions:
# 1. Execute on all ranks: with uw.selective_ranks(None):
# 2. Move collective operation outside selective context
# 3. Use a non-collective alternative if available
```

## 5. API Summary

```python
# Main function
uw.pprint(ranks, *args, prefix=True)

# Convenience shortcuts  
uw.print0(*args)          # Rank 0, no prefix
uw.printall(*args)        # All ranks, with prefix
uw.debug(*args)           # All ranks, debug prefix

# Context manager
with uw.selective_ranks(ranks):
    # Code executes only on specified ranks
    # Collective operations raise helpful errors

# Rank specifications
uw.pprint(0, "msg")              # Single rank
uw.pprint(slice(0, 4), "msg")    # Range: 0, 1, 2, 3
uw.pprint([0, 2, 5], "msg")      # Specific ranks
uw.pprint(None, "msg")           # All ranks
uw.pprint(slice(None), "msg")    # Also all ranks
```

## Benefits

1. **Simple API**: Just `uw.pprint(ranks, message)`
2. **Flexible rank selection**: Slices, lists, or single ranks
3. **Automatic safety**: Detects collective operations that would hang
4. **Clear errors**: Tells you exactly what would go wrong and how to fix it
5. **No MPI knowledge needed**: Users just specify which ranks they want

This is much simpler and focused on your actual request!