# Rank Selection Specification

## Overview

Multiple ways to specify which ranks execute/print, from simple to complex patterns.

## 1. Basic Selection Types

### Single Rank (int)
```python
uw.pprint(0, "Only rank 0")
uw.pprint(5, "Only rank 5")

with uw.selective_ranks(0):
    # Only rank 0 executes
```

### All Ranks (None or 'all')
```python
uw.pprint(None, "Everyone")
uw.pprint('all', "Also everyone")

with uw.selective_ranks(None):
    # All ranks execute
```

### Range of Ranks (slice)
```python
# Standard Python slice notation
uw.pprint(slice(0, 4), "Ranks 0, 1, 2, 3")
uw.pprint(slice(2, 8), "Ranks 2 through 7")
uw.pprint(slice(0, 10, 2), "Even ranks 0, 2, 4, 6, 8")
uw.pprint(slice(1, 10, 2), "Odd ranks 1, 3, 5, 7, 9")

# With None for open-ended
uw.pprint(slice(5, None), "Rank 5 onwards")
uw.pprint(slice(None, 5), "Ranks 0-4")
uw.pprint(slice(None, None, 2), "All even ranks")
```

### Specific Ranks (list/tuple)
```python
uw.pprint([0, 3, 7], "Ranks 0, 3, and 7")
uw.pprint((1, 2, 5, 8), "Ranks 1, 2, 5, and 8")
uw.pprint([0], "Just rank 0 (as list)")

# Can be dynamically generated
important_ranks = [0, uw.mpi.size // 2, uw.mpi.size - 1]
uw.pprint(important_ranks, "First, middle, and last rank")
```

## 2. Advanced Selection Patterns

### Named Patterns (string aliases)
```python
# Predefined convenient patterns
uw.pprint('all', "Everyone")
uw.pprint('first', "Rank 0 only")  
uw.pprint('last', "Highest rank only")
uw.pprint('even', "Even-numbered ranks")
uw.pprint('odd', "Odd-numbered ranks")
uw.pprint('corners', "First and last rank")

# Implementation
def _interpret_ranks(ranks, mpi_size):
    """Convert rank specification to list of ranks."""
    if ranks is None or ranks == 'all':
        return list(range(mpi_size))
    elif ranks == 'first':
        return [0]
    elif ranks == 'last':
        return [mpi_size - 1]
    elif ranks == 'even':
        return list(range(0, mpi_size, 2))
    elif ranks == 'odd':
        return list(range(1, mpi_size, 2))
    elif ranks == 'corners':
        return [0, mpi_size - 1] if mpi_size > 1 else [0]
    # ... handle other types
```

### Percentage-Based Selection
```python
# Select by percentage of ranks
uw.pprint('10%', "First 10% of ranks")
uw.pprint('25%', "First quarter of ranks")
uw.pprint('50%', "First half of ranks")

# Implementation
elif isinstance(ranks, str) and ranks.endswith('%'):
    percent = float(ranks[:-1]) / 100.0
    count = max(1, int(mpi_size * percent))
    return list(range(count))
```

### Set Operations (combining selections)
```python
# Define a RankSet class for complex selections
class RankSet:
    def __init__(self, ranks):
        self.ranks = _interpret_ranks(ranks, uw.mpi.size)
    
    def __or__(self, other):  # Union
        return RankSet(list(set(self.ranks) | set(other.ranks)))
    
    def __and__(self, other):  # Intersection
        return RankSet(list(set(self.ranks) & set(other.ranks)))
    
    def __sub__(self, other):  # Difference
        return RankSet(list(set(self.ranks) - set(other.ranks)))

# Usage
first_half = RankSet(slice(0, uw.mpi.size // 2))
even_ranks = RankSet('even')
first_half_even = first_half & even_ranks

uw.pprint(first_half_even.ranks, "First half, even ranks only")
```

## 3. Function-Based Selection

### Lambda/Callable Selection
```python
# Pass a function that returns True/False for each rank
uw.pprint(lambda r: r % 3 == 0, "Every third rank")
uw.pprint(lambda r: r < 5 or r > 10, "First 5 or after 10")
uw.pprint(lambda r: r in [0, 3, 7], "Custom selection")

# Implementation addition
elif callable(ranks):
    return [r for r in range(mpi_size) if ranks(r)]
```

### Conditional Selection Based on Data
```python
# Select ranks based on runtime conditions
def ranks_with_work():
    """Select ranks that have work to do."""
    if local_data.size > 0:
        return uw.mpi.rank
    return None

active_ranks = uw.mpi.gather(ranks_with_work())
uw.pprint(active_ranks, "I have data to process")
```

## 4. Convenience Methods for Common Patterns

### Helper Functions
```python
def first_n_ranks(n):
    """First n ranks."""
    return slice(0, n)

def last_n_ranks(n):
    """Last n ranks."""
    return slice(uw.mpi.size - n, None)

def every_nth_rank(n, offset=0):
    """Every nth rank starting from offset."""
    return slice(offset, None, n)

def rank_range(start, end):
    """Ranks from start to end (inclusive)."""
    return slice(start, end + 1)

# Usage
uw.pprint(first_n_ranks(4), "First 4 ranks")
uw.pprint(last_n_ranks(2), "Last 2 ranks")
uw.pprint(every_nth_rank(3), "Every third rank")
uw.pprint(rank_range(2, 6), "Ranks 2-6 inclusive")
```

## 5. Enhanced Implementation

```python
def _should_rank_execute(rank, ranks, mpi_size):
    """
    Comprehensive rank selection logic.
    
    Args:
        rank: Current rank number
        ranks: Rank specification (many types supported)
        mpi_size: Total number of ranks
    
    Returns:
        bool: True if this rank should execute
    """
    # None or 'all' - everyone
    if ranks is None or ranks == 'all':
        return True
    
    # Single integer
    elif isinstance(ranks, int):
        return rank == ranks
    
    # Slice object
    elif isinstance(ranks, slice):
        start = ranks.start if ranks.start is not None else 0
        stop = ranks.stop if ranks.stop is not None else mpi_size
        step = ranks.step if ranks.step is not None else 1
        # Handle negative indices
        if start < 0:
            start = mpi_size + start
        if stop < 0:
            stop = mpi_size + stop
        return rank in range(start, stop, step)
    
    # List or tuple
    elif isinstance(ranks, (list, tuple)):
        return rank in ranks
    
    # String patterns
    elif isinstance(ranks, str):
        if ranks == 'first':
            return rank == 0
        elif ranks == 'last':
            return rank == mpi_size - 1
        elif ranks == 'even':
            return rank % 2 == 0
        elif ranks == 'odd':
            return rank % 2 == 1
        elif ranks == 'corners':
            return rank in [0, mpi_size - 1]
        elif ranks.endswith('%'):
            percent = float(ranks[:-1]) / 100.0
            count = max(1, int(mpi_size * percent))
            return rank < count
    
    # Callable (function)
    elif callable(ranks):
        return bool(ranks(rank))
    
    # Set
    elif isinstance(ranks, set):
        return rank in ranks
    
    # Range object
    elif isinstance(ranks, range):
        return rank in ranks
    
    # NumPy array (boolean mask or indices)
    elif hasattr(ranks, '__array__'):
        import numpy as np
        arr = np.asarray(ranks)
        if arr.dtype == bool:
            # Boolean mask
            return arr[rank] if rank < len(arr) else False
        else:
            # Array of indices
            return rank in arr
    
    # RankSet or custom object with .ranks attribute
    elif hasattr(ranks, 'ranks'):
        return rank in ranks.ranks
    
    else:
        raise ValueError(f"Unsupported rank specification type: {type(ranks)}")
```

## 6. Usage Examples

### Example 1: Debugging Different Groups
```python
# First few ranks for initial debugging
uw.pprint(slice(0, 3), "Starting computation")

# Sample from middle
mid = uw.mpi.size // 2
uw.pprint(slice(mid-1, mid+2), "Middle ranks status")

# Last rank for summary
uw.pprint('last', "Final rank complete")
```

### Example 2: Load Balancing Diagnostics
```python
# Every 10th rank reports load
uw.pprint(slice(0, None, 10), f"Load: {local_workload}")

# Ranks with imbalanced load
overloaded = [r for r in range(uw.mpi.size) if workload[r] > threshold]
uw.pprint(overloaded, "WARNING: Overloaded")
```

### Example 3: Hierarchical Output
```python
# Level 0: Just root
uw.pprint(0, "=== SUMMARY ===")

# Level 1: Quadrant leaders (for 16+ ranks)
if uw.mpi.size >= 16:
    leaders = [0, 4, 8, 12]
    uw.pprint(leaders, "Quadrant status")

# Level 2: All ranks (verbose)
if verbose:
    uw.pprint('all', f"Detailed status: {status}")
```

### Example 4: Complex Selection
```python
# Combine multiple criteria
def select_debug_ranks(rank):
    # First 3, last 1, and any rank with errors
    if rank < 3:
        return True
    if rank == uw.mpi.size - 1:
        return True
    if has_errors[rank]:
        return True
    return False

uw.pprint(select_debug_ranks, "Debug output")
```

### Example 5: NumPy Integration
```python
import numpy as np

# Boolean mask
mask = np.zeros(uw.mpi.size, dtype=bool)
mask[0:4] = True  # First 4
mask[-2:] = True  # Last 2
uw.pprint(mask, "Selected by mask")

# Index array
indices = np.array([0, 2, 4, 6, 8])
uw.pprint(indices, "Even ranks up to 8")
```

## 7. Error Handling

```python
# Clear error messages for invalid specifications
try:
    uw.pprint("invalid", "Message")
except ValueError as e:
    print(e)
    # "Unsupported rank specification type: <class 'str'>"
    # "Did you mean 'all', 'first', 'last', 'even', or 'odd'?"

# Bounds checking
uw.pprint(100, "Rank 100")  # If only 16 ranks exist
# Simply doesn't print (rank 100 doesn't exist)

# Empty selection
uw.pprint([], "Nobody")  # Valid - no output

# Negative indices (Python-style)
uw.pprint(-1, "Last rank")  # Could support this
uw.pprint(slice(-3, None), "Last 3 ranks")  # Pythonic!
```

## 8. Performance Considerations

```python
# Pre-compute rank lists for repeated use
class RankGroups:
    def __init__(self):
        size = uw.mpi.size
        self.io_ranks = [0]  # I/O on rank 0
        self.compute_ranks = list(range(1, size))  # Compute on others
        self.checkpoint_ranks = list(range(0, size, size // 4))  # Checkpointing subset

groups = RankGroups()

# Use throughout simulation
uw.pprint(groups.io_ranks, "Saving data...")
uw.pprint(groups.compute_ranks, "Computing...")
uw.pprint(groups.checkpoint_ranks, "Checkpoint")
```

## Summary

The rank selection system supports:

1. **Basic**: int, None/'all', slice, list/tuple
2. **Named**: 'first', 'last', 'even', 'odd', 'corners'  
3. **Percentage**: '10%', '25%', '50%'
4. **Functional**: lambda/callable for complex logic
5. **NumPy**: boolean masks, index arrays
6. **Pythonic**: Negative indices, range objects
7. **Extensible**: Custom classes with .ranks attribute

This gives maximum flexibility while keeping simple cases simple!