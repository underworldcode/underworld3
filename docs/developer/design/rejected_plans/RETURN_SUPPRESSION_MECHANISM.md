# Return Value Suppression in rank_conditional()

## The Challenge

We want code like this to work:
```python
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # All ranks execute
    print(stats)         # Only rank 0 prints
```

**Problem**: `print(stats)` needs `stats` to exist on all ranks (for syntax), but we only want rank 0 to actually print.

## Option 1: Manual Check (Simplest)

```python
@contextmanager
def rank_conditional(self, rank=0):
    old_conditional = self._conditional_rank
    self._conditional_rank = rank
    try:
        yield (self._rank == rank)  # Return True/False
    finally:
        self._conditional_rank = old_conditional

# User code
with uw.mpi.rank_conditional(0) as is_active:
    stats = var.stats()  # All ranks execute
    if is_active:
        print(stats)  # Explicit check
```

**Pros**: Simple, explicit, no magic
**Cons**: User must remember `if is_active:`

## Option 2: Suppress stdout (What We Want for print)

```python
import sys
import io

@contextmanager
def rank_conditional(self, rank=0):
    old_conditional = self._conditional_rank
    self._conditional_rank = rank
    
    # Suppress stdout on non-active ranks
    if self._rank != rank:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Redirect to nowhere
    
    try:
        yield (self._rank == rank)
    finally:
        # Restore stdout
        if self._rank != rank:
            sys.stdout = old_stdout
        self._conditional_rank = old_conditional

# User code - works automatically!
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # All ranks execute
    print(stats)         # All ranks execute, but only rank 0 output visible
```

**Pros**: Natural - print just works
**Cons**: Doesn't suppress return values, only stdout

## Option 3: Decorator Integration (For @collective_operation)

The `@collective_operation` decorator can intercept returns:

```python
def collective_operation(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Execute on all ranks
        result = func(self, *args, **kwargs)
        
        # Suppress return on non-active ranks
        if uw.mpi.is_conditional:
            if not uw.mpi.is_active_rank:
                return None  # Non-active ranks get None
        
        return result
    
    wrapper._is_collective = True
    return wrapper

# Usage
@collective_operation
def stats(self):
    # ... calculate stats ...
    return stats_dict

# In user code
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # rank 0 gets dict, others get None
    print(stats)         # rank 0 prints dict, others print None
```

**Pros**: Works with decorated methods automatically
**Cons**: Only works on decorated methods, `print(None)` on other ranks

## Option 4: Hybrid Approach (RECOMMENDED)

Combine stdout suppression + explicit check option:

```python
@contextmanager
def rank_conditional(self, rank=0, suppress_stdout=True):
    """
    Context for rank-conditional code.
    
    Args:
        rank: Which rank should be "active"
        suppress_stdout: If True, suppress stdout on non-active ranks
    
    Yields:
        bool: True if current rank is active rank
    """
    old_conditional = self._conditional_rank
    self._conditional_rank = rank
    is_active = (self._rank == rank)
    
    # Optionally suppress stdout on non-active ranks
    if suppress_stdout and not is_active:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    
    try:
        yield is_active
    finally:
        # Restore output streams
        if suppress_stdout and not is_active:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self._conditional_rank = old_conditional

# Usage patterns:

# Pattern 1: Automatic (stdout suppressed)
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # All execute
    print(stats)         # Only rank 0 visible

# Pattern 2: Explicit check (for non-print operations)
with uw.mpi.rank_conditional(0) as is_active:
    stats = var.stats()  # All execute
    if is_active:
        result = process_stats(stats)  # Only rank 0 executes
        save_to_file(result)

# Pattern 3: No suppression (debugging)
with uw.mpi.rank_conditional(0, suppress_stdout=False) as is_active:
    stats = var.stats()
    print(f"Rank {uw.mpi.rank}: {stats}")  # All ranks print
```

## Option 5: Context-Aware Print Functions (CLEANEST)

Don't suppress at context level - make print functions context-aware:

```python
# Context manager - simple, just tracks state
@contextmanager
def rank_conditional(self, rank=0):
    old_conditional = self._conditional_rank
    self._conditional_rank = rank
    try:
        yield (self._rank == rank)
    finally:
        self._conditional_rank = old_conditional

# Print function - checks context
def print0(*args, **kwargs):
    """Print only on rank 0, respects rank_conditional context."""
    # Check if we're in a conditional context
    if uw.mpi.is_conditional:
        # Inside rank_conditional - only print if active
        if uw.mpi.is_active_rank:
            print(*args, **kwargs)
    else:
        # Not in context - old behavior (check rank)
        if uw.mpi.rank == 0:
            print(*args, **kwargs)

# Usage - explicit uw.print0 instead of print
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # All execute
    uw.print0(stats)     # Only rank 0 prints

# Or use regular print with explicit check
with uw.mpi.rank_conditional(0) as is_active:
    stats = var.stats()
    if is_active:
        print(stats)
```

## Recommended Implementation

**Use Option 4 (Hybrid)** with these defaults:

```python
class MPIContext:
    @contextmanager
    def rank_conditional(self, rank=0, suppress_stdout=True):
        """
        Context for rank-specific execution with collective operation safety.
        
        All ranks execute code inside the context, but only the specified rank
        sees output (if suppress_stdout=True) or should process results.
        
        Args:
            rank: Which rank is "active" (default 0)
            suppress_stdout: Suppress print output on non-active ranks (default True)
        
        Yields:
            bool: True if current rank is the active rank
        
        Examples:
            # Simple printing (automatic)
            with uw.mpi.rank_conditional(0):
                print(f"Stats: {var.stats()}")  # All execute, only rank 0 prints
            
            # Conditional logic (explicit)
            with uw.mpi.rank_conditional(0) as is_active:
                data = var.stats()  # All ranks execute
                if is_active:
                    save_to_file(data)  # Only rank 0 saves
        """
        old_conditional = self._conditional_rank
        self._conditional_rank = rank
        is_active = (self._rank == rank)
        
        # Suppress stdout/stderr on non-active ranks
        if suppress_stdout and not is_active:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        
        try:
            yield is_active
        finally:
            # Restore streams
            if suppress_stdout and not is_active:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            self._conditional_rank = old_conditional

# The @collective_operation decorator also suppresses returns
def collective_operation(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        
        # If in rank_conditional, suppress return on non-active ranks
        if uw.mpi.is_conditional and not uw.mpi.is_active_rank:
            return None
        
        return result
    
    wrapper._is_collective = True
    return wrapper
```

## Why This Works

1. **stdout suppression**: Handles all print statements automatically
2. **Explicit check**: `yield is_active` lets users do conditional logic
3. **Decorator integration**: `@collective_operation` returns None on non-active ranks
4. **Debug mode**: Can disable suppression with `suppress_stdout=False`

## Edge Cases Handled

```python
# Mixed print and operations
with uw.mpi.rank_conditional(0) as is_active:
    stats = var.stats()        # All ranks execute, @collective returns None on ranks != 0
    print(f"Stats: {stats}")   # All ranks execute, only rank 0 output visible
    
    if is_active:
        save_results(stats)    # Only rank 0 executes (stats is not None)

# Nested contexts (advanced)
with uw.mpi.rank_conditional(0):
    data = collect_data()
    
    # Different rank for sub-operation
    with uw.mpi.rank_conditional(1):
        process_data(data)  # Now rank 1 is active

# Debugging - see all output
with uw.mpi.rank_conditional(0, suppress_stdout=False):
    stats = var.stats()
    print(f"Rank {uw.mpi.rank}: {stats}")  # All ranks print
```

## Summary

**stdout suppression** (Option 4 hybrid approach) is the best solution because:

1. ✅ Natural `print()` statements work automatically
2. ✅ Explicit `if is_active:` available when needed  
3. ✅ Compatible with `@collective_operation` decorator
4. ✅ Debug mode available
5. ✅ No magic - clear what's happening

The suppression is **per-context**, so it's easy to understand and control.