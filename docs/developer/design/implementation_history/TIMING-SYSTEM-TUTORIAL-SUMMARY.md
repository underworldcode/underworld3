# Underworld3 Timing System Tutorial - Summary

**Location**: `docs/examples/Tutorial_Timing_System.ipynb`

## Quick Start

```python
import underworld3 as uw

# 1. Enable timing (once at start - no environment variables needed!)
uw.timing.start()

# 2. Run your simulation
mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.05)
# ... do your work ...

# 3. View results - clean UW3-focused summary
uw.timing.print_summary()
```

## Key Features Demonstrated

### 1. User-Friendly Summary (New!)
- **`uw.timing.print_summary()`** - Shows only UW3 operations
- Filters out hundreds of low-level PETSc events
- Perfect for quick performance checks
- Customizable sorting and filtering

### 2. Detailed Profiling
- **`uw.timing.print_table()`** - Full PETSc profiling data
- Comprehensive view of all operations
- Use for deep performance analysis

### 3. Programmatic Access
- **`uw.timing.get_summary()`** - Returns dict with timing data
- Integrate timing into your analysis workflows
- Build custom performance dashboards

### 4. Practical Examples
- Poisson equation solver
- Time-stepping loop
- Function evaluation
- Real-world performance optimization

## Tutorial Structure

The notebook covers:

1. **Basic Usage** - How to enable and use timing
2. **Example Workflow** - Poisson equation setup and solve
3. **User-Friendly Summary** - Clean, filtered view
4. **Sorting & Filtering** - Customize the output
5. **Programmatic Access** - Use timing data in code
6. **Full PETSc Details** - When you need deep profiling
7. **All Events View** - Alternative to full table
8. **Time-Stepping Example** - Real simulation timing
9. **Saving Results** - Export to CSV/text files
10. **Summary & Tips** - Quick reference guide

## Comparison: Before vs After

### Before (Old System)
```python
# Required environment variable
import os
os.environ['UW_TIMING_ENABLE'] = '1'

import underworld3 as uw
uw.timing.start()

# ... work ...

# Output: Overwhelming mix of UW3 + PETSc events
uw.timing.print_table()  # 100+ lines of mixed information
```

### After (New System)
```python
# No environment variable needed!
import underworld3 as uw
uw.timing.start()  # Works immediately in Jupyter

# ... work ...

# Output: Clean UW3-focused view
uw.timing.print_summary()  # ~10 lines of relevant information

# Still available when needed:
uw.timing.print_table()  # Full PETSc details
```

## Example Output

### User-Friendly Summary
```
====================================================================================================
UNDERWORLD3 TIMING SUMMARY (UW3 Operations Only)
====================================================================================================
Total time: 1.234 seconds
Showing 7 of 7 events (min time: 1.0ms)
====================================================================================================
Event Name                                            Count     Time (s)    % Total
----------------------------------------------------------------------------------------------------
Poisson.solve                                             1     0.856234      69.4%
UnstructuredSimplexBox                                    1     0.234156      19.0%
evaluate                                                 10     0.089234       7.2%
Mesh.__init__                                             1     0.034567       2.8%
...
====================================================================================================

💡 Tip: Use uw.timing.print_summary(filter_uw=False) to see all PETSc events
    Use uw.timing.print_table() for full PETSc profiling details
```

Clean, focused, actionable!

## Usage Patterns

### Quick Performance Check
```python
uw.timing.start()
# ... run simulation ...
uw.timing.print_summary()  # See where time is spent
```

### Find Frequently Called Operations
```python
uw.timing.print_summary(sort_by='count', max_events=10)
# Identify optimization opportunities
```

### Deep Profiling
```python
uw.timing.print_summary(filter_uw=False, min_time=0.001)
# See both UW3 and PETSc operations > 1ms
```

### Export for Analysis
```python
uw.timing.print_table("results.csv")
# Analyze in Excel or with pandas
```

## Benefits

### For Users
- ✅ **No setup required** - works immediately in Jupyter
- ✅ **Clean output** - see only what matters
- ✅ **Actionable insights** - identify bottlenecks quickly
- ✅ **Still comprehensive** - full PETSc data when needed

### For Developers
- ✅ **Easy to use** - `uw.timing.start()` and you're done
- ✅ **Extensible** - add custom events easily
- ✅ **Well-documented** - complete tutorial notebook
- ✅ **Integrated** - unified UW3 + PETSc timing

## Implementation Details

### What Gets Timed

**Automatically tracked:**
- All decorated UW3 operations (mesh creation, solvers, evaluation, etc.)
- All PETSc operations (matrix ops, vector ops, solver internals)
- Memory usage and FLOP counts
- MPI communication (in parallel runs)

**Phase 1 decorator coverage (completed):**
- ✅ `evaluate()` and `global_evaluate()`
- ✅ `solve()` methods
- ✅ Mesh creation

**Future phases:**
- Mesh variable operations
- Swarm operations
- Caching operations

### Filtering Logic

`print_summary()` filters events using regex patterns:
- `Function.*` - Function evaluation operations
- `Mesh.*` - Mesh operations
- `*Solver.*` - Solver operations
- Custom decorated operations

This removes ~100 low-level PETSc events while keeping ~10 relevant UW3 operations.

## Tips for Best Results

1. **Start timing early** - Call `uw.timing.start()` at the beginning
2. **Use summary first** - `print_summary()` for quick checks
3. **Sort by count** - Find operations called many times
4. **Filter by time** - Use `min_time` to ignore trivial operations
5. **Save for comparison** - Export CSV to compare across runs
6. **Full table for debugging** - Use `print_table()` only when needed

## Related Documentation

- **Review Document**: `docs/reviews/2025-11/TIMING-SYSTEM-REFACTOR-REVIEW.md`
- **Implementation**: `src/underworld3/timing.py`
- **PETSc Documentation**: https://petsc.org/release/manual/profiling/

---

**The tutorial notebook provides a complete, hands-on guide with working examples!**

Run it to see the timing system in action with real UW3 code.
