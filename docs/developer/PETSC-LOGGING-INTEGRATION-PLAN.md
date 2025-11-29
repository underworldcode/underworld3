# PETSc Logging Integration Plan

**Date**: 2025-11-16
**Status**: Planning Phase
**Related**: Current UW3 timing system (`src/underworld3/timing.py`)

## Executive Summary

Current UW3 timing system (`timing.py`) only captures ~15% of actual computational work (Python API calls). PETSc provides comprehensive built-in logging that captures ~95% of work including all solver operations, MPI communication, and memory usage.

**Recommendation**: Integrate PETSc logging as primary timing mechanism, keep current UW3 timing for Python-layer overhead.

## Current UW3 Timing Limitations

### What It Captures (~15% of work)
- Python API calls: `stokes.solve()`, `projection.solve()`
- Mesh creation time
- Variable initialization
- Python-layer overhead

### What It MISSES (~85% of work)
- **All PETSc operations**: MatMult, KSPSolve, VecNorm, PCApply
- **MPI communication**: All message passing, synchronization
- **Memory operations**: Allocations, copies, ghost updates
- **Assembly time**: Jacobian/residual construction
- **Preconditioner setup**: ILU, AMG, etc.

### Example: Stokes Solve Timing Breakdown
```
UW3 timing shows:
  stokes.solve() → 10.5s (entire solve)

PETSc would show:
  MatMult         → 3.2s (30%)    # Matrix-vector products
  PCApply         → 4.1s (39%)    # Preconditioner application
  VecNorm         → 0.8s (8%)     # Vector norms
  VecAXPY         → 0.9s (9%)     # Vector operations
  KSPSolve        → 10.1s (96%)   # Total SNES/KSP time
  SNESJacobianEval → 0.4s (4%)   # Jacobian assembly
```

**Insight**: UW3 timing tells you solve took 10.5s, PETSc tells you WHY it took 10.5s.

## PETSc Logging Capabilities

### 1. Automatic Performance Profiling
PETSc instruments ALL operations automatically once enabled:

- **Time per operation**: MatMult, KSPSolve, VecNorm, PCApply, etc.
- **Flop counting**: Floating point operations performed
- **Memory tracking**: Peak memory, allocation counts, current usage
- **Call counts**: How many times each operation was called
- **Per-stage breakdown**: Setup vs Solve vs Postprocess

**No code changes required** - just enable logging and run.

### 2. MPI-Aware Statistics
PETSc provides parallel statistics automatically:

- **Max/Min/Avg** across all ranks
- **Load imbalance detection**: Max/Min ratio (>1.2 indicates imbalance)
- **Communication costs**: MPI message counts and sizes
- **Synchronization overhead**: Time spent in MPI barriers
- **Per-rank breakdown**: Identify slow ranks

**Example Output**:
```
Event                Count      Time (sec)     Flops/sec      --- MPI Stats ---
                              Max    Ratio   Max     Ratio   Max/Min   Messages
MatMult              1000    3.21e+00  1.15  2.1e+09  1.08      1.12    1.2e+05
  → Rank 0: 3.21s
  → Rank 7: 2.79s  ← 15% slower (load imbalance!)
```

### 3. Custom Instrumentation
Users can add custom timing to their own code:

**Stages**: Group related operations
```python
from petsc4py import PETSc

stage = PETSc.Log.Stage('initialization')
stage.push()
# ... initialization code ...
stage.pop()
```

**Events**: Time specific code blocks
```python
event = PETSc.Log.Event('custom_computation')
event.begin()
# ... custom code ...
event.end()
```

**Manual Flop Logging**:
```python
PETSc.Log.logFlops(num_flops)  # Track custom operations
```

### 4. Export Formats
PETSc supports multiple output formats:

- **ASCII table** (default, human-readable)
- **CSV**: `-log_view :file.csv:ascii_csv`
- **Python dict**: `-log_view :file.py:ascii_python`
- **JSON**: Via custom parsing or PETSc 3.21+

## Integration Approach

### Phase 1: Minimal Viable Integration (HIGH VALUE, LOW EFFORT)

**Goal**: Enable PETSc logging with minimal UW3 code changes

**Implementation**:
```python
# In src/underworld3/timing.py (add new functions)

def enable_petsc_logging():
    """Enable PETSc performance logging.

    Captures all PETSc operations (solve, assembly, MPI, memory).
    Much more comprehensive than UW3 timing alone.

    Usage:
        import underworld3 as uw
        uw.timing.enable_petsc_logging()

        # ... run simulation ...

        uw.timing.print_petsc_log()  # Print to console
        uw.timing.print_petsc_log("timing.txt")  # Save to file
    """
    from petsc4py import PETSc
    PETSc.Log.begin()


def print_petsc_log(filename=None):
    """Display PETSc performance summary.

    Parameters
    ----------
    filename : str, optional
        If provided, write log to file. Otherwise print to console.

    Examples
    --------
    >>> uw.timing.print_petsc_log()  # Print to console
    >>> uw.timing.print_petsc_log("performance.txt")  # Save to file
    >>> uw.timing.print_petsc_log("perf.csv")  # CSV format (if .csv extension)
    """
    from petsc4py import PETSc

    if filename:
        if filename.endswith('.csv'):
            # CSV format for spreadsheet analysis
            viewer = PETSc.Viewer().createASCII(filename, 'w')
            viewer.pushFormat(PETSc.Viewer.Format.ASCII_CSV)
        else:
            # Standard ASCII table
            viewer = PETSc.Viewer().createASCII(filename, 'w')
        PETSc.Log.view(viewer)
        viewer.destroy()
    else:
        # Print to console
        PETSc.Log.view()
```

**Estimated Effort**: 2 hours (add functions, test, document)

**Benefits**:
- ✅ Captures 95% of computational work (vs 15% currently)
- ✅ Zero user code changes (opt-in via API call)
- ✅ Works immediately with existing code
- ✅ MPI-aware statistics built-in
- ✅ Export to CSV for analysis

### Phase 2: Context Manager Integration (MEDIUM VALUE, MEDIUM EFFORT)

**Goal**: Pythonic API with automatic stage management

**Implementation**:
```python
# In src/underworld3/timing.py

from contextlib import contextmanager

class PETScTiming:
    """Wrapper around PETSc logging with UW3 integration."""

    def __init__(self):
        self._enabled = False
        self._stages = {}
        self._events = {}

    def enable(self):
        """Enable PETSc logging."""
        from petsc4py import PETSc
        PETSc.Log.begin()
        self._enabled = True

    def create_stage(self, name):
        """Create a named stage for grouping operations."""
        from petsc4py import PETSc
        if name not in self._stages:
            self._stages[name] = PETSc.Log.Stage(name)
        return self._stages[name]

    def create_event(self, name):
        """Create a named event for timing specific operations."""
        from petsc4py import PETSc
        if name not in self._events:
            self._events[name] = PETSc.Log.Event(name)
        return self._events[name]

    @contextmanager
    def stage(self, name):
        """Context manager for PETSc stages.

        Examples
        --------
        >>> with uw.petsc_timing.stage("initialization"):
        ...     mesh = uw.meshing.StructuredQuadBox(...)
        ...     solver = uw.systems.Stokes(...)
        """
        stage = self.create_stage(name)
        stage.push()
        try:
            yield stage
        finally:
            stage.pop()

    @contextmanager
    def event(self, name):
        """Context manager for PETSc events.

        Examples
        --------
        >>> with uw.petsc_timing.event("custom_assembly"):
        ...     # ... custom assembly code ...
        """
        event = self.create_event(name)
        event.begin()
        try:
            yield event
        finally:
            event.end()

    def view(self, filename=None):
        """Display PETSc performance summary."""
        from petsc4py import PETSc
        if filename:
            viewer = PETSc.Viewer().createASCII(filename)
            PETSc.Log.view(viewer)
            viewer.destroy()
        else:
            PETSc.Log.view()

# Create global instance
petsc_timing = PETScTiming()
```

**Usage Example**:
```python
import underworld3 as uw

uw.petsc_timing.enable()

with uw.petsc_timing.stage("initialization"):
    mesh = uw.meshing.StructuredQuadBox(...)
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    stokes = uw.systems.Stokes(mesh, ...)

with uw.petsc_timing.stage("solve"):
    with uw.petsc_timing.event("stokes_solve"):
        stokes.solve()

    with uw.petsc_timing.event("projection"):
        proj = uw.systems.Projection(mesh, gradT, ...)
        proj.solve()

uw.petsc_timing.view("performance.txt")
```

**Estimated Effort**: 1 day (class implementation, integration, testing, docs)

### Phase 3: Automatic UW3 Stage Integration (HIGH VALUE, HIGH EFFORT)

**Goal**: Automatically instrument UW3 operations with PETSc stages

**Implementation**: Modify solver classes to use PETSc stages internally:

```python
# In src/underworld3/systems/solvers.py

class Stokes(Solver):
    def solve(self, zero_init_guess=True, _force_setup=False):
        """Solve the Stokes system with automatic timing."""
        from petsc4py import PETSc

        # Create stage if PETSc logging enabled
        if PETSc.Log.isActive():
            stage = PETSc.Log.Stage(f"Stokes_{self.u_Field.name}")
            stage.push()

        try:
            # ... existing solve implementation ...
            return super().solve(zero_init_guess, _force_setup)
        finally:
            if PETSc.Log.isActive():
                stage.pop()
```

**Benefits**:
- ✅ Users get detailed timing WITHOUT any code changes
- ✅ Automatic stage names (Stokes_u, Projection_gradT, etc.)
- ✅ Works across all solvers (Stokes, AdvDiff, Poisson, etc.)

**Estimated Effort**: 2-3 days (modify all solvers, test, document)

### Phase 4: Combined UW3 + PETSc Reporting (LOW PRIORITY)

**Goal**: Unified timing report combining both systems

**Example Output**:
```
=== UW3 Performance Summary ===

Python Layer (UW3 timing):
  Mesh creation        → 0.12s
  Variable init        → 0.05s
  Solver setup         → 0.31s

Computational Work (PETSc timing):
  Stokes solve         → 10.5s
    ├─ MatMult         → 3.2s (30%)
    ├─ PCApply         → 4.1s (39%)
    ├─ VecNorm         → 0.8s (8%)
    └─ Other           → 2.4s (23%)

MPI Statistics:
  Load balance (max/min) → 1.15 (15% imbalance)
  Communication overhead → 5% of total time

Total runtime: 11.0s (95% in PETSc operations)
```

**Estimated Effort**: 2-3 days (parsing, formatting, testing)

## Comparison: Current vs Proposed

| Feature | Current UW3 Timing | PETSc Logging | Combined |
|---------|-------------------|---------------|----------|
| **Coverage** | ~15% | ~95% | ~100% |
| **Python overhead** | ✅ Yes | ❌ No | ✅ Yes |
| **Solver internals** | ❌ No | ✅ Yes | ✅ Yes |
| **MPI statistics** | ❌ No | ✅ Yes | ✅ Yes |
| **Memory tracking** | ❌ No | ✅ Yes | ✅ Yes |
| **Flop counting** | ❌ No | ✅ Yes | ✅ Yes |
| **Custom stages** | ❌ No | ✅ Yes | ✅ Yes |
| **Export formats** | ❌ Text only | ✅ CSV/Python/JSON | ✅ Multiple |
| **Load balancing** | ❌ No | ✅ Yes | ✅ Yes |

## Worked Examples

See `examples/timing_petsc_integration.py` for complete worked example.

## Implementation Checklist

### Phase 1: Minimal Integration (RECOMMENDED START)
- [ ] Add `enable_petsc_logging()` to `timing.py`
- [ ] Add `print_petsc_log(filename=None)` to `timing.py`
- [ ] Test with simple Stokes problem
- [ ] Test with MPI (2, 4, 8 ranks)
- [ ] Test CSV export
- [ ] Add docstrings with examples
- [ ] Update user documentation (Notebook 14?)

### Phase 2: Context Managers (OPTIONAL)
- [ ] Implement `PETScTiming` class
- [ ] Add `stage()` context manager
- [ ] Add `event()` context manager
- [ ] Create `uw.petsc_timing` global instance
- [ ] Test with nested stages
- [ ] Document usage patterns

### Phase 3: Automatic Integration (FUTURE)
- [ ] Modify `Stokes.solve()` to use stages
- [ ] Modify `AdvDiff.solve()` to use stages
- [ ] Modify `Poisson.solve()` to use stages
- [ ] Modify `Projection.solve()` to use stages
- [ ] Test all solvers with logging
- [ ] Verify stage names are meaningful

### Phase 4: Combined Reporting (FUTURE)
- [ ] Parse PETSc log output
- [ ] Combine with UW3 timing data
- [ ] Create unified report format
- [ ] Add visualization (optional)

## Testing Strategy

### Unit Tests
```python
# tests/test_petsc_logging.py

def test_enable_petsc_logging():
    """Test that PETSc logging can be enabled."""
    import underworld3 as uw
    uw.timing.enable_petsc_logging()
    # Should not raise

def test_petsc_log_output():
    """Test that PETSc log can be printed."""
    import underworld3 as uw
    uw.timing.enable_petsc_logging()

    # Run simple problem
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4,4))
    u = uw.discretisation.MeshVariable("u", mesh, 1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.solve()

    # Print log (should not crash)
    uw.timing.print_petsc_log("/tmp/test_log.txt")

    # Verify file exists
    assert os.path.exists("/tmp/test_log.txt")
```

### Integration Tests
- Test with Stokes solver (complex)
- Test with MPI (verify parallel statistics)
- Test with custom stages
- Test CSV export format

## References

- **PETSc Documentation**: https://petsc.org/release/manual/profiling/
- **petsc4py Logging**: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Log.html
- **Current UW3 Timing**: `src/underworld3/timing.py`
- **Worked Example**: `examples/timing_petsc_integration.py`

## Decision Log

**2025-11-16**: Initial planning document created
- Identified current timing gaps (~85% of work not captured)
- Proposed 4-phase integration approach
- Recommended Phase 1 as minimal viable integration (2 hours effort)

## Next Steps

1. **Review** this document with UW3 team
2. **Decide** on implementation phases (recommend Phase 1 only)
3. **Implement** Phase 1 (2 hours)
4. **Test** with example notebooks
5. **Document** in user guide (create Notebook 14: Performance Profiling)
6. **Gather feedback** before proceeding to Phase 2+
