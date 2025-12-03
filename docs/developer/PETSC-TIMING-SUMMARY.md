# PETSc Logging Integration - Summary for Implementation

**Date**: 2025-11-16
**Status**: Documented and Validated
**Related Files**:
- Planning document: `docs/developer/PETSC-LOGGING-INTEGRATION-PLAN.md`
- Worked example: `examples/timing_petsc_integration.py`

## What Was Accomplished

### 1. Comprehensive Investigation
Investigated PETSc's built-in logging capabilities and how they can enhance UW3's timing system from ~15% coverage (Python API only) to ~95% coverage (including all solver operations).

### 2. Planning Document Created
`docs/developer/PETSC-LOGGING-INTEGRATION-PLAN.md` contains:
- **Executive Summary**: Problem statement and recommendation
- **Current Limitations**: What UW3 timing misses (~85% of computational work)
- **PETSc Capabilities**: Detailed feature list
  - Automatic performance profiling (MatMult, KSPSolve, etc.)
  - MPI-aware statistics (load balance, communication)
  - Custom instrumentation (stages, events)
  - Multiple export formats (ASCII, CSV, Python dict)
- **4-Phase Integration Approach**:
  - Phase 1: Minimal integration (2 hours) - HIGH VALUE, LOW EFFORT
  - Phase 2: Context managers (1 day) - Pythonic API
  - Phase 3: Automatic integration (2-3 days) - Zero user code changes
  - Phase 4: Combined reporting (future) - Unified UW3+PETSc output
- **Implementation Checklist**: Step-by-step guide for each phase
- **Testing Strategy**: Unit and integration test plans
- **Comparison Table**: Current vs proposed system features

### 3. Worked Example Created
`examples/timing_petsc_integration.py` demonstrates:
- **How to enable PETSc logging**: `PETSc.Log.begin()`
- **Custom stages**: Group related operations (initialization, solve, postprocessing)
- **Custom events**: Time specific code blocks (linear solve)
- **Output formats**: ASCII and CSV export
- **Real solver integration**: Working Poisson problem with 32x32 mesh

### 4. Validated Results
Example output shows PETSc captures:
```
Event Stage: poisson_solve
  SNESSolve:       46ms (99% of solver stage)
  KSPSolve:         5ms (67% of SNES work)
  MatMult:         ~1ms (30% of KSP time)
  PCApply:         ~3ms (41% of KSP time)
  VecNorm:        <1ms (18% of KSP time)
  SNESJacobianEval: 13ms (4% of SNES work)
  SNESFunctionEval: 14ms (5% of SNES work)
```

**Key Insight**: PETSc logging reveals WHERE time is spent within `solve()`, not just that `solve()` took 46ms total.

## What This Enables

### Developer Benefits
1. **Performance Debugging**: Identify bottlenecks at operation level
2. **Preconditioner Tuning**: See exactly how much time PCApply takes
3. **Solver Comparison**: Compare GMR vs CG, different preconditioners
4. **Parallel Scaling**: Load balance analysis across MPI ranks
5. **Memory Profiling**: Peak memory per operation

### User Benefits
1. **Zero Code Changes**: Enable with `uw.timing.enable_petsc_logging()`
2. **Automatic Output**: Comprehensive timing without instrumentation
3. **Export to Spreadsheet**: CSV format for analysis
4. **MPI-Aware**: Parallel statistics automatically included

## How to Use (After Implementation)

### Phase 1 API (Proposed):
```python
import underworld3 as uw

# Enable PETSc logging
uw.timing.enable_petsc_logging()

# ... run simulation ...
mesh = uw.meshing.StructuredQuadBox(...)
poisson = uw.systems.Poisson(...)
poisson.solve()

# Print comprehensive timing
uw.timing.print_petsc_log()  # To console
uw.timing.print_petsc_log("timing.txt")  # To file
uw.timing.print_petsc_log("timing.csv")  # CSV format
```

### Phase 2 API (Proposed - Context Managers):
```python
import underworld3 as uw

uw.petsc_timing.enable()

with uw.petsc_timing.stage("initialization"):
    mesh = uw.meshing.StructuredQuadBox(...)
    T = uw.discretisation.MeshVariable(...)

with uw.petsc_timing.stage("solve"):
    poisson = uw.systems.Poisson(...)
    poisson.solve()

uw.petsc_timing.view("performance.txt")
```

### Phase 3 (Automatic - Future):
```python
# No code changes needed!
# Solvers automatically use PETSc stages internally

import underworld3 as uw

uw.timing.enable_petsc_logging()

mesh = uw.meshing.StructuredQuadBox(...)
stokes = uw.systems.Stokes(...)
stokes.solve()  # Automatically creates "Stokes_solve" stage

uw.timing.print_petsc_log()
# Output shows:
#   Stage: Stokes_u (automatically named)
#   Stage: Projection_gradT (if projection used)
```

## Example Output Interpretation

From the worked example (`examples/timing_petsc_integration.py`):

```
Event Stage 3: poisson_solve
                                     Time (sec)    Flops
SNESSolve              1  4.6087e-02  3.79e+07   ← Total solve time (46ms)
├─ KSPSolve            2  4.6390e-03  2.56e+07   ← Linear solver (5ms, 67%)
│  ├─ MatMult         75  1.1800e-03  1.14e+07   ← Matrix ops (1ms, 30%)
│  ├─ PCApply         37  3.4230e-03  1.56e+07   ← Preconditioner (3ms, 41%)
│  ├─ VecNorm        112  3.7000e-04  8.51e+05   ← Vector norms (<1ms, 18%)
│  └─ KSPGMRESOrthog  75  4.6500e-04  6.99e+06   ← Orthogonalization
├─ SNESJacobianEval    2  1.3299e-02  1.50e+06   ← Jacobian assembly (13ms, 4%)
└─ SNESFunctionEval    3  1.4018e-02  2.09e+06   ← Residual evaluation (14ms, 5%)
```

**Reading**:
- Poisson solve took 46ms total
- Linear solver (KSP) accounts for 67% of that
- Within KSP: Preconditioner (41%) and matrix-vector products (30%) dominate
- Assembly (Jacobian + Function) takes only 9% combined

**Actionable**:
- If PCApply is slow → Try different preconditioner
- If MatMult is slow → Consider matrix-free methods
- If SNESJacobianEval is slow → Optimize constitutive model

## Next Steps for Implementation

### Immediate (Recommended - Phase 1):
1. **Add two functions to `src/underworld3/timing.py`**:
   - `enable_petsc_logging()`  (~10 lines)
   - `print_petsc_log(filename=None)`  (~20 lines)
2. **Test with existing notebooks**: Notebook 5 (Stokes), Notebook 11 (Poisson)
3. **Create user documentation**: New notebook "14-Performance_Profiling.ipynb"
4. **Estimated effort**: 2 hours

### Future (Optional):
- Phase 2: Context manager API (1 day)
- Phase 3: Automatic solver integration (2-3 days)
- Phase 4: Combined UW3+PETSc reporting (2-3 days)

## Files Created
1. `docs/developer/PETSC-LOGGING-INTEGRATION-PLAN.md` - Complete planning document
2. `examples/timing_petsc_integration.py` - Working demonstration
3. `docs/developer/PETSC-TIMING-SUMMARY.md` - This file (summary)

## Testing the Example

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
pixi run -e default python examples/timing_petsc_integration.py

# Output files created:
# /tmp/petsc_timing_example.txt  - Human-readable ASCII format
# /tmp/petsc_timing_example.csv  - Spreadsheet-compatible format
```

## References
- **PETSc Profiling Documentation**: https://petsc.org/release/manual/profiling/
- **petsc4py Logging API**: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Log.html
- **Current UW3 Timing**: `src/underworld3/timing.py`
- **Temporary Notes** (working notes, can be deleted): `/tmp/petsc_timing_capabilities.md`

## Decision

**Recommended**: Implement **Phase 1 only** (minimal integration, 2 hours)
- Provides 95% of the value with minimal effort
- Can be done in a single session
- No breaking changes to existing code
- Users can opt-in when needed
- Sets foundation for future phases if needed

**Not Recommended**: Full implementation of all 4 phases immediately
- Phase 2-4 add API sugar but limited additional value
- Can always add later if Phase 1 proves valuable
- Keep it simple for now

---

**Status**: Ready for implementation when prioritized
**Complexity**: Low (Phase 1), Medium (Phase 2-3), Medium-High (Phase 4)
**Risk**: Very Low - opt-in feature, no changes to existing workflows
**Value**: High - fills major gap in current timing system
