# Timing Decorator Coverage Analysis

## Current Status (2025-11-16)

### Existing Decorators: MINIMAL (2 decorators in timing.py only)

**Files with decorators:**
- `src/underworld3/timing.py` - 2 decorators (examples/tests only)

**Coverage**: ~0% - No production code decorated

---

## Recommended Decorator Coverage

### Priority 1: CRITICAL Performance Paths (Must Have)

#### 1. Function Evaluation (`src/underworld3/function/functions_unit_system.py`)
**Why**: Previously identified as 99.5% of bottleneck (now optimized with caching)
**Functions to decorate:**
- `evaluate()` - Main evaluation function
- `global_evaluate()` - Global evaluation across processes

**Expected benefit**: Track if caching is working, identify remaining bottlenecks

#### 2. Solver Operations (`src/underworld3/systems/solvers.py`)
**Why**: Core computational work - users want to know solve times
**Classes and methods to decorate:**

**Poisson Solver (SNES_Poisson):**
- `solve()` - Main solve operation

**Darcy Solver (SNES_Darcy):**
- `solve()` - Main solve operation

**Stokes Solver (SNES_Stokes):**
- `solve()` - Main solve operation
- `_setup_pointwise_functions()` - Setup overhead
- `_setup_problem_description()` - Problem setup

**Projection (SNES_Projection, SNES_Vector_Projection):**
- `solve()` - Projection solve

**Expected benefit**: Understand solver performance, identify setup vs solve time

#### 3. Mesh Creation (`src/underworld3/meshing/`)
**Why**: Mesh creation can be expensive, especially for complex geometries
**Functions to decorate:**

**Cartesian meshes (`cartesian.py`):**
- `StructuredQuadBox.__init__()`
- `UnstructuredSimplexBox.__init__()`

**Spherical meshes (`spherical.py`):**
- `SphericalShell.__init__()`

**Annulus meshes (`annulus.py`):**
- `Annulus.__init__()`

**Expected benefit**: Track mesh creation overhead, especially for large/complex meshes

#### 4. Mesh Variable Operations (`src/underworld3/discretisation/discretisation_mesh_variables.py`)
**Why**: Variable creation and data operations are common
**Functions to decorate:**
- `_MeshVariable.__init__()` - Variable creation
- `_MeshVariable._update_lvec()` - Data synchronization (if not already tracked by PETSc)

**Expected benefit**: Understand variable creation cost, data sync overhead

---

### Priority 2: USEFUL Performance Insights (Nice to Have)

#### 5. Swarm Operations (`src/underworld3/swarm.py`)
**Why**: Particle operations can be expensive
**Functions to decorate:**
- `Swarm.advection()` - Particle advection
- `Swarm.populate()` - Swarm population
- `SwarmVariable._update()` - Proxy variable updates (RBF interpolation)

**Expected benefit**: Track particle advection cost, RBF overhead

#### 6. DMInterpolation Cache (`src/underworld3/function/dminterpolation_cache.py`)
**Why**: Already optimized - validate cache is working
**Functions to decorate:**
- `DMInterpolationCache.get_structure()` - Cache lookup
- `CachedDMInterpolationInfo.create_structure()` - Cache miss (expensive)
- `CachedDMInterpolationInfo.evaluate()` - Cache hit (cheap)

**Expected benefit**: Confirm caching effectiveness, measure hit/miss costs

---

### Priority 3: OPTIONAL Deep Profiling (Advanced Users)

#### 7. Constitutive Models (`src/underworld3/constitutive_models/`)
**Why**: Material behavior calculations can be complex
**Note**: Use `uw.timing.add_timing_to_module()` for automatic decoration

**Expected benefit**: Detailed stress/strain calculation timing for model development

#### 8. Integration/Assembly (`src/underworld3/cython/`)
**Why**: Cython-level operations are low-level but important
**Note**: May require Cython decorator support

**Expected benefit**: Low-level performance profiling for developers

---

## Implementation Strategy

### Phase 1: Add Decorators to Key Functions (Quick Win)
**Target files:**
1. `src/underworld3/function/functions_unit_system.py` - evaluate()
2. `src/underworld3/systems/solvers.py` - all solve() methods
3. `src/underworld3/meshing/*.py` - mesh __init__ methods

**Estimated effort**: 30 minutes
**Expected benefit**: 80% of user-visible performance insights

### Phase 2: Add Decorators to Secondary Functions (Completeness)
**Target files:**
4. `src/underworld3/discretisation/discretisation_mesh_variables.py`
5. `src/underworld3/swarm.py`
6. `src/underworld3/function/dminterpolation_cache.py`

**Estimated effort**: 30 minutes
**Expected benefit**: Complete picture of UW3 performance

### Phase 3: Deep Profiling with Module Decoration (Advanced)
**Strategy**: Use `uw.timing.add_timing_to_module(uw.constitutive_models)`
**Target**: Constitutive models, advanced operations
**Estimated effort**: 10 minutes (just add module decoration calls)
**Expected benefit**: Detailed profiling for model/solver developers

---

## Example: Adding Decorators to Solvers

### Before (no timing)
```python
# src/underworld3/systems/solvers.py
class SNES_Poisson(SNES_Scalar):
    def solve(self, zero_init_guess=True, _force_setup=False):
        # ... solve implementation ...
        return self
```

### After (with timing)
```python
# src/underworld3/systems/solvers.py
import underworld3 as uw

class SNES_Poisson(SNES_Scalar):
    @uw.timing.routine_timer_decorator
    def solve(self, zero_init_guess=True, _force_setup=False):
        # ... solve implementation ...
        return self
```

**Result**: PETSc log will show "SNES_Poisson.solve" with:
- Call count
- Total time
- Average time per call
- Memory usage
- Flops (if applicable)

---

## Testing Strategy

### Validation Test
Create `test_timing_coverage.py`:
```python
import underworld3 as uw
import numpy as np

uw.timing.start()

# Test decorated operations
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Solver
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.f = 1.0
poisson.solve()

# Evaluation
coords = np.random.random((100, 2))
result = uw.function.evaluate(T.sym, coords, rbf=False)

# View results
uw.timing.print_table()
```

**Expected output**: Should show decorated functions in PETSc log with timing data

---

## Benefits

### User Benefits
1. **Identify bottlenecks**: See exactly where time is spent
2. **Optimize workflows**: Focus effort on expensive operations
3. **Track performance changes**: Compare timing before/after code changes
4. **Validate caching**: Confirm optimizations are working (e.g., DMInterpolation cache)

### Developer Benefits
1. **Performance regression detection**: CI can track timing changes
2. **Optimization guidance**: Data-driven decisions about what to optimize
3. **Comprehensive profiling**: PETSc captures ~95% of computational work
4. **Low overhead**: PETSc events are lightweight (~0.1% overhead)

---

## Current Gaps

**Missing coverage:**
- Solver solve() methods - **CRITICAL GAP**
- Function evaluate() - **CRITICAL GAP**
- Mesh creation - **MAJOR GAP**
- Variable operations - **MODERATE GAP**
- Swarm operations - **MODERATE GAP**

**Recommendation**: Implement Phase 1 (30 minutes) to close CRITICAL and MAJOR gaps.
