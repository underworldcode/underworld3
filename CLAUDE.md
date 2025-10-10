# Underworld3 Data Access Migration

> **Note**: This file provides context for AI assistants working on the codebase. Human-readable developer documentation is being migrated to `docs/developer/` in Quarto format. See `docs/developer/subsystems/data-access.qmd` and `docs/developer/subsystems/model-orchestration.qmd` for the current documentation.

## CRITICAL BUILD CONSTRAINTS

### PETSc Directory Location (DO NOT MOVE)
**WARNING**: The `petsc/` directory in `/Users/lmoresi/+Underworld/underworld-pixi-2/petsc/` MUST NOT be moved or relocated.
- PETSc is NOT relocatable after compilation
- The build contains hardcoded paths that cannot be changed
- Moving the directory will break petsc4py bindings
- All pixi tasks depend on this fixed location
- The PETSC_DIR environment variable points to this specific path
- **Future Plan**: Convert to pixi package installed in underworld3 directory (pending ~1hr build time solution)

### underworld3-documentation Status
**Migration in Progress**: The `underworld3-documentation` directory contains legacy documentation being migrated.
- **Already migrated**: Constitutive models theory documentation
- **To migrate**: Solver documentation, key example notebooks, benchmarks
- **See**: `MIGRATION_FROM_UW3_DOCUMENTATION.md` for detailed status
- **Goal**: Single source of truth in underworld3 repository

### CRITICAL REBUILD REQUIREMENT ⚠️
**After modifying source files, always run `pixi run underworld-build` to see changes!**
- **Why**: Underworld3 is installed as a package in the pixi environment
- **Location**: Changes go to `/Users/lmoresi/+Underworld/underworld-pixi-2/.pixi/envs/default/lib/python3.12/site-packages/underworld3/`
- **Build tasks**: `pixi run underworld-build` rebuilds and reinstalls the package
- **Verification**: Check `uw.model.__file__` to confirm source location

### TEST QUALITY AND VALIDATION PRINCIPLES ⚠️
**CRITICAL**: New tests must be validated before making code changes to fix them!
- **Validate test correctness**: Ensure new tests are properly structured and test real functionality
- **Independent verification**: Before changing main code, verify there's actually a problem with the core system
- **Test isolation**: New regression tests should not drive changes to working core functionality
- **Core system priority**: If core tests (0000-0599) pass, the system is working correctly
- **Example failure (2025-10-09)**: test_06*_regression.py tests were incorrectly structured, causing `AttributeError: 'MutableDenseMatrix' object has no attribute 'u'` in constitutive models. Core system was working fine - the new tests had wrong assumptions about API usage.
- **Resolution approach**: Disable problematic new tests, validate core functionality, then fix test structure rather than changing working code

### JOSS Paper - Publication of Record
**Location**: `publications/joss-paper/` (moved from `docs/joss-paper/`)
- **Status**: FROZEN - Publication of record, preserve as-is
- **DO NOT UPDATE**: This directory should not be modified even if code changes significantly
- **Content**: Journal of Open Source Software publication source (paper.md, paper.pdf, paper.bib)
- **Useful for documentation**: Contains high-level overview, design rationale, key features explanation
- **Mining potential**: Executive summaries, architecture descriptions, comparison with other tools

### Planning Documents
**Location**: `planning/`
- **Purpose**: Architecture plans, design documents, feature roadmaps
- **Status**: Mix of historical plans and current/future designs

#### Feature Plans (Historical - verify against implementation)
- `parameter_system_plan.md` - Parameter system design (note: current implementation differs)
- `material_properties_plan.md` - Material properties architecture
- `mathematical_objects_plan.md` - Mathematical objects design (✅ IMPLEMENTED)
- `claude_examples_plan.md` - Example usage patterns
- `units_system_plan.md` - Units and dimensional analysis system
- `MultiMaterial_ConstitutiveModel_Plan.md` - Multi-material constitutive models

#### Parallel Safety System (✅ IMPLEMENTED 2025-01-24)
- `PARALLEL_PRINT_SIMPLIFIED.md` - **Main design**: `uw.pprint()` and `selective_ranks()` (✅ **IMPLEMENTED**)
- `RANK_SELECTION_SPECIFICATION.md` - Comprehensive rank selection syntax (✅ **IMPLEMENTED**)
- `COLLECTIVE_OPERATIONS_CLASSIFICATION.md` - Classification of collective vs local operations
- `AUTOMATIC_COLLECTIVE_DETECTION.md` - How to detect collective ops automatically (future enhancement)
- `CODEBASE_COLLECTIVE_ANALYSIS.md` - Using existing code patterns to identify operations
- `PARALLEL_SAFETY_ANALYSIS.md` - Analysis of current issues (resolved)
- `PARALLEL_SAFETY_DESIGN.md` - Early comprehensive design (superseded by simplified version)
- `RETURN_SUPPRESSION_MECHANISM.md` - Technical details of stdout suppression
- `PARALLEL_PRINTING_DESIGN.md` - Alternative design approach

**Status**: Core parallel safety system implemented and integrated. Codebase migrated to new patterns. See implementation in `src/underworld3/mpi.py` and documentation in `docs/advanced/parallel-computing.qmd`.

**Documentation Strategy**: Mine planning documents for important information to consolidate into developer guide (`docs/developer/`), then clean up planning directory to avoid repository clutter. Developer guide should serve dual purpose as implementation reference and code patterns guide.

## Project Context
Migrating Underworld3 from access context manager pattern to direct data access using NDArray_With_Callback for backward compatibility.

### Parallel Computing in Underworld3
**CRITICAL UNDERSTANDING**: Underworld3 rarely uses MPI directly - PETSc handles all parallel synchronization.

**Key Principles:**
- **PETSc manages parallelism**: All mesh operations, solvers, vector updates are inherently parallel via PETSc
- **Collective operations**: Many PETSc operations require all processes to participate (collective calls)
- **UW3 API is parallel-safe**: Use only UW3 wrapper functions which properly wrap PETSc collective operations
- **MPI usage is minimal**: The main use of `uw.mpi.rank` is for conditional logic (e.g., "only rank 0 prints output")
- **Avoid direct MPI**: Don't use mpi4py directly unless absolutely necessary - use UW3 API instead

**Writing Parallel-Safe Code:**
- Use UW3 mesh/variable/solver API - these handle collective operations correctly
- Be careful about rank-conditional code that might skip collective operations
- Understand which UW3 operations are collective (most mesh/solver operations)
- Don't introduce raw MPI barriers or communications unless you understand the PETSc context

**Common Pattern:**
```python
# CORRECT - using UW3 API (PETSc handles parallelism)
mesh.access(var)  # Collective operation via PETSc
var.data[...] = values

# CORRECT - conditional output only
if uw.mpi.rank == 0:
    print("Status message")  # Non-collective, safe

# INCORRECT - using raw MPI for what PETSc handles
from mpi4py import MPI
MPI.COMM_WORLD.barrier()  # Usually unnecessary, PETSc manages this
```

**PARALLEL SAFETY PATTERNS (New - Use These!):**
```python
# OLD PATTERN (deprecated)
if uw.mpi.rank == 0:
    print(f"Stats: {var.stats()}")  # DANGEROUS if stats() is collective!

# NEW PATTERN (safe) - Use uw.pprint()
uw.pprint(0, f"Stats: {var.stats()}")  # All ranks execute stats(), only rank 0 prints

# OLD PATTERN (deprecated)
if uw.mpi.rank == 0:
    import pyvista as pv
    plotter = pv.Plotter()
    # visualization code...

# NEW PATTERN (safe) - Use selective_ranks()
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import pyvista as pv
        plotter = pv.Plotter()
        # visualization code...
```

**Parallel Safety API (✅ IMPLEMENTED):**
- **`uw.pprint(ranks, *args, **kwargs)`** - Parallel-safe printing (all ranks evaluate args, selected ranks print)
- **`uw.selective_ranks(ranks)`** - Context manager for rank-specific execution
- **Rank selection**: Supports int, slice, list, str patterns ('all', 'first', 'even', etc.), functions, numpy arrays

**See**: 
- `src/underworld3/mpi.py` - **Implementation** of `pprint()` and `selective_ranks()`
- `docs/advanced/parallel-computing.qmd` - **User documentation** with comprehensive examples and migration guide
- `planning/PARALLEL_PRINT_SIMPLIFIED.md` - Original design document
- `planning/RANK_SELECTION_SPECIFICATION.md` - Complete rank selection syntax specification

## Architecture Priorities & Module Purposes

### Core Design Principles
1. **Solver Stability is Paramount**: The PETSc-based solvers (Stokes, advection-diffusion, etc.) are the core of the system. They have been carefully optimized, benchmarked, and validated over many years. Any changes must preserve their integrity.

2. **Conservative Migration Strategy**:
   - **User-facing code** (tests, examples): Use new `array` property with automatic sync
   - **Solver internals**: Keep using `vec` property with direct PETSc access
   - **Gradual transition**: Only make changes when driven by actual needs, not cleanup

### Module Purposes & Boundaries

#### Solvers (`underworld3.cython.petsc_generic_snes_solvers`)
- **Purpose**: High-performance numerical solving using PETSc
- **Access Pattern**: Direct PETSc vector access via `vec` property
- **Change Policy**: NO CHANGES without extensive benchmarking
- **Why**: These handle matrix assembly, preconditioners, field splitting, BC application, parallel ghost exchanges

#### Mesh Variables (`discretisation_mesh_variables.py`)
- **Purpose**: User-facing interface for field data
- **Access Pattern**: Transitioning to direct `array` property
- **Key Features**:
  - `array` property: NDArray_With_Callback for automatic PETSc sync
  - `vec` property: Preserved for solver compatibility
  - `_available=True` by default: Ensures solvers always have access

#### Swarm Variables (`swarm.py`)
- **Purpose**: Particle-based data with mesh proxy variables
- **Access Pattern**: Direct `data` property with lazy proxy updates
- **Key Features**:
  - Lazy evaluation for proxy variables (avoid PETSc conflicts)
  - Migration control for particle redistribution
  - RBF interpolation to mesh when needed

### Critical Compatibility Requirements

1. **Vector Availability**: Setting `_available=True` by default ensures solvers can access vectors without modification
2. **Lazy Initialization**: Vectors are created on first access via `_set_vec()`
3. **Backward Compatibility**: Old `with mesh.access()` patterns still work but are not required

## Key Changes Made

### 1. Backward Compatible Data Property
- **Files**: `discretisation_mesh_variables.py`, `swarm.py`
- **Implementation**: `@property def data(self): return self.array.reshape(-1, self.num_components)` with custom callback
- **Purpose**: Eliminates need for `with mesh.access(var): var.data[...] = values` pattern

### 2. Method Renaming (Completed)
- `unpack_uw_data_to_petsc` → `unpack_uw_data_from_petsc`
- `unpack_raw_data_to_petsc` → `unpack_raw_data_from_petsc`
- **Rationale**: We unpack FROM PETSc, not TO PETSc

### 3. DM Initialization Flag (Completed)
- **Change**: `_accessed` → `_dm_initialized` 
- **Files**: `discretisation_mesh.py` (line 546 init, line 1140 setting)
- **Purpose**: Tracks when PETSc DM has been built, not just access state
- **Fixed**: DM rebuild triggering when adding new variables

### 4. Dummy Access Manager (Completed)
- **Implementation**: Uses `NDArray_With_Callback.delay_callbacks_global()`
- **Files**: `discretisation_mesh.py` (`access()` → `_legacy_access()`)
- **Purpose**: Enables testing both old and new patterns

### 5. Update_lvec Fix (Completed)
- **Location**: `discretisation_mesh.py:1057-1063`
- **Changes**: 
  - Removed `with self.access():` context, changed `var.vec` → `var._lvec`
  - Added vector initialization check: `if var._lvec is None: var._set_vec(available=True)`
- **Rationale**: Direct PETSc vector access for internal mesh operations eliminates availability check requirement, but must ensure vectors are initialized

## Locking Hierarchy
mesh → swarm → variables (variables must be locked through their container)

## Array Formats
- **array**: (N,a,b) where scalar=(N,1,1), vector=(N,1,3), tensor=(N,3,3)  
- **data**: (-1, num_components) flat format for backward compatibility

## Vestigial Code Identified
- `Stateful` mixin class and `_increment()` method in `_api_tools.py`
- Various access-related flags that may no longer be needed

## Recent Progress
- Fixed recursion error in data property by using direct PETSc access
- Resolved callback index conflicts between array and data properties
- Fixed DM initialization flag not being set with direct access
- Eliminated access context manager requirement in update_lvec() by using direct `_lvec` access
- Fixed symmetric tensor data property shape issue
- Implemented Mathematical Objects with complete SymPy integration
- **LATEST**: Implemented elegant `to_model_units()` using Pint dimensional analysis (2025-10-08)

## Current Status (Phase 1 Complete)
✅ Core data property implementation complete
✅ Method naming corrected  
✅ DM initialization tracking fixed
✅ Dummy access manager working
✅ update_lvec() access requirement eliminated
✅ Symmetric tensor data property fixed
✅ Swarm points setter migration bug fixed
✅ PETSc field access conflicts resolved
✅ Swarm proxy variable lazy evaluation implemented
✅ Vector availability issues fixed for solver compatibility
✅ Most Stokes tests now passing
✅ Mathematical Objects implementation complete
✅ Advection-diffusion bug fixed (mesh._stale_lvec flag issue resolved)
✅ Test suite reorganized by complexity (0000-0199 simple, 0500-0699 intermediate, 1000+ complex)
✅ **Model auto-registration system implemented** (2025-09-23)
✅ **Obsolete migration validation test removed** (test_0560_migration_validation.py)
✅ **Private variables system implemented** (2025-09-30) - `_register=False` parameter for non-persistent variables
✅ **Units capability tests fixed** (2025-09-30) - 79/81 tests passing, core units functionality working
✅ **Elegant to_model_units() implementation** (2025-10-08) - Uses Pint dimensional analysis for composite dimensions, returns dimensionless UWQuantity objects

## Pending Cleanup (Future Phase)
🔄 **Remove legacy array interface methods** - When migration is complete:
   - Remove `use_legacy_array()` and `use_enhanced_array()` from SwarmVariable and MeshVariable
   - Remove associated tests that validate interface switching:
     - `test_0530_array_migration.py` - Tests legacy/enhanced interface migration
     - `test_0550_direct_pack_unpack.py` - Tests both interface modes (partial removal)
     - ~~`test_0560_migration_validation.py`~~ - ✅ **REMOVED** (2025-09-23) - Obsolete migration scaffolding
   - Review and potentially remove `test_0540_coordinate_change_locking.py` - May be migration scaffolding
   - Keep only tests that validate current array interface functionality
   - Archive migration tests for historical reference

## Test Suite Organization (Latest)
Tests reorganized by complexity level for better execution order:

**SIMPLE (0000-0199)**: Basic functionality, imports, simple operations
- test_0100_backward_compatible_data.py (was test_0002_*)
- test_0110_basic_swarm.py (was test_0002_*)
- test_0120_data_property_access.py (was test_0002_*)
- test_0130_field_creation.py (was test_0002_*)
- test_0140_synchronised_updates.py (was test_0002_*)

**INTERMEDIATE (0500-0699)**: Data structures, transformations, enhanced interfaces
- test_0500_enhanced_array_structure.py (was test_0002_*)
- test_0510_enhanced_swarm_array.py (was test_0002_*)
- test_0520_mathematical_mixin_enhanced.py (was test_0002_*)
- test_0530_array_migration.py (was unnumbered)
- test_0540_coordinate_change_locking.py (was unnumbered)
- test_0550_direct_pack_unpack.py (was unnumbered)
- ~~test_0560_migration_validation.py~~ (REMOVED - obsolete)

**UNITS AND ENHANCED CAPABILITIES (0700-0799)**: Units system, dimensional analysis, enhanced variables
- test_0700_units_system.py - Core units system tests (79/81 passing)
- test_0710_units_utilities.py - Units utility functions and integration tests
- test_0720_mathematical_mixin_comprehensive.py - Enhanced mathematical operations

**COMPLEX (1000+)**: Physics solvers, time-stepping, coupled systems
- Poisson solvers (1000-1009)
- Stokes solvers (1010-1050)
- Advection-diffusion (1100-1120)

## Symmetric Tensor Fix (Latest)
**Problem**: For symmetric tensors, `num_components` (6 in 3D) ≠ array components (9 in 3D)
- `array` shape: `(N, 3, 3)` = 9 components (full tensor)
- `data` should be: `(N, 6)` = 6 components (packed symmetric format)
- Previous implementation: `self.array.reshape(-1, self.num_components)` failed

**Solution**: Direct PETSc access in data property
- Changed data property to access PETSc vector directly: `self.vec.array.reshape(-1, self.num_components)`
- This gives the correct packed format with proper component count
- Maintains pack/unpack logic for write operations via callback

## Swarm Migration Understanding

### What Migration Does
**Migration moves particles between processors** based on their spatial location. When particles move in space (e.g., `swarm.points` is updated), they may now belong to a different processor's spatial domain. Migration:
- Redistributes particles to the processor that owns their spatial domain
- Changes local array sizes as particles arrive/leave each processor  
- Updates all swarm variable arrays to match the new particle distribution
- Is essential for parallel correctness

### Migration Patterns
1. **Default behavior**: Migration should happen automatically when particles move
2. **Deferred migration**: Use `migration_disabled()` context for batch operations:
   ```python
   with swarm.migration_disabled():
       for i in range(n):
           swarm.points[mask[i]] += deltas[i]
   # Migration could happen here if context manager supports it
   ```
3. **Why defer**: Avoid repeated redistributions during complex multi-step updates

### Current Bug (swarm.points setter) - FIXED
**Problem**: The swarm.points callback incorrectly wrapped migration in `with self.migration_disabled():`, making migration a no-op. This prevented essential particle redistribution, causing test failures.

**Solution Applied**:
1. **Fixed callback** (line 1496-1520): Removed `migration_disabled()` wrapper, added check `if not self._migration_disabled:` before migrate()
2. **Enhanced setter** (line 1545-1551): Added direct PETSc DM field update for immediate consistency
3. **New context manager**: Added `migration_control(disable=False)` with deferred migration support
4. **Backward compatibility**: `migration_disabled()` now calls `migration_control(disable=True)`

## Swarm Proxy Variable Understanding

### Critical Insight: Proxy Mesh Variables
Swarm variables with `proxy_degree > 0` create **proxy mesh variables** that interpolate swarm data using radial basis functions. These proxies are used for:
- Integration and derivative calculations (integrals work on the mesh proxy, not raw swarm data)
- Symbolic operations that require continuous field representations

### Proxy Update Requirements
The proxy mesh variable MUST be updated via `swarmVar._update()` whenever:
1. **Swarm data changes**: `swarmVar.data[...] = values` (fixed in pack methods)
2. **Particle positions change**: Migration, advection, manual position updates
3. **Particle count changes**: Population, deletion, resampling
4. **Swarm topology changes**: Remeshing, recreation

### Current Implementation
- ✅ `pack_raw_data_to_petsc()` and `pack_uw_data_to_petsc()` call `_update()`
- ✅ Migration calls `_update()` for all variables after particle redistribution  
- ✅ Points setter (`swarm.points = new_pos`) calls migrate + `_update()` for all variables
- ✅ Advection ends with `migrate()` call (triggers proxy updates)
- ✅ Add particles methods call `migrate()` (triggers proxy updates)

### Potential Issues Found
- ⚠️  **`populate()` method** (line 1682): Adds new particles but doesn't call migrate or update proxies
  - **Impact**: Existing proxy variables may not reflect new particle distribution
  - **Solution**: Add `migrate()` call or explicit proxy updates at end of populate
- ⚠️  **Direct DM operations**: Any direct PETSc DM field modifications bypass proxy updates
- ⚠️  **Swarm recreation**: If swarms are recreated, proxy variables need full rebuild

## Final Implementation Summary

### Key Technical Solutions
1. **Vector Availability Fix**: Set `_available=True` by default to ensure solver compatibility
2. **Lazy Vector Initialization**: `vec` property creates vectors on first access if needed
3. **Swarm Proxy Lazy Evaluation**: Mark proxy as stale on data changes, update only when `sym` property accessed
4. **PETSc Field Access Protection**: Use proper field registration and avoid nested field access

### Testing Notes
- Direct data access working: `var.data[...] = values` triggers automatic PETSc sync
- Both patterns supported: Legacy `with mesh.access(var)` and new direct access
- Solver integration: Stokes tests pass with preserved PETSc vector access patterns
- Conservative approach: No solver modifications, only interface compatibility layers

## Mathematical Objects Implementation (Latest)

### Overview
Successfully implemented natural mathematical notation for variables, enabling direct arithmetic operations without requiring explicit `.sym` access. Variables now work exactly like SymPy matrices in mathematical contexts while preserving computational functionality.

### Key Features Implemented ✅
1. **Direct Arithmetic Operations**: `var * 2`, `2 * var`, `var + 1`, `-var`, `var / 2`, `var ** 2`
2. **Component Access**: `velocity[0]` instead of `velocity.sym[0]`
3. **Complete SymPy Matrix API**: `var.T`, `var.dot()`, `var.norm()`, `var.cross()`, etc.
4. **JIT Compatibility**: All operations return pure SymPy objects, preserving compilation
5. **Backward Compatibility**: All existing `.sym` usage continues to work
6. **Preserved Display**: Variables show computational view by default, not symbolic

### Implementation Architecture
**MathematicalMixin Class** (`utilities/mathematical_mixin.py`):
- **`_sympify_()` protocol**: Enables SymPy integration for mathematical operations
- **Explicit arithmetic methods**: `__add__`, `__mul__`, `__sub__`, `__truediv__`, `__pow__`, `__neg__`
- **Right-hand operations**: `__radd__`, `__rmul__`, etc. for operations like `2 * var`
- **`__getitem__()` method**: Component access without `.sym`
- **`__getattr__() delegation`**: Automatic access to all SymPy Matrix methods
- **Display control**: `__repr__()` preserves computational view, `sym_repr()` for symbolic

### Integration Points
**Variable Classes Enhanced**:
- `_MeshVariable(MathematicalMixin, Stateful, uw_object)` in `discretisation_mesh_variables.py`
- `SwarmVariable(MathematicalMixin, Stateful, uw_object)` in `swarm.py`

### Critical Technical Insights
1. **Dual Operation Support Required**: 
   - `_sympify_()` handles SymPy-initiated operations (`sympy.Symbol * var`)
   - Explicit methods handle Python-initiated operations (`var * 2`)
   - Both needed for complete mathematical integration

2. **SymPy API Delegation Success**:
   - `__getattr__()` automatically delegates to `self.sym` for missing methods
   - Provides hundreds of SymPy Matrix methods without individual implementation
   - Future-proof for new SymPy methods, scales automatically

3. **JIT Compatibility Verified**:
   - `_sympify_()` returns identical SymPy atoms as `.sym` property
   - JIT compilation unchanged: same Function identification and PETSc mapping
   - No performance impact, identical expression trees

4. **Display Behavior Balance**:
   - Users prefer computational view by default (data, mesh info)
   - Mathematical display available via `sym_repr()` when needed
   - Jupyter LaTeX rendering for mathematical contexts

### Usage Examples
```python
# Before: Required .sym for mathematical operations
momentum = density * velocity.sym
strain_rate = velocity.sym[0].diff(x) + velocity.sym[1].diff(y)
velocity_magnitude = velocity.sym.norm()

# After: Natural mathematical syntax
momentum = density * velocity              # Direct arithmetic
strain_rate = velocity[0].diff(x) + velocity[1].diff(y)  # Component access  
velocity_magnitude = velocity.norm()       # Direct method access

# All SymPy Matrix methods available:
velocity.T                    # Transpose
velocity.dot(other)          # Dot product
velocity.cross(other)        # Cross product  
velocity.diff(x)             # Differentiation
velocity.subs(x, 1)          # Substitution
# And hundreds more...
```

### Benefits Achieved
- **Natural Mathematical Expressions**: Code looks like mathematical equations
- **Complete SymPy Integration**: Full Matrix API automatically available
- **Zero Breaking Changes**: All existing code continues to work
- **JIT Compatibility Maintained**: Identical compilation paths and performance
- **Minimal Implementation**: Simple mixin provides maximum functionality
- **Future-Proof Design**: Automatically supports new SymPy methods

## Legacy Access Pattern Removal (Phase 2 Complete)

### Successfully Removed Safe Patterns
✅ **From `swarm.py`:**
1. KDTree creation: `with self.access(): self._index = uw.kdtree.KDTree(self.data)` (line 2664)
2. Velocity evaluation: `with self.access(): vel = uw.function.evaluate(...)` (line 2933)
3. Data display: `with self.swarm.access(): display(self.data)` (line 757)
4. HDF5 save operations: `with self.swarm.access(self): h5f.create_dataset(...)` (lines 909, 917)
5. KDTree queries for level sets: `with self.swarm.access(): kd = uw.kdtree.KDTree(...)` (lines 1174, 1214)
6. RBF interpolation: `with self.swarm.mesh.access(meshVar), self.swarm.access(): meshVar.data[...] = ...` (lines 505, 1220)

✅ **From `discretisation_mesh_variables.py` (previous):**
7. RBF interpolation data copy: `with self.mesh.access(): D = self.data.copy()` (line 780)
8. H5 vector loading: `with self.mesh.access(): self.data[...] = ...` (lines 1003-1016)
9. Swarm RBF to mesh: `with meshVar.mesh.access(meshVar): meshVar.data[...] = Values[...]` (line 455)

### Testing Environment Discovery
**Critical Learning**: Must use `pixi run -e default python script.py` to execute code
- **Why**: Pixi manages all dependencies and builds necessary PETSc components
- **Without pixi**: `ModuleNotFoundError: No module named 'underworld3'`
- **With pixi**: Full environment with compiled PETSc, MPI, and all dependencies

### Validation Results
✅ **Field creation test passed**:
- Variable u: field_id=0 ✓
- Variable p: field_id=1 ✓  
- Variable s: field_id=2 ✓
- Array access: `s.array` ✓

**Test command**: `pixi run -e default python debug_field_test.py`
**Result**: All legacy access patterns successfully removed without breaking functionality

### Pixi Environment Information
- **Available environments**: `default`, `dev`
- **Default environment**: 37 conda dependencies + PyPI packages
- **Key components**: python, mpich, petsc stack, numpy, scipy, sympy, h5py
- **Build tasks**: petsc-build, underworld-build, petsc4py-build
- **Test tasks**: underworld-test, petsc-test

### Pattern Classification (Final)
**✅ SAFE TO REMOVE - Confirmed Working**:
- Simple data access for calculations
- Display/visualization operations  
- File I/O operations (HDF5 save/load)
- KDTree construction and queries
- RBF interpolation with data assignment

**⚠️ PRESERVED - Require Further Analysis**:
- Solver-adjacent operations in `discretisation_mesh.py`
- Complex migration and swarm operations
- Direct PETSc DM field manipulations
- Any pattern involving `update_lvec()` calls

### Key Technical Insights
1. **Data Property Success**: Direct `var.data[...] = values` works without access contexts
2. **Pixi Integration**: All testing must use pixi environments for proper dependency resolution
3. **Conservative Approach Validated**: Solver interfaces remain untouched and functional
4. **Callback System Working**: NDArray_With_Callback automatically syncs to PETSc

## Direct Array Access Migration (Phase 3 Complete)

### New User-Friendly Wrapper Function
Added `uw.synchronised_array_update()` context manager for batch operations:
```python
def synchronised_array_update(context_info="user operations"):
    """
    Context manager for synchronised array updates across multiple variables.
    
    Batches multiple array assignments together and defers PETSc synchronization
    until the end of the context, ensuring atomic updates and better performance.
    """
    return utilities.NDArray_With_Callback.delay_callbacks_global(context_info)
```

### Migration Patterns Applied

#### Single Variable Updates → Direct `.array` Access
```python
# OLD PATTERN
with mesh.access(var):
    var.data[...] = values

# NEW PATTERN  
var.array[...] = values
```

#### Multiple Variable Updates → `synchronised_array_update`
```python
# OLD PATTERN
with mesh.access(var1, var2, var3):
    var1.data[...] = values1
    var2.data[...] = values2
    var3.data[...] = values3

# NEW PATTERN
with uw.synchronised_array_update():
    var1.array[...] = values1
    var2.array[...] = values2
    var3.array[...] = values3
```

### Files Successfully Updated

#### Notebooks (4 files)
- **`2-Variables.ipynb`**: Multi-variable setup using `synchronised_array_update()`
- **`5-Solvers-ii-Stokes.ipynb`**: Direct array access for null space removal
- **`8-Particle_Swarms.ipynb`**: Batch swarm variable initialization

#### Tests (9+ files)
**Converted to direct `.array` access:**
- `test_0503_evaluate.py`: Single variable assignments
- `test_0002_basic_swarm.py`: Shape checking patterns  
- `test_1100_AdvDiffCartesian.py`: Velocity field initialization
- `test_0005_IndexSwarmVariable.py`: Material property assignments
- `test_0003_save_load.py`: Data setup for save/load tests

**Converted to `synchronised_array_update()`:**
- `test_0503_evaluate2.py`: Multi-variable function evaluation tests
- `test_0505_rbf_swarm_mesh.py`: Vector component assignments
- `test_1110_advDiffAnnulus.py`: Temperature and velocity field setup

### Benefits Achieved
1. **Simplified Syntax**: `uw.synchronised_array_update()` vs verbose utility path
2. **Clear Intent**: Function name describes exactly what it does
3. **Better Performance**: Batch operations avoid redundant synchronization
4. **Parallel Safe**: Includes MPI barriers for proper coordination
5. **Backward Compatible**: Old `mesh.access()` patterns still work

### User Guidelines
- **Single variable**: Use direct `var.array[...] = values`
- **Multiple variables**: Use `with uw.synchronised_array_update():` context
- **Shape checking/inspection**: Direct access without context manager
- **Prefer `.array` over `.data`**: Array property is the recommended interface

### Next Phase Approach
- **Test suite validation**: Ensure all changes work correctly
- **Advection-diffusion analysis**: Careful examination of time-stepping and particle transport
- **No solver changes**: Continue preserving benchmarked solver implementations
- **Surgical fixes only**: Address specific issues without architectural changes

## Model Auto-Registration System (2025-09-23)

### Overview
Implemented automatic registration of all UW3 objects (meshes, swarms, variables, solvers) with a global default Model for serialization and orchestration support.

### Implementation Details

**Files Modified:**
1. **`src/underworld3/model.py`**:
   - Added global `_default_model` singleton pattern
   - Added `get_default_model()` function to get/create default model
   - Added `reset_default_model()` function to start fresh
   - Simplified Model class to use plain dicts instead of complex registries

2. **`src/underworld3/__init__.py`** (line 134):
   - Imported model functions: `from .model import Model, create_model, get_default_model, reset_default_model`

3. **`src/underworld3/discretisation/discretisation_mesh.py`** (line 627):
   - Added auto-registration: `uw.get_default_model()._register_mesh(self)`
   - Removed redundant import (was causing UnboundLocalError)

4. **`src/underworld3/discretisation/discretisation_mesh_variables.py`** (line 406):
   - Added auto-registration: `uw.get_default_model()._register_variable(self.name, self)`
   - Removed redundant import

5. **`src/underworld3/swarm.py`**:
   - Line 208: SwarmVariable auto-registration
   - Line 1481: Swarm auto-registration
   - Removed redundant imports

### Key Features
- **Automatic Registration**: All objects register with default model on creation
- **Serialization Support**: `model.to_dict()` exports to JSON-serializable format
- **Simple API**: `uw.get_default_model()` and `uw.reset_default_model()`
- **No Breaking Changes**: Existing code works without modification
- **Weak References**: Swarms use WeakValueDictionary to prevent circular refs

### Testing
- Created `tests/test_model_basic.py` with 4 test cases
- All tests passing ✅
- Validates auto-registration, serialization, and model tracking

### Bug Fixes
- **Fixed UnboundLocalError**: Removed redundant `import underworld3 as uw` statements that shadowed the top-level import
- **Files affected**: discretisation_mesh.py, discretisation_mesh_variables.py, swarm.py (3 locations)

## Coding Conventions and Best Practices (2025-10-10)

### Variable Naming: Avoid Ambiguous 'model'
**IMPORTANT**: The variable name `model` is now ambiguous and should be avoided in new code.

**Why**: We have two different "model" concepts in UW3:
1. **`uw.model`** / **`uw.Model`**: Serialization/orchestration system for managing UW3 objects
2. **Constitutive models**: Material behavior models (ViscousFlowModel, DiffusionModel, etc.)

**Recommended Patterns**:
```python
# GOOD - Clear and unambiguous
constitutive_model = stokes.constitutive_model
diffusion_model = adv_diff.constitutive_model
orchestration_model = uw.get_default_model()

# AVOID - Ambiguous
model = stokes.constitutive_model  # Which kind of model?
model = uw.get_default_model()     # Not clear what this is
```

**In Tests**: Use descriptive names
- `constitutive_model` for material models accessed via `solver.constitutive_model`
- `orchestration_model` or `uw_model` for `uw.Model` instances
- Add comments when accessing constitutive models: `# Note: Use 'constitutive_model' not 'model' to avoid confusion with uw.model`

**Legacy Code**: Fix variable names opportunistically as we encounter them, not in bulk refactoring