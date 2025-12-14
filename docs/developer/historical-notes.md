# Underworld3 Development History

> **Purpose**: This file preserves the development history, completed migrations, fixed bugs, and implementation details. This content was moved from CLAUDE.md to reduce AI assistant context load while maintaining a valuable historical record.
>
> **For AI assistants**: This file is referenced from CLAUDE.md. Read it when you need historical context about past bugs, migrations, or implementation decisions.

---

## Table of Contents
- [Documentation Migration Status](#documentation-migration-status)
- [Parallel Safety System Implementation](#parallel-safety-system-implementation)
- [Units System Bug Fixes](#units-system-bug-fixes)
- [Data Access Migration History](#data-access-migration-history)
- [Test Suite Reorganization](#test-suite-reorganization)
- [Mathematical Objects Implementation](#mathematical-objects-implementation)
- [Legacy Access Pattern Removal](#legacy-access-pattern-removal)
- [Model Auto-Registration Implementation](#model-auto-registration-implementation)

---

## Documentation Migration Status

**Migration in Progress** (as of 2025-09): The `underworld3-documentation` directory contains legacy documentation being migrated.
- **Already migrated**: Constitutive models theory documentation
- **To migrate**: Solver documentation, key example notebooks, benchmarks
- **See**: `MIGRATION_FROM_UW3_DOCUMENTATION.md` for detailed status
- **Goal**: Single source of truth in underworld3 repository

---

## Parallel Safety System Implementation

### Status: IMPLEMENTED (2025-01-24)

The parallel safety system is now fully implemented and integrated into the codebase.

**Design Documents** (historical reference):
- `PARALLEL_PRINT_SIMPLIFIED.md` - Main design: `uw.pprint()` and `selective_ranks()`
- `RANK_SELECTION_SPECIFICATION.md` - Comprehensive rank selection syntax
- `COLLECTIVE_OPERATIONS_CLASSIFICATION.md` - Classification of collective vs local operations
- `AUTOMATIC_COLLECTIVE_DETECTION.md` - How to detect collective ops automatically (future enhancement)
- `CODEBASE_COLLECTIVE_ANALYSIS.md` - Using existing code patterns to identify operations
- `PARALLEL_SAFETY_ANALYSIS.md` - Analysis of issues (resolved)
- `PARALLEL_SAFETY_DESIGN.md` - Early comprehensive design (superseded by simplified version)
- `RETURN_SUPPRESSION_MECHANISM.md` - Technical details of stdout suppression
- `PARALLEL_PRINTING_DESIGN.md` - Alternative design approach

**Implementation**: `src/underworld3/mpi.py`
**Documentation**: `docs/advanced/parallel-computing.qmd`

---

## Units System Bug Fixes

### Bug Fix: String Return from .units Property (2025-11-19)

**Problem**: Added type annotation `-> str` to `UWQuantity.units` property, which forced string conversion.

**Impact**:
- Broke Rayleigh number calculations
- Broke all unit arithmetic
- `model.get_scale_for_dimensionality(qty.units)` failed with `AttributeError: 'str' object has no attribute 'items'`

**Root Cause**:
```python
# WRONG - type hint forced string conversion
@property
def units(self) -> str:
    return str(self._pint_qty.units)  # Breaks dimensional analysis!
```

**Fix**: Removed type hint, return raw Pint object instead.

---

### Bug Fix: Unit Conversion on Composite Expressions (2025-11-25)

**Problem**: `.to_base_units()` and `.to_reduced_units()` were causing evaluation errors on composite expressions.

**Root Cause**:
- Methods embedded conversion factors in expression tree: `new_expr = expr * 5617615.15`
- During nondimensional evaluation cycles, factors were **double-applied**
- Example: `sqrt((kappa * t_now))**0.5` would evaluate to wrong value after conversion

**Fix Applied**:
- Composite expressions (containing UWexpression symbols): Only change display units, no factor embedding
- Simple expressions (no symbols): Apply conversion factors as before
- Issues UserWarning when display-only conversion occurs

**Verification**: All evaluation bugs fixed
- `evaluate(expr.to_base_units())` now equals `evaluate(expr)`
- System is "bulletproof" for evaluation with nondimensional scaling
- See: `docs/reviews/2025-11/UNITS-EVALUATION-FIXES-2025-11-25.md`

---

### Historical Issue: L.units.to_compact() (2025-11-19)

**User Report**: `L.units.to_compact()` raised AttributeError

**Resolution**: This is **correct behavior** - Units alone can't be compacted. Only full Quantities (value + units) support conversion methods.

```python
# WRONG - .units is a Unit object, not a Quantity
L = uw.quantity(2900, "km")
L.units.to_base_units()     # AttributeError - Unit has no to_base_units method

# CORRECT - use conversion methods on the UWQuantity itself
L.to_base_units()           # Returns UWQuantity(2900000, "m")
```

---

## Data Access Migration History

### Phase 1 Complete: Core Implementation

**Key Changes Made**:

1. **Backward Compatible Data Property**
   - Files: `discretisation_mesh_variables.py`, `swarm.py`
   - Implementation: `@property def data(self): return self.array.reshape(-1, self.num_components)` with custom callback
   - Purpose: Eliminates need for `with mesh.access(var): var.data[...] = values` pattern

2. **Method Renaming**
   - `unpack_uw_data_to_petsc` → `unpack_uw_data_from_petsc`
   - `unpack_raw_data_to_petsc` → `unpack_raw_data_from_petsc`
   - Rationale: We unpack FROM PETSc, not TO PETSc

3. **DM Initialization Flag**
   - Change: `_accessed` → `_dm_initialized`
   - Files: `discretisation_mesh.py` (line 546 init, line 1140 setting)
   - Purpose: Tracks when PETSc DM has been built, not just access state
   - Fixed: DM rebuild triggering when adding new variables

4. **Dummy Access Manager**
   - Implementation: Uses `NDArray_With_Callback.delay_callbacks_global()`
   - Files: `discretisation_mesh.py` (`access()` → `_legacy_access()`)
   - Purpose: Enables testing both old and new patterns

5. **Update_lvec Fix**
   - Location: `discretisation_mesh.py:1057-1063`
   - Changes: Removed `with self.access():` context, changed `var.vec` → `var._lvec`
   - Added vector initialization check: `if var._lvec is None: var._set_vec(available=True)`

### Vestigial Code Identified
- `Stateful` mixin class and `_increment()` method in `_api_tools.py`
- Various access-related flags that may no longer be needed

### Progress Milestones

- Fixed recursion error in data property by using direct PETSc access
- Resolved callback index conflicts between array and data properties
- Fixed DM initialization flag not being set with direct access
- Eliminated access context manager requirement in update_lvec() by using direct `_lvec` access
- Fixed symmetric tensor data property shape issue
- Implemented Mathematical Objects with complete SymPy integration
- Implemented elegant `to_model_units()` using Pint dimensional analysis (2025-10-08)

### Coordinate Units System (2025-10-15)

**FIXED**: Model synchronization via auto-registration in `Model.__init__()` and `set_reference_quantities()`
**FIXED**: Coordinate unit detection via enhanced `get_units()` that searches inside SymPy expressions

Implementation details:
- Implemented `patch_coordinate_units()` to add unit awareness to mesh coordinates (x, y, z)
- Created `UnitAwareBaseScalar` subclass for future native unit support
- Added `Model.set_as_default()` for explicit control in advanced workflows
- Result: `uw.get_units(mesh.X[0])` now correctly returns coordinate units
- Documentation: See `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`

### Current Status Checklist (Phase 1)

- [x] Core data property implementation complete
- [x] Method naming corrected
- [x] DM initialization tracking fixed
- [x] Dummy access manager working
- [x] update_lvec() access requirement eliminated
- [x] Symmetric tensor data property fixed
- [x] Swarm points setter migration bug fixed
- [x] PETSc field access conflicts resolved
- [x] Swarm proxy variable lazy evaluation implemented
- [x] Vector availability issues fixed for solver compatibility
- [x] Most Stokes tests now passing
- [x] Mathematical Objects implementation complete
- [x] Advection-diffusion bug fixed (mesh._stale_lvec flag issue resolved)
- [x] Test suite reorganized by complexity
- [x] Model auto-registration system implemented (2025-09-23)
- [x] Obsolete migration validation test removed (test_0560_migration_validation.py)
- [x] Private variables system implemented (2025-09-30)
- [x] Units capability tests fixed (2025-09-30)
- [x] Elegant to_model_units() implementation (2025-10-08)
- [x] Coordinate units system complete (2025-10-15)

---

## Pending Cleanup (Future Phase)

**Remove legacy array interface methods** - When migration is complete:
- Remove `use_legacy_array()` and `use_enhanced_array()` from SwarmVariable and MeshVariable
- Remove associated tests that validate interface switching:
  - `test_0530_array_migration.py` - Tests legacy/enhanced interface migration
  - `test_0550_direct_pack_unpack.py` - Tests both interface modes (partial removal)
  - ~~`test_0560_migration_validation.py`~~ - REMOVED (2025-09-23) - Obsolete migration scaffolding
- Review and potentially remove `test_0540_coordinate_change_locking.py` - May be migration scaffolding
- Keep only tests that validate current array interface functionality
- Archive migration tests for historical reference

---

## Test Suite Reorganization

### Original Reorganization (by complexity level)

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

**UNITS AND ENHANCED CAPABILITIES (0700-0799)**: Units system, dimensional analysis
- test_0700_units_system.py - Core units system tests (79/81 passing)
- test_0710_units_utilities.py - Units utility functions
- test_0720_mathematical_mixin_comprehensive.py - Enhanced mathematical operations

**COMPLEX (1000+)**: Physics solvers, time-stepping, coupled systems
- Poisson solvers (1000-1009)
- Stokes solvers (1010-1050)
- Advection-diffusion (1100-1120)

### Test Classification Status (2025-11-15)

**Completed Actions**:
1. [x] FIXED: JIT unwrapping bug (test_0818_stokes_nd.py: all 5 tests passing)

**In Progress**:
2. [ ] Classify 79 failing units tests into Tiers B or C

**TODO**:
3. [ ] Mark all Tier A tests with `@pytest.mark.tier_a`
4. [ ] Mark incomplete features as Tier C with `@pytest.mark.xfail`

---

## Mathematical Objects Implementation

### Overview
Successfully implemented natural mathematical notation for variables, enabling direct arithmetic operations without requiring explicit `.sym` access.

### Features Implemented

1. **Direct Arithmetic Operations**: `var * 2`, `2 * var`, `var + 1`, `-var`, `var / 2`, `var ** 2`
2. **Component Access**: `velocity[0]` instead of `velocity.sym[0]`
3. **Complete SymPy Matrix API**: `var.T`, `var.dot()`, `var.norm()`, `var.cross()`, etc.
4. **JIT Compatibility**: All operations return pure SymPy objects, preserving compilation
5. **Backward Compatibility**: All existing `.sym` usage continues to work
6. **Preserved Display**: Variables show computational view by default, not symbolic

### Implementation Architecture

**MathematicalMixin Class** (`utilities/mathematical_mixin.py`):
- `_sympify_()` protocol: Enables SymPy integration for mathematical operations
- Explicit arithmetic methods: `__add__`, `__mul__`, `__sub__`, `__truediv__`, `__pow__`, `__neg__`
- Right-hand operations: `__radd__`, `__rmul__`, etc. for operations like `2 * var`
- `__getitem__()` method: Component access without `.sym`
- `__getattr__() delegation`: Automatic access to all SymPy Matrix methods
- Display control: `__repr__()` preserves computational view, `sym_repr()` for symbolic

### Integration Points

**Variable Classes Enhanced**:
- `_MeshVariable(MathematicalMixin, Stateful, uw_object)` in `discretisation_mesh_variables.py`
- `SwarmVariable(MathematicalMixin, Stateful, uw_object)` in `swarm.py`

### Technical Insights

1. **Dual Operation Support Required**:
   - `_sympify_()` handles SymPy-initiated operations (`sympy.Symbol * var`)
   - Explicit methods handle Python-initiated operations (`var * 2`)
   - Both needed for complete mathematical integration

2. **SymPy API Delegation Success**:
   - `__getattr__()` automatically delegates to `self.sym` for missing methods
   - Provides hundreds of SymPy Matrix methods without individual implementation
   - Future-proof for new SymPy methods

3. **JIT Compatibility Verified**:
   - `_sympify_()` returns identical SymPy atoms as `.sym` property
   - JIT compilation unchanged: same Function identification and PETSc mapping
   - No performance impact, identical expression trees

### Benefits Achieved
- Natural Mathematical Expressions: Code looks like mathematical equations
- Complete SymPy Integration: Full Matrix API automatically available
- Zero Breaking Changes: All existing code continues to work
- JIT Compatibility Maintained: Identical compilation paths and performance
- Minimal Implementation: Simple mixin provides maximum functionality
- Future-Proof Design: Automatically supports new SymPy methods

---

## Legacy Access Pattern Removal

### Phase 2 Complete: Safe Pattern Removal

**Successfully Removed from `swarm.py`:**
1. KDTree creation: `with self.access(): self._index = uw.kdtree.KDTree(self.data)` (line 2664)
2. Velocity evaluation: `with self.access(): vel = uw.function.evaluate(...)` (line 2933)
3. Data display: `with self.swarm.access(): display(self.data)` (line 757)
4. HDF5 save operations: `with self.swarm.access(self): h5f.create_dataset(...)` (lines 909, 917)
5. KDTree queries for level sets: `with self.swarm.access(): kd = uw.kdtree.KDTree(...)` (lines 1174, 1214)
6. RBF interpolation: `with self.swarm.mesh.access(meshVar), self.swarm.access(): meshVar.data[...] = ...` (lines 505, 1220)

**Successfully Removed from `discretisation_mesh_variables.py`:**
7. RBF interpolation data copy: `with self.mesh.access(): D = self.data.copy()` (line 780)
8. H5 vector loading: `with self.mesh.access(): self.data[...] = ...` (lines 1003-1016)
9. Swarm RBF to mesh: `with meshVar.mesh.access(meshVar): meshVar.data[...] = Values[...]` (line 455)

### Testing Environment Discovery

**Critical Learning**: Must use `pixi run -e default python script.py` to execute code
- **Why**: Pixi manages all dependencies and builds necessary PETSc components
- **Without pixi**: `ModuleNotFoundError: No module named 'underworld3'`
- **With pixi**: Full environment with compiled PETSc, MPI, and all dependencies

### Validation Results

Field creation test passed:
- Variable u: field_id=0
- Variable p: field_id=1
- Variable s: field_id=2
- Array access: `s.array` works

### Pattern Classification (Final)

**SAFE TO REMOVE - Confirmed Working**:
- Simple data access for calculations
- Display/visualization operations
- File I/O operations (HDF5 save/load)
- KDTree construction and queries
- RBF interpolation with data assignment

**PRESERVED - Require Further Analysis**:
- Solver-adjacent operations in `discretisation_mesh.py`
- Complex migration and swarm operations
- Direct PETSc DM field manipulations
- Any pattern involving `update_lvec()` calls

---

## Phase 3 Complete: Direct Array Access Migration

### Files Successfully Updated

**Notebooks (4 files)**:
- `2-Variables.ipynb`: Multi-variable setup using `synchronised_array_update()`
- `5-Solvers-ii-Stokes.ipynb`: Direct array access for null space removal
- `8-Particle_Swarms.ipynb`: Batch swarm variable initialization

**Tests (9+ files)**:

Converted to direct `.array` access:
- `test_0503_evaluate.py`: Single variable assignments
- `test_0002_basic_swarm.py`: Shape checking patterns
- `test_1100_AdvDiffCartesian.py`: Velocity field initialization
- `test_0005_IndexSwarmVariable.py`: Material property assignments
- `test_0003_save_load.py`: Data setup for save/load tests

Converted to `synchronised_array_update()`:
- `test_0503_evaluate2.py`: Multi-variable function evaluation tests
- `test_0505_rbf_swarm_mesh.py`: Vector component assignments
- `test_1110_advDiffAnnulus.py`: Temperature and velocity field setup

---

## Model Auto-Registration Implementation

### Overview (2025-09-23)
Implemented automatic registration of all UW3 objects (meshes, swarms, variables, solvers) with a global default Model.

### Files Modified

1. **`src/underworld3/model.py`**:
   - Added global `_default_model` singleton pattern
   - Added `get_default_model()` function
   - Added `reset_default_model()` function
   - Simplified Model class to use plain dicts

2. **`src/underworld3/__init__.py`** (line 134):
   - Imported model functions

3. **`src/underworld3/discretisation/discretisation_mesh.py`** (line 627):
   - Added auto-registration: `uw.get_default_model()._register_mesh(self)`

4. **`src/underworld3/discretisation/discretisation_mesh_variables.py`** (line 406):
   - Added auto-registration for variables

5. **`src/underworld3/swarm.py`**:
   - Line 208: SwarmVariable auto-registration
   - Line 1481: Swarm auto-registration

### Key Features
- Automatic Registration: All objects register with default model on creation
- Serialization Support: `model.to_dict()` exports to JSON-serializable format
- Simple API: `uw.get_default_model()` and `uw.reset_default_model()`
- No Breaking Changes: Existing code works without modification
- Weak References: Swarms use WeakValueDictionary to prevent circular refs

### Bug Fixes
- **Fixed UnboundLocalError**: Removed redundant `import underworld3 as uw` statements that shadowed the top-level import
- Files affected: discretisation_mesh.py, discretisation_mesh_variables.py, swarm.py

### Testing
- Created `tests/test_model_basic.py` with 4 test cases
- All tests passing
- Validates auto-registration, serialization, and model tracking

---

## Swarm Bug Fix: Migration in Points Setter

### Problem (FIXED)
The swarm.points callback incorrectly wrapped migration in `with self.migration_disabled():`, making migration a no-op. This prevented essential particle redistribution, causing test failures.

### Solution Applied
1. **Fixed callback** (line 1496-1520): Removed `migration_disabled()` wrapper, added check `if not self._migration_disabled:` before migrate()
2. **Enhanced setter** (line 1545-1551): Added direct PETSc DM field update for immediate consistency
3. **New context manager**: Added `migration_control(disable=False)` with deferred migration support
4. **Backward compatibility**: `migration_disabled()` now calls `migration_control(disable=True)`

---

*Last updated: 2025-12-13*
*Moved from CLAUDE.md to reduce context load*
