# Underworld3 TODO List

## High Priority

### Unwrapping Logic Unification
**Status**: Planned
**Priority**: Medium-High
**Effort**: 1-2 days
**Created**: 2025-11-14

**Objective**: Unify JIT compilation and evaluate unwrapping pathways into a single robust core with thin wrappers.

**Context**:
- After fixing the variable scaling bug, both paths do essentially the same thing
- Current separation is historical, not fundamental
- Code duplication creates fragility and maintenance burden

**Proposed Solution**:
- Create `_unwrap_expression_core()` with all common logic
- Thin wrapper `unwrap_for_evaluate()` adds dimensionality tracking
- Thin wrapper `_unwrap_for_compilation()` returns just expression
- Expected: 56% code reduction, single point of maintenance

**Documentation**:
- See `UNWRAPPING_UNIFICATION_PROPOSAL.md` for complete design
- See `UNWRAPPING_COMPARISON_REPORT.md` for current state analysis

**Migration Strategy**:
1. Create and test core function independently
2. Switch evaluate path first (simpler)
3. Switch JIT path second (more complex)
4. Remove old code
5. Validate full test suite

**Success Criteria**:
- All tests pass (especially `test_0818_stokes_nd.py`)
- No performance regression
- Simplified codebase
- Single source of truth for unwrapping logic

**Related Work**:
- Variable scaling bug fix (completed 2025-11-14)
- Refactoring of UWQuantity handling (completed 2025-11-14)

---

## Medium Priority

### Complete Regression Test Fixes
**Status**: In Progress (3/49 failing)
**Location**: `tests/test_06*_regression.py`

### Validate Complex Solver Tests
**Status**: Not validated after recent changes
**Location**: `tests/test_1*_*.py`

### Remove Remaining Legacy Access Patterns
**Status**: Mostly complete, some patterns preserved in solver code
**Action**: Search and classify remaining `with mesh.access()` patterns

### Example Notebooks Validation
**Status**: Some updated, full validation needed
**Location**: `docs/examples/*.ipynb`

### Simplify Notebook 13 (Coordinate Units Demo)
**Status**: Pending
**Priority**: Medium
**Location**: `docs/beginner/tutorials/13-Scaling-problems-with-physical-units.ipynb`

**Issues to Address**:
- Review and fix arrow diagrams (clarity issues)
- Simplify explanations - remove verbose print statements
- Remove excessive comments - let code speak for itself
- Ensure "it just works" message through simplicity
- Use UK/Australian spelling (metres, kilometres)

**Goal**: Demonstrate natural elegance of units system without over-explaining.
Show that working in different coordinate units (metres vs kilometres) with
automatic scaling "just works".

---

## Low Priority / Future Work

### Consider Performance Optimization
- Profile unwrapping pathways
- Benchmark before/after unification
- Ensure no slowdown in solver compilation

### Documentation Updates
- Update developer guide after unification
- Add examples of unwrapping behavior
- Document when each pathway is used

---

## Completed ✓

### Variable Scaling Bug Fix (2025-11-14)
Fixed double non-dimensionalization of variables in JIT compilation path.

### UWQuantity Unwrapping Refactoring (2025-11-14)
Consolidated UWQuantity handling into `_unwrap_expressions()` for better visibility.

### Data Access Migration (2025-10-10)
Complete migration from access context managers to direct `.array` property.

### Units System Implementation (2025-10-08)
100% working units system with dimensional analysis.

### DM State Corruption Bug Fix (2025-10-14)
Variables can now be created after `solve()` without corruption.

### Coordinate Units System (2025-10-15)
Mesh coordinates carry unit information with dimensional analysis.
