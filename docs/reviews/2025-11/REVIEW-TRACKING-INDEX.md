# November 2025 Review Tracking Index

**Created**: 2025-11-17
**Status**: Planning Phase
**Purpose**: Track formal code reviews for major systems revised in November 2025

## Overview

This document tracks 9 major system reviews that require external approval. These reviews document significant refactoring and new implementations completed in November 2025, establishing time-sensitive snapshots of current state, design rationale, and community approval.

## Review Process

Each review follows the template in [`CODE-REVIEW-PROCESS.md`](../../developer/CODE-REVIEW-PROCESS.md) and is archived permanently in `docs/reviews/2025-11/` upon sign-off.

## Review Items

### 1. Function Evaluation and Global Evaluate Merger

**File**: `EVALUATE-FUNCTION-SYSTEM-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Large

**Scope**:
- Merger of `evaluate()` and `global_evaluate()` code paths
- Automatic lambdification optimization system (10,000x speedup)
- Function detection logic (UW3 vs SymPy functions)
- Expression unwrapping and substitution
- DMInterpolation caching system
- Non-dimensionalization handling in evaluation
- Performance improvements and benchmarks

**Key Changes**:
- `src/underworld3/function/functions_unit_system.py` - Main evaluation logic
- `src/underworld3/function/pure_sympy_evaluator.py` - Lambdification system
- `src/underworld3/function/expressions.py` - UWexpression integration
- `src/underworld3/function/dminterpolation_cache.py` - Caching
- `tests/test_0720_lambdify_optimization_paths.py` - 20 comprehensive tests

**Documentation**:
- `LAMBDIFY-DETECTION-BUG-FIX.md`
- `UWEXPRESSION-LAMBDIFY-FIX.md`
- `LAMBDIFY-OPTIMIZATION-TEST-COVERAGE.md`
- `AUTOMATIC-LAMBDIFICATION-OPTIMIZATION.md`

**Dependencies**: None
**Blocks**: None

---

### 2. Array System and Mathematical Mixins

**File**: `ARRAY-SYSTEM-MATHEMATICAL-MIXINS-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Large

**Scope**:
- NDArray_With_Callback system for automatic sync
- MathematicalMixin for natural mathematical operations
- Direct `.array` property eliminating access contexts
- Backward-compatible `.data` property
- Operator overloading (_sympify_ protocol)
- Component access without .sym
- Enhanced array structure (N, a, b) format

**Key Changes**:
- `src/underworld3/utilities/nd_array_with_callback.py`
- `src/underworld3/utilities/mathematical_mixin.py`
- `src/underworld3/discretisation/discretisation_mesh_variables.py`
- `src/underworld3/swarm.py`

**Documentation**:
- Migration guides for old `with mesh.access()` patterns
- Mathematical operations guide
- Array shape conventions

**Dependencies**: None
**Blocks**: Review #3 (Units awareness integrates with array system)

---

### 3. Units-Awareness System

**File**: `UNITS-AWARENESS-SYSTEM-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Very Large

**Scope**:
- UWQuantity and unit-aware objects
- Integration with NDArray_With_Callback
- Closure principles and analysis
- Automatic unit propagation
- Unit extraction from SymPy expressions
- Coordinate units system
- Model auto-registration for units
- Dimensional analysis system

**Key Changes**:
- `src/underworld3/function/quantities.py`
- `src/underworld3/utilities/unit_aware_array.py`
- `src/underworld3/model.py` - Auto-registration
- Coordinate patching system

**Documentation**:
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
- User guide for units system
- Migration guide for unit-aware code

**Dependencies**: Review #2 (Array system)
**Blocks**: Review #6 (Non-dimensionalization uses units)

---

### 4. Timing System

**File**: `TIMING-SYSTEM-REVIEW.md`
**Status**: ✅ COMPLETE (Exists)
**Priority**: MEDIUM
**Estimated Effort**: Small

**Scope**:
- PETSc-based unified timing
- User-friendly `uw.timing.start()` API
- `print_summary()` filtered output
- `print_table()` detailed profiling
- Programmatic access via `get_summary()`
- Performance monitoring integration

**Key Changes**:
- `src/underworld3/timing.py`
- Integration with PETSc.Log system

**Documentation**:
- `docs/examples/Tutorial_Timing_System.ipynb`
- `docs/examples/Tutorial_Timing_System.py`

**Note**: Review already exists at `docs/reviews/2025-11/TIMING-SYSTEM-REFACTOR-REVIEW.md`

**Dependencies**: None
**Blocks**: None

---

### 5. Testing Suite Organization

**File**: `TESTING-SUITE-ORGANIZATION-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: MEDIUM
**Estimated Effort**: Medium

**Scope**:
- Test numbering system (0000-9999)
- Test levels (Level 1, 2, 3) with pytest markers
- Reliability tiers (Tier A, B, C) for TDD
- Test organization by complexity
- Regression test integration
- Performance test baselines
- Test documentation standards

**Key Changes**:
- `tests/` directory reorganization
- `pytest.ini` configuration
- Test marker system
- CI/CD integration

**Documentation**:
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md`
- `docs/developer/TEST-CLASSIFICATION-2025-11-15.md`
- `scripts/test_levels.sh`

**Dependencies**: None
**Blocks**: None

---

### 6. Non-Dimensionalization System

**File**: `NON-DIMENSIONALIZATION-SYSTEM-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Large

**Scope**:
- Separation from units system (distinct concerns)
- `to_model_units()` using Pint dimensional analysis
- Reference quantity system
- Scaling factors and transformations
- Integration with solvers
- Dimensionless UWQuantity objects
- Model reference scale management

**Key Changes**:
- `src/underworld3/function/quantities.py` - to_model_units()
- `src/underworld3/model.py` - Reference quantities
- Solver integration for dimensionless equations

**Documentation**:
- Non-dimensionalization user guide
- Reference quantity specification guide
- Solver scaling documentation

**Dependencies**: Review #3 (Units system)
**Blocks**: None

---

### 7. Parallel-Safe System

**File**: `PARALLEL-SAFE-SYSTEM-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Large

**Scope**:
- `uw.pprint()` for parallel-safe printing
- `selective_ranks()` context manager
- Rank selection syntax (int, slice, str patterns, functions)
- `@collective` decorators for operation classification
- Migration from `if uw.mpi.rank == 0:` patterns
- Collective operation safety

**Key Changes**:
- `src/underworld3/mpi.py` - pprint() and selective_ranks()
- Decorator system for collective operations
- Documentation of parallel patterns

**Documentation**:
- `docs/advanced/parallel-computing.qmd`
- `planning/PARALLEL_PRINT_SIMPLIFIED.md`
- `planning/RANK_SELECTION_SPECIFICATION.md`
- Migration guide from old patterns

**Dependencies**: None
**Blocks**: None

---

### 8. Variable Reduction Operators System

**File**: `REDUCTION-OPERATORS-SYSTEM-REVIEW.md`
**Status**: ✅ COMPLETE (Exists)
**Priority**: MEDIUM
**Estimated Effort**: Medium

**Scope**:
- Unified reduction operations (max, min, mean, sum, std)
- Global vs local operations
- Mesh and swarm variable support
- Parallel-correct implementations
- Integration statistics for swarms
- RBF-based spatial integration

**Key Changes**:
- Reduction operation implementation
- Statistics methods
- Integration system

**Note**: Review already exists at `docs/reviews/2025-10/REDUCTION-OPERATIONS-REVIEW.md` and `SWARM-INTEGRATION-STATISTICS-REVIEW.md`

**Dependencies**: None
**Blocks**: None

---

### 9. Expression Unwrapping System

**File**: `EXPRESSION-UNWRAPPING-SYSTEM-REVIEW.md`
**Status**: 📋 Not Started
**Priority**: HIGH
**Estimated Effort**: Large

**Scope**:
- New expression unwrapping architecture
- `unwrap_for_evaluate()` for evaluation path
- `unwrap_for_solver()` for solver path
- Constant handling and non-dimensionalization
- Variable symbol preservation
- Dimensionality tracking
- JIT compatibility preservation

**Key Changes**:
- Expression unwrapping logic
- Unit handling during unwrapping
- Solver integration
- Evaluation path optimization

**Documentation**:
- Technical notes on unwrapping
- Design rationale documentation
- Integration guide for new expression types

**Dependencies**: Review #3 (Units system), Review #6 (Non-dimensionalization)
**Blocks**: None

---

### 10. Review System Infrastructure

**File**: `REVIEW-SYSTEM-INFRASTRUCTURE-REVIEW.md`
**Status**: 🔍 Under Review (Pilot Review)
**Priority**: MEDIUM
**Estimated Effort**: Small (Process Documentation)

**Scope**:
- Formal architectural review process definition
- GitHub integration (templates, labels, workflows, automation)
- Review document structure and templates
- Team workflow and training materials
- Permanent archive system in `docs/reviews/`

**Key Changes**:
- `docs/developer/CODE-REVIEW-PROCESS.md` (pre-existing)
- `docs/developer/GITHUB-REVIEW-INTEGRATION.md` (NEW)
- `docs/developer/REVIEW-WORKFLOW-QUICK-START.md` (NEW)
- `.github/ISSUE_TEMPLATE/architectural-review.yml` (NEW)
- `.github/PULL_REQUEST_TEMPLATE/architectural-review.md` (NEW)
- `.github/workflows/architectural-review-validation.yml` (NEW)
- 12 GitHub labels created

**Documentation**:
- Complete process documentation (~42KB)
- GitHub integration guide
- Quick start reference for authors/reviewers/leads
- This review tests the system on itself

**Dependencies**: None
**Blocks**: None

**Note**: This is the pilot review that will validate the review system by being the first review submitted through it. Meta! 🔄

---

## Review Status Summary

| # | Review Topic | Status | Priority | Estimated Effort | Dependencies |
|---|-------------|--------|----------|------------------|--------------|
| 1 | Function Evaluation System | 📋 Not Started | HIGH | Large | None |
| 2 | Array System & Math Mixins | 📋 Not Started | HIGH | Large | None |
| 3 | Units-Awareness System | 📋 Not Started | HIGH | Very Large | #2 |
| 4 | Timing System | ✅ Complete | MEDIUM | Small | None |
| 5 | Testing Suite | 📋 Not Started | MEDIUM | Medium | None |
| 6 | Non-Dimensionalization | 📋 Not Started | HIGH | Large | #3 |
| 7 | Parallel-Safe System | 📋 Not Started | HIGH | Large | None |
| 8 | Reduction Operators | ✅ Complete | MEDIUM | Medium | None |
| 9 | Expression Unwrapping | 📋 Not Started | HIGH | Large | #3, #6 |
| 10 | Review System Infrastructure | 🔍 Pilot Review | MEDIUM | Small | None |

**Legend**:
- ✅ Complete: Review document exists and approved
- 📋 Not Started: Review document needs to be created
- 🔄 In Progress: Review document being written
- 🔍 Under Review: Submitted for approval
- ⏸️ Blocked: Waiting on dependencies

## Suggested Review Order

### Phase 1: Independent Systems (No Dependencies)
1. **Timing System** (✅ Already complete)
2. **Testing Suite Organization** - Establishes test framework
3. **Array System & Mathematical Mixins** - Core infrastructure
4. **Parallel-Safe System** - Independent utility system

### Phase 2: Dependent Systems
5. **Units-Awareness System** - Depends on array system (#2)
6. **Non-Dimensionalization** - Depends on units (#3)

### Phase 3: Advanced Systems
7. **Expression Unwrapping** - Depends on units (#3) and non-dim (#6)
8. **Function Evaluation System** - Can incorporate learnings from other reviews

### Phase 4: Already Complete
9. **Reduction Operators** (✅ Already complete)

### Phase 5: Process Infrastructure (Meta-Review)
10. **Review System Infrastructure** (🔍 Pilot - tests the system on itself!)

## Progress Tracking

### Completed Reviews: 3/10 (30%)
- ✅ Timing System
- ✅ Reduction Operators
- ✅ Expression Unwrapping (existed previously)

### Submitted for Review: 7/10 (70%)
- 🔍 Function Evaluation System (2025-11-17)
- 🔍 Array System & Mathematical Mixins (2025-11-17)
- 🔍 Units-Awareness System (2025-11-17)
- 🔍 Testing Suite Organization (2025-11-17)
- 🔍 Non-Dimensionalization System (2025-11-17)
- 🔍 Parallel-Safe System (2025-11-17)
- 🔍 Review System Infrastructure (2025-11-17) **← PILOT REVIEW**

### Not Started: 0/10 (0%)

**Target Completion**: All reviews submitted 2025-11-17
**Pilot Review**: Review System Infrastructure will be first review submitted through new GitHub workflow

## Review Creation Guidelines

When creating a new review document:

1. **Copy template** from `CODE-REVIEW-PROCESS.md`
2. **Create in** `docs/reviews/2025-11/[TOPIC]-REVIEW.md`
3. **Include sections**:
   - Overview and scope
   - Changes made (code, docs, tests)
   - Testing verification
   - Sign-off table
   - Known limitations
4. **Link all** related files and documentation
5. **Provide** testing instructions
6. **Update** this tracking index

## Notes

- **Iterative Process**: Reviews may need updates as requirements evolve
- **Time-Sensitive**: These represent November 2025 state; future changes get new reviews
- **Community Approval**: External sign-off required for each review
- **Permanent Archive**: Once approved, reviews are preserved indefinitely

## Contact

For questions about specific reviews:
- **Review Coordinator**: [To be assigned]
- **Project Lead**: [Name/Contact]

For general process questions:
- See [`CODE-REVIEW-PROCESS.md`](../../developer/CODE-REVIEW-PROCESS.md)

---

**Last Updated**: 2025-11-17
**Maintained By**: Project Leadership
**Next Review**: Update after each review completion
