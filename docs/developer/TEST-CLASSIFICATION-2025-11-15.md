# Test Classification Analysis - 2025-11-15

## Integration of Test Levels and Reliability Tiers

Underworld3 uses a **dual classification system** for tests:

### Test Levels (Number Prefix) - Complexity/Scope
Based on existing `scripts/test_levels.sh`:

- **Level 1 (0000-0499)**: Quick tests - core functionality (~2-5 min)
  - Imports, mesh creation, basic operations
  - Essential unit tests

- **Level 2 (0500-0899)**: Intermediate tests (~5-10 min)
  - 0500-0599: Enhanced arrays and data migration
  - 0600-0699: Regression tests (critical for stability)
  - 0700-0799: Units system (mathematical objects, core units)
  - 0800-0899: Unit-aware functionality (integration tests)

- **Level 3 (1000+)**: Physics/solver tests (~10-15 min)
  - 1000-1009: Poisson solvers
  - 1010-1050: Stokes flow solvers
  - 1100-1120: Advection-diffusion and time-stepping
  - Application-driven, not classical unit tests

### Reliability Tiers (Letter Suffix) - Trust Level
New system for test quality/maturity:

- **Tier A**: Production-ready, trusted for TDD
- **Tier B**: Validated but needs more production time
- **Tier C**: Experimental, development only

**Combined Example**: `test_0700_units_system.py` could be marked as:
- Level 2 (intermediate complexity, 07xx range)
- Tier A (production-ready, core functionality)

## Current Status Analysis (2025-11-15)

### After Unwrapping Bug Fix

**Fixed Issues**:
- ✅ test_0818_stokes_nd.py: All 5 tests now PASSING (JIT unwrapping bug fixed)

**Remaining Failures**: 79 tests (running fresh analysis now)

### Preliminary Classification

Based on test file analysis and failure patterns:

#### **Tier A Candidates** (Long-lived, Proven, Core Functionality)

**Level 1 (Core - 0000-0499)**:
- test_0000_imports.py
- test_0001_basic_model.py
- test_0002_basic_swarm.py
- test_0003_save_load.py
- test_0004_simple_meshes.py
- test_0005_IndexSwarmVariable.py
- test_0100_backward_compatible_data.py
- test_0110_basic_swarm.py
- test_0120_data_property_access.py
- test_0130_field_creation.py
- test_0140_synchronised_updates.py

**Level 2 (Intermediate - Core Units)**:
- test_0700_units_system.py (core units, 21 tests - IF passing)
- test_0701_units_dimensionless.py
- test_0702_units_temperature.py
- test_0703_units_pressure.py

**Level 3 (Physics - Established Solvers)**:
- test_1000_PoissonCartesian.py
- test_1001_PoissonAnnulus.py
- test_1010_StokesCartesian.py
- test_1011_StokesCylindrical.py

#### **Tier B Candidates** (Validated, Needs More Time)

**Units Integration (0800-0899)** - Recently implemented:
- test_0801_units_utilities.py (10 tests)
- test_0803_units_workflow_integration.py (3 tests)
- test_0818_stokes_nd.py (NOW PASSING - candidate for A promotion!)

**Regression Tests (0600-0699)** - Created recently:
- test_06xx_regression.py files (46/49 passing last check)
- Need individual review

**Enhanced Features (0500-0599)**:
- test_0500_enhanced_array_structure.py
- test_0510_enhanced_swarm_array.py
- test_0520_mathematical_mixin_enhanced.py

#### **Tier C Candidates** (Experimental, Needs Work)

**Comparison Operators (Not Implemented?)**:
- test_0810_uwquantity_comparison_operators.py (6 failures)
  - **LIKELY**: Feature not fully implemented
  - **ACTION**: Review if comparison operators are supposed to work

**Units Propagation (Advanced Features)**:
- test_0850_units_propagation.py (11 failures)
- test_0850_units_closure_comprehensive.py
  - **LIKELY**: Advanced units features still in development
  - **ACTION**: Mark as xfail if feature incomplete

**Reduction Operations**:
- test_0850_comprehensive_reduction_operations.py (8 failures)
- test_0851_std_reduction_method.py (2 failures)
- test_0852_swarm_integration_statistics.py (6 failures)
  - **STATUS**: Recently added (2025-10-25), documented as "All Passing" in TEST-COVERAGE-ANALYSIS.md
  - **LIKELY**: Environmental issue or recent breakage
  - **ACTION**: Investigate what changed

**Poisson with Units**:
- test_0812_poisson_with_units.py (3 failures)
  - **LIKELY**: Integration issue with units enforcement
  - **ACTION**: Test after units mode clarification

**Mesh Variable Ordering ("No Batman")**:
- test_0813_mesh_variable_ordering_regression.py (3 failures)
  - **CONTEXT**: Tests the fix for DM state corruption bug  - **STATUS**: Should be working per CLAUDE.md
  - **ACTION**: Investigate - these are important regression tests

**Coordinate Units**:
- test_0815_variable_coords_units.py (1 failure)
  - **CONTEXT**: Coordinate units system marked complete in CLAUDE.md
  - **ACTION**: Check if test assumptions are correct

**Global ND Flag**:
- test_0816_global_nd_flag.py (1 failure)
  - **ACTION**: Check interaction with strict units mode

## Next Steps

### Immediate (2025-11-15)

1. ✅ **Fixed**: JIT unwrapping bug (test_0818_stokes_nd.py now passing)
2. 🔄 **Running**: Fresh test run to get current status
3. 📋 **TODO**: Analyze each failure category:
   - Comparison operators: Feature status?
   - Reduction operations: What broke?
   - Mesh variable ordering: Why failing?
   - Units propagation: Expected behavior?

### Short-term (This Week)

1. **Mark Tier A tests**: Add `@pytest.mark.tier_a` to proven tests
2. **Mark Tier C tests**: Add `@pytest.mark.tier_c` + `@pytest.mark.xfail` to incomplete features
3. **Tier B by default**: Everything else gets `@pytest.mark.tier_b`
4. **Update test_levels.sh**: Add tier filtering support

### Medium-term (Next 2 Weeks)

1. **Fix or document all failures**: Each test either passes, is marked xfail with reason, or is removed
2. **Promote test_0818_stokes_nd.py**: Move from B to A (now consistently passing)
3. **Review regression tests**: Validate all test_06xx files
4. **CI Integration**: Set up Tier A as the CI suite

## Test File Inventory

### Level 2 Units Tests (test_07*_units*.py, test_08*_*.py)

**Total**: 78 test files in this range
**Current Status**: 180 passed, 79 failed (before unwrapping fix)

**Categories**:
1. Core units (0700-0709): Foundational units system
2. Units utilities (0800-0809): Helper functions, conversions
3. Units integration (0810-0819): Integration with solvers/features
4. Advanced units (0850-0899): Propagation, closure, reduction ops

### Detailed File List with Preliminary Classification

```
test_0700_units_system.py                               - Tier A candidate (core)
test_0701_units_dimensionless.py                        - Tier A candidate
test_0702_units_temperature.py                          - Tier A candidate
test_0703_units_pressure.py                             - Tier A candidate
test_0704_units_viscosity.py                            - Tier A candidate
test_0801_units_utilities.py                            - Tier B (utilities)
test_0803_units_workflow_integration.py                 - Tier B (integration)
test_0810_uwquantity_comparison_operators.py            - Tier C (not implemented?)
test_0812_poisson_with_units.py                         - Tier B (integration)
test_0813_mesh_variable_ordering_regression.py          - Tier B → A (critical regression)
test_0814_strict_units_enforcement.py                   - Tier B (new feature)
test_0815_variable_coords_units.py                      - Tier B (recently completed)
test_0816_global_nd_flag.py                             - Tier B (integration)
test_0818_stokes_nd.py                                  - Tier B → A (NOW FIXED!)
test_0850_comprehensive_reduction_operations.py         - Tier C (investigate failures)
test_0850_units_closure_comprehensive.py                - Tier C (advanced)
test_0850_units_propagation.py                          - Tier C (advanced)
test_0851_std_reduction_method.py                       - Tier C (investigate)
test_0852_swarm_integration_statistics.py               - Tier C (investigate)
```

## Decision Rules for Classification

### When to Mark Tier A
- [ ] Test has existed for >3 months
- [ ] Test passes consistently
- [ ] Functionality is stable and production-used
- [ ] Test is clear and well-documented
- [ ] Failure would indicate definite bug

### When to Mark Tier B
- [ ] Test passes at least once
- [ ] Functionality appears correct
- [ ] Needs more production validation
- [ ] OR: Test is new (<3 months)

### When to Mark Tier C + xfail
- [ ] Feature is not fully implemented
- [ ] Test documents expected future behavior
- [ ] Test is being developed alongside feature
- [ ] Known issues exist

### When to Remove Test
- [ ] Feature is permanently abandoned
- [ ] Test is redundant with better test
- [ ] Test assumptions are fundamentally wrong
