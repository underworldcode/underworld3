# Test Classification System - Implementation Complete
## 2025-11-15

## Executive Summary

Successfully implemented a flexible dual-classification system for tests that separates **what** is being tested (levels) from **how much to trust it** (tiers), with explicit level markers independent of number prefixes.

## System Design

### Three-Dimensional Classification

1. **Number Prefix (0000-9999)**: Organization/ordering by topic
   - 0000-0499: Core functionality
   - 0700-0799: Units system
   - 1010-1099: Stokes solvers
   - etc.

2. **Level Markers** (`@pytest.mark.level_N`): Complexity **independent** of number
   - Level 1: Quick tests (~seconds) - imports, setup, no solving
   - Level 2: Intermediate (~minutes) - integration, simple solves
   - Level 3: Physics (~minutes to hours) - benchmarks, complex solves

3. **Tier Markers** (`@pytest.mark.tier_X`): Reliability/trust
   - Tier A: Production-ready (TDD-safe)
   - Tier B: Validated (use with caution)
   - Tier C: Experimental (development only)

### Key Innovation: Decoupled Organization from Complexity

**Example**: File `test_1010_stokes_basic.py` (Stokes topic, 1010 range)
- Can contain Level 1 tests (just setup, no solve) - **fast**
- Can contain Level 3 tests (full benchmark) - **slow**
- Both live together, organized by topic
- Can run all Level 1 tests across ALL topics: `pytest -m level_1`

**Benefits**:
- Thematic organization: All Stokes tests together regardless of complexity
- Flexible execution: "Run all quick tests" works across entire suite
- Natural workflow: Add simple validation tests to complex solver files

## Implementation Status

### ✅ Completed

1. **pytest.ini Updated**:
   - Added `level_1`, `level_2`, `level_3` markers
   - Added `tier_a`, `tier_b`, `tier_c` markers
   - Documentation strings for each marker

2. **Documentation Created**:
   - `docs/developer/TESTING-RELIABILITY-SYSTEM.md` - Complete specification
   - `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` - Analysis
   - `CLAUDE.md` - Integrated overview with examples
   - `.claude/commands/` - Slash commands for tier-based testing

3. **JIT Unwrapping Bug Fixed**:
   - `src/underworld3/function/expressions.py` - Fixed `unwrap()` to respect parameters
   - `src/underworld3/utilities/_jitextension.py` - Enhanced debugging output
   - **Result**: test_0818_stokes_nd.py ALL 5 TESTS PASSING ✅

### 🔄 Next Steps (Ready to Execute)

1. **Mark tests with both level AND tier** (can start immediately):
   ```python
   @pytest.mark.level_1  # Quick - no solving
   @pytest.mark.tier_a   # Production-ready
   def test_stokes_create_mesh():
       ...

   @pytest.mark.level_3  # Physics - full benchmark
   @pytest.mark.tier_a   # Production-ready
   def test_stokes_sinking_block():
       ...
   ```

2. **Classify failing tests** (see analysis below)

3. **Promote test_0818_stokes_nd.py to Tier A** (after 1 week validation)

## Current Test Status (After Unwrapping Fix)

### Units Test Suite (test_07*, test_08*)

**Total**: 259 tests
**Passing**: 197 (76%)
**Failing**: 62 (24%)

**Major Win**: test_0818_stokes_nd.py ✅
- Before: 0/5 passing (JIT compilation failure)
- After: 5/5 passing (100%)

### Failure Categories

#### 1. Comparison Operators (28 failures) - **TIER C**
**Files**: test_0810_uwquantity_comparison_operators.py
**Status**: Feature appears unimplemented
**Action**: Mark as `@pytest.mark.tier_c` + `@pytest.mark.xfail(reason="...")`
**Rationale**: Comparison operators (<, >, ==, !=) for UWQuantity not fully working

#### 2. Reduction Operations (12 failures) - **INVESTIGATE**
**Files**: test_0850_comprehensive_reduction_operations.py, test_0851_std_reduction_method.py
**Status**: Was documented as "All Passing" in October, now failing
**Action**: Investigate what changed - likely Tier B (real bug to fix)
**Rationale**: These were working, something broke them

#### 3. Swarm Integration Statistics (6 failures) - **INVESTIGATE**
**Files**: test_0852_swarm_integration_statistics.py
**Status**: Similar to reduction operations
**Action**: Investigate - likely Tier B
**Rationale**: Recently added feature, may have environmental issues

#### 4. Units Propagation (11 failures) - **TIER C**
**Files**: test_0850_units_propagation.py
**Status**: Advanced units features
**Action**: Review if feature is complete; if not, mark Tier C + xfail
**Rationale**: Symbolic units propagation may still be in development

#### 5. Mesh Variable Ordering (3 failures) - **HIGH PRIORITY TIER B**
**Files**: test_0813_mesh_variable_ordering_regression.py
**Status**: Tests "No Batman" fix, should be working
**Action**: **URGENT** investigation required
**Rationale**: CLAUDE.md says this is fixed, but tests fail

#### 6. Poisson with Units (3 failures) - **TIER B**
**Files**: test_0812_poisson_with_units.py
**Status**: Integration test
**Action**: Investigate - likely related to strict units mode
**Rationale**: Core units + solver integration, should work

#### 7. Misc Units Integration (7 failures) - **TIER B/C**
**Files**: Various (coordinate units, unit conversion utilities, etc.)
**Status**: Mixed - some features incomplete, some real bugs
**Action**: Case-by-case analysis

## Usage Examples

### Quick Pre-Commit Check
```bash
# Run all Level 1 tests, Tier A only (~1-2 min, ultra-safe)
pytest -m "level_1 and tier_a"
```

### Fast TDD Cycle
```bash
# Run Level 1+2, Tier A (skip heavy physics, skip experimental)
pytest -m "(level_1 or level_2) and tier_a" -v
```

### Full Validation Before Release
```bash
# Run all Tier A+B tests (exclude experimental)
pytest -m "tier_a or tier_b" -v
```

### Topic-Specific Testing
```bash
# All Stokes tests that are production-ready
pytest tests/test_1*stokes*.py -m tier_a

# Quick Stokes validation (setup tests only)
pytest tests/test_1*stokes*.py -m level_1
```

### By Number Range (Existing System Still Works)
```bash
# Run by test level using pixi
pixi run underworld-test 1    # 0000-0499
pixi run underworld-test 2    # 0500-0899
pixi run underworld-test 3    # 1000+
```

## Classification Decision Matrix

| Test Characteristics | Number Range | Level | Tier | Example |
|---------------------|--------------|-------|------|---------|
| Basic imports, stable | 0000-0199 | 1 | A | test_0000_imports.py |
| Core units, proven | 0700-0799 | 2 | A | test_0700_units_system.py |
| Units integration, new | 0800-0899 | 2 | B | test_0818_stokes_nd.py |
| Advanced units, incomplete | 0850-0899 | 2 | C | test_0850_units_propagation.py |
| Stokes setup, stable | 1010-1099 | 1 | A | test_1010_stokes_create_vars.py |
| Stokes benchmark, proven | 1010-1099 | 3 | A | test_1010_stokes_benchmark.py |
| Stokes variant, new | 1010-1099 | 3 | B | test_1015_stokes_variant.py |

## Immediate Action Plan

### Phase 1: Triage Failures (This Week)

1. **Quick Wins** - Mark as Tier C + xfail:
   - test_0810_uwquantity_comparison_operators.py (28 tests)
   - test_0850_units_propagation.py (11 tests if incomplete)

2. **Investigate** - Determine Tier B or C:
   - test_0850_comprehensive_reduction_operations.py (12 tests)
   - test_0852_swarm_integration_statistics.py (6 tests)

3. **High Priority** - Fix or understand:
   - test_0813_mesh_variable_ordering_regression.py (3 tests - should work!)
   - test_0812_poisson_with_units.py (3 tests - core integration)

### Phase 2: Mark Passing Tests (This Week)

1. **Tier A Candidates** (high confidence):
   - test_0700_units_system.py (18/21 passing)
   - test_0710_units_utilities.py (40/43 passing)
   - test_0818_stokes_nd.py (5/5 passing - just fixed!)
   - test_0814_strict_units_enforcement.py (9/9 passing)

2. **Apply markers systematically**:
   - Start with clearly passing, stable tests
   - Add level markers based on actual runtime
   - Document any skipped tests

### Phase 3: Promote and Refine (Ongoing)

1. **Promote test_0818_stokes_nd.py to Tier A** (after 1 week)
2. **Review Tier B tests monthly** for promotion to Tier A
3. **Fix or remove Tier C tests** quarterly

## Benefits Demonstrated

1. **Flexible Organization**: Stokes tests can have both quick setup tests (Level 1) and slow benchmarks (Level 3) in same file
2. **Precise Execution**: Can run "all quick tests" or "all trusted tests" across entire suite
3. **Clear Communication**: Tiers tell developers exactly how much to trust failures
4. **Supports Development**: Can add tests for incomplete features without breaking CI (Tier C + xfail)
5. **Natural Workflow**: Organization by topic (numbers), execution by complexity/trust (markers)

## Key Documents

- **System Specification**: `docs/developer/TESTING-RELIABILITY-SYSTEM.md`
- **Current Analysis**: `docs/developer/TEST-CLASSIFICATION-2025-11-15.md`
- **Quick Reference**: `CLAUDE.md` (Test Classification section)
- **Unwrapping Fix**: `UNWRAPPING_BUG_FIX_2025-11-15.md`
- **Setup Summary**: `TEST-RELIABILITY-SYSTEM-SETUP-2025-11-15.md`
