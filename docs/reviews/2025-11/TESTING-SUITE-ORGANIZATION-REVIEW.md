# Testing Suite Organization Review

**Review ID**: UW3-2025-11-005
**Date**: 2025-11-17
**Status**: Submitted for Review
**Component**: Testing Infrastructure
**Reviewer**: [To be assigned]

## Overview

This review covers the reorganization and classification of the Underworld3 test suite to support efficient development workflows, Test-Driven Development (TDD), and CI/CD integration. The new system introduces dual classification (test levels and reliability tiers) with pytest markers, enabling flexible test execution strategies.

## Changes Made

### Code Changes

**Test Organization**:
- `tests/test_0000_imports.py` → Simple import tests (Level 1, Tier A)
- `tests/test_0100-0199_*.py` → Basic functionality tests (Level 1)
- `tests/test_0500-0699_*.py` → Intermediate tests (Level 2)
- `tests/test_0700-0899_*.py` → Units system tests (Level 2)
- `tests/test_1000+_*.py` → Physics/solver tests (Level 3)

**Configuration**:
- `pytest.ini` - Added marker definitions for levels and tiers
- `scripts/test_levels.sh` - Test execution script by complexity

### Documentation Changes

**Created**:
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md` - Complete system documentation
- `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` - Classification analysis
- Updated test docstrings with classification information

### Test Coverage

**Total Tests**: ~150 tests across all levels
**Classification Coverage**: All tests now have level markers
**New Test Markers**:
```python
@pytest.mark.level_1  # Quick core tests
@pytest.mark.level_2  # Intermediate tests
@pytest.mark.level_3  # Physics/solver tests
@pytest.mark.tier_a   # Production-ready (TDD-safe)
@pytest.mark.tier_b   # Validated (use with caution)
@pytest.mark.tier_c   # Experimental (development only)
```

## System Architecture

### Dual Classification System

#### Dimension 1: Test Levels (Complexity/Scope)

**Level 1 - Quick Core Tests**:
- **Purpose**: Fast validation of basic functionality
- **Runtime**: Seconds
- **Examples**: Imports, basic setup, simple operations
- **No Solving**: Tests that don't run physics solvers
- **Use Case**: Pre-commit checks, rapid feedback

**Level 2 - Intermediate Tests**:
- **Purpose**: Integration, units, regression validation
- **Runtime**: Minutes
- **Examples**: Units system, enhanced arrays, simple solves
- **Use Case**: Feature validation, integration testing

**Level 3 - Physics/Solver Tests**:
- **Purpose**: Full physics validation, benchmarks
- **Runtime**: Minutes to hours
- **Examples**: Stokes benchmarks, time-stepping, coupled systems
- **Use Case**: Release validation, comprehensive testing

#### Dimension 2: Reliability Tiers (Trust Level)

**Tier A - Production-Ready**:
- **Criteria**:
  - Long-lived (>3 months), consistently passing
  - Trusted for TDD and CI
  - Failure indicates DEFINITE regression
- **Use Case**: Safe for test-driven development
- **Examples**: Core Stokes tests, basic mesh creation

**Tier B - Validated**:
- **Criteria**:
  - Passed at least once, not battle-tested
  - New features (<3 months) or recently refactored
  - Failure could be test OR code issue
- **Use Case**: Full validation suites
- **Examples**: New units integration, recent features

**Tier C - Experimental**:
- **Criteria**:
  - Feature may not be fully implemented
  - Test OR code (or both) may be incorrect
  - Mark with `@pytest.mark.xfail` if expected to fail
- **Use Case**: Development only, not CI
- **Examples**: Unimplemented features, active development

### Test Numbering System

**Organization by Topic** (0000-9999):
- `0000-0499`: Core functionality (imports, meshes, data access)
- `0500-0599`: Enhanced arrays and migration
- `0600-0699`: Regression tests
- `0700-0799`: Units system
- `0800-0899`: Unit-aware integration
- `1000-1099`: Poisson/Darcy solvers
- `1100-1199`: Stokes flow
- `1200+`: Advection-diffusion, coupled systems

**Key Insight**: Number prefix organizes by topic, NOT complexity. A file like `test_1010_stokes_*.py` can contain both Level 1 (setup) and Level 3 (benchmark) tests.

### Execution Strategies

#### By Complexity Level
```bash
# Quick validation (< 2 min)
pytest -m level_1

# Intermediate validation (< 10 min)
pytest -m level_2

# Skip heavy physics
pytest -m "level_1 or level_2"

# Full suite
pytest -m "level_1 or level_2 or level_3"
```

#### By Reliability Tier
```bash
# TDD-safe only
pytest -m tier_a

# Full validation (exclude experimental)
pytest -m "tier_a or tier_b"
pytest -m "not tier_c"
```

#### Combined Filtering
```bash
# Fast TDD cycle
pytest -m "(level_1 or level_2) and tier_a"

# All Stokes tests that are production-ready
pytest tests/test_1*stokes*.py -m tier_a

# Quick checks excluding experimental
pytest -m "level_1 and not tier_c"
```

#### By Number Range (Legacy Support)
```bash
# Still works via pixi tasks
pixi run underworld-test 1  # 0000-0499
pixi run underworld-test 2  # 0500-0899
pixi run underworld-test 3  # 1000+
```

## Key Features

### 1. Flexible Test Selection

**Problem Solved**: Previously, no way to run "quick tests only" or "TDD-safe tests only"

**Solution**: Pytest markers enable arbitrary filtering:
- Want quick feedback? Run Level 1 only
- Want TDD safety? Run Tier A only
- Want comprehensive validation? Run Tier A + B

### 2. Clear Trust Levels

**Problem Solved**: Hard to know which test failures indicate real regressions vs test issues

**Solution**: Tier system explicitly marks trust level:
- Tier A failure → Investigate code immediately
- Tier B failure → Could be test or code
- Tier C failure → Expected during development

### 3. Development Workflow Support

**Problem Solved**: Running full suite takes too long for iterative development

**Solution**: Developers can run relevant subsets:
- Working on units? `pytest -m level_2 tests/test_07*.py`
- Pre-commit check? `pytest -m "level_1 and tier_a"`
- Feature validation? `pytest -m "tier_a or tier_b"`

### 4. CI/CD Integration

**Recommended CI Stages**:
```yaml
# Stage 1: Fast feedback (< 2 min)
- pytest -m "level_1 and tier_a"

# Stage 2: Intermediate (< 10 min)
- pytest -m "level_2 and (tier_a or tier_b)"

# Stage 3: Full validation (nightly)
- pytest -m "(level_1 or level_2 or level_3) and not tier_c"
```

## Test Migration Examples

### Before (No Classification)
```python
def test_stokes_solver():
    """Test Stokes solver."""
    # Could be quick setup or slow benchmark - no way to tell
    pass
```

### After (With Classification)
```python
@pytest.mark.level_1
@pytest.mark.tier_a
def test_stokes_mesh_creation():
    """Test creating Stokes mesh and variables (no solving)."""
    # Clear: Quick setup, production-ready
    pass

@pytest.mark.level_3
@pytest.mark.tier_a
def test_stokes_benchmark():
    """Test Stokes solver against analytical solution."""
    # Clear: Full physics, production-ready
    pass
```

## Testing Instructions

### Verify Marker System
```bash
# Check markers are defined
pytest --markers | grep -E "level_|tier_"

# Should show:
# level_1: Quick core tests (seconds)
# level_2: Intermediate tests (minutes)
# level_3: Physics/solver tests (minutes to hours)
# tier_a: Production-ready (TDD-safe)
# tier_b: Validated (use with caution)
# tier_c: Experimental (development only)
```

### Run Classification Tests
```bash
# Quick tests only
pytest -m level_1 -v

# TDD-safe tests only
pytest -m tier_a -v

# Combined: Fast TDD validation
pytest -m "level_1 and tier_a" -v
```

### Verify Legacy Support
```bash
# Number-based execution still works
pixi run underworld-test 1
pixi run underworld-test 2
```

## Known Limitations

### 1. Incomplete Tier Classification

**Status**: Not all tests have tier markers yet
**Impact**: Some tests only have level markers
**Plan**: Gradually add tier markers as tests mature

### 2. Tier Assignment Subjectivity

**Challenge**: "When is a test Tier A vs Tier B?" requires judgment
**Guideline**:
- Tier A: >3 months old, consistently passing, failure is alarming
- Tier B: <3 months old OR recently modified OR complex feature
- Tier C: Known issues or incomplete implementation

### 3. Level vs Number Confusion

**Potential Issue**: Number prefix ≠ Level marker can be confusing
**Clarification**:
- Number: Topic organization (1000-1099 = Poisson)
- Level: Complexity (Level 1 = fast, Level 3 = slow)
- A Poisson file can have both Level 1 and Level 3 tests

## Benefits

### For Developers
- **Fast feedback**: Run Level 1 in < 2 minutes
- **Confident TDD**: Run Tier A tests knowing failures matter
- **Focused testing**: Test only relevant areas during development

### For CI/CD
- **Staged validation**: Fast→Medium→Slow pipeline stages
- **Resource optimization**: Don't run slow tests on every commit
- **Clear failure signals**: Tier A failures block merges

### For Project Management
- **Test quality metrics**: Track Tier A coverage growth
- **Regression detection**: Tier A tests are regression sentinels
- **Development velocity**: Fast tests enable rapid iteration

## Related Documentation

- `docs/developer/TESTING-RELIABILITY-SYSTEM.md` - Complete system guide
- `docs/developer/TEST-CLASSIFICATION-2025-11-15.md` - Classification analysis
- `scripts/test_levels.sh` - Test execution script
- `pytest.ini` - Marker definitions

## Recommendations

### For New Tests
1. **Always add level marker**: Choose level_1, level_2, or level_3
2. **Add tier marker when ready**: Start as tier_b, promote to tier_a after 3 months
3. **Use descriptive names**: Test name should indicate what it tests
4. **Add xfail for incomplete**: `@pytest.mark.xfail` with reason if not working

### For CI/CD
1. **Stage 1 (Fast)**: `pytest -m "level_1 and tier_a"` (< 2 min)
2. **Stage 2 (Medium)**: `pytest -m "level_2 and tier_a"` (< 10 min)
3. **Stage 3 (Full)**: `pytest -m "not tier_c"` (nightly)

### For Test Promotion
1. **Tier B → Tier A**: After 3 months of consistent passing
2. **Tier C → Tier B**: When feature implementation complete
3. **Level adjustment**: If test runtime changes significantly

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | AI Assistant | 2025-11-17 | Submitted |
| Primary Reviewer | [To be assigned] | | Pending |
| Secondary Reviewer | [To be assigned] | | Pending |
| Project Lead | [To be assigned] | | Pending |

## Review Comments and Resolutions

[To be filled during review process]

---

**Review Status**: Awaiting assignment of reviewers
**Expected Completion**: [TBD]
**Priority**: MEDIUM

This review documents a critical testing infrastructure improvement that enables efficient development workflows and reliable quality assurance.
