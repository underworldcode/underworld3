# Test Reliability Classification System

**Last Updated**: 2025-11-15
**Status**: Active - Use for all new tests and test reviews

## Overview

Underworld3 uses a three-tier reliability classification system (A/B/C) to ensure tests are trustworthy and appropriate for their intended use. This system prevents test-driven development from being derailed by unreliable tests and provides clear guidelines for test maturation.

## Reliability Tiers

### Tier A: Production-Ready (Trusted)
**Use for**: Test-Driven Development (TDD), Continuous Integration (CI), Release Validation

**Characteristics**:
- ✅ Long-lived tests with proven track record (>3 months in codebase)
- ✅ Consistently passing across multiple environments
- ✅ Clear, well-documented test intent
- ✅ Tests stable, well-understood functionality
- ✅ Failure indicates DEFINITE regression in production code
- ✅ No known flakiness or environmental sensitivity
- ✅ Reviewed and approved by core maintainers

**Examples**:
- Core Stokes solver tests (test_101*_Stokes*.py)
- Basic mesh creation and data access (test_0100-0199_*.py)
- Fundamental units system tests (test_0700_units_system.py)

**Pytest Marker**: `@pytest.mark.tier_a`

**When to Use**:
- Running full CI pipeline before merging
- Test-driven development sprints
- Release validation
- Bisecting regressions (these tests can be trusted to find the problem)

### Tier B: Validated (Use with Caution)
**Use for**: Feature Validation, Exploratory Testing, Manual Review

**Characteristics**:
- ⚠️ Successfully run at least once, but not yet battle-tested
- ⚠️ Test appears correct but functionality may still be evolving
- ⚠️ Limited production usage or edge case coverage
- ⚠️ May have environmental dependencies not fully documented
- ⚠️ Failure could indicate test OR code issue - requires investigation
- ⚠️ Not yet reviewed for promotion to Tier A

**Examples**:
- Recently added units integration tests (test_08*_*.py - many currently failing)
- New reduction operation tests (test_0850_*.py)
- Feature tests for newly implemented capabilities

**Pytest Marker**: `@pytest.mark.tier_b`

**When to Use**:
- Manual feature validation after implementation
- Exploratory testing of new capabilities
- Code review process (validate test works as intended)
- NOT for automated TDD sprints (unless explicitly monitoring)

**Promotion Path**: B → A
1. Test passes consistently for 3+ months
2. Functionality confirmed stable in production
3. Core maintainer review confirms test quality
4. Add to Tier A suite via PR review

### Tier C: Experimental (Development)
**Use for**: Feature Development, Debugging, Test Development

**Characteristics**:
- 🚧 Test OR code (or both!) may be incorrect
- 🚧 Actively under development
- 🚧 Used to explore expected behavior
- 🚧 May test unimplemented or partially implemented features
- 🚧 Failures are EXPECTED and informative
- 🚧 Not suitable for any automated testing

**Examples**:
- Tests written for not-yet-implemented features
- Exploratory tests to understand API design
- Tests for actively debugged features
- Tests with known issues (mark with `@pytest.mark.xfail` + reason)

**Pytest Markers**:
- `@pytest.mark.tier_c`
- `@pytest.mark.xfail(reason="Feature not yet implemented")`
- `@pytest.mark.skip(reason="Waiting for X to be fixed")`

**When to Use**:
- Feature development (write test first, then implement)
- Debugging complex issues (write test to reproduce bug)
- API design exploration (what SHOULD the behavior be?)
- NEVER for automated CI/TDD

**Promotion Path**: C → B
1. Feature fully implemented
2. Test passes consistently
3. Developer confirms test is correct
4. Remove xfail/skip markers
5. Promote to Tier B for further validation

## Implementation in Pytest

### pytest.ini Configuration

```ini
[pytest]
markers =
    # Reliability tiers (how much to trust the test)
    tier_a: Production-ready tests (trusted, use for TDD and CI)
    tier_b: Validated tests (use with caution, manual review recommended)
    tier_c: Experimental tests (development only, not for automation)

    # Complexity levels (what kind of test, independent of number prefix)
    level_1: Quick core tests - imports, basic setup, no solving (~seconds)
    level_2: Intermediate tests - integration, units, regression (~minutes)
    level_3: Physics tests - solvers, time-stepping, coupled systems (~minutes to hours)

    # Other markers
    mpi: marks tests as requiring MPI
    slow: marks tests as slow (>10s)
```

### Level vs Number Prefix

**IMPORTANT**: The number prefix (0000-9999) is for **organization/ordering only**. The actual complexity level is marked explicitly with `@pytest.mark.level_N`.

**Why this matters**:
- A file `test_1010_stokes_basic.py` can contain both Level 1 (setup) and Level 3 (benchmark) tests
- Allows thematic organization: All Stokes tests in 1010-1099 regardless of complexity
- Can run "all quick tests" across all topics: `pytest -m level_1`

**Example**:
```python
# File: test_1010_stokes_basic.py (number 1010 = Stokes topic)

@pytest.mark.level_1  # Quick - just setup
@pytest.mark.tier_a   # Production-ready
def test_stokes_create_mesh_and_variables():
    """Test creating Stokes mesh and variables (no solving)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    # Just creation - very fast!

@pytest.mark.level_3  # Physics - full solve + benchmark
@pytest.mark.tier_a   # Production-ready
def test_stokes_sinking_block_benchmark():
    """Test Stokes solver against analytical solution."""
    # Complex benchmark with large mesh, comparison to theory
    # Could take minutes!
```

Both tests live in the same file (organized by topic), but have different levels (organized by complexity).

### Marking Tests

```python
import pytest

# Level 1 + Tier A: Quick, production-ready
@pytest.mark.level_1
@pytest.mark.tier_a
def test_basic_mesh_creation():
    \"\"\"Test mesh creation with default parameters.\"\"\"
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    assert mesh.dim == 2
    assert mesh.elementCount > 0

# Level 2 + Tier B: Intermediate, validated but new
@pytest.mark.level_2
@pytest.mark.tier_b
def test_units_integration_with_stokes():
    \"\"\"Test Stokes solver with unit-aware variables.

    Status: Passing locally, needs more production validation.
    \"\"\"
    # Test implementation...

# Level 3 + Tier A: Complex physics, production-ready
@pytest.mark.level_3
@pytest.mark.tier_a
def test_stokes_benchmark_sinking_block():
    \"\"\"Test Stokes solver against analytical solution.

    Validated against Gerya (2019) textbook solution.
    \"\"\"
    # Benchmark implementation...

# Level 2 + Tier C: Intermediate complexity, experimental feature
@pytest.mark.level_2
@pytest.mark.tier_c
@pytest.mark.xfail(reason="Advanced units propagation not yet implemented")
def test_symbolic_units_propagation():
    \"\"\"Test automatic unit propagation through symbolic operations.

    This test documents expected behavior for future implementation.
    \"\"\"
    # Test for future feature...
```

### Running Tests by Tier

```bash
# Run only Tier A tests (safe for TDD)
pytest -m tier_a

# Run Tier A and B tests (full validation)
pytest -m "tier_a or tier_b"

# Run all tests including experimental (for development)
pytest

# Exclude experimental tests
pytest -m "not tier_c"
```

## Current Test Classification Status

### 2025-11-15 Audit Results

**Units Test Suite (test_07*_units*.py, test_08*_*.py)**:
- **Total**: 259 tests
- **Passing**: 180 (69.5%)
- **Failing**: 79 (30.5%)

**Immediate Actions Required**:
1. ✅ **DONE**: Fixed Stokes JIT unwrapping bug (test_0818_stokes_nd.py now passing)
2. 🔄 **IN PROGRESS**: Classify remaining 79 failures as B or C
3. 📋 **TODO**: Eliminate tests for unimplemented features (move to C or remove)
4. 📋 **TODO**: Fix legitimate test failures or mark as xfail with clear reasons

## Test Review Process

### For New Tests (PR Review)

**Checklist**:
- [ ] Test has clear docstring explaining intent
- [ ] Test has appropriate tier marker (start at C, promote through review)
- [ ] If Tier C/xfail: Reason clearly documented
- [ ] Test follows project conventions (naming, structure)
- [ ] Test is not redundant with existing tests
- [ ] If testing edge case: Edge case clearly documented

### For Promoting Tests (C → B → A)

**C → B Promotion**:
- [ ] Feature fully implemented
- [ ] Test passes consistently (developer verified)
- [ ] Test correctly validates intended behavior
- [ ] Remove xfail/skip markers
- [ ] Update tier marker to `@pytest.mark.tier_b`

**B → A Promotion** (Requires Core Maintainer Review):
- [ ] Test has passed for 3+ months without modification
- [ ] Functionality confirmed stable in production use
- [ ] Test quality reviewed (clear, maintainable, appropriate assertions)
- [ ] No known environmental flakiness
- [ ] Update tier marker to `@pytest.mark.tier_a`

## Guidelines for Test Development

### When Writing a New Test

1. **Start at Tier C**: All new tests begin as experimental
2. **Document Intent**: Clear docstring explaining what behavior is tested
3. **Use xfail Appropriately**: If testing unimplemented feature, mark with xfail
4. **Don't Break CI**: Tier C tests with xfail won't break automated testing
5. **Promote Deliberately**: Don't rush promotion - let tests prove reliability

### When a Test Fails

**If Tier A Test Fails**:
- 🚨 **HIGH PRIORITY**: Definite regression in production code
- Investigate immediately
- Bisect to find breaking commit
- Fix production code or demote test if it was incorrectly promoted

**If Tier B Test Fails**:
- ⚠️ **MEDIUM PRIORITY**: Could be test OR code issue
- Investigate to determine root cause
- If code issue: Fix code
- If test issue: Fix test or demote to Tier C
- Document findings in test or issue tracker

**If Tier C Test Fails**:
- ℹ️ **EXPECTED**: Tier C tests may fail
- No immediate action required
- Useful for tracking development progress
- Update xfail reason if expectations change

## Integration with Slash Commands

The following slash commands help manage test reliability:

- **`/test-solvers`**: Run Tier A solver tests (trusted for validation)
- **`/test-units`**: Run Tier A+B units tests (quick validation)
- **`/test-regression`**: Run full Tier A suite (check for regressions)
- **`/validate-docs`**: Check documentation test coverage

## Migration Plan for Existing Tests

### Phase 1: Initial Classification (2025-11-15 to 2025-12-01)

1. **Classify all existing tests**:
   - Simple tests (0000-0199): Review for Tier A
   - Intermediate tests (0500-0699): Review for Tier A/B
   - Regression tests (0600-0699): Review for Tier B (→A after validation)
   - Units tests (0700-0799): Classify failures as B or C
   - Complex tests (1000+): Review for Tier A/B

2. **Add markers to all test files**
3. **Update pytest.ini with tier markers**
4. **Document known issues with xfail**

### Phase 2: Stabilization (2025-12-01 to 2026-01-01)

1. **Fix or document all Tier B test failures**
2. **Remove or mark xfail for Tier C tests**
3. **Promote stable Tier B tests to Tier A**
4. **Establish CI pipeline using Tier A tests**

### Phase 3: Maintenance (Ongoing)

1. **Regular review of Tier B tests for promotion**
2. **Continuous monitoring of Tier A test stability**
3. **Documentation updates for test coverage**

## Rationale

**Why This System?**

1. **Prevents TDD Confusion**: Developers know which tests to trust
2. **Documents Test Maturity**: Clear progression from experimental to production
3. **Reduces False Alarms**: Tier C failures expected, Tier A failures are urgent
4. **Guides Review Process**: Clear criteria for test promotion
5. **Supports Development**: Can write tests for future features without breaking CI

**Comparison to Other Approaches**:
- **Better than xfail alone**: Three tiers provide more nuanced classification
- **Better than skip**: Tests still run, failures are informative
- **Better than no classification**: Prevents "all tests are equal" assumption

## References

- **Pytest Markers**: https://docs.pytest.org/en/stable/how-to/mark.html
- **Test Coverage Analysis**: `docs/reviews/2025-10/TEST-COVERAGE-ANALYSIS.md`
- **Project Test Organization**: `CLAUDE.md` (Test Suite Organization section)
