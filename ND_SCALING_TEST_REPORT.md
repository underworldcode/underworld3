# Non-Dimensional Scaling Test Suite Report
**Date**: 2025-10-23
**Status**: Overall Quality **B** (31/33 tests passing, 94% pass rate)

---

## Test Quality Rating System

- **A Quality**: Production-ready, comprehensive, all passing
- **B Quality**: Good coverage, mostly passing, minor issues
- **C Quality**: Incomplete, debug code, needs work

---

## Official Test Suite (`tests/` directory)

### ✅ A-Quality Tests (All Passing)

#### `test_0814_dimensionality_nondimensional.py` - **12/12 PASSING** ✅
**Quality**: **A** - Production ready

Tests comprehensive dimensionality tracking and ND transformations:
- ✅ Basic dimensionality tracking
- ✅ Manual reference scaling
- ✅ Scalar unwrap preserves function symbols
- ✅ Vector unwrap operations
- ✅ Derivative unwrap
- ✅ Multi-variable expressions
- ✅ Gradient unwrap
- ✅ Mixed dimensional/non-dimensional expressions
- ✅ Scaling coefficient visibility
- ✅ Automatic scale derivation
- ✅ UWQuantity dimensionality
- ✅ Roundtrip conversion

**Issues**: None - excellent coverage of core functionality

---

#### `test_0817_poisson_nd.py` - **4/4 PASSING** ✅
**Quality**: **A** - Production ready

Tests Poisson solver with ND scaling:
- ✅ Dimensional vs non-dimensional comparison (0.25 diffusivity)
- ✅ Dimensional vs non-dimensional comparison (0.1 diffusivity)
- ✅ Poisson with source term
- ✅ Scaling improves conditioning

**Issues**: None - validates solver integration

---

#### `test_0818_stokes_nd.py` - **5/5 PASSING** ✅
**Quality**: **A-** - Production ready with deprecation warnings

Tests Stokes solver with ND scaling:
- ✅ Dimensional vs non-dimensional (resolution=8)
- ✅ Dimensional vs non-dimensional (resolution=16)
- ✅ Buoyancy-driven flow
- ✅ Variable viscosity
- ✅ Scaling derives pressure scale

**Issues**:
- ⚠️ **Deprecation warnings**: Uses old BC API `add_dirichlet_bc((1.0, 0.0), "Top", (0, 1))`
- **Action**: Update to new API: `add_dirichlet_bc((1.0, 0.0), "Top")` or use `sympy.oo`

---

### ⚠️ B-Quality Tests (Partial Failures)

#### `test_0816_global_nd_flag.py` - **10/12 PASSING** ⚠️
**Quality**: **B** - Good but 2 test failures

**Passing Tests** (10):
- ✅ Global flag default state
- ✅ Global flag can be enabled
- ✅ Global flag can be disabled
- ✅ Global flag toggle
- ✅ Unwrap without scaling
- ✅ Scaling coefficient always computed
- ✅ Backward compatibility
- ✅ Flag state persistence
- ✅ No scaling without reference quantities
- ✅ Vector variable scaling

**Failing Tests** (2):
- ❌ `test_unwrap_with_scaling` - Scaling factors not appearing in unwrapped expression
  ```
  Expected: "1000" or "0.001" in unwrapped string
  Got: "{ \hspace{ 0.0077pt } {T} }(N.x, N.y)" (no scaling factor)
  ```

- ❌ `test_multiple_variables_scaling` - Combined scaling factors missing
  ```
  Expected: "e-" or "1.0e-12" in expression
  Got: "{ \hspace{ 0.0085pt } {T} }(N.x, N.y)*{ \hspace{ 0.0085pt } {p} }(N.x, N.y)"
  ```

**Root Cause**: The `unwrap_nd_scaling()` function is not including scaling coefficients in the output expression when global ND flag is active.

**Impact**: Medium - unwrap functionality works but doesn't expose scaling factors for inspection/validation

**Action Required**:
1. Fix `unwrap_nd_scaling()` to multiply by scaling coefficients
2. Or update tests to match actual unwrap behavior if current behavior is intended

---

## Development/Debug Tests (root directory)

### C-Quality Tests (Debug/Exploration Code)

#### `test_nd_unwrap_validation.py` - **7/8 PASSING** ⚠️
**Quality**: **C** - Debug script, one failure

**Passing Tests** (7):
- ✅ Basic scalar variable unwrap
- ✅ Vector variable unwrap
- ✅ Derivative unwrap
- ✅ Multi-variable expression
- ✅ Gradient unwrap
- ✅ Mixed dimensional/non-dimensional
- ✅ Scaling coefficient visibility

**Failing Test** (1):
- ❌ `test_complex_equation` - Shape mismatch
  ```python
  equation = grad_T + p_nd.sym  # (1,2) + (1,1) shape error
  ```

**Issues**:
- Shape mismatch: trying to add gradient (vector) to scalar pressure
- Not a pytest test - runs as script
- Has MPI finalization errors in stderr

**Action Required**:
- Fix shape mismatch (add scalar to scalar, or clarify test intent)
- Convert to proper pytest format or remove if redundant
- Located in root dir - should move to `tests/` if keeping

---

#### Script-Based Validation Tests
**Quality**: **C** - Not integrated into pytest

- `test_reference_quantity_validation.py` - **Script** (not pytest)
- `test_model_validation_method.py` - **Script** (not pytest)

**Issues**:
- Execute as standalone scripts, not pytest tests
- No pass/fail reporting to pytest
- Located in root directory

**Action Required**:
- Convert to pytest format with proper test functions
- Move to `tests/` directory
- Add to CI/CD pipeline

---

#### Debug/Exploration Scripts
**Quality**: **C** - Temporary development code

Files in root directory:
- `test_array_scaling_debug.py` - Development script
- `test_bc_scaling.py` - BC scaling exploration
- `test_derivative_scaling_analysis.py` - Derivative scaling analysis
- `test_nd_array_storage.py` - Array storage testing
- `test_nd_comprehensive_suite.py` - Comprehensive tests (script)
- `test_pressure_scaling_pattern.py` - Pressure scaling exploration
- `test_projection_nd_scaling.py` - Projection scaling tests
- `test_stokes_manual_vs_auto_nd.py` - Manual vs auto comparison
- `test_stokes_nd_debug.py` - Stokes ND debugging

**Issues**:
- Development/debug code left in root directory
- Not integrated into pytest suite
- Unclear which are still relevant
- Should be in `tests/` or archived

**Action Required**:
1. **Review each script** - determine if still needed
2. **Convert useful tests** to pytest format and move to `tests/`
3. **Archive obsolete tests** to `tests/archived/` or delete
4. **Keep only active** development in root with clear naming

---

## Summary of Issues

### Critical (Blocks Production)
None - core functionality working

### High Priority (Should Fix Soon)
1. **test_0816_global_nd_flag.py failures** - Unwrap not exposing scaling factors (2 tests)
2. **Deprecation warnings** - Update BC API in test_0818_stokes_nd.py
3. **Root directory clutter** - ~12 test scripts not in pytest suite

### Medium Priority (Nice to Have)
4. **test_nd_unwrap_validation.py** - Fix shape mismatch, convert to pytest
5. **Validation scripts** - Convert to pytest format
6. **Test organization** - Move/archive debug scripts

---

## Recommendations

### Immediate Actions (This Week)
1. **Fix test_0816 failures**: Update `unwrap_nd_scaling()` or adjust test expectations
2. **Update BC API**: Remove deprecated component specification in test_0818
3. **Archive decision**: Move obsolete debug scripts out of root directory

### Short Term (This Month)
4. **Pytest conversion**: Convert validation scripts to proper pytest tests
5. **Test cleanup**: Organize tests/ directory, remove duplicates
6. **Documentation**: Add docstrings explaining what each test validates

### Long Term (Next Quarter)
7. **Integration tests**: Add end-to-end workflow tests with ND scaling
8. **Performance tests**: Benchmark ND vs dimensional solve times
9. **Error testing**: Comprehensive testing of edge cases and error paths

---

## Test Coverage Assessment

### Well Covered ✅
- Basic ND transformations and unwrap
- Poisson solver with ND scaling
- Stokes solver with ND scaling
- Reference quantity validation (warnings)
- Dimensionality tracking

### Needs Coverage ⚠️
- Advection-diffusion with ND scaling
- Time-dependent problems with ND scaling
- Complex multi-material ND problems
- Swarm variables with ND scaling
- Projection system with ND scaling (has debug test, not in suite)

### Missing Coverage ❌
- Error recovery and edge cases
- Performance/conditioning benchmarks
- Documentation examples as tests
- Tutorial notebook validation

---

## Overall Grade: **B** (94% pass rate)

**Strengths**:
- Core ND functionality well tested (31/33 tests passing)
- Good solver integration coverage
- Comprehensive dimensionality tracking tests

**Weaknesses**:
- Root directory cluttered with debug scripts
- Some tests not in pytest suite
- Missing coverage for advanced features
- Deprecation warnings need addressing

**Path to A Grade**:
1. Fix 2 failing tests in test_0816
2. Clean up root directory test scripts
3. Convert validation scripts to pytest
4. Add missing coverage areas
5. Update deprecated API usage
