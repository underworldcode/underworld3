# Lambdification Optimization Test Coverage

**Test File**: `tests/test_0720_lambdify_optimization_paths.py`
**Created**: 2025-11-17
**Purpose**: Document and validate automatic lambdification optimization paths

## Test Summary

**Total Tests**: 20
**Status**: ✅ All passing
**Run Time**: ~0.88 seconds

## Test Categories

### 1. Pure SymPy Expressions (3 tests)
Tests that simple mathematical expressions use the fast lambdified path:
- `test_simple_polynomial` - Polynomial expressions (x**2 + 2*x + 1)
- `test_multiple_variables` - Multiple symbols (x**2 + y**2)
- `test_constant_expression` - Constant values (3.14)

**Expected**: All should use lambdification (~10,000x faster than substitution)

### 2. SymPy Built-in Functions (3 tests)
Tests that SymPy library functions are recognized and lambdified:
- `test_erf_function` - Error function erf(5*x - 2)
- `test_trigonometric_functions` - sin() and cos() functions
- `test_exponential_function` - exp() function

**Expected**: Should NOT be rejected as UW3 Functions (module-based detection)

### 3. Mesh Coordinates (2 tests)
Tests that BaseScalar mesh coordinates use lambdification:
- `test_mesh_coordinates_simple` - Basic coordinate expressions
- `test_mesh_coordinates_complex` - Complex coordinate expressions

**Expected**: BaseScalars are pure sympy, should be lambdified

### 4. UW3 MeshVariables (2 tests)
Tests that actual mesh data uses RBF interpolation (NOT lambdified):
- `test_mesh_variable_symbol` - Direct MeshVariable access (T.sym[0])
- `test_mesh_variable_in_expression` - Mixed with coordinates

**Expected**: Should use RBF interpolation path (correct for actual data)

### 5. UWexpression Parameters (2 tests)
Tests automatic substitution of UWexpression symbols:
- `test_uwexpression_numeric` - Numeric UWexpression (alpha = 0.1)
- `test_uwexpression_in_sympy_function` - UWexpression in functions

**Expected**: UWexpression symbols substituted, then lambdified

### 6. rbf Flag Behavior (2 tests)
Tests that rbf flag doesn't affect pure sympy optimization:
- `test_rbf_false_pure_sympy` - Pure sympy with rbf=False should still be fast
- `test_rbf_false_mesh_variable` - MeshVariable with rbf=False should use RBF

**Expected**: rbf flag only matters for actual mesh data, not pure math

### 7. Detection Mechanism (4 tests)
Tests the `is_pure_sympy_expression()` detection logic:
- `test_detection_pure_sympy` - Detects pure sympy correctly
- `test_detection_mesh_coordinates` - Detects BaseScalar as pure
- `test_detection_sympy_function` - Detects SymPy functions as pure
- `test_detection_uw3_variable` - Detects UW3 variables as NOT pure

**Expected**: Accurate classification of expression types

### 8. Performance Expectations (2 tests)
Tests that performance is as expected:
- `test_lambdify_caching` - Cached evaluations should be fast
- `test_rbf_false_not_slow` - rbf=False should not bypass optimization

**Expected**:
- Cached calls < 10ms for small evaluations
- rbf=False with pure sympy should be fast (< 1s)

## Key Optimization Paths Documented

### Path 1: Lambdification (Fast - ~0.001s for 100 points)
**When**: Pure sympy expressions, SymPy functions, mesh coordinates
**Detection**: `is_pure_sympy_expression()` returns True
**Performance**: ~10,000x faster than substitution
**Examples**:
- `x**2 + y**2`
- `sympy.erf(5*x - 2)`
- `mesh.X[0]**2 + mesh.X[1]**2`

### Path 2: RBF Interpolation (Correct for data - ~0.01s for 100 points)
**When**: Expressions with UW3 MeshVariable/SwarmVariable data
**Detection**: `is_pure_sympy_expression()` returns False
**Performance**: Slower but necessary for interpolating mesh data
**Examples**:
- `T.sym[0]` (where T is a MeshVariable)
- `T.sym[0] + mesh.X[0]**2` (mixed expression)

### Path 3: Old Substitution Path (Slow - ~20s for 100 points - BYPASSED)
**When**: Should never happen with current implementation
**Why avoided**: Fixed rbf flag logic ensures pure sympy always uses lambdification
**Previous bug**: `rbf=False` would bypass lambdification incorrectly

## Running the Tests

```bash
# Run all lambdify optimization tests
pytest tests/test_0720_lambdify_optimization_paths.py -v

# Run specific test class
pytest tests/test_0720_lambdify_optimization_paths.py::TestSympyFunctions -v

# Run specific test
pytest tests/test_0720_lambdify_optimization_paths.py::TestRBFFlagBehavior::test_rbf_false_pure_sympy -v
```

## Regression Prevention

These tests prevent regressions in:

1. **Function Detection** - Ensures UW3 Functions distinguished from SymPy functions
2. **rbf Flag Logic** - Ensures pure sympy always optimized regardless of rbf flag
3. **UWexpression Substitution** - Ensures automatic parameter substitution works
4. **Performance** - Ensures optimizations actually provide speedup

## Related Documentation

- `LAMBDIFY-DETECTION-BUG-FIX.md` - Function detection fix details
- `UWEXPRESSION-LAMBDIFY-FIX.md` - UWexpression integration
- `AUTOMATIC-LAMBDIFICATION-OPTIMIZATION.md` - Overall system documentation

## Future Monitoring

When revisiting performance periodically:

1. **Run these tests** to ensure all paths still working
2. **Check timing benchmarks** in performance expectation tests
3. **Add new test cases** if new expression types discovered
4. **Update performance thresholds** if infrastructure changes

**Expected performance characteristics:**
- Lambdified evaluation: < 0.01s for 1000 points (after caching)
- RBF interpolation: ~0.01-0.1s for 1000 points (data-dependent)
- Caching speedup: 10-100x for first vs cached evaluation

---

**Status**: Production ready, all tests passing
**Coverage**: Documents all known optimization paths
**Maintenance**: Run periodically to catch performance regressions
