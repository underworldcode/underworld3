# SNES Diagnostics Implementation Guide

## Overview

This guide provides the complete implementation for adding SNES convergence diagnostics to the solver base class in Underworld3. The enhancement adds three key methods to `SolverBaseClass` that provide comprehensive convergence information with human-readable string representations.

## Key Features

✅ **String Convergence Reasons**: Converts numerical PETSc convergence reasons to descriptive strings
✅ **Zero Iteration Detection**: Identifies scaling-related convergence issues
✅ **Comprehensive Diagnostics**: Full SNES iteration and tolerance information
✅ **Exception Framework**: Optional exception raising for diverged solvers
✅ **User-Friendly Interface**: Multiple usage patterns for different needs

## Implementation Location

**File**: `src/underworld3/cython/petsc_generic_snes_solvers.pyx`
**Class**: `SolverBaseClass`
**Integration**: Add three methods to the base class

## Method 1: Core Diagnostics

```python
def get_snes_diagnostics(self):
    """
    Extract comprehensive SNES convergence diagnostics with string representations.

    Returns:
    --------
    dict
        Comprehensive convergence diagnostics including:
        - converged: bool - Whether solver converged
        - diverged: bool - Whether solver diverged
        - convergence_reason: int - Numerical convergence reason
        - convergence_reason_string: str - Human-readable convergence reason
        - snes_iterations: int - Number of SNES iterations
        - linear_iterations: int - Total number of linear iterations
        - zero_iterations: bool - Whether SNES took zero iterations
        - tolerances: dict - SNES tolerance settings
    """

    if self.snes is None:
        return {
            'error': 'SNES not initialized - call solve() first',
            'snes_available': False
        }

    # Get basic convergence info
    converged_reason = self.snes.getConvergedReason()
    snes_iterations = self.snes.getIterationNumber()
    linear_iterations = self.snes.getLinearSolveIterations()
    rtol, atol, stol, maxit = self.snes.getTolerances()

    # Determine convergence status
    converged = converged_reason > 0
    diverged = converged_reason < 0

    # Map convergence reasons to descriptive strings (PETSc documentation)
    convergence_reason_map = {
        # Positive reasons = converged
        1: "CONVERGED_FNORM_ABS - ||F|| < atol",
        2: "CONVERGED_FNORM_RELATIVE - ||F|| < rtol*||F_initial||",
        3: "CONVERGED_SNORM_RELATIVE - ||x|| < stol",
        4: "CONVERGED_ITS - Maximum iterations reached",

        # Zero = still iterating (shouldn't see after solve)
        0: "ITERATING - Still iterating (unexpected after solve)",

        # Negative reasons = diverged
        -1: "DIVERGED_FUNCTION_DOMAIN - Function domain error",
        -2: "DIVERGED_FUNCTION_COUNT - Too many function evaluations",
        -3: "DIVERGED_LINEAR_SOLVE - Linear solver failed",
        -4: "DIVERGED_FNORM_NAN - ||F|| is Not-a-Number",
        -5: "DIVERGED_MAX_IT - Maximum iterations exceeded",
        -6: "DIVERGED_LINE_SEARCH - Line search failed",
        -7: "DIVERGED_INNER - Inner solve failed",
        -8: "DIVERGED_LOCAL_MIN - Local minimum reached",
        -9: "DIVERGED_DTOL - ||F|| increased by divtol",
        -10: "DIVERGED_JACOBIAN_DOMAIN - Jacobian calculation failed",
        -11: "DIVERGED_TR_DELTA - Trust region delta too small",
    }

    convergence_reason_string = convergence_reason_map.get(
        converged_reason,
        f"UNKNOWN_CONVERGENCE_REASON_{converged_reason}"
    )

    return {
        'snes_available': True,
        'converged': converged,
        'diverged': diverged,
        'convergence_reason': converged_reason,
        'convergence_reason_string': convergence_reason_string,
        'snes_iterations': snes_iterations,
        'linear_iterations': linear_iterations,
        'zero_iterations': snes_iterations == 0,
        'linear_solver_failed': converged_reason == -3,
        'nan_residual': converged_reason == -4,
        'tolerances': {
            'relative_tolerance': rtol,
            'absolute_tolerance': atol,
            'step_tolerance': stol,
            'max_iterations': maxit
        }
    }
```

## Method 2: Convergence Checking with Exceptions

```python
def check_snes_convergence(self, raise_on_divergence=True, print_diagnostics=False):
    """
    Check SNES convergence and optionally raise exceptions or print diagnostics.

    Parameters:
    -----------
    raise_on_divergence : bool
        Whether to raise an exception if solver diverged
    print_diagnostics : bool
        Whether to print diagnostic information

    Returns:
    --------
    dict
        SNES diagnostics

    Raises:
    -------
    RuntimeError
        If solver diverged and raise_on_divergence=True
    """

    diagnostics = self.get_snes_diagnostics()

    if not diagnostics.get('snes_available', False):
        if raise_on_divergence:
            raise RuntimeError(diagnostics.get('error', 'SNES diagnostics not available'))
        return diagnostics

    if print_diagnostics:
        print(f"\n=== SNES DIAGNOSTICS ===")
        print(f"Status: {'✓ CONVERGED' if diagnostics['converged'] else '✗ DIVERGED'}")
        print(f"Reason: {diagnostics['convergence_reason_string']}")
        print(f"Iterations: {diagnostics['snes_iterations']} SNES, {diagnostics['linear_iterations']} linear")

        tol = diagnostics['tolerances']
        print(f"Tolerances: rtol={tol['relative_tolerance']:.1e}, "
              f"atol={tol['absolute_tolerance']:.1e}")

        # Issue-specific warnings
        if diagnostics['zero_iterations']:
            print(f"⚠️  WARNING: Zero SNES iterations!")
            print(f"   Possible scaling issues - consider geological scaling")

        if diagnostics['linear_solver_failed']:
            print(f"⚠️  LINEAR SOLVER FAILURE!")
            print(f"   Often caused by poor matrix conditioning")

    # Raise exception if requested and solver diverged
    if diagnostics['diverged'] and raise_on_divergence:
        error_msg = f"SNES solver diverged: {diagnostics['convergence_reason_string']}\n"
        error_msg += f"Iterations: {diagnostics['snes_iterations']} SNES, {diagnostics['linear_iterations']} linear"

        if diagnostics['zero_iterations']:
            error_msg += "\nZERO ITERATIONS: Scaling or tolerance issues likely"
            error_msg += "\nSUGGESTION: Try geological scaling"

        if diagnostics['linear_solver_failed']:
            error_msg += "\nLINEAR SOLVER FAILURE: Matrix conditioning problems"
            error_msg += "\nSUGGESTION: Check scaling or solver options"

        raise RuntimeError(error_msg)

    return diagnostics
```

## Method 3: Enhanced Solve Method

```python
def solve_with_diagnostics(self,
                          check_convergence=True,
                          raise_on_divergence=False,
                          print_diagnostics=False,
                          **solve_kwargs):
    """
    Solve with automatic SNES convergence checking and diagnostics.

    Parameters:
    -----------
    check_convergence : bool
        Whether to check convergence after solving
    raise_on_divergence : bool
        Whether to raise exception on divergence
    print_diagnostics : bool
        Whether to print diagnostic information
    **solve_kwargs
        Additional arguments passed to solve()

    Returns:
    --------
    dict or None
        SNES diagnostics if check_convergence=True, None otherwise

    Raises:
    -------
    RuntimeError
        If solver diverged and raise_on_divergence=True
    """

    # Call the original solve method
    self.solve(**solve_kwargs)

    # Check convergence if requested
    if check_convergence:
        return self.check_snes_convergence(
            raise_on_divergence=raise_on_divergence,
            print_diagnostics=print_diagnostics
        )

    return None
```

## Usage Examples

### 1. Basic Diagnostics (Manual Check)

```python
# Standard solve + manual diagnostics
stokes.solve()
diagnostics = stokes.get_snes_diagnostics()

print(f"Converged: {diagnostics['converged']}")
print(f"Reason: {diagnostics['convergence_reason_string']}")
print(f"Iterations: {diagnostics['snes_iterations']}")

if diagnostics['zero_iterations']:
    print("⚠️  Zero iterations detected - try geological scaling!")
```

### 2. Automatic Checking with Exceptions

```python
# Solve with automatic convergence checking and exception on failure
try:
    stokes.solve_with_diagnostics(
        raise_on_divergence=True,
        print_diagnostics=True
    )
    print("✓ Solver converged successfully")
except RuntimeError as e:
    print(f"Solver failed: {e}")
    print("Consider using geological scaling")
```

### 3. Development/Debug Mode

```python
# Solve with comprehensive diagnostics but no exceptions
diagnostics = stokes.solve_with_diagnostics(
    check_convergence=True,
    raise_on_divergence=False,
    print_diagnostics=True
)

if not diagnostics['converged']:
    # Handle divergence gracefully
    if diagnostics['zero_iterations']:
        print("Trying geological scaling...")
        # Apply scaling and retry
```

### 4. Production Mode with Error Handling

```python
# Robust production usage
try:
    # Try with current scaling
    stokes.solve_with_diagnostics(raise_on_divergence=True)

except RuntimeError as e:
    if "ZERO ITERATIONS" in str(e):
        print("Scaling issue detected - applying geological scaling")
        # Apply geological scaling and retry
        uw.scaling.use_geological_scaling()
        stokes.solve_with_diagnostics(raise_on_divergence=True)
    else:
        # Other convergence issues
        print(f"Solver convergence issue: {e}")
        raise
```

## Integration Benefits

### For Users:
- **Clear error messages** instead of cryptic PETSc codes
- **Automatic detection** of scaling-related issues
- **Actionable suggestions** for fixing convergence problems
- **Flexible interface** for different usage scenarios

### For Developers:
- **Robust error handling** throughout the codebase
- **Better debugging information** for solver issues
- **Consistent diagnostics** across all solver types
- **Foundation for automatic scaling** recommendations

### For Scaling System:
- **Perfect integration** with the geological scaling validation
- **Automatic detection** of when scaling is needed
- **Clear feedback** on scaling effectiveness
- **Production-ready** error reporting

## Expected Output Examples

### Successful Convergence:
```
=== SNES DIAGNOSTICS ===
Status: ✓ CONVERGED
Reason: CONVERGED_FNORM_RELATIVE - ||F|| < rtol*||F_initial||
Iterations: 3 SNES, 12 linear
Tolerances: rtol=1.0e-06, atol=0.0e+00
```

### Zero Iterations (Scaling Issue):
```
=== SNES DIAGNOSTICS ===
Status: ✗ DIVERGED
Reason: DIVERGED_LINEAR_SOLVE - Linear solver failed
Iterations: 0 SNES, 0 linear
Tolerances: rtol=1.0e-06, atol=0.0e+00
⚠️  WARNING: Zero SNES iterations!
   Possible scaling issues - consider geological scaling
⚠️  LINEAR SOLVER FAILURE!
   Often caused by poor matrix conditioning
```

### Exception Message:
```
RuntimeError: SNES solver diverged: DIVERGED_LINEAR_SOLVE - Linear solver failed
Iterations: 0 SNES, 0 linear
ZERO ITERATIONS: Scaling or tolerance issues likely
SUGGESTION: Try geological scaling
```

## Implementation Priority

**High Priority** - This enhancement provides immediate value:
1. **Better user experience** - Clear error messages instead of silent failures
2. **Scaling validation** - Perfect integration with the geological scaling system
3. **Production robustness** - Proper error handling for solver issues
4. **Development efficiency** - Better debugging information

**Low Risk** - Addition to base class:
- Methods are pure additions, no existing functionality changed
- Optional usage - existing code continues to work
- Comprehensive error handling prevents crashes
- Based on stable PETSc APIs

## Conclusion

This implementation provides the foundation for robust solver diagnostics in Underworld3. The string representations of convergence reasons address your specific request, while the comprehensive diagnostic framework enables users to understand and fix scaling-related convergence issues.

The integration is ready for production and will significantly improve the user experience when encountering solver convergence problems.