# MathematicalMixin Symbolic Behavior Design Document
**Date**: 2025-10-26
**Author**: Claude
**Status**: Design Phase

---

## Executive Summary

The MathematicalMixin class currently causes premature numeric substitution in arithmetic operations, breaking lazy evaluation and symbolic expression display. This document provides a comprehensive analysis and unified solution that preserves symbolic behavior across all classes while maintaining backward compatibility.

---

## Problem Statement

### Current Behavior (BROKEN)
```python
Ra = uw.expression(r"\mathrm{Ra}", 72435330.0)
T = uw.MeshVariable("T", mesh, 1)

# Current: Ra * T → 72435330.0 * T (numeric substitution)
# Expected: Ra * T → Ra * T (symbolic preservation)
```

### Root Cause
In `mathematical_mixin.py` lines 256-258:
```python
def __mul__(self, other):
    # Convert MathematicalMixin arguments to their symbolic form
    if hasattr(other, "_sympify_"):
        other = other.sym  # BUG: Substitutes numeric value immediately!
```

This pattern appears in ALL arithmetic operations: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`, and their right-hand equivalents.

---

## Affected Classes

### 1. SwarmVariable
- **Inheritance**: `DimensionalityMixin, MathematicalMixin, Stateful, uw_object`
- **Purpose**: Particle data with optional mesh proxy variables
- **Symbolic Behavior**: Should remain symbolic in expressions
- **Units**: Inherits from DimensionalityMixin for unit tracking
- **Unwrapping**: Via `unwrap()` when evaluating expressions

### 2. EnhancedMeshVariable (exported as MeshVariable)
- **Inheritance**: `DimensionalityMixin, UnitAwareMixin, MathematicalMixin`
- **Purpose**: Mesh-based field variables
- **Symbolic Behavior**: Must remain symbolic for solver expressions
- **Units**: Full unit support via UnitAwareMixin
- **Unwrapping**: Critical for solver compilation

### 3. UWexpression
- **Inheritance**: `MathematicalMixin, UWQuantity, uw_object, Symbol`
- **Purpose**: Named constants and parameters with LaTeX display
- **Symbolic Behavior**: MUST remain symbolic (inherits from Symbol!)
- **Units**: Full UWQuantity unit support
- **Unwrapping**: Substitutes numeric value during compilation
- **Special**: Already overrides arithmetic operations to preserve symbolism

---

## Design Principles

### 1. Lazy Evaluation (CRITICAL)
Expressions must remain symbolic until explicitly unwrapped for compilation:
```python
# Building expressions (symbolic)
momentum = density * velocity  # Should be ρ * v, not numeric
strain_rate = velocity.diff(x)  # Should be ∂v/∂x, not evaluated

# Unwrapping for compilation (numeric substitution)
compiled = unwrap(momentum)  # NOW substitute values
```

### 2. Symbolic Display
Users expect to see meaningful symbolic representations:
```python
Ra * T  # Should display as "Ra * T" for pedagogical clarity
        # Not "72435330.0 * T" which obscures meaning
```

### 3. Unit Propagation
Units must flow through arithmetic operations:
```python
L0 = uw.quantity(2900, "km")
L0**3  # Must yield 'kilometer ** 3' units
```

### 4. Backward Compatibility
Existing code using `.sym` directly must continue working.

---

## Proposed Solution

### Core Fix: Conditional Substitution

**Strategy**: Only substitute `.sym` for non-MathematicalMixin objects.

```python
class MathematicalMixin:
    def __mul__(self, other):
        """Multiplication preserving symbolic expressions."""
        sym = self._validate_sym()

        # KEY CHANGE: Don't substitute .sym for MathematicalMixin objects
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()  # Use protocol, not .sym directly

        try:
            return sym * other
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot multiply {type(self).__name__} and {type(other).__name__}: {e}")
```

### Why This Works

1. **MathematicalMixin * MathematicalMixin**: Preserves both as symbols
2. **MathematicalMixin * SymPy**: Works via SymPy's dispatch
3. **MathematicalMixin * Number**: Works via SymPy's handling
4. **MathematicalMixin * Other**: Calls `_sympify_()` protocol

### Apply to All Operations

This pattern must be applied to:
- `__add__`, `__radd__`
- `__sub__`, `__rsub__`
- `__mul__`, `__rmul__`
- `__truediv__`, `__rtruediv__`
- `__pow__`, `__rpow__`

---

## Class-Specific Behaviors

### SwarmVariable & MeshVariable
```python
# Current (broken)
velocity * pressure  # → velocity.sym * pressure.sym (numeric arrays!)

# Fixed
velocity * pressure  # → V * P (symbolic Functions)

# Unwrapping
unwrap(velocity * pressure)  # → Substitutes actual values
```

### UWexpression
```python
# Already has overrides to preserve symbolic behavior
def __mul__(self, other):
    return sympy.Symbol.__mul__(self, other)  # Uses Symbol's method

# This is correct! But MathematicalMixin breaks it from the other side
```

### Mixed Operations
```python
# Expression * Variable
Ra * T  # → Ra * T (both remain symbolic)

# Expression * Expression
Ra * alpha  # → Ra * α (symbolic product)

# Variable * Variable
velocity * density  # → v * ρ (symbolic)
```

---

## Implementation Plan

### Phase 1: Fix MathematicalMixin (PRIORITY)
1. Update all arithmetic operations to use conditional substitution
2. Preserve `_sympify_()` protocol usage
3. Add comprehensive tests for symbolic preservation

### Phase 2: Remove UWexpression Overrides
1. Once MathematicalMixin is fixed, remove redundant overrides
2. Simplify inheritance chain
3. Verify all operations still work

### Phase 3: Validate Unwrapping
1. Ensure `unwrap()` correctly substitutes all values
2. Test with complex nested expressions
3. Verify solver compilation still works

### Phase 4: Unit Integration
1. Ensure units propagate through symbolic operations
2. Test dimensional analysis with symbolic expressions
3. Verify unit cancellation in unwrapped expressions

---

## Test Cases

### 1. Symbolic Preservation
```python
def test_symbolic_preservation():
    Ra = uw.expression("Ra", 1e7)
    T = uw.MeshVariable("T", mesh, 1)

    expr = Ra * T
    # Should contain Ra symbol, not 1e7
    assert Ra in expr.atoms()
    assert 1e7 not in expr.atoms()
```

### 2. Lazy Evaluation
```python
def test_lazy_evaluation():
    Ra = uw.expression("Ra", 1e7)
    T = uw.MeshVariable("T", mesh, 1)

    expr = Ra * T
    Ra.sym = 2e7  # Change value

    unwrapped = unwrap(expr)
    # Should use new value 2e7
    assert 2e7 in str(unwrapped)
```

### 3. Unit Propagation
```python
def test_unit_propagation():
    L0 = uw.quantity(100, "km")
    v = uw.MeshVariable("v", mesh, 1, units="m/s")

    expr = v * L0
    units = uw.get_units(expr)
    assert units == "meter * kilometer / second"
```

---

## Migration Strategy

### Backward Compatibility
- All existing `.sym` usage continues working
- `unwrap()` function remains unchanged
- Solver interfaces unchanged

### Gradual Rollout
1. **Week 1**: Implement and test MathematicalMixin fix
2. **Week 2**: Remove UWexpression overrides if safe
3. **Week 3**: Comprehensive testing with notebooks
4. **Week 4**: Documentation and user guide updates

### Risk Mitigation
- Keep old behavior behind feature flag initially
- Extensive test coverage before deployment
- Monitor solver performance (no changes expected)

---

## Alternative Approaches Considered

### 1. Always Use SymPy Protocol (Rejected)
```python
# Always return self from _sympify_()
def _sympify_(self):
    return self  # Not self.sym
```
**Problem**: Causes recursion in operations like `atoms()`

### 2. Separate Symbolic and Numeric Mixins (Rejected)
```python
class SymbolicMixin: ...  # For expressions
class NumericMixin: ...   # For variables
```
**Problem**: Too much code duplication, breaks existing inheritance

### 3. Wrapper Objects (Rejected)
```python
class SymbolicWrapper:
    def __init__(self, obj): ...
```
**Problem**: Adds complexity, breaks isinstance() checks

---

## Conclusion

The proposed solution of conditional substitution in MathematicalMixin arithmetic operations:
1. **Preserves symbolic behavior** for all MathematicalMixin objects
2. **Maintains backward compatibility** with existing code
3. **Enables proper lazy evaluation** for expression changes
4. **Supports unit propagation** through symbolic operations
5. **Requires minimal code changes** (only in MathematicalMixin)

This approach aligns with the original design intent where expressions remain symbolic as long as possible, with `unwrap()` providing explicit control over when numeric substitution occurs.

---

## Appendix: Detailed Code Analysis

### Current Call Flow (Broken)
```
Ra * T
├─> MathematicalMixin.__mul__(Ra, T)
│   ├─> if hasattr(T, "_sympify_"): T = T.sym  # BUG!
│   └─> return Ra.sym * T.sym  # Returns 1e7 * T.sym
```

### Fixed Call Flow
```
Ra * T
├─> MathematicalMixin.__mul__(Ra, T)
│   ├─> if isinstance(T, MathematicalMixin): keep T as-is
│   └─> return Ra.sym * T  # Returns Ra_symbol * T_symbol
```

### Unwrap Flow (Unchanged)
```
unwrap(Ra * T)
├─> Extract atoms: {Ra, T}
├─> Substitute: Ra → 1e7, T → T.sym
└─> Return: 1e7 * T.sym  # Numeric substitution happens here
```