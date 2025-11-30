# Plan to Fix Closure Inconsistencies

**Date:** 2025-11-08
**Goal:** Achieve complete and consistent closure for all unit-aware operations
**Status:** READY TO IMPLEMENT

---

## Executive Summary

**The Problem:** Unit information is **lost during expression composition**.

When you compose expressions like `2 * radius * velocity[1]`, the result is a plain SymPy object that has lost all unit/dimension information. You can't ask it "what are your units?" - you have to call `uw.get_units()` and rely on dimensional analysis to reconstruct what the units *should* be.

**The Core Issue:** Variables return plain SymPy instead of unit-aware objects:
```python
T = MeshVariable("T", mesh, units="kelvin")
v = MeshVariable("v", mesh, 2, units="m/s")

# This composition loses unit-awareness:
expr = 2 * T * v[1]  # Returns: plain sympy.Mul ❌
expr.units            # AttributeError: no .units property ❌
uw.get_units(expr)    # Works but indirect ⚠️
```

**The Solution: `UnitAwareExpression` - The Closure Wrapper**

`UnitAwareExpression` is **specifically designed** to wrap compound expressions and preserve units during composition:

```python
# What SHOULD happen (after fix):
expr = 2 * T * v[1]  # Returns: UnitAwareExpression ✓
expr.units            # 'kelvin * meter / second' ✓
expr.sym              # Pure SymPy for JIT ✓
```

**Purpose of `UnitAwareExpression`:**
- **Catches composition** - Wraps results when unit-aware objects combine
- **Preserves units** - Carries unit metadata alongside SymPy expression
- **Maintains closure** - Operations on unit-aware objects return unit-aware objects
- **Enables JIT** - Pure SymPy accessible via `.sym` property

**Current Status:**
- ✅ `UnitAwareExpression` class **exists** with complete arithmetic support
- ✅ Designed specifically for closure (wrapping compound expressions)
- ❌ **Not being used!** Variables return plain SymPy instead
- **Impact:** 5/30 closure tests failing (25/30 passing = 83%)

---

## The Inconsistency

### What Works (Complete Closure)

```python
# UnitAwareArray - WORKS ✓
arr1 = UnitAwareArray([1, 2, 3], units="m")
arr2 = UnitAwareArray([4, 5, 6], units="s")
result = arr1 / arr2
print(result.units)  # 'm / s' ✓
print(type(result))  # UnitAwareArray ✓

# UWQuantity - WORKS ✓
length = uw.quantity(5, "m")
time = uw.quantity(2, "s")
speed = length / time
print(speed.units)  # 'm / s' ✓
print(type(speed))  # UWQuantity ✓
```

### What Doesn't Work (Incomplete Closure)

```python
# Variables - DOESN'T WORK ❌
T = uw.discretisation.MeshVariable("T", mesh, units="kelvin")
v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

result = T * v[0]
print(type(result))  # sympy.Mul ❌ (not unit-aware!)
print(hasattr(result, 'units'))  # False ❌

# Can extract units, but not convenient:
units = uw.get_units(result)  # Works, returns 'kelvin * meter / second' ✓
# But result itself doesn't carry units as a property ❌
```

---

## Why UnitAwareExpression is Critical for Closure

### The Composition Problem

**Closure property:** Operations on unit-aware objects should return unit-aware objects.

**What breaks closure:** Object composition creates plain SymPy:

```python
# Each component is unit-aware:
scalar = 2                                    # Dimensionless constant
radius = MeshVariable("r", mesh, units="m")   # Has .units property ✓
velocity = MeshVariable("v", mesh, 2, units="m/s")  # Has .units property ✓

# But composition loses unit-awareness:
expr = scalar * radius * velocity[1]
# Returns: sympy.Mul (plain SymPy) ❌
# Lost: No .units property, no direct way to query units
```

**Why plain SymPy breaks closure:**
1. **No unit storage** - SymPy doesn't know about physical units
2. **Indirect access** - Must call `uw.get_units(expr)` to reconstruct units
3. **Analysis required** - Need to traverse expression tree and compute units
4. **Not self-describing** - The object doesn't "know" its own units

### UnitAwareExpression: The Closure Wrapper

**Purpose:** Wrap compound expressions to preserve unit information during composition.

```python
# UnitAwareExpression wraps SymPy + units:
class UnitAwareExpression:
    def __init__(self, expr, units):
        self._expr = expr      # Pure SymPy expression
        self._units = units    # Unit metadata (pint.Unit)

    @property
    def sym(self):
        """Pure SymPy for JIT compilation."""
        return self._expr

    @property
    def units(self):
        """Direct unit access - self-describing!"""
        return self._units
```

**How it achieves closure:**

```python
# After fix - composition returns UnitAwareExpression:
expr = scalar * radius * velocity[1]
# Returns: UnitAwareExpression(sympy.Mul(...), pint.Unit("m**2/s"))

# Now the object IS self-describing:
expr.units       # Direct access ✓
expr.sym         # Pure SymPy for JIT ✓
expr * 2         # Returns UnitAwareExpression (closure!) ✓
```

### Composition Flow: Before vs After

**BEFORE (broken closure):**
```
scalar (2)  ×  radius (MeshVar)  ×  velocity[1] (MeshVar)
   ↓               ↓                      ↓
   int         .sym (SymPy)          .sym (SymPy)
   ↓               ↓                      ↓
   └───────────────┴──────────────────────┘
                   ↓
           Plain sympy.Mul ❌
           - No .units property
           - Not self-describing
           - Breaks closure
```

**AFTER (complete closure):**
```
scalar (2)  ×  radius (MeshVar)  ×  velocity[1] (MeshVar)
   ↓               ↓                      ↓
   int         .__mul__()            .__mul__()
   ↓               ↓                      ↓
   ↓         Computes SymPy        Computes SymPy
   ↓         Gets units via        Gets units via
   ↓         uw.get_units()        uw.get_units()
   ↓               ↓                      ↓
   └───────────────┴──────────────────────┘
                   ↓
        UnitAwareExpression ✓
        - Has .units property
        - Has .sym property
        - Self-describing
        - Maintains closure!
```

### Why Not Just Use `uw.get_units()`?

**Current workaround:**
```python
expr = T * v[0]  # Plain SymPy
units = uw.get_units(expr)  # Reconstruct units via analysis
```

**Problems with indirect access:**
1. **Not closure** - `expr` itself doesn't carry units
2. **Computational cost** - Must traverse tree and analyze every time
3. **Not discoverable** - Users don't know to call `uw.get_units()`
4. **Inconsistent** - `UnitAwareArray` has `.units`, but expressions don't

**With UnitAwareExpression (direct access):**
```python
expr = T * v[0]  # UnitAwareExpression
units = expr.units  # Direct property access ✓
# Closure: unit-aware in → unit-aware out
```

---

## Root Cause Analysis

### Where Variable Operations Come From

**File:** `src/underworld3/utilities/mathematical_mixin.py`

```python
class MathematicalMixin:
    """Mixin providing natural mathematical notation for variables."""

    def _sympify_(self):
        """Enable SymPy integration."""
        return self.sym  # Returns plain SymPy Matrix

    def __mul__(self, other):
        """Multiplication operator."""
        return self.sym * other  # Returns plain SymPy! ❌

    def __add__(self, other):
        """Addition operator."""
        return self.sym + other  # Returns plain SymPy! ❌

    # etc...
```

**Problem:** These methods return `self.sym` (plain SymPy), not `UnitAwareExpression`.

---

## The Fix: Three-Step Solution

### Step 1: Enhance `MathematicalMixin` to Return `UnitAwareExpression`

**File:** `src/underworld3/utilities/mathematical_mixin.py`

**Current (broken):**
```python
def __mul__(self, other):
    """Multiplication operator."""
    return self.sym * other  # Plain SymPy ❌
```

**Fixed:**
```python
def __mul__(self, other):
    """Multiplication operator with unit preservation."""
    # Compute SymPy result
    sympy_result = self.sym * other

    # If this variable has units, wrap in UnitAwareExpression
    if hasattr(self, 'has_units') and self.has_units:
        from underworld3.expression.unit_aware_expression import UnitAwareExpression
        import underworld3 as uw

        # Get units of result via dimensional analysis
        result_units = uw.get_units(sympy_result)

        # Return unit-aware expression
        if result_units is not None:
            # Convert string units to pint.Unit if needed
            if isinstance(result_units, str):
                result_units = uw.units(result_units)
            return UnitAwareExpression(sympy_result, result_units)

    # No units, return plain SymPy
    return sympy_result
```

**Apply this pattern to:**
- `__add__`, `__radd__`
- `__sub__`, `__rsub__`
- `__mul__`, `__rmul__`
- `__truediv__`, `__rtruediv__`
- `__pow__`
- `__neg__`
- `__getitem__` (component access)

### Step 2: Update `_sympify_()` for SymPy Integration

**Problem:** SymPy calls `_sympify_()` during operations, which returns plain `self.sym`.

**Current:**
```python
def _sympify_(self):
    """Enable SymPy integration for mathematical operations."""
    return self.sym  # Plain SymPy ❌
```

**Consideration:** We can't change `_sympify_()` to return `UnitAwareExpression` because SymPy expects pure SymPy objects.

**Solution:** Keep `_sympify_()` as-is. The explicit operator methods (`__mul__`, etc.) will handle wrapping.

### Step 3: Ensure `UnitAwareExpression` Integration

**Already done!** `UnitAwareExpression` has:
- ✅ `.sym` property for SymPy expression
- ✅ `.units` property for unit metadata
- ✅ All arithmetic operators with unit preservation
- ✅ Unit compatibility checking

**Just needs to be used by `MathematicalMixin`!**

---

## Implementation Details

### Modified Methods in `MathematicalMixin`

#### 1. Multiplication

```python
def __mul__(self, other):
    """Multiplication with automatic unit handling."""
    sympy_result = self.sym * other
    return self._wrap_if_unit_aware(sympy_result)

def __rmul__(self, other):
    """Right multiplication."""
    sympy_result = other * self.sym
    return self._wrap_if_unit_aware(sympy_result)
```

#### 2. Division

```python
def __truediv__(self, other):
    """Division with automatic unit handling."""
    sympy_result = self.sym / other
    return self._wrap_if_unit_aware(sympy_result)

def __rtruediv__(self, other):
    """Right division."""
    sympy_result = other / self.sym
    return self._wrap_if_unit_aware(sympy_result)
```

#### 3. Addition/Subtraction

```python
def __add__(self, other):
    """Addition with unit compatibility checking."""
    sympy_result = self.sym + other
    return self._wrap_if_unit_aware(sympy_result)

def __sub__(self, other):
    """Subtraction with unit compatibility checking."""
    sympy_result = self.sym - other
    return self._wrap_if_unit_aware(sympy_result)
```

#### 4. Power

```python
def __pow__(self, exponent):
    """Power operation with unit exponentiation."""
    sympy_result = self.sym ** exponent
    return self._wrap_if_unit_aware(sympy_result)
```

#### 5. Component Access

```python
def __getitem__(self, index):
    """Component access with unit preservation."""
    # For vector/tensor variables
    sympy_result = self.sym[index]
    return self._wrap_if_unit_aware(sympy_result)
```

### Helper Method: `_wrap_if_unit_aware()`

```python
def _wrap_if_unit_aware(self, sympy_expr):
    """
    Wrap SymPy expression in UnitAwareExpression if this variable has units.

    Parameters
    ----------
    sympy_expr : sympy.Basic
        SymPy expression result from operation

    Returns
    -------
    UnitAwareExpression or sympy.Basic
        Wrapped if has units, plain SymPy otherwise
    """
    # Check if this variable has units
    if not (hasattr(self, 'has_units') and self.has_units):
        return sympy_expr  # No units, return plain SymPy

    # Get units of result via dimensional analysis
    import underworld3 as uw
    result_units = uw.get_units(sympy_expr)

    if result_units is None:
        return sympy_expr  # Couldn't determine units, return plain

    # Wrap in UnitAwareExpression
    from underworld3.expression.unit_aware_expression import UnitAwareExpression

    # Convert string units to pint.Unit if needed
    if isinstance(result_units, str):
        result_units = uw.units(result_units)

    return UnitAwareExpression(sympy_expr, result_units)
```

---

## Testing Strategy

### Failing Tests to Fix (5 tests)

1. **`test_closure_variable_multiply_variable`**
   - Tests: `T * v` returns unit-aware object
   - Fix: `__mul__` returns `UnitAwareExpression`

2. **`test_closure_scalar_times_variable`**
   - Tests: `2 * T` returns unit-aware object
   - Fix: `__rmul__` returns `UnitAwareExpression`

3. **`test_closure_second_derivative`**
   - Tests: `d²T/dx²` returns unit-aware object
   - Fix: Derivative operations preserve unit-awareness

4. **`test_units_addition_incompatible_units_fails`**
   - Tests: Adding incompatible units raises error
   - Fix: `UnitAwareExpression.__add__` already checks this ✓

5. **`test_closure_evaluate_returns_unit_aware`**
   - Tests: `uw.function.evaluate()` returns `UnitAwareArray`
   - Fix: Ensure evaluation converts `UnitAwareExpression` properly

### New Tests to Add

**Test `UnitAwareExpression` directly:**
```python
def test_unit_aware_expression_arithmetic():
    """Test UnitAwareExpression arithmetic operations."""
    from underworld3.expression.unit_aware_expression import UnitAwareExpression
    import sympy
    import underworld3 as uw

    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    expr1 = UnitAwareExpression(x, uw.units("m"))
    expr2 = UnitAwareExpression(y, uw.units("s"))

    # Multiplication
    result = expr1 * expr2
    assert isinstance(result, UnitAwareExpression)
    assert result.units == uw.units("m * s")

    # Division
    result = expr1 / expr2
    assert isinstance(result, UnitAwareExpression)
    assert result.units == uw.units("m / s")
```

---

## Backward Compatibility

### Will This Break Existing Code?

**No!** Because:

1. **SymPy compatibility preserved:**
   - `UnitAwareExpression` has `.sym` property that returns pure SymPy
   - JIT compilation uses `.sym` internally, will continue to work

2. **User code still works:**
   ```python
   # Old pattern (still works):
   result = T.sym * v.sym  # Plain SymPy ✓

   # New pattern (also works):
   result = T * v  # UnitAwareExpression with .sym and .units ✓
   ```

3. **`.sym` access unchanged:**
   - Existing `var.sym` accesses return plain SymPy
   - New operations on variables return `UnitAwareExpression`
   - Can always extract `.sym` if needed

---

## Performance Considerations

### Cost of Wrapping

**Wrapping overhead:**
- Call to `uw.get_units()` to determine result units
- Construction of `UnitAwareExpression` object
- **Estimate:** < 1 microsecond per operation

**When it happens:**
- Only when creating new expressions (during model setup)
- NOT during numerical evaluation (JIT uses pure SymPy)

**Impact:** Negligible - model setup is not performance-critical

---

## Implementation Checklist

### Phase 1: Core Implementation (2-3 hours)

- [ ] Add `_wrap_if_unit_aware()` helper to `MathematicalMixin`
- [ ] Update `__mul__` and `__rmul__`
- [ ] Update `__truediv__` and `__rtruediv__`
- [ ] Update `__add__`, `__radd__`, `__sub__`, `__rsub__`
- [ ] Update `__pow__`
- [ ] Update `__neg__`
- [ ] Update `__getitem__` (component access)

### Phase 2: Testing (1-2 hours)

- [ ] Run closure tests: `pytest tests/test_0850_units_closure_comprehensive.py`
- [ ] Verify 5 failing tests now pass
- [ ] Add tests for `UnitAwareExpression` directly
- [ ] Verify JIT still works (SymPy extraction)

### Phase 3: Edge Cases (1 hour)

- [ ] Test mixed operations (Variable + UWQuantity, etc.)
- [ ] Test operations with non-unit-aware variables
- [ ] Test derivative operations
- [ ] Test evaluation with `UnitAwareExpression`

### Phase 4: Documentation (30 minutes)

- [ ] Update `UNITS_CLOSURE_ANALYSIS.md` with new status
- [ ] Document `UnitAwareExpression` in method table
- [ ] Add examples to docstrings

**Total estimated time:** 4-7 hours

---

## Success Criteria

### Quantitative

- ✅ 30/30 closure tests passing (currently 25/30)
- ✅ All existing tests still pass
- ✅ No performance regression in JIT compilation

### Qualitative

- ✅ User can write: `result = T * v` and get `.units` property
- ✅ Consistent behavior across all object types
- ✅ Clear error messages for unit incompatibility
- ✅ Natural mathematical notation throughout

---

## Answer to Your Questions

### Q1: How do you propose we address the inconsistencies?

**A:** Modify `MathematicalMixin` to return `UnitAwareExpression` instead of plain SymPy.

**Key insight:** `UnitAwareExpression` already exists with complete arithmetic support! We just need to use it.

**Implementation:** Add `_wrap_if_unit_aware()` helper that:
1. Checks if variable has units
2. Computes result units via `uw.get_units()`
3. Wraps in `UnitAwareExpression` if units exist
4. Returns plain SymPy if no units

### Q2: What about unit aware expressions, are they tested in this closure matrix?

**A:** Not explicitly, but they should be!

**Current status:**
- ✅ `UnitAwareExpression` class exists with full arithmetic
- ❌ Not tested in closure test suite
- ❌ Not used by Variables (returns plain SymPy instead)

**Fix:**
1. Add explicit tests for `UnitAwareExpression` arithmetic
2. Make Variables use `UnitAwareExpression` for operation results
3. Update closure matrix to show `UnitAwareExpression` status

---

## Next Steps

**Ready to implement?** This plan:
1. Uses existing `UnitAwareExpression` (no new class needed)
2. Minimal changes to `MathematicalMixin` (~100 lines)
3. Clear path to 100% closure
4. Backward compatible
5. Well-tested (5 failing tests → passing)

**Recommendation:** Implement Phase 1 (core functionality) first, test incrementally, then proceed.

---

**This achieves complete and consistent closure across all unit-aware types!**
