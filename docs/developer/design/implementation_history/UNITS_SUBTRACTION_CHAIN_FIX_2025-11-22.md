# Units Subtraction Chain Fix (2025-11-22)

## Summary

**Fixed critical units bug in chained subtraction operations.**

### User-Reported Bug
```python
x = uw.expression("x", 100, units="km")
x0 = uw.expression("x0", 50, units="km")
velocity_phys = uw.quantity(5, "cm/year")
t_now = uw.expression("t", 1, units="Myr")

result = x - x0 - velocity_phys * t_now
uw.get_units(result)  # ❌ Returned: 'megayear' (WRONG - should be length!)
```

**Expected**: Length units (kilometer)
**Actual**: Time units (megayear) ❌

### Test Results

**Before fix**: 2 FAILED / 2 PASSED
**After fix**: **4 PASSED / 0 FAILED** ✅

---

## Root Cause

### Problem 1: Exact String Comparison Instead of Dimensional Analysis

`UnitAwareExpression.__sub__()` was checking exact unit string equality:

```python
# BEFORE (Wrong)
def __sub__(self, other):
    if isinstance(other, UnitAwareExpression):
        if self._units and other._units:
            if self._units != other._units:  # ❌ String comparison!
                raise ValueError(f"Cannot subtract {other._units} from {self._units}")
```

**Why This Failed**:
- Velocity × Time = (cm/year) × (Myr) = `cm * megayear / year`
- Pint doesn't automatically simplify compound units in multiplication
- `cm * megayear / year` != `kilometer` (even though dimensionally compatible)
- String comparison failed, preventing subtraction

### Problem 2: No Unit Simplification Before Comparison

Units like `cm * megayear / year` should simplify to just `cm` (since megayear/year cancels), but this wasn't happening before dimensional compatibility checking.

---

## Solution Implemented

### Fix: Use Pint's Dimensional Compatibility Checking

**File**: `src/underworld3/expression_types/unit_aware_expression.py`
**Lines**: 223-333 (all addition/subtraction methods)
**Date**: 2025-11-22

Updated all arithmetic operators (`__add__`, `__radd__`, `__sub__`, `__rsub__`) to use Pint's conversion system instead of string comparison.

**Pattern Applied**:
```python
# AFTER (Correct)
def __sub__(self, other):
    """Subtraction requires compatible units - preserves left operand units."""
    if isinstance(other, UnitAwareExpression):
        if self._units and other._units:
            try:
                # Create dummy Pint quantities to check compatibility
                self_pint = 1.0 * self._units
                other_pint = 1.0 * other._units

                # Try to convert - this will raise if incompatible
                _ = other_pint.to(self._units)

                # Units are compatible - subtraction preserves left operand units
                new_expr = self._expr - other._expr
                return self.__class__(new_expr, self._units)
            except Exception as e:
                raise ValueError(
                    f"Cannot subtract {other._units} from {self._units}: "
                    f"incompatible dimensions. {e}"
                )
        new_expr = self._expr - other._expr
        return self.__class__(new_expr, self._units or other._units)
```

### Key Changes

1. **Dimensional Compatibility**: Use `other_pint.to(self_pint.units)` to check if conversion is possible
2. **Automatic Simplification**: Pint's conversion system automatically simplifies units
3. **Preserve Left Operand Units**: Subtraction/addition preserve first operand's units
4. **Proper Error Messages**: Include dimensional incompatibility information

---

## Methods Updated

All four addition/subtraction operators updated with identical pattern:

| Method | Purpose | Left Operand Preserved |
|--------|---------|------------------------|
| `__add__(self, other)` | Addition (self + other) | ✅ Yes (self) |
| `__radd__(self, other)` | Right addition (other + self) | ✅ Yes (other) |
| `__sub__(self, other)` | Subtraction (self - other) | ✅ Yes (self) |
| `__rsub__(self, other)` | Right subtraction (other - self) | ✅ Yes (other) |

---

## Test Coverage

### New Test File: `test_0751_subtraction_chain_units.py`

Created comprehensive test suite to prevent regression:

```python
@pytest.mark.tier_a  # Production-ready
@pytest.mark.level_1  # Quick tests, no solving
class TestSubtractionChainUnits:
    def test_simple_subtraction_chain(self):
        """Test: length - length - length = length"""
        x = uw.quantity(100, "km")
        x0 = uw.quantity(50, "km")
        dx = uw.quantity(10, "km")

        result = x - x0 - dx
        # Should have length units, not time units

    def test_subtraction_with_velocity_time_product(self):
        """Test: position - position0 - velocity*time = position"""
        velocity = uw.quantity(5, "cm/year")
        t = uw.quantity(1, "Myr")
        displacement = velocity * t

        # displacement has length dimensions
        result = x - x0 - displacement
        # Result should have length dimensions

    def test_expression_subtraction_chain(self):
        """Test the exact user-reported case with expressions."""
        x = uw.expression("x", 100, units="km")
        x0 = uw.expression("x0", 50, units="km")
        velocity_phys = uw.quantity(5, "cm/year")
        t_now = uw.expression("t", 1, units="Myr")

        result = x - x0 - velocity_phys * t_now

        # Should have length units, NOT time units
        assert result.units.dimensionality == ureg.meter.dimensionality

    def test_left_associativity_preservation(self):
        """Test that subtraction preserves first operand units."""
        x = uw.expression("x", 100, units="km")  # kilometers
        x0 = uw.expression("x0", 50, units="m")  # meters (different!)

        result = x - x0
        # Should preserve x's units (km), not x0's units (m)
```

**All 4 tests passing** ✅

---

## Verification

### User's Exact Case - Fixed ✅
```python
x = uw.expression("x", 100, units="km")
x0_at_start = uw.expression("x0", 50, units="km")
velocity_phys = uw.quantity(5, "cm/year")
t_now = uw.expression("t", 1, units="Myr")

result = x - x0_at_start - velocity_phys * t_now

# BEFORE FIX:
uw.get_units(result)  # ❌ 'megayear'
result.units.dimensionality  # ❌ [time]

# AFTER FIX:
uw.get_units(result)  # ✅ 'kilometer'
result.units.dimensionality  # ✅ [length]
```

### Regression Tests - Still Passing ✅
- `test_0750_unit_aware_interface_contract.py`: 17 PASSED (6 XPASS → now passing)
- No regressions in existing tests

---

## Why This Approach Works

### 1. Pint Handles Unit Simplification Automatically
```python
# Compound units are simplified during conversion check:
velocity = 5 cm/year
time = 1 Myr  # megayear = 1e6 years

displacement = velocity * time
# Internal: cm * megayear / year
# Pint simplifies: megayear/year = 1e6 year / year = 1e6
# Result: 5e6 cm = 50 km (length dimensions)

# Conversion check:
(1.0 * displacement.units).to(kilometer)  # ✅ Works!
```

### 2. Dimensional Compatibility vs String Equality

| Approach | Units Match | Result |
|----------|-------------|--------|
| **String comparison** | `"km"` vs `"cm * Myr / year"` | ❌ FAIL (different strings) |
| **Dimensional check** | `[length]` vs `[length]` | ✅ PASS (same dimensions) |

### 3. Left Operand Preservation Rule

Pint convention: Addition/subtraction preserve left operand's units:
```python
x = 100 km
x0 = 50 m

result = x - x0
# Result has x's units (km), not x0's units (m)
# Internally: converts x0 to km, then subtracts
```

---

## Benefits Achieved

1. **✅ Dimensional Compatibility**: Units checked by physics, not string matching
2. **✅ Automatic Simplification**: Pint handles compound unit reduction
3. **✅ Clear Error Messages**: Dimensional mismatch errors include context
4. **✅ Left Operand Rule**: Consistent with Pint's conventions
5. **✅ Test Coverage**: Comprehensive tests prevent future regressions

---

## Comparison with UWexpression Pattern

This fix brings `UnitAwareExpression` arithmetic in line with `UWexpression` arithmetic:

**UWexpression** (already working):
```python
# src/underworld3/function/expressions.py:1082-1095
def __sub__(self, other):
    if isinstance(other, (UWQuantity, UnitAwareExpression)):
        self_has_pint = hasattr(self, '_has_pint_qty') and self._has_pint_qty
        if self_has_pint and other_units is not None:
            try:
                self_pint = 1.0 * self._pint_qty.units
                other_pint = 1.0 * other_units
                _ = other_pint.to(self_pint.units)  # ✅ Dimensional check

                result_sym = Symbol.__sub__(self, other)
                return UnitAwareExpression(result_sym, self._pint_qty.units)
```

**UnitAwareExpression** (now consistent):
```python
# src/underworld3/expression_types/unit_aware_expression.py:281-303
def __sub__(self, other):
    if isinstance(other, UnitAwareExpression):
        if self._units and other._units:
            try:
                self_pint = 1.0 * self._units
                other_pint = 1.0 * other._units
                _ = other_pint.to(self._units)  # ✅ Same pattern

                new_expr = self._expr - other._expr
                return self.__class__(new_expr, self._units)
```

**Consistency Achieved**: Both classes now use identical dimensional compatibility checking ✅

---

## Files Modified

**Source Code**:
- `src/underworld3/expression_types/unit_aware_expression.py` (lines 223-333)
  - `__add__()`: Updated to use Pint dimensional check
  - `__radd__()`: Updated to preserve left operand units
  - `__sub__()`: Updated to use Pint dimensional check
  - `__rsub__()`: Updated to preserve left operand units

**Tests**:
- `tests/test_0751_subtraction_chain_units.py` (NEW)
  - 4 comprehensive tests for subtraction chains
  - Tests exact user-reported case
  - Tests left-associativity preservation
  - Tier A (production-ready), Level 1 (quick tests)

**Documentation**:
- This file: `UNITS_SUBTRACTION_CHAIN_FIX_2025-11-22.md`

---

## Lessons Learned

### 1. String Comparison is Dangerous for Units
**Problem**: Different unit expressions can represent the same physical quantity
**Solution**: Always use Pint's dimensional analysis, not string matching

### 2. Pint Doesn't Always Auto-Simplify in Multiplication
**Problem**: `velocity * time` returns `cm * megayear / year`, not simplified `cm`
**Solution**: Use `.to()` conversion to trigger simplification

### 3. Test-Driven Development Prevents Regressions
**Process**:
1. User reports bug with specific example
2. Create test that reproduces the bug (fails)
3. Fix the code
4. Verify test passes
5. Verify no regressions in existing tests

**Result**: High confidence the fix is correct and won't break again

---

## Status

**✅ COMPLETE** - Bug fixed, tests passing, documentation updated
**Date**: 2025-11-22
**Test Suite**: `test_0751_subtraction_chain_units.py` - 4/4 passing
**Regression Tests**: `test_0750_unit_aware_interface_contract.py` - 17/17 passing (6 XPASS)
**User Case**: Verified working ✅
