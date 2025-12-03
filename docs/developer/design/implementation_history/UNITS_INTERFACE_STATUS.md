# Units System Interface Status (2025-11-21)

## Test-Driven Design Complete

We now have a comprehensive Level 1 test suite (`test_0750_unit_aware_interface_contract.py`) that defines the **required interface** for all unit-aware objects.

### Test Results Summary: 11 PASSED / 6 XFAIL

**✅ PASSING (What Currently Works)**:
1. `test_units_property_returns_pint_unit_uwquantity` - UWQuantity.units returns Pint Unit ✅
2. `test_units_property_returns_pint_unit_uwexpression` - UWexpression.units returns Pint Unit ✅
3. `test_conversion_methods_present_uwquantity` - UWQuantity has full conversion API ✅
4. `test_conversion_methods_present_uwexpression` - UWexpression has full conversion API ✅
5. `test_lazy_evaluation_uwexpression_basic` - .sym setter works correctly ✅
6. `test_lazy_evaluation_preserves_symbolic_structure` - Arithmetic preserves symbols ✅
7. `test_lazy_evaluation_updates_propagate` - Updates to expressions work ✅
8. `test_multiplication_closure_quantity_quantity` - UWQuantity * UWQuantity works ✅
9. `test_multiplication_combines_units_correctly` - Pint dimensional analysis works ✅
10. `test_time_stepping_lazy_update_pattern` - Time-stepping pattern works ✅
11. `test_multiple_expressions_share_updated_variable` - Shared variable updates work ✅

**❌ XFAIL (What Needs Fixing)**:
1. `test_units_property_returns_pint_unit_arithmetic_result` - **UnitAwareExpression.units returns STRING** ❌
2. `test_conversion_methods_present_arithmetic_result` - **UnitAwareExpression missing .to_base_units() etc.** ❌
3. `test_lazy_evaluation_subtraction_preserves_units` - **Subtraction returns wrong units** ❌
4. `test_multiplication_closure_quantity_expression` - **UWQuantity * UWexpression missing interface** ❌
5. `test_multiplication_closure_expression_expression` - **UWexpression * UWexpression missing interface** ❌
6. `test_get_units_consistency` - **uw.get_units() returns string, not Pint Unit** ❌

---

## Architecture Issues Identified

### Critical Bug #1: UnitAwareExpression.units Returns String

**File**: `src/underworld3/expression_types/unit_aware_expression.py` lines 79-82

**Current Code**:
```python
if hasattr(self._units, 'dimensionality'):
    # It's a pint.Unit - convert to string
    return str(self._units)  # ❌ VIOLATES PRINCIPLE
```

**Violates**: CLAUDE.md principle "ALWAYS store and return Pint objects internally"

**Impact**: All arithmetic operations now return objects with string units instead of Pint Units

---

### Critical Bug #2: UnitAwareExpression Missing Conversion Methods

**File**: `src/underworld3/expression_types/unit_aware_expression.py`

**Missing Methods**:
- `.to_base_units()`
- `.to_compact()`
- `.to_reduced_units()`
- `.to_nice_units()`

**Impact**: Arithmetic results don't have the same interface as UWQuantity/UWexpression, breaking closure

**User Expectation**:
```python
result = velocity * time  # Returns UnitAwareExpression
result.to_compact()       # Should work - currently AttributeError
```

---

### Bug #3: uw.get_units() Doesn't Normalize

**Problem**: `uw.get_units()` returns whatever the object's `.units` property returns, with no normalization

**Current Behavior**:
```python
uw.get_units(uw.quantity(5, "cm"))    # Returns pint.Unit ✅
uw.get_units(velocity * time)         # Returns string ❌
```

**Expected**: Always return `pint.Unit` objects

---

## The Root Problem

We have **three classes** that should have **identical interfaces** but don't:

| Feature | UWQuantity | UWexpression | UnitAwareExpression |
|---------|------------|--------------|---------------------|
| `.units` returns | `pint.Unit` ✅ | `pint.Unit` ✅ | `str` ❌ |
| `.to_base_units()` | ✅ | ✅ (inherited) | ❌ Missing |
| `.to_compact()` | ✅ | ✅ (inherited) | ❌ Missing |
| `._pint_qty` storage | ✅ | ✅ (inherited) | ❌ No Pint |
| **Used for** | Constants | Named variables | **Arithmetic results** |

**The Issue**: `UnitAwareExpression` is the **return type for all arithmetic** but lacks the full interface!

---

## Fix Strategy (Test-Driven)

### Phase 1: Fix UnitAwareExpression Interface (PRIORITY)

**Goal**: Make all 6 XFAIL tests pass

1. **Fix `.units` property to return `pint.Unit`**
   - Remove `str()` conversion in unit_aware_expression.py:79-82
   - Ensure consistent Pint Unit returns

2. **Add missing conversion methods to UnitAwareExpression**
   - Implement `.to_base_units()` → returns new UnitAwareExpression
   - Implement `.to_compact()` → returns new UnitAwareExpression
   - Implement `.to_reduced_units()` → returns new UnitAwareExpression
   - Implement `.to_nice_units()` → returns new UnitAwareExpression

3. **Fix subtraction unit inference**
   - Ensure `x - (velocity * time)` preserves units of `x`
   - Add Pint compatibility checking

4. **Normalize uw.get_units()**
   - Always return `pint.Unit`, never strings
   - Add conversion if object returns string

### Phase 2: Run Full Test Suite

After Phase 1 fixes:
```bash
pytest tests/test_0750_unit_aware_interface_contract.py -v
# Should show: 17 PASSED, 0 XFAIL
```

### Phase 3: Verify No Regressions

Run existing units tests to ensure fixes don't break anything:
```bash
pytest tests/test_074*.py -v  # All units tests
pytest tests/test_0700_units_system.py -v  # Core units
```

---

## Design Principles (From Tests)

### 1. Interface Consistency
**All unit-aware objects must**:
- Return `pint.Unit` from `.units` property (never string)
- Provide conversion methods: `to()`, `to_base_units()`, `to_compact()`, `to_reduced_units()`, `to_nice_units()`
- Support arithmetic with proper unit combination

### 2. Lazy Evaluation Preservation
**Arithmetic operations must**:
- Preserve symbolic structure (not evaluate immediately)
- Allow updates to symbolic variables to propagate
- Support time-stepping pattern: define once, update many times

### 3. Type Safety
**Throughout the codebase**:
- Internally: Always `pint.Unit` or `pint.Quantity` objects
- User input: Accept strings, immediately convert to Pint
- User output: Can be string (via `str(units)`) but internally must be Pint

---

## Next Steps

1. ✅ **DONE**: Comprehensive interface tests created
2. **TODO**: Fix UnitAwareExpression.units to return pint.Unit
3. **TODO**: Add conversion methods to UnitAwareExpression
4. **TODO**: Fix subtraction unit inference
5. **TODO**: Normalize uw.get_units()
6. **TODO**: Verify all 17 tests pass
7. **TODO**: Run regression tests
8. **TODO**: Update BUG_QUEUE_UNITS_REGRESSIONS.md with results

---

## Success Criteria

**Phase 1 Complete When**:
```bash
pytest tests/test_0750_unit_aware_interface_contract.py -v
# Shows: 17 passed, 0 xfailed
```

**Phase 2 Complete When**:
```bash
pytest tests/test_074*.py tests/test_070*.py -v
# All units tests pass with no regressions
```

**Architecture Fixed When**:
- All unit-aware objects have identical interfaces ✅
- Lazy evaluation works correctly ✅
- No string units returned internally ✅
- Arithmetic closure property holds ✅
