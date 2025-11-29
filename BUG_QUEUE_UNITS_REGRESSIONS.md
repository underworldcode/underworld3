# Units System Architecture Bugs (2025-11-21)

## Critical: Inconsistent Return Types from .units Property

### Bug 1: UnitAwareExpression.units Returns String
**Location**: `src/underworld3/expression_types/unit_aware_expression.py` lines 79-82

**Violation**: Returns string instead of Pint Unit object
```python
velocity_phys = uw.quantity(5, "cm/year")
t_now = uw.expression("t", 1, "time", units="Myr")
result = velocity_phys * t_now

# WRONG: Returns string
uw.get_units(result)  # 'centimeter * megayear / year'

# EXPECTED: Should return Pint Unit object
# ureg.parse_expression('centimeter * megayear / year')
```

**Architecture Principle Violated**: From CLAUDE.md:
> "Accept strings for user convenience, but ALWAYS store and return Pint objects internally."

**Current Code** (unit_aware_expression.py:79-82):
```python
if hasattr(self._units, 'dimensionality'):
    # It's a pint.Unit - convert to string
    return str(self._units)  # ❌ WRONG - returns string!
```

**Should Be**:
```python
if hasattr(self._units, 'dimensionality'):
    return self._units  # ✅ Return Pint object
```

---

### Bug 2: Subtraction Returns Wrong Units
**Symptom**:
```python
x = uw.expression("x", 5, "distance", units="km")
velocity_phys = uw.quantity(5, "cm/year")
t_now = uw.expression("t", 1, "time", units="Myr")

result = x - velocity_phys * t_now

# WRONG: Returns 'megayear'
uw.get_units(result)  # 'megayear'

# EXPECTED: Should return 'kilometer' (units of x)
```

**Root Cause**: Likely in `UnitAwareExpression.__sub__()` - doesn't properly combine/check unit compatibility

---

### Bug 3: Missing Unit Conversion Methods
**Location**: `UnitAwareExpression` class

**Missing Methods**:
- `.to_base_units()` 
- `.to_compact()`
- `.to_reduced_units()`
- `.to_nice_units()`

**Impact**: UnitAwareExpression objects (returned from arithmetic) don't have the same interface as UWQuantity objects, breaking the closure property.

**User Expectation**:
```python
result = velocity_phys * t_now  # Returns UnitAwareExpression
result.to_compact()  # Should simplify 'cm * Myr / year' → 'km'
# Currently: AttributeError!
```

---

## Root Cause Analysis

### Fragile Architecture
Every fix introduces new regressions because we're patching at the wrong level:

1. **Fixed**: `.sym` setter didn't update `._pint_qty`
2. **Broke**: Nothing (good fix)
3. **Fixed**: `.copy()` didn't update `._pint_qty`
4. **Broke**: Nothing (good fix)
5. **Fixed**: Expression arithmetic lost units (UWQuantity * UWexpression)
6. **Broke**: 
   - Now returns UnitAwareExpression with string units
   - Subtraction has wrong unit inference
   - Missing conversion methods

### The Pattern
We have **three different unit-aware classes** with **inconsistent interfaces**:

| Feature | UWQuantity | UWexpression | UnitAwareExpression |
|---------|------------|--------------|---------------------|
| `.units` returns | Pint Unit ✅ | Pint Unit ✅ | String ❌ |
| `.to_base_units()` | ✅ | ✅ (inherited) | ❌ Missing |
| `.to_compact()` | ✅ | ✅ (inherited) | ❌ Missing |
| `._pint_qty` | ✅ | ✅ (inherited) | ❌ No Pint storage |
| Arithmetic | Returns UWQuantity | Returns UnitAwareExpression | Returns UnitAwareExpression |

### The Problem
**UnitAwareExpression** was designed as a lightweight wrapper (SymPy expr + units), but it's now the **return type for all arithmetic**, so it needs the **full UWQuantity interface**.

---

## Proposed Solutions (DO NOT IMPLEMENT YET)

### Option A: Make UnitAwareExpression Consistent
Add missing methods and fix return types:
```python
class UnitAwareExpression:
    @property
    def units(self):
        # Return Pint Unit, not string
        return self._units  # Don't convert to string!
    
    def to_base_units(self):
        # Implement conversion methods
        ...
```

**Pro**: Minimal changes
**Con**: Still have three different classes with duplicated logic

### Option B: Unified Units Protocol
Define a protocol/abstract base class:
```python
class UnitAwareProtocol(Protocol):
    @property
    def units(self) -> ureg.Unit:  # Always Pint Unit
        ...
    
    def to_base_units(self) -> Self:
        ...
    
    def to_compact(self) -> Self:
        ...
```

Then ensure all three classes implement it.

### Option C: Composition Over Inheritance
Extract unit storage/operations into a shared component:
```python
class UnitsStorage:
    """Handles all unit storage, conversion, arithmetic"""
    def __init__(self, units: str | ureg.Unit):
        self._pint_qty = ...
    
    @property
    def units(self) -> ureg.Unit:
        return self._pint_qty.units
    
    def to_base_units(self):
        ...

class UWQuantity:
    def __init__(self, value, units):
        self._units_storage = UnitsStorage(units)
    
    @property
    def units(self):
        return self._units_storage.units

class UnitAwareExpression:
    def __init__(self, expr, units):
        self._expr = expr
        self._units_storage = UnitsStorage(units)
    
    @property  
    def units(self):
        return self._units_storage.units
```

**Pro**: Single source of truth, consistent behavior
**Con**: Significant refactoring

---

## Recommendation

**STOP PATCHING. Do a comprehensive units interface audit:**

1. Document the expected interface for ALL unit-aware objects
2. Audit all three classes for compliance
3. Write comprehensive interface tests FIRST
4. Then fix systematically, testing after each change

**Test Coverage Needed**:
- Return type of `.units` property (must be Pint Unit, never string)
- Presence of all conversion methods (to_base_units, to_compact, etc.)
- Arithmetic closure (result has same interface as operands)
- Unit inference for all operations (+, -, *, /, **)
- Compatibility with `uw.get_units()` (should normalize if needed)

---

## Status: DOCUMENTED, NOT FIXED
**Date**: 2025-11-21
**Severity**: Critical - Architecture violation
**Next Step**: Comprehensive interface audit before attempting fixes
