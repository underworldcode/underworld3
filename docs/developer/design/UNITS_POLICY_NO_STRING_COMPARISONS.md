# Units System Policy: Pint-Only Arithmetic

## CRITICAL POLICY

**ONLY Pint performs unit conversions and arithmetic. NEVER manual fallbacks.**

### The Danger: Losing Numerical Scaling

**Why this matters**: String comparisons and manual arithmetic **lose scale factors**.

```python
x = 100 km
y = 50 m

# WRONG - Manual arithmetic after dimension check:
if dimensions_compatible(x, y):  # ✅ Check passes
    result = 100 + 50  # ❌ WRONG: 150 km (should be 100.05 km!)

# CORRECT - Let Pint handle conversion:
result = x + y  # ✅ Pint converts 50m → 0.05km → 100.05 km
```

**An error is better than wrong physics.**

This policy is **non-negotiable** and must be enforced in all code reviews, testing, and development.

---

## The Rule

### ❌ NEVER Do This:
```python
# WRONG #1: String comparison
if str(self.units) == str(other.units):
    result = self.value + other.value  # Lost scale factor!

# WRONG #2: Dimension check without Pint conversion
if self.units.dimensionality == other.units.dimensionality:
    result = self.value + other.value  # Lost scale factor!

# WRONG #3: Manual conversion attempt
factor = get_conversion_factor(self.units, other.units)
result = self.value + other.value * factor  # Fragile, error-prone!
```

### ✅ ONLY Do This:
```python
# CORRECT - Let Pint handle ALL conversion
try:
    # Option 1: Direct Pint arithmetic (BEST)
    result_pint = self._pint_qty + other._pint_qty
    return UWQuantity.from_pint(result_pint)

    # Option 2: Use .to() method (Pint does conversion)
    other_converted = other.to(self.units)
    result = self.value + other_converted.value

except Exception as e:
    # If Pint can't handle it, FAIL
    raise ValueError(f"Cannot add {self.units} and {other.units}: {e}")

# NO FALLBACKS - Pint or nothing
```

---

## Why This Matters

### Problem: String Comparison Fails for Dimensionally Equivalent Units

**Example**:
```python
velocity = 5 cm/year
time = 1 Myr  # megayear = 1e6 years

displacement = velocity * time
# Internal representation: "cm * megayear / year"
# (Pint doesn't auto-simplify in multiplication)

# String comparison would say these are DIFFERENT:
str(displacement.units)  # "cm * megayear / year"
str(kilometer)           # "kilometer"
# ❌ "cm * megayear / year" != "kilometer"  (strings are different)

# But Pint knows they're the SAME dimension:
displacement.units.dimensionality  # [length]
kilometer.dimensionality           # [length]
# ✅ Both are [length] - dimensionally compatible!
```

### Real Bug This Caused

**User-Reported (2025-11-22)**:
```python
x = uw.expression("x", 100, units="km")
x0 = uw.expression("x0", 50, units="km")
velocity_phys = uw.quantity(5, "cm/year")
t_now = uw.expression("t", 1, units="Myr")

result = x - x0 - velocity_phys * t_now

# With string comparison:
uw.get_units(result)  # ❌ Returned 'megayear' (WRONG!)

# After fix with Pint comparison:
uw.get_units(result)  # ✅ Returns 'kilometer' (CORRECT!)
```

The bug occurred because:
1. `velocity * time` created compound units: `cm * Myr / year`
2. String comparison: `"kilometer" != "cm * Myr / year"` → Rejected subtraction
3. Dimensional check: `[length] == [length]` → Allows subtraction ✅

---

## Strings Are ONLY For Input and Display

### Where Strings Are Acceptable

**1. User Input** (parse immediately):
```python
def __init__(self, value, units: str = None):
    if units is not None:
        # Convert string to Pint immediately
        from ..scaling import units as ureg
        self._pint_qty = value * ureg.parse_expression(units)  # ✅
        self._has_pint_qty = True
```

**2. Display/Repr** (human-readable output):
```python
def __repr__(self):
    unit_str = str(self.units)  # ✅ For display only
    return f"{self.value} {unit_str}"
```

**3. Serialization** (file I/O):
```python
def to_dict(self):
    return {
        'value': self.value,
        'units': str(self.units)  # ✅ For JSON/HDF5 storage
    }
```

### Where Strings Are FORBIDDEN

**1. Return values**:
```python
@property
def units(self):
    return str(self._pint_qty.units)  # ❌ WRONG - return Pint Unit!
    return self._pint_qty.units       # ✅ CORRECT
```

**2. Intermediate storage**:
```python
self._units_string = str(units)  # ❌ WRONG - store Pint!
self._units = ureg(units)        # ✅ CORRECT
```

**3. Comparisons/conversions**:
```python
if str(self.units) == str(other.units):  # ❌ WRONG
    other_converted = other.to(self.units)  # ✅ CORRECT
```

**Rule**: Strings at API boundaries only. Pint everywhere else.

---

## ONLY Acceptable Optimization: Pint Unit Equality

### The ONLY Pattern That's Safe

**ACCEPTABLE** (but MUST have Pint fallback):
```python
# Optimization: Check if Pint Unit objects are identical
if query_units == self.coord_units:  # Comparing Pint Unit objects
    return coords  # SAME OBJECT - skip conversion (optimization)
else:
    # DIFFERENT objects - MUST use Pint conversion
    coords_qty = ureg.Quantity(coords, query_units)
    coords_converted = coords_qty.to(self.coord_units)  # REQUIRED
    return coords_converted.magnitude
```

**Rules for this optimization**:
1. ✅ Both operands MUST be Pint Unit objects (not strings!)
2. ✅ MUST have Pint conversion in the `else` branch
3. ✅ Only use as optimization to skip work, not for correctness
4. ✅ If Pint conversion fails in `else`, let it raise

**Why This Is Safe**:
- `==` on Pint Units uses Pint's `__eq__` (checks object identity/equivalence)
- If units are identical Pint objects: `km == km` → skip conversion (safe)
- If units differ: `km == m` → False → **Pint MUST do conversion**

### Type Checking Is OK (Input Sanitization)

**ACCEPTABLE** (at API boundaries only):
```python
# Defensive: Check if we received string or Pint
if isinstance(units, str):
    pint_unit = ureg.parse_expression(units)  # Parse to Pint immediately
else:
    pint_unit = units  # Already Pint, use directly
```

**Rules**:
1. ✅ Only at API boundaries (accepting user input)
2. ✅ Immediately convert strings to Pint
3. ✅ Never use string comparison of unit values

---

## Implementation Checklist

When writing or reviewing code involving units:

### ✅ MUST Do:
1. **Accept strings in public API** (user convenience)
2. **Convert strings to Pint immediately** upon receipt
3. **Store Pint objects internally** (never store strings)
4. **Return Pint objects** to users (preserve functionality)
5. **Let Pint perform ALL conversions** (no manual arithmetic)
6. **Fail loudly** if Pint can't handle it
7. **Convert to strings** only in `__repr__`, `__str__`, or serialization

### ❌ NEVER Do:
1. **Store units as strings** internally
2. **Return strings** from `.units` property (return Pint Unit!)
3. **Compare unit strings** for compatibility
4. **Manual arithmetic after dimension check** (loses scale factors!)
5. **Manual conversion calculations** (fragile and error-prone)
6. **Fallbacks that don't use Pint conversion** (wrong physics!)
7. **Convert to strings** as return values (users can call `str()` themselves)

---

## Code Review Questions

When reviewing units-related code, ask:

1. **Is this comparing units using strings?**
   - If yes: REJECT (unless it's display/serialization)

2. **Does this store units as strings internally?**
   - If yes: REJECT (only accept strings at API boundary)

3. **Does this return strings from `.units` property?**
   - If yes: REJECT (return Pint Unit objects)

4. **Does error handling fall back to string comparison?**
   - If yes: REJECT (use Pint fallback instead)

5. **Is this optimization using `==` without Pint fallback?**
   - If yes: REJECT (must have Pint conversion fallback)

---

## Historical Violations (Fixed)

### Fix #1: UnitAwareExpression String Equality (2025-11-22)

**Before** (WRONG):
```python
def __sub__(self, other):
    if self._units != other._units:  # ❌ String comparison
        raise ValueError(...)
```

**After** (CORRECT):
```python
def __sub__(self, other):
    try:
        self_pint = 1.0 * self._units
        other_pint = 1.0 * other._units
        _ = other_pint.to(self._units)  # ✅ Pint conversion check
        # Compatible - proceed
    except Exception:
        raise ValueError(f"Incompatible dimensions: {e}")
```

**Files**: `src/underworld3/expression_types/unit_aware_expression.py` (lines 223-333)
**Date**: 2025-11-22

### Fix #2: UWQuantity Removed Dangerous Fallback (2025-11-22)

**Before** (WRONG - TWICE!):
```python
# First version: String comparison (loses scale factors)
except (AttributeError, ValueError):
    if str(self.units) == str(other.units):  # ❌ String fallback
        result = self.value + other.value  # ❌ No conversion!

# Second version: Dimension check without conversion (STILL loses scale factors!)
except (AttributeError, ValueError):
    try:
        _ = other_pint.to(self_pint.units)  # ✅ Check compatibility
        result = self.value + other.value  # ❌ DIDN'T APPLY CONVERSION!
    except Exception:
        raise ValueError("Incompatible dimensions")
```

**After** (CORRECT):
```python
# Use .to() for conversion - let Pint handle ALL scaling
try:
    other_converted = other.to(str(self.units))  # ✅ Pint does conversion
    result = self.value + other_converted.value  # ✅ Converted value
    return UWQuantity(result, str(self.units))
except (AttributeError, ValueError) as e:
    # If Pint can't handle it, FAIL - don't try manual conversion
    raise ValueError(f"Cannot add {other.units} and {self.units}. Pint conversion failed: {e}")
```

**Key Fix**: Removed fallback entirely. Either Pint does the conversion or we fail.

**Files**: `src/underworld3/function/quantities.py` (lines 665-676, 711-722)
**Date**: 2025-11-22

---

## Testing Requirements

### Test Coverage for Dimensionally Compatible Units

All unit-aware classes **MUST** have tests for:

1. **Different units, same dimension**:
   ```python
   def test_different_units_same_dimension(self):
       x = uw.quantity(100, "km")
       y = uw.quantity(50, "m")  # Different units!

       result = x + y
       # Should succeed - both are [length]
       assert result.units.dimensionality == ureg.meter.dimensionality
   ```

2. **Compound units from multiplication**:
   ```python
   def test_compound_units_subtraction(self):
       velocity = uw.quantity(5, "cm/year")
       time = uw.quantity(1, "Myr")
       displacement = velocity * time  # Creates "cm * Myr / year"

       distance = uw.quantity(100, "km")
       result = distance - displacement  # Should work!
       assert result.units.dimensionality == ureg.meter.dimensionality
   ```

3. **Incompatible dimensions (should raise)**:
   ```python
   def test_incompatible_dimensions_raise(self):
       length = uw.quantity(100, "m")
       time = uw.quantity(5, "s")

       with pytest.raises(ValueError, match="incompatible"):
           result = length + time  # Should fail: can't add [length] + [time]
   ```

### Continuous Integration

- **All units tests** must pass before merging
- **Regression suite** (`test_0750_*.py`, `test_0751_*.py`) must be green
- **New features** must include dimensional compatibility tests

---

## Summary

### The Core Principle

**Pint is better than string comparison because:**
1. **Dimensional analysis**: Recognizes `cm`, `meter`, `km` are all `[length]`
2. **Automatic simplification**: Simplifies `cm * Myr / year` to `cm`
3. **Unit conversion**: Handles conversion between compatible units
4. **Physics-based**: Only fails on physically incompatible operations

**Strings are dumb text matching:**
1. **No dimensional analysis**: `"km" != "meter"` even though both are `[length]`
2. **No simplification**: Can't simplify `"cm * Myr / year"` to `"cm"`
3. **No conversion**: Can't convert between unit systems
4. **Text-based**: Fails on trivial differences (`"km"` vs `"kilometer"`)

### The Policy

```
User Input (str) → [PARSE] → Pint Objects → [INTERNAL OPERATIONS] → Pint Objects → User Output (Pint)
                     ↑                                ↑                                    ↑
                  BOUNDARY                         EVERYWHERE                          RETURN Pint
                  (Accept str)                  (Use Pint Only)                    (Users can call str() if needed)
```

**Key Points**:
1. **Accept strings** from users (convenience)
2. **Parse to Pint immediately** at boundary
3. **Use Pint everywhere** internally
4. **Return Pint objects** to users (preserve functionality)

**Only convert to strings**:
- In `__repr__()` / `__str__()` for display
- When serializing to files (JSON, HDF5, etc.)
- NEVER in the middle of calculations or as return values from `.units` property

---

## Enforcement

This policy is **mandatory** and will be enforced through:
1. **Code reviews**: All PRs checked for string comparisons
2. **Test coverage**: Tests must verify dimensional compatibility
3. **Documentation**: This file is the policy of record
4. **Architecture**: Unit-aware classes must follow this pattern

**Violations will be rejected in code review.**

---

**Status**: ✅ **ACTIVE POLICY**
**Date**: 2025-11-22
**Authority**: Core architecture principle
**Scope**: All units-related code in Underworld3
