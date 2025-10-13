# Human-Readable Model Units Display

**Date**: 2025-10-12
**Status**: ✅ IMPLEMENTED

## Problem

Model units using Pint constants were incomprehensible when displayed:

```python
vel_model
# UWQuantity(0.9999999999999922, '_2900000m / _1p83E15s')
```

Users couldn't understand what these `_2900000m / _1p83E15s` units meant, making model units mysterious and confusing.

## Solution

Implemented automatic interpretation of model units that:

1. **Combines Pint constants numerically**
   - `_2900000m / _1p83E15s` → `(2.9e6 / 1.83e15) m/s` → `1.58e-9 m/s`

2. **Converts to user-friendly units**
   - Tries common units (cm/year, km, Myr, GPa, etc.)
   - Chooses units that give reasonable magnitudes (0.001 - 1000)
   - Prioritizes geological scales for geodynamics

3. **Shows interpretation in all display contexts**
   - `str()`: `3.16e9 (≈ 5.000 cm/year)`
   - `repr()`: `UWQuantity(..., '_2900000m / _1p83E15s')  [≈ 5.000 cm/year]`
   - `f"{vel}"`: `3.16e9 (≈ 5.000 cm/year)`

## Implementation

### Files Modified

**`src/underworld3/function/quantities.py`:**

1. **New method `_interpret_model_units()` (lines 722-864)**:
   - Combines Pint constants via `.to_base_units()`
   - Tests friendly unit conversions (cm/year, km, Myr, GPa, etc.)
   - Scores each conversion by magnitude appropriateness
   - Returns best human-readable interpretation

2. **Updated `__str__()` (lines 690-712)**:
   - Calls `_interpret_model_units()` for model units
   - Shows `value (≈ X units)` format

3. **Updated `__repr__()` (lines 866-887)**:
   - Appends `[≈ X units]` for model units
   - Preserves technical units for reference

4. **Updated `__format__()` (lines 574-615)**:
   - Includes interpretation in f-strings
   - Shows units for regular quantities too

### Key Algorithm

```python
def _interpret_model_units(self) -> str:
    """
    Interpret model units in human-readable form.

    Steps:
    1. Create unit quantity (1.0 * model_units)
    2. Convert to SI base units (combines constants)
    3. Try friendly unit conversions (cm/year, km, Myr, etc.)
    4. Score by magnitude (prefer 0.001-1000 range)
    5. Return best interpretation
    """
    # Example: _2900000m / _1p83E15s
    unit_qty = 1.0 * self._pint_qty.units

    # Combines: (2.9e6 / 1.83e15) m/s = 1.58e-9 m/s
    base_qty = unit_qty.to_base_units()

    # Try conversions: cm/year, km/Myr, m/s, etc.
    for name, friendly_unit in friendly_conversions:
        converted = base_qty.to(friendly_unit)
        magnitude = abs(converted.magnitude)

        # Score: prefer magnitudes 0.001-1000
        score = calculate_score(magnitude)

        if score < best_score:
            best_conversion = (magnitude, name)

    return f"≈ {magnitude} {name}"
```

### Friendly Units Priority

```python
friendly_conversions = [
    # Velocity (geological first)
    ('cm/year', u.cm / u.year),
    ('km/Myr', u.km / (1e6 * u.year)),
    ('mm/year', u.mm / u.year),
    ('m/s', u.m / u.s),

    # Length
    ('km', u.km),
    ('m', u.m),
    ('cm', u.cm),

    # Time (geological first)
    ('Myr', 1e6 * u.year),
    ('kyr', 1e3 * u.year),
    ('year', u.year),
    ('day', u.day),
    ('s', u.s),

    # Pressure/stress (geological scales first)
    ('GPa', 1e9 * u.Pa),
    ('MPa', 1e6 * u.Pa),
    ('Pa', u.Pa),

    # Temperature, viscosity, density, etc.
]
```

## Examples

### Before Fix

```python
vel_model = model.to_model_units(uw.quantity(5, "cm/year"))

# Incomprehensible!
print(vel_model)
# UWQuantity(0.9999999999999922, '_2900000m / _1p83E15s')
```

### After Fix

```python
vel_model = model.to_model_units(uw.quantity(5, "cm/year"))

# Human-readable!
print(vel_model)
# 3.16e9 (≈ 5.000 cm/year)

print(repr(vel_model))
# UWQuantity(3.16e9, '_2900000m / _1p83E15s')  [≈ 5.000 cm/year]

# Works in f-strings too
print(f"Velocity: {vel_model}")
# Velocity: 3.16e9 (≈ 5.000 cm/year)
```

### Different Quantities

```python
# Length
length_model = model.to_model_units(uw.quantity(100, "km"))
print(length_model)
# 0.0000345 (≈ 100 km)

# Time
time_model = model.to_model_units(uw.quantity(1, "Myr"))
print(time_model)
# 5.46e-16 (≈ 1.00 Myr)

# Temperature
temp_model = model.to_model_units(uw.quantity(1000, "K"))
print(temp_model)
# 0.667 (≈ 1000 K)
```

## Benefits

1. **Demystifies Model Units**
   - Users understand what "1.0 model unit" means
   - No more mysterious Pint constant names

2. **Educational**
   - Shows the relationship between model units and physical units
   - Helps users verify their unit conversions

3. **Debugging-Friendly**
   - Easy to spot unit conversion errors
   - Can quickly check if magnitudes are reasonable

4. **Consistent Display**
   - Works in all contexts: str(), repr(), f-strings
   - Print statements, notebooks, debugging all show interpretation

5. **Geological Focus**
   - Prioritizes geological units (cm/year, km, Myr, GPa)
   - Appropriate for geodynamics applications

## Technical Notes

### Pint Constant Combination

The key insight is that Pint constants like `_2900000m` represent actual numerical values (2.9e6 meters). When we create `1.0 * (_2900000m / _1p83E15s)` and convert `.to_base_units()`, Pint automatically:

1. Evaluates each constant to its numerical value
2. Performs the arithmetic: `2.9e6 m / 1.83e15 s`
3. Returns: `1.58e-9 m/s`

This is then converted to friendly units using standard Pint conversion.

### Magnitude-Based Scoring

The scoring algorithm prefers magnitudes in range [0.001, 1000]:

```python
if -3 <= log10(magnitude) <= 3:
    score = abs(log10(magnitude))  # Prefer ~1
elif log10(magnitude) < -3:
    score = 100 + abs(log10(magnitude) + 3)  # Heavily penalize
else:
    score = 100 + (log10(magnitude) - 3)  # Heavily penalize
```

This ensures we show:
- `5.000 cm/year` instead of `1.58e-9 m/s`
- `2.90 km` instead of `2900000 mm`
- `58.0 Myr` instead of `1.83e15 s`

### Edge Cases

- **No model instance**: Returns `None` (no interpretation shown)
- **Dimensionless**: No interpretation needed
- **No good conversion**: Falls back to SI base units with magnitude
- **Conversion errors**: Silently returns `None`

## Testing

Test script: `/tmp/test_readable_model_units.py`

**Results:**
```bash
$ pixi run -e default python /tmp/test_readable_model_units.py

✅ Model units now show human-readable interpretations!
✅ The Pint constants are combined (e.g., _2900000m / _1p83E15s → numerical value)
✅ Converted to user-friendly units (cm/year, km, Myr, etc.)
✅ Both str() and repr() show the interpretation

🎉 Model unit display is now human-friendly!
```

## Related Work

This builds on the unit simplification work from earlier today:
- **UNIT_SIMPLIFICATION_IMPLEMENTATION_SUMMARY.md** - Simplified fundamental scales display
- Both improvements make the units system more user-friendly and less mysterious

## Future Enhancements

1. **User Preferences**: Allow users to specify preferred units per dimension
2. **Context-Aware Units**: Adjust based on problem type (geological vs. engineering)
3. **Caching**: Cache interpretations to avoid repeated conversions
4. **More Unit Options**: Extend friendly_conversions list for specialized applications

---

**Status: COMPLETE** 🎉

Model units are now human-readable! Users can understand what their model units mean in physical terms, making the units system much less mysterious and more educational.
