# Unit Simplification Issue in Reference Quantities

**Date**: 2025-10-12
**Status**: ✅ FIXED (2025-10-12)
**Implementation**: Dual unit simplification system in `model.py`

## Problem

When using `model.set_reference_quantities()` with "human friendly" units, derived scales are not simplified:

```python
model.set_reference_quantities(
    mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,
    plate_velocity=5 * uw.units.cm / uw.units.year,
    domain_depth=2000 * uw.units.km,
    mantle_temperature=2000 * uw.units.K
)

scales = model.get_fundamental_scales()
# Output shows:
#   length: 2000 kilometer                    ✓ Good
#   temperature: 2000 kelvin                  ✓ Good
#   time: 500 kilometer * year / centimeter   ✗ NOT SIMPLIFIED
#   mass: 2e+27 kilometer ** 2 * pascal * second * year / centimeter  ✗ NOT SIMPLIFIED
```

## Expected Behavior

Derived units should be simplified to canonical forms:

```python
# What we want:
#   length: 2000 kilometer
#   temperature: 2000 kelvin
#   time: 1.26e+14 second  (or "40.0 Myr")
#   mass: 2.0e+27 kilogram
```

## Root Cause

The `get_fundamental_scales()` method computes derived quantities using Pint arithmetic, but doesn't call `.to_base_units()` or `.to_compact()` to simplify the result.

**Location**: `src/underworld3/scaling/model_scaling.py` (or wherever `get_fundamental_scales()` is implemented)

## Solution

Add unit simplification in `get_fundamental_scales()`:

```python
def get_fundamental_scales(self):
    """Return fundamental scales in simplified units."""
    scales = {}

    # Direct scales
    scales['length'] = self._length_scale
    scales['temperature'] = self._temperature_scale

    # Derived scales - SIMPLIFY THESE
    scales['time'] = (self._length_scale / self._velocity_scale).to_base_units()
    scales['mass'] = (self._viscosity_scale * self._time_scale / self._length_scale).to_base_units()

    return scales
```

Or for even better readability, convert to "nice" units:

```python
# Convert time to most appropriate time unit
time_seconds = time_scale.to('second').magnitude
if time_seconds > 3.156e13:  # > 1 Myr
    scales['time'] = time_scale.to('megayear')
elif time_seconds > 3.156e10:  # > 1000 years
    scales['time'] = time_scale.to('year')
else:
    scales['time'] = time_scale.to('second')
```

## Impact

This affects:
- `model.view()` output (hard to read)
- User understanding of what scales are being used
- Debugging and validation of scaling choices
- Documentation and tutorials showing complex unsimplified units

## Priority

**Medium-High** - Doesn't affect correctness (the math is right), but significantly impacts user experience and understanding.

## Workaround

Users can manually simplify:
```python
scales = model.get_fundamental_scales()
time_readable = scales['time'].to_base_units()
mass_readable = scales['mass'].to_base_units()
```

## Related Files

- `src/underworld3/scaling/model_scaling.py` (or similar)
- `src/underworld3/model.py` (`get_fundamental_scales()` method)
- Tutorial notebooks showing `model.view()` output

## Solution Implemented (2025-10-12)

### Dual Unit Simplification System

Implemented a two-stage simplification approach in `src/underworld3/model.py`:

#### Stage 1: At Model Creation (`derive_fundamental_scalings()`)

**Modified lines 1119-1127, 1142-1149, and similar for all derived scales:**

```python
# Derive time from length / velocity
time_raw = scalings['[length]'] / qty
# TWO-STAGE SIMPLIFICATION:
# Stage 1: Rationalize to SI base units to eliminate compound nonsense
time_base = time_raw.to_base_units()
# Stage 2: Convert to user-appropriate time unit (infer from velocity)
time_user = self._convert_to_user_time_unit(time_base, qty)
scalings['[time]'] = time_user
```

**New helper method `_convert_to_user_time_unit()` (lines 1004-1071):**
- Infers appropriate time unit from velocity's unit system
- Geological scales (cm/year) → year/kiloyear/megayear
- Engineering scales (m/s) → second/hour/day
- Magnitude-based selection within inferred unit family

#### Stage 2: At Display Time (`get_fundamental_scales()`)

**Modified lines 1447-1455:**

```python
for internal_name, scale_qty in scalings.items():
    if internal_name in dimension_map:
        friendly_name = dimension_map[internal_name]
        # Apply magnitude-based unit selection for display
        display_qty = self._choose_display_units(scale_qty, internal_name)
        result[friendly_name] = display_qty
```

**New helper method `_choose_display_units()` (lines 1073-1162):**
- Chooses units that give values in range [1, 1000]
- Dimension-aware unit options (km for length, Myr for time, etc.)
- Scoring algorithm finds optimal magnitude representation

### Results

With the same input:
```python
model.set_reference_quantities(
    mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,
    plate_velocity=5 * uw.units.cm / uw.units.year,
    domain_depth=2000 * uw.units.km,
    mantle_temperature=2000 * uw.units.K
)
```

**Before fix:**
```
length: 2000 kilometer
temperature: 2000 kelvin
time: 500 kilometer * year / centimeter   ✗ NOT SIMPLIFIED
mass: 2e+27 kilometer ** 2 * pascal * second * year / centimeter  ✗ NOT SIMPLIFIED
```

**After fix:**
```
length: 2.0 megameter                     ✅ Magnitude-appropriate
temperature: 2000.0 kelvin                ✅ Direct mapping
time: 40.0 megayear                       ✅ Simplified and readable!
mass: 2.52e+42 kilogram                   ✅ Simplified to SI base!
```

### Testing

Test script created at `/tmp/test_unit_simplification.py`:

```bash
$ pixi run -e default python /tmp/test_unit_simplification.py
✅ Time scale simplified
✅ Mass scale simplified
🎉 ALL TESTS PASSED! Unit simplification is working correctly.
```

### Files Modified

1. **`src/underworld3/model.py`**:
   - Added `_convert_to_user_time_unit()` method (lines 1004-1071)
   - Added `_choose_display_units()` method (lines 1073-1162)
   - Modified `derive_fundamental_scalings()` to apply two-stage simplification (lines 1119-1275)
   - Modified `get_fundamental_scales()` to apply display unit selection (lines 1444-1455)

2. **Build**: Rebuilt with `pixi run underworld-build`

### Benefits

1. **No Compound Units**: Derived scales always simplified to rational units
2. **Context-Aware**: Time units inferred from velocity unit system
3. **Magnitude-Appropriate**: Display uses units that avoid scientific notation when possible
4. **User-Friendly**: Geological scales show in years/megayears, not seconds
5. **Automatic**: No user intervention required

## Recommended Usage Pattern

### Use `uw.pprint()` for Parallel-Safe Display

When displaying fundamental scales in notebooks, examples, and tests, **always use `uw.pprint()` instead of plain `print()`**:

```python
import underworld3 as uw

# Set up model
model = uw.Model("my_model")
model.set_reference_quantities(
    mantle_viscosity=1e21 * uw.scaling.units.Pa * uw.scaling.units.s,
    plate_velocity=5 * uw.scaling.units.cm / uw.scaling.units.year,
    domain_depth=2900 * uw.scaling.units.km,
    mantle_temperature=1500 * uw.scaling.units.K
)

# ❌ DON'T: Unsafe in parallel
if uw.mpi.rank == 0:
    print(f"Scales: {model.get_fundamental_scales()}")

# ✅ DO: Parallel-safe
scales = model.get_fundamental_scales()
uw.pprint(0, "\nFundamental Scales:")
for name, value in scales.items():
    uw.pprint(0, f"  {name:15s}: {value}")
```

**Why `uw.pprint()` is better:**
- All ranks execute collective operations (no deadlocks)
- Only specified ranks print (no duplicate output)
- Cleaner code (no `if uw.mpi.rank == 0:` conditionals)
- **This is the recommended pattern for all UW3 notebooks/examples/tests**

See `planning/UNIT_SIMPLIFICATION_IMPLEMENTATION_SUMMARY.md` for complete examples.

## Testing (Original)

After fix, verify:
```python
model.set_reference_quantities(
    mantle_viscosity=1e21 * uw.units.Pa * uw.units.s,
    plate_velocity=5 * uw.units.cm / uw.units.year,
    domain_depth=2000 * uw.units.km,
    mantle_temperature=2000 * uw.units.K
)

scales = model.get_fundamental_scales()
assert 'kilometer * year / centimeter' not in str(scales['time'])
assert scales['time'].units in ['second', 'year', 'kiloyear', 'megayear']
assert scales['mass'].units == 'kilogram'
# ✅ ALL ASSERTIONS PASS
```
