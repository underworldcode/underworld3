# Unit Rounding Modes - Summary

## Problem Solved

**Before:** Unit names had ugly floating point precision artifacts:
```
UWQuantity(2.0000000000000107, '_499p9999999999974m')
```

**After:** Clean, readable unit names:
```
UWQuantity(1.0, '_1km')  [≈ 0.5000 km]
```

## Implementation

### Two Rounding Modes

#### 1. Powers of 10 (Default)
Rounds reference quantities to the nearest power of 10:
- Only produces: 1, 10, 100, 1000, 10000, ... (no 2, 5 intermediates)
- Threshold at 5.0 with floating point tolerance

**Examples:**
- 499.999 m → 100 m → `_100m`
- 500 m → 1000 m → `_1km`
- 1500 m → 1000 m → `_1km`
- 7500 m → 10000 m → `_10km`

#### 2. Engineering Mode
Rounds to powers of 1000 (aligns with SI prefixes):
- Only produces: 1, 1000, 1e6, 1e9, ... (10^(3n))
- Threshold at 500 with floating point tolerance
- Advantage: Aligns with SI prefixes (k, M, G, T, P, E, Z, Y)

**Examples:**
- 499.999 m → 1 m → `_1m`
- 500 m → 1000 m → `_1km`
- 1500 m → 1000 m → `_1km`
- 7500 m → 1000 m → `_1km` (different from powers_of_10!)

### Usage

**Default (Powers of 10):**
```python
import underworld3 as uw

# Uses powers_of_10 mode by default
model = uw.get_default_model()
model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(500, "m"),  # → rounds to 1km
    mantle_temperature=uw.quantity(1000, "K"),
)
```

**Engineering Mode:**
```python
import underworld3 as uw

# Enable engineering mode
model = uw.get_default_model()
model.unit_rounding_mode = "engineering"  # Set BEFORE set_reference_quantities()

model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(7500, "m"),  # → rounds to 1km (not 10km!)
    mantle_temperature=uw.quantity(1000, "K"),
)
```

## Technical Details

### Floating Point Tolerance

The key fix was adding tolerance to threshold checks to handle floating point errors from dimensional analysis:

**Problem:**
- Dimensional analysis uses `10**log10(value)` operations
- For 500m: `10**log10(500)` = `10**2.69897...` = `499.9999999...`
- Old threshold `normalized >= 5.0` would fail, rounding DOWN to 100 instead of UP to 1000

**Solution:**
- Added tolerance: `normalized >= 5.0 - 1e-9` (powers of 10)
- Added tolerance: `normalized >= 500 - 1e-6` (engineering)
- Now 499.99999... correctly rounds to 1000

### Files Modified

**`src/underworld3/model.py`:**
1. Added `unit_rounding_mode` Pydantic field (default: "powers_of_10")
2. Modified `_round_to_nice_value()` to support both modes with tolerance
3. Modified threshold checks in both modes

## Testing

All tests passing:
- ✅ `test_clean_unit_names.py` - No floating point artifacts
- ✅ `test_mesh_coords_units.py` - Mesh coordinates have units
- ✅ `test_unit_rounding_modes.py` - Both modes work correctly
- ✅ `tests/test_0812_poisson_with_units.py` - Solver integration works

## Benefits

1. **Clean unit names:** No more `_499p9999999999974m`
2. **User control:** Choose between powers of 10 or engineering mode
3. **SI prefix alignment:** Engineering mode aligns with standard prefixes
4. **Robust:** Floating point tolerance handles numerical errors gracefully
5. **Backward compatible:** Default behavior produces clean names automatically

## Migration

**No migration needed!** Existing code automatically gets cleaner unit names. Users who want engineering mode can opt in by setting `model.unit_rounding_mode = "engineering"`.
