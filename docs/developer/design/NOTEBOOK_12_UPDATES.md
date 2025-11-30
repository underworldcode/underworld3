# Notebook 12 Updates - Human-Readable Model Units

**Date**: 2025-10-12
**Status**: ✅ CODE UPDATED - Needs notebook execution to refresh outputs

## Changes Made

### 1. Converted All Print Statements to uw.pprint()
Replaced all `print()` calls with `uw.pprint(0, ...)` for parallel safety.

**Cells updated**:
- Cell 4: Physical quantities display (4 statements)
- Cell 6: Model creation output (1 statement)
- Cell 8: Model units conversions (6 statements)
- Cell 11: Fundamental scales (4 statements)
- Cell 14: Scale summary (multiple statements)
- Cell 17: Extended model scales (multiple statements)
- Cell 18: Multi-dimensional model (multiple statements)
- Cell 20: Error detection examples (multiple statements)
- Cell 22: Rayleigh number calculation (multiple statements)
- Cell 23: Temperature conversions (multiple statements)
- Cell 24: Dimensionlessness check (2 statements)
- Cell 29: UWQuantity arithmetic (multiple statements)
- Cell 31: Pint-native units (multiple statements)
- Cell 34: Mesh creation (3 statements)
- Cell 35: Variable initialization (2 statements)
- Cell 37: Scaling mode comparison (multiple statements)
- Cell 39: User-friendly features (multiple statements)
- Cell 40: Physics-based analysis (multiple statements)
- Cell 42: Validation features (multiple statements)
- Cell 43: Domain-agnostic analysis (multiple statements)
- Cell 45: Natural language conversions (multiple statements)

### 2. Added Human-Readable Model Units Documentation
Inserted new markdown cell after cell 7 explaining the new feature:

> **Human-Readable Model Units**: When you display quantities in model units, Underworld3 automatically shows both:
> - The numerical value in model units
> - An approximate interpretation in familiar units (like cm/year, km, Myr, GPa)
>
> This makes it easy to understand what "1.0 model unit" actually means in physical terms!

## Expected Output Changes

When the notebook is re-run, outputs will show human-readable interpretations:

### Before (Old Format)
```python
vel_model
# UWQuantity(0.9999999999999922, '_2900000m / _1p83E15s')
```

### After (New Format)
```python
vel_model
# UWQuantity(1.0, '_2900000m / _1p83E15s')  [≈ 5.000 cm/year]

print(f"Velocity: {vel_model:.1f}")
# Velocity: 1.0 (≈ 5.000 cm/year)
```

### Example Model Units Display
```
Reference quantities in model units:
  Depth: 1.0 (≈ 2.90e+03 km)
  Temperature: 1.0 (≈ 1.50e+03 K)
  Viscosity: 1.0 (≈ 1.00e+21 Pa*s)
  Velocity: 1.0 (≈ 5.000 cm/year)

Other quantities in model units:
  Half depth (1450 km): 0.50 (≈ 2.90e+03 km)
  Fast velocity (10 cm/year): 2.0 (≈ 5.000 cm/year)
```

## Next Steps

**To complete the update:**
1. Run the notebook: `jupyter nbconvert --to notebook --execute 12-Units_System.ipynb`
2. Or open in Jupyter and use "Run All" to refresh all outputs
3. Verify all cells execute without errors
4. Verify outputs show human-readable interpretations

**Test command**:
```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/docs/beginner/tutorials
pixi run -e default jupyter nbconvert --to notebook --execute 12-Units_System.ipynb --inplace
```

## Benefits

1. **Parallel Safety**: All print statements now use `uw.pprint()`, ensuring safe execution in parallel environments
2. **User Understanding**: Model units are no longer mysterious - users can see what they mean in familiar units
3. **Consistent Display**: Human-readable interpretations appear in all contexts (str, repr, f-strings)
4. **Educational**: Shows relationship between model units and physical units clearly

## Related Work

- **Implementation**: `src/underworld3/function/quantities.py` - `_interpret_model_units()` method
- **Planning**: `planning/HUMAN_READABLE_MODEL_UNITS.md` - Complete implementation details
- **Tests**: `/tmp/test_readable_model_units.py` - Validation of human-readable display

---

**Status**: ✅ Code changes complete, ready for notebook execution to refresh outputs
