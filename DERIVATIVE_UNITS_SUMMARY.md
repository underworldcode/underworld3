# Derivative Units Implementation Summary

**Date**: 2025-10-15
**Status**: ✅ COMPLETE

## Problem Statement

The user identified that derivatives were not computing units correctly due to the chain rule:

```python
T.sym.diff(mesh.N.x)  # Was returning 'kelvin' instead of 'kelvin / kilometer'
```

The question was: **Do we need to patch in a special diff method to make this actually work?**

## Answer: No Special Diff Method Needed!

The solution was to enhance the existing unit detection system to recognize derivatives and compute their units correctly using dimensional analysis.

## Implementation

### 1. Derivative Detection

Derivatives in Underworld/SymPy are identified by the `diffindex` attribute. This tells us:
- It's a derivative function (not a regular UnderworldFunction)
- Which coordinate it's derived with respect to (the index in `args`)

### 2. Units Computation

For a derivative `dT/dx`:
- Get variable units: `T.units` → `'kelvin'`
- Get coordinate units: `get_units(x)` → `'kilometer'`
- Compute derivative units using Pint: `kelvin / kilometer`

### 3. Code Changes

**File**: `src/underworld3/function/unit_conversion.py`

**Change 1** - Enhanced `get_units()` (lines 232-238):
```python
# Priority 2a: Check for DERIVATIVES first (before general UnderworldFunction)
# Derivatives have diffindex attribute and need special handling: var_units / coord_units
if hasattr(obj, 'diffindex'):
    # Delegate to compute_expression_units which has derivative handling
    computed_units = compute_expression_units(obj)
    if computed_units:
        return computed_units
```

**Change 2** - Enhanced `compute_expression_units()` (lines 337-359):
```python
# Priority -1: Check for DERIVATIVES first (before general UnderworldFunction)
# Derivatives are UnderworldFunctions with a diffindex attribute
# Units of derivative = units(variable) / units(coordinate)
if hasattr(expr, 'diffindex'):
    try:
        # Get variable units from meshvar
        variable = expr.meshvar()
        if variable and hasattr(variable, 'units'):
            var_units = variable.units
            if var_units is not None:
                # Get the coordinate it's derived with respect to
                deriv_index = expr.diffindex
                if deriv_index < len(expr.args):
                    coord = expr.args[deriv_index]
                    coord_units = get_units(coord)
                    if coord_units is not None:
                        # Use Pint to compute derivative units: var_units / coord_units
                        var_pint = ureg.parse_expression(str(var_units))
                        coord_pint = ureg.parse_expression(coord_units)
                        deriv_units = var_pint / coord_pint
                        return str(deriv_units.units)
    except Exception:
        pass
```

## Chain Rule Consideration

The user's original question included: "if x is scaled, then the chain rule should also mean the derivative is scaled."

### Answer: Coordinates Already Have Physical Units

In Underworld3's current implementation:
- `mesh.N.x` is treated as PHYSICAL coordinates (kilometers)
- NOT model coordinates (0-1 dimensionless)
- Therefore, `dT/dN.x` gives the correct physical gradient directly
- No additional scaling needed!

See `test_derivative_units.py` section 7 for detailed analysis.

## Validation

### Comprehensive Tests Passing ✅

**File**: `test_complete_units_system.py`

All tests pass:
1. ✅ Coordinate units from mesh
2. ✅ Variable units from mesh variables
3. ✅ Compound expressions (*, /, **)
4. ✅ Unit cancellation in expressions
5. ✅ **Derivative units with chain rule**
6. ✅ Complex expressions with derivatives

### Example Results

```python
import underworld3 as uw

# Coordinates
uw.get_units(mesh.X[0])  # → 'kilometer' ✅

# Variable
uw.get_units(T.sym)  # → 'kelvin' ✅

# Compound expressions
uw.get_units(T.sym / y)  # → 'kelvin / kilometer' ✅
uw.get_units(T.sym * x / y)  # → 'kelvin' (cancellation) ✅

# Derivatives
grad = T.sym.diff(mesh.N.x)
uw.get_units(grad)  # → 'kelvin / kilometer' ✅

# Complex expressions
uw.get_units(grad[0,0] * x)  # → 'kelvin' (cancellation) ✅
```

## Benefits

1. **Correct Physics**: Derivatives now have correct physical dimensions
2. **Automatic**: No special syntax or methods needed - just works
3. **Complete System**: Works seamlessly with compound expressions and unit cancellation
4. **Uses Pint**: Leverages existing dimensional analysis infrastructure
5. **No Breaking Changes**: Existing code continues to work

## Technical Insight

The key insight is that **derivatives are just another expression type** that needs dimensional analysis. By detecting them early in the priority chain (before checking for general UnderworldFunctions), we can apply the correct chain rule arithmetic.

The implementation leverages:
- SymPy's derivative metadata (`diffindex`, `args`)
- Existing coordinate unit patching system
- Pint's dimensional arithmetic
- Unified `compute_expression_units()` function

## Limitations

- Second derivatives are not supported by Underworld (throws RuntimeError)
- This is a limitation of the underlying function system, not the units system

## Documentation

- Technical details: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
- Project status: `CLAUDE.md` section "Coordinate Units and Dimensional Analysis"
- Validation: `test_complete_units_system.py`
- Chain rule analysis: `test_derivative_units.py`
