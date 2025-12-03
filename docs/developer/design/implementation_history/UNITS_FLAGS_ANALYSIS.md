# Units and Scaling Flags Analysis

**Date**: 2025-11-15
**Author**: Claude (AI Assistant)
**Status**: Analysis for Review

## Executive Summary

Analysis of three critical issues:
1. **Flag system interaction** with JIT vs evaluate unwrappers
2. **Merging plan implications** for unwrapping unification
3. **Half-way zone problem** - units without reference quantities

**Key Finding**: Current system has a fundamental bug where both unwrappers fail to convert units when scaling is OFF. The "half-way zone" allows creation of inconsistent states that produce silent errors.

---

## 1. Flag System Analysis

### Current Behavior

**Single Global Flag**: `_USE_NONDIMENSIONAL_SCALING` (default: `False`)

**Both unwrappers use the same flag**:
- JIT: `if uw._is_scaling_active():` (line 186 in expressions.py)
- Evaluate: `scaling_active = uw.is_nondimensional_scaling_active()` (line 107 in functions_unit_system.py)

### Test Results

Input: `uw.quantity(0.0001, "km/yr")` = `3.17e-9 m/s` in SI

| Scaling Mode | JIT unwrap | Evaluate unwrap | Expected |
|--------------|------------|-----------------|----------|
| **OFF** (default) | `0.0001` | `0.0001` | `3.17e-9` (SI) |
| **ON** | `2.0` (ND) | `2.0` (ND) | `2.0` (ND) |

**Finding**: Both unwrappers return the same values, but **both are wrong when scaling is OFF**.

### The Bug

**Location**: `unwrap_for_evaluate()` lines 398-402 (same logic in `_unwrap_expressions()`)

```python
# No scaling: just return the value
if hasattr(expr, 'value'):
    return sympy.sympify(expr.value), result_dimensionality
```

**Problem**: Returns raw magnitude (`0.0001`) without unit conversion to SI.

**Impact**:
- `0.0001 km/yr` → returns `0.0001`
- `0.0001 cm/yr` → returns `0.0001`
- **Ratio**: 1:1 (should be 100,000:1)

### Does This Break JIT Compilation?

**Answer: YES, but only for constants in equations**

**Scenario**: User writes equation with mixed units:
```python
# Reference quantities in m/s
model.set_reference_quantities(plate_velocity=uw.quantity(5, "cm/year"))

# Equation uses km/yr constant (NOT scaled OFF mode)
buoyancy = temperature * uw.quantity(0.0001, "km/yr")
```

**What happens**:
1. JIT unwrapper sees `0.0001 km/yr`, scaling OFF
2. Returns `0.0001` (wrong! should be `3.17e-9`)
3. PETSc gets equation with wrong coefficient
4. **Silent error** - solution is off by 100,000x

**Severity**: HIGH - affects solver correctness

---

## 2. Merging Plan Implications

### Original Merging Plan (TODO.md)

**Objective**: Unify JIT and evaluate unwrappers into single core

**Assumption**: "After fixing variable scaling bug, both do essentially the same thing"

**Proposed**:
```
_unwrap_expression_core()  ← Common logic
├── unwrap_for_evaluate()  ← Adds dimensionality tracking
└── _unwrap_for_compilation()  ← Just returns expression
```

### Current Reality

**They DO the same thing** (both wrong!):
- Both use `_unwrap_expressions()` as core
- Both have same bug with constants when scaling OFF
- Difference is **only** dimensionality tracking for re-dimensionalization

### Merging Impact

**The current bug makes merging EASIER, not harder**:

1. **Shared bug location**: Both use `_unwrap_expressions()` → fix once, fixes both
2. **Shared logic**: Consolidation is still valid
3. **Key difference**: Return signature only
   - JIT: `return expr` (pure SymPy)
   - Evaluate: `return (expr, dimensionality)` (SymPy + metadata)

### Updated Merging Recommendation

**BEFORE merging**: Fix the units bug in shared core `_unwrap_expressions()`

**Corrected flow**:
```python
def _unwrap_expressions(fn, keep_constants=True, return_self=True):
    """Core unwrapping logic."""

    if isinstance(fn, UWQuantity):
        if uw._is_scaling_active() and model.has_units():
            # Scaling ON: non-dimensionalize
            nondim = uw.non_dimensionalise(fn)
            return sympy.sympify(nondim.value)
        else:
            # Scaling OFF: convert to SI base units
            si_qty = fn.to_base_units()  # ← FIX: Convert to SI
            return sympy.sympify(si_qty.magnitude)
```

**Then merge**:
```
_unwrap_expressions_core(fn) → SymPy in SI or ND
├── unwrap_for_evaluate(fn) → (expr, dims)
└── _unwrap_for_compilation(fn) → expr
```

**Status**: Merging plan is STILL VALID, just needs bug fix first.

---

## 3. Half-Way Zone Problem

### The Problem

**Definition**: Variables have units but model has no reference quantities

**Example**:
```python
# No reference quantities
mesh = uw.meshing.StructuredQuadBox(...)
v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")  # ← Units but no scales

# What happens?
v.scaling_coefficient = 1.0  # Default (wrong conditioning)
```

### Current Behavior

**Creation**: Allowed with warning
```
UserWarning: Variable 'v' has units 'm/s' but no reference quantities are set.
Variable will use scaling_coefficient=1.0, which may lead to poor numerical conditioning.
```

**Evaluation**: Silent errors
```python
q = uw.quantity(100, "m/s")
result = uw.function.evaluate(q, coords)
# Returns: 100.0 (plain number, units dropped!)
```

### Why This Is Dangerous

1. **Silent errors**: Units disappear, user doesn't notice
2. **Inconsistent state**: System has partial unit information
3. **Mixed signals**: Warning on creation, silence on usage
4. **No enforcement**: User can proceed with broken state

### Use Cases Analysis

| Scenario | Units? | Ref Quantities? | Valid? | Behavior |
|----------|--------|----------------|--------|----------|
| **A**: Plain numbers | ❌ | ❌ | ✅ YES | Works correctly |
| **B**: Full units system | ✅ | ✅ | ✅ YES | Works correctly |
| **C**: Units, no ref quant | ✅ | ❌ | ❌ **NO** | Half-way zone (broken) |
| **D**: No units, has ref quant | ❌ | ✅ | ⚠️ Odd | Scales computed but unused |

**Legitimate use case for C?** **NO**
- If you want units → need scales for conditioning
- If you don't care about conditioning → don't use units
- Middle ground serves no purpose

### Proposed Solutions

#### Option 1: Strict Mode (Recommended)

**Enforce**: Units require reference quantities

```python
class MeshVariable:
    def __init__(self, ..., units=None):
        if units is not None:
            model = uw.get_default_model()
            if not model.has_units():
                raise ValueError(
                    f"Cannot create variable with units='{units}' "
                    f"without reference quantities.\n"
                    f"Either:\n"
                    f"  1. Set reference quantities first: "
                    f"model.set_reference_quantities(...)\n"
                    f"  2. Remove units parameter (use plain numbers)"
                )
```

**Pros**:
- Forces correct usage
- Prevents silent errors
- Clear error messages
- Eliminates half-way zone

**Cons**:
- Breaking change (some existing code may fail)
- Less flexible (but flexibility here is harmful)

#### Option 2: Soft Mode (Current + Enhancement)

**Keep warning but add evaluation checks**:

```python
def evaluate(expr, coords, ...):
    # Check for inconsistent state
    if result_dimensionality and not model.has_units():
        raise ValueError(
            f"Cannot evaluate expression with units when model has no "
            f"reference quantities.\n"
            f"Call model.set_reference_quantities() first."
        )
```

**Pros**:
- Backward compatible
- Catches errors at usage time
- Clear when something is wrong

**Cons**:
- Still allows creation of bad state
- Error happens later (less intuitive)
- Users may ignore warnings

#### Option 3: Auto-Complete Mode

**Automatically set default reference quantities**:

```python
# When user sets units without reference quantities
# Auto-generate sensible defaults based on units
if units and not model.has_units():
    model.set_reference_quantities(
        **_infer_defaults_from_units(units)
    )
```

**Pros**:
- "Just works" for simple cases
- No breaking changes

**Cons**:
- Magic behavior (confusing)
- Wrong defaults may be worse than no defaults
- Hides the underlying requirement

### Recommendation: Option 1 (Strict Mode)

**Rationale**:
1. **Clear contract**: Units → scales required
2. **Fail fast**: Catch errors at creation, not usage
3. **No silent errors**: System either works or fails loudly
4. **Educational**: Forces users to understand scaling

**Migration path**:
1. Add strict check in next version with clear error messages
2. Document in migration guide
3. Show examples of correct patterns
4. Provide helper: `model.set_standard_geodynamics_scales()`

---

## Immediate Actions Required

### 1. Fix Units Bug (High Priority)

**File**: `src/underworld3/function/expressions.py`

**Location**: `_unwrap_expressions()` lines 124-136

**Change**:
```python
if isinstance(fn, UWQuantity):
    if uw._is_scaling_active() and fn.has_units:
        nondim = uw.non_dimensionalise(fn)
        # ... return ND value
    # FIX: No scaling - convert to SI base units
    if hasattr(fn, '_pint_qty'):
        si_qty = fn._pint_qty.to_base_units()
        return sympy.sympify(si_qty.magnitude)
    elif hasattr(fn, 'value'):
        return sympy.sympify(fn.value)  # Fallback
```

**Impact**: Fixes both JIT and evaluate paths

### 2. Fix Evaluate Return Logic (High Priority)

**File**: `src/underworld3/function/functions_unit_system.py`

**Location**: Lines 154-179

**Change**: Always dimensionalize if `result_dimensionality` exists

```python
# Step 5: Handle dimensionalization
# If we have dimensionality info, ALWAYS wrap with units
if result_dimensionality is not None:
    dimensionalized_result = uw.dimensionalise(raw_values, result_dimensionality)
    if check_extrapolated:
        return dimensionalized_result, extrapolated
    else:
        return dimensionalized_result

# No dimensionality - return plain array
if check_extrapolated:
    return raw_values, extrapolated
else:
    return raw_values
```

### 3. Enforce Units-Scales Contract (Medium Priority)

**File**: `src/underworld3/discretisation/enhanced_variables.py`

**Add strict check** (after community discussion):
```python
# In EnhancedMeshVariable.__init__()
if units is not None:
    model = uw.get_default_model()
    if not model.has_units():
        raise ValueError(...)  # Strict mode
```

---

## Testing Strategy

### 1. Unit Tests for Bug Fix

```python
def test_constant_evaluation_units_conversion():
    """Constants should convert to SI regardless of scaling mode."""
    model.set_reference_quantities(plate_velocity=uw.quantity(5, "cm/year"))

    # Scaling OFF - should still convert units
    uw.use_nondimensional_scaling(False)
    q_km = uw.quantity(0.0001, "km/yr")
    q_cm = uw.quantity(0.0001, "cm/yr")

    result_km = uw.function.evaluate(q_km, coords)
    result_cm = uw.function.evaluate(q_cm, coords)

    # Different units should give different results
    ratio = result_km.max() / result_cm.max()
    assert abs(ratio - 1e5) < 1e3  # 100,000x ratio
```

### 2. Half-Way Zone Tests

```python
def test_units_require_reference_quantities():
    """Creating variables with units should require reference quantities."""
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(...)

    # Should raise if strict mode enabled
    with pytest.raises(ValueError, match="reference quantities"):
        v = uw.discretisation.MeshVariable("v", mesh, 1, units="m/s")
```

---

## Conclusion

1. **Flag system works correctly** - both unwrappers use same flag consistently
2. **Both have same bug** - fail to convert units when scaling OFF
3. **Merging plan still valid** - fix bug first, then merge
4. **Half-way zone is dangerous** - should be eliminated via strict mode

**Next Steps**:
1. Fix units bug (both paths simultaneously)
2. Add strict units-scales enforcement
3. Proceed with merging plan
4. Update documentation and migration guide
