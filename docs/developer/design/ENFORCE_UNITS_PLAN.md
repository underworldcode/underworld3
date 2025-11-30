# Plan: Enforce Units Require Reference Quantities

**Date**: 2025-11-15
**Objective**: Eliminate the "half-way zone" where variables have units but no reference quantities

---

## Current Problem

**Half-Way Zone**: Variables can be created with units but no reference quantities
```python
# Currently ALLOWED (with warning)
mesh = uw.meshing.StructuredQuadBox(...)
v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")  # ← No reference quantities!

# Results in:
# - scaling_coefficient = 1.0 (wrong conditioning)
# - Silent errors in evaluate() and operations
# - Inconsistent state
```

**Why This Is Bad**:
1. Units become meaningless labels (no dimensional analysis)
2. Scaling doesn't work (conditioning problems)
3. Silent errors propagate through calculations
4. Users don't understand requirement until something breaks

---

## Proposed Solution: Phased Strict Mode

### Phase 1: Opt-In Strict Mode (Immediate)

**Add global flag with default OFF** (backward compatible):

```python
# In src/underworld3/__init__.py
_STRICT_UNITS_MODE = False  # Default: allow half-way zone (current behavior)

def use_strict_units(enabled=True):
    """
    Enable or disable strict units enforcement.

    When enabled, variables with units REQUIRE reference quantities to be set.
    When disabled (default), variables with units are allowed without reference
    quantities but will get scaling_coefficient=1.0 and a warning.

    Parameters
    ----------
    enabled : bool, default=True
        True to enforce strict units (recommended for production)
        False to allow units without reference quantities (legacy)

    Examples
    --------
    Enable strict mode at start of script:

    >>> import underworld3 as uw
    >>> uw.use_strict_units(True)  # Enforce units-scales contract
    >>>
    >>> model = uw.get_default_model()
    >>> # This will raise if no reference quantities:
    >>> v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

    See Also
    --------
    is_strict_units_active : Check current strict mode
    """
    global _STRICT_UNITS_MODE
    _STRICT_UNITS_MODE = bool(enabled)

def is_strict_units_active():
    """Check if strict units enforcement is enabled."""
    return _STRICT_UNITS_MODE
```

**Check in variable creation**:

```python
# In src/underworld3/discretisation/enhanced_variables.py
# In EnhancedMeshVariable.__init__()

if units is not None:
    model = uw.get_default_model()

    # Check if strict mode is enabled
    if uw.is_strict_units_active() and not model.has_units():
        raise ValueError(
            f"Strict units mode: Cannot create variable with units='{units}' "
            f"when model has no reference quantities.\n\n"
            f"Options:\n"
            f"  1. Set reference quantities FIRST:\n"
            f"     model = uw.get_default_model()\n"
            f"     model.set_reference_quantities(\n"
            f"         domain_depth=uw.quantity(1000, 'km'),\n"
            f"         plate_velocity=uw.quantity(5, 'cm/year')\n"
            f"     )\n\n"
            f"  2. Remove units parameter (use plain numbers):\n"
            f"     uw.discretisation.MeshVariable('{name}', mesh, ...)\n\n"
            f"  3. Disable strict mode (not recommended):\n"
            f"     uw.use_strict_units(False)\n"
        )

    # If not strict mode and no reference quantities, warn as before
    if not model.has_units():
        warnings.warn(
            f"Variable '{name}' has units '{units}' but no reference quantities are set.\n"
            f"Call model.set_reference_quantities() before creating variables with units.\n"
            f"Variable will use scaling_coefficient=1.0, which may lead to poor numerical conditioning.\n"
            f"Consider enabling strict mode: uw.use_strict_units(True)",
            UserWarning
        )
```

**Export from main module**:
```python
# In src/underworld3/__init__.py (exports section)
from . import (
    ...,
    use_strict_units,
    is_strict_units_active,
)
```

---

### Phase 2: Enable By Default (Future Version)

**Change default to strict mode**:
```python
# In __init__.py
_STRICT_UNITS_MODE = True  # ← Default changes to strict
```

**Add deprecation notice** for old behavior:
```python
def use_strict_units(enabled=True):
    """..."""
    if not enabled:
        warnings.warn(
            "Disabling strict units mode is deprecated and will be removed in v1.0. "
            "The half-way zone (units without reference quantities) is not supported.",
            DeprecationWarning,
            stacklevel=2
        )
    global _STRICT_UNITS_MODE
    _STRICT_UNITS_MODE = bool(enabled)
```

**Migration period**: 1-2 releases

---

### Phase 3: Remove Opt-Out (v1.0)

**Remove the flag entirely** - always strict:
```python
# Remove use_strict_units() entirely
# Always enforce units require reference quantities
```

---

## Implementation Details

### 1. Where To Check

**Primary Location**: `src/underworld3/discretisation/enhanced_variables.py`
- `EnhancedMeshVariable.__init__()` - Check at variable creation

**Secondary Locations** (optional - could enforce at other entry points):
- `Model.set_reference_quantities()` - Could validate existing variables
- `evaluate()` - Could check before evaluating constants with units
- Assignment operations - Could check when assigning UWQuantity to variable

**Recommendation**: Start with primary location only (variable creation). This is:
- Simplest to implement
- Clearest to users (fail fast)
- Least intrusive

### 2. Error Message Design

**Bad Error** (vague):
```
ValueError: Variable with units requires reference quantities
```

**Good Error** (actionable):
```
ValueError: Strict units mode: Cannot create variable with units='m/s' when model has no reference quantities.

Options:
  1. Set reference quantities FIRST:
     model = uw.get_default_model()
     model.set_reference_quantities(
         domain_depth=uw.quantity(1000, 'km'),
         plate_velocity=uw.quantity(5, 'cm/year')
     )

  2. Remove units parameter (use plain numbers):
     uw.discretisation.MeshVariable('v', mesh, 2)  # No units

  3. Disable strict mode (not recommended):
     uw.use_strict_units(False)
```

**Key Elements**:
- **What's wrong**: Clear statement of the problem
- **Why it matters**: Brief explanation or link to docs
- **How to fix**: Multiple concrete solutions with code examples
- **Escape hatch**: Opt-out if really needed

### 3. Testing Strategy

**Test strict mode ON**:
```python
def test_strict_units_requires_reference_quantities():
    """Strict mode: variables with units require reference quantities."""
    uw.reset_default_model()
    uw.use_strict_units(True)

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Should raise - no reference quantities
    with pytest.raises(ValueError, match="Strict units mode.*reference quantities"):
        v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

    # Should work - set reference quantities first
    model = uw.get_default_model()
    model.set_reference_quantities(plate_velocity=uw.quantity(5, "cm/year"))

    v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")  # ✓ OK
    assert v.scaling_coefficient != 1.0  # Proper scaling
```

**Test strict mode OFF** (backward compatibility):
```python
def test_non_strict_allows_units_without_reference():
    """Non-strict mode: variables with units allowed (with warning)."""
    uw.reset_default_model()
    uw.use_strict_units(False)

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

    # Should work but warn
    with pytest.warns(UserWarning, match="no reference quantities"):
        v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

    assert v.scaling_coefficient == 1.0  # Identity scaling (half-way zone)
```

**Test default behavior**:
```python
def test_strict_units_default_state():
    """Default is non-strict for backward compatibility (Phase 1)."""
    # In Phase 1: default is False
    assert uw.is_strict_units_active() == False

    # In Phase 2+: default would be True
    # assert uw.is_strict_units_active() == True
```

---

## Migration Guide

### For Users

**Old Code** (may have half-way zone):
```python
import underworld3 as uw

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, units="K")  # ← Half-way zone!
```

**New Code** (fix half-way zone):
```python
import underworld3 as uw

# Option 1: Add reference quantities (recommended)
model = uw.get_default_model()
model.set_reference_quantities(
    temperature_diff=uw.quantity(1000, "K"),
    domain_depth=uw.quantity(1000, "km")
)

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, units="K")  # ✓ OK

# Option 2: Remove units (if you don't need them)
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1)  # No units, plain numbers
```

**Enable Strict Mode** (recommended for new projects):
```python
import underworld3 as uw

# Enable at start of script
uw.use_strict_units(True)  # Enforce contract

# Rest of code...
```

---

## Documentation Changes

### 1. Add to User Guide

**New Section**: "Units and Reference Quantities"

```markdown
## Units and Reference Quantities

### The Units-Scales Contract

Underworld3 enforces a simple principle: **variables with units require reference quantities**.

This ensures:
- Proper dimensional analysis
- Correct non-dimensional scaling
- Predictable numerical conditioning

### Strict Mode (Recommended)

Enable strict mode to catch errors early:

```python
import underworld3 as uw

# Enable at start of script
uw.use_strict_units(True)
```

With strict mode:
- ✓ Variables with units require reference quantities
- ✓ Clear error messages with fix instructions
- ✓ Prevents silent errors

### Setting Reference Quantities

```python
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    temperature_diff=uw.quantity(1000, "K"),
    plate_velocity=uw.quantity(5, "cm/year")
)
```

Now you can create variables with units:
```python
T = uw.discretisation.MeshVariable("T", mesh, 1, units="K")
```
```

### 2. Update API Documentation

**Add to `MeshVariable` docstring**:
```python
def __init__(self, name, mesh, num_components, degree=1, units=None, ...):
    """
    ...

    Parameters
    ----------
    ...
    units : str, optional
        Physical units for the variable (e.g., "m/s", "Pa", "K").
        **Requires reference quantities**: If units are specified, you must
        call `model.set_reference_quantities()` BEFORE creating the variable
        (enforced in strict mode via `uw.use_strict_units(True)`).

    Raises
    ------
    ValueError
        If units are specified but no reference quantities are set (strict mode only)

    Examples
    --------
    >>> # Set reference quantities first
    >>> model = uw.get_default_model()
    >>> model.set_reference_quantities(plate_velocity=uw.quantity(5, "cm/year"))
    >>>
    >>> # Now create variable with units
    >>> v = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")
    """
```

---

## Rollout Plan

### Week 1: Implementation
- [ ] Add `use_strict_units()` and `is_strict_units_active()` to `__init__.py`
- [ ] Add check in `EnhancedMeshVariable.__init__()`
- [ ] Write comprehensive tests
- [ ] Update error messages

### Week 2: Testing
- [ ] Run full test suite with strict mode ON
- [ ] Identify and fix any broken tests
- [ ] Add migration examples
- [ ] Test backward compatibility (strict mode OFF)

### Week 3: Documentation
- [ ] Add user guide section
- [ ] Update API documentation
- [ ] Create migration guide
- [ ] Add examples to tutorials

### Week 4: Review & Release
- [ ] Code review
- [ ] Community feedback
- [ ] Release notes
- [ ] Announce in changelog

---

## Alternative: Immediate Strict Mode

**If we want to be more aggressive** (skip phased approach):

```python
# Just enforce immediately, no flag:
if units is not None and not model.has_units():
    raise ValueError(...)  # Always strict, no opt-out
```

**Pros**:
- Simpler implementation (no flag to manage)
- Clearer contract (no confusion)
- Forces correct usage

**Cons**:
- Breaking change (will break existing code)
- No migration period
- May frustrate users with legacy code

**Recommendation**: Use phased approach for better adoption

---

## Edge Cases

### 1. Creating Variables Before Setting Reference Quantities

**Problem**: User might want to create variables first, set references later

**Solution**: Allow this pattern explicitly:
```python
# Create variables without units first
T = uw.discretisation.MeshVariable("T", mesh, 1)  # No units yet

# Set reference quantities
model.set_reference_quantities(...)

# Set units on existing variable? (currently not supported)
# T.units = "K"  # Not implemented - would need property setter
```

**Recommendation**: Document that reference quantities must come FIRST, or variables must be created without units

### 2. Multiple Models

**Problem**: Different models may have different reference quantities

**Solution**: Each model tracks its own units state:
```python
model1 = uw.Model()
model1.set_reference_quantities(...)

model2 = uw.Model()  # No reference quantities

# Variables check their model:
T1 = uw.discretisation.MeshVariable("T", mesh1, 1, units="K")  # Uses model1 ✓
T2 = uw.discretisation.MeshVariable("T", mesh2, 1, units="K")  # Uses model2 ✗
```

This already works with `model.has_units()` check

### 3. Swarm Variables

**Problem**: SwarmVariable also supports units

**Solution**: Apply same check in `SwarmVariable.__init__()`
```python
# In swarm.py, SwarmVariable.__init__()
if units is not None:
    model = uw.get_default_model()
    if uw.is_strict_units_active() and not model.has_units():
        raise ValueError(...)  # Same check as MeshVariable
```

---

## Success Criteria

1. **No half-way zone**: Cannot create variables with units unless reference quantities set
2. **Clear errors**: Error messages tell users exactly how to fix
3. **Backward compatible**: Phased rollout doesn't break existing code immediately
4. **Well documented**: Users understand the contract and how to comply
5. **Tests pass**: Full test suite works in both strict and non-strict modes

---

## Timeline

**Phase 1** (Immediate - this PR):
- Implement opt-in strict mode
- Default: OFF (backward compatible)
- Release in next minor version (e.g., v0.99.1)

**Phase 2** (6-12 months):
- Change default: ON
- Add deprecation warning for opt-out
- Release in next major version (e.g., v1.0.0)

**Phase 3** (12-18 months):
- Remove opt-out entirely
- Always strict
- Release in next major version (e.g., v1.1.0)

---

## Implementation Checklist

- [x] Add global flag `_STRICT_UNITS_MODE` - **DONE** (2025-11-15)
- [x] Implement `use_strict_units(enabled)` - **DONE** (2025-11-15)
- [x] Implement `is_strict_units_active()` - **DONE** (2025-11-15)
- [x] Add check in `EnhancedMeshVariable.__init__()` - **DONE** (2025-11-15)
- [x] Add check in `SwarmVariable.__init__()` - **DONE** (2025-11-15)
- [x] Write error message template - **DONE** (2025-11-15)
- [x] Add tests for strict mode - **DONE** (2025-11-15) - 9 comprehensive tests
- [x] Add tests for non-strict mode - **DONE** (2025-11-15) - included in test suite
- [x] Reset strict mode in `reset_default_model()` - **DONE** (2025-11-15)
- [ ] Update user guide
- [ ] Update API documentation
- [ ] Create migration examples
- [ ] Add to CHANGELOG.md
- [ ] Code review
- [ ] Merge and release

## Implementation Summary (Strict Mode ON by Default)

**Status**: ✅ FULLY IMPLEMENTED (2025-11-15)
**Decision**: Skipped phased rollout - strict mode ON by default from day one

**Files Modified**:

1. **`src/underworld3/__init__.py` (lines 375-444)**:
   - Added global flag `_STRICT_UNITS_MODE = True` (DEFAULT: ON)
   - Implemented `use_strict_units(enabled=True)` function
   - Implemented `is_strict_units_active()` function
   - Comprehensive docstrings emphasizing strict mode is default

2. **`src/underworld3/discretisation/enhanced_variables.py` (lines 144-177)**:
   - Added enforcement check in `EnhancedMeshVariable.__init__()`
   - Clear, actionable error messages with 3 solution options
   - Checks strict mode AND model.has_units() status
   - Properly handles variable name formatting

3. **`src/underworld3/swarm.py` (lines 127-158)**:
   - Added identical enforcement check in `SwarmVariable.__init__()`
   - Same error messages as MeshVariable for consistency
   - Keeps swarm and mesh variables in sync (critical for proxy variables)

4. **`src/underworld3/model.py` (lines 4521-4547)**:
   - Modified `reset_default_model()` to reset strict mode to default (ON)
   - Ensures test isolation and proper global state management
   - Documented in docstring

5. **`tests/test_0814_strict_units_enforcement.py` (NEW)**:
   - 9 comprehensive tests covering all scenarios
   - Tests default state (ON - strict from the start)
   - Tests strict mode enforcement
   - Tests non-strict mode (expert/debugging use)
   - Tests toggle functionality
   - Tests error message quality
   - Tests variables without units (always allowed)
   - Tests with reference quantities (works in both modes)
   - Tests multiple variables
   - Tests backward compatibility path

**Test Results**: ✅ 9/9 PASSING

**Key Design Decisions**:
1. **Default ON**: Strict mode enforced by default - no production users to break yet
2. **Opt-out for experts**: Can disable with `uw.use_strict_units(False)` for debugging
3. **Clear errors**: Error messages provide 3 concrete solutions with code examples
4. **Reference quantities MUST be set BEFORE mesh**: Enforces correct workflow
5. **Strict mode applies globally**: All variables (mesh + swarm) checked consistently
6. **Reset restores default**: `reset_default_model()` resets strict mode to ON

**Rationale for Default ON**:
- Units system hasn't been rolled out to production yet
- No external users or deprecated examples in the wild
- Perfect time to enforce best practices from day one
- Prevents users from developing bad habits
- Makes the "correct" path the easiest path
- Still allows opt-out for edge cases

**Validation**:
- All 9 strict units tests passing ✅
- Original bug fix test passing ✅ (backward compatibility confirmed)
- Units system tests still passing ✅
- SwarmVariable and MeshVariable enforcement in sync ✅

**Next Steps** (Future PRs):
1. Update user guide with strict mode as default behavior
2. Add examples showing correct workflow (reference quantities first)
3. Add to CHANGELOG.md
4. Code review and merge
