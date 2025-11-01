# Enhanced Units Principles for CLAUDE.md

## INSERT THIS SECTION - Enhancing "Units Everywhere or Nowhere Principle"

**Location in CLAUDE.md**: Replace/enhance section starting at line 307

---

## 🎯 Core UW3 Principles: Units, Dimensionality, and Code Standards

### Principle 1: Units Everywhere or Nowhere

When working with physical problems in Underworld3, dimensional consistency is critical. The system enforces a simple but powerful principle:

**Units Everywhere or Nowhere**: Either all quantities in a model have explicit physical units (when reference quantities are set), OR all quantities are plain numbers (when reference quantities are not set). No mixing.

### Principle 1a: Dimensionality Always Present (NEW - 2025-10-29)

**CRITICAL**: When units are enabled (`model.has_units() == True`), ALL quantities must have **both**:
1. **Units** (even if dimensionless, e.g., `uw.quantity(1.0, "dimensionless")`)
2. **Dimensionality metadata** (the `[length]`, `[time]`, `[mass]`, etc. tags)

**Why Dimensionality Matters**:
```python
# WITHOUT dimensionality (WRONG)
dt = 0.001  # Just a number - can't convert between unit systems

# WITH dimensionality (CORRECT)
dt = UWQuantity(0.001, dimensionless=True, dimensionality='[time]')
# Can now convert: model.from_model_magnitude(dt, '[time]') works!
```

**Dimensionless ≠ Dimension-free**:
- ❌ WRONG: "Dimensionless means no dimensionality"
- ✓ CORRECT: "Dimensionless means dimensionality = 1" (but still carries `[time]`, `[length]`, etc. metadata)

### Principle 1b: Reduction Operations Must Preserve Dimensionality (NEW - 2025-10-29)

**CRITICAL VIOLATION - Currently Unfixed**:

When `model.has_units() == True`, **ALL** reduction operations (`.max()`, `.min()`, `.mean()`, etc.) must return `UWQuantity` objects with correct dimensionality.

**Current Violations**:
1. **`v.max()`** returns bare `float` or `tuple` ✗
   - Should return: `UWQuantity` with `[length]/[time]` dimensionality
2. **`stokes.estimate_dt()`** returns bare `float` ✗
   - Should return: `UWQuantity` with `[time]` dimensionality

**Why This Violates Architecture**:
```python
# Current broken behavior
v_max = v.max()  # Returns 1.23e-4 (bare float)
model.to_model_magnitude(v_max)  # FAILS - no dimensionality!

# Correct behavior (needs implementation)
v_max = v.max()  # Returns UWQuantity(1.23e-4, 'm/s', dimensionality='[length]/[time]')
v_max_physical = model.from_model_magnitude(v_max, '[length]/[time]')
print(v_max_physical.to('cm/year'))  # Works!
```

**Files Requiring Fixes**:
- `src/underworld3/discretisation/discretisation_mesh_variables.py`: `.max()`, `.min()`, `.mean()`, `.std()`
- `src/underworld3/systems/solvers.py`: `Stokes.estimate_dt()`

See detailed fix specifications at end of this document.

### Principle 2: Tutorial Notebooks Must Not Use Workarounds (NEW - 2025-10-29)

**CRITICAL**: Tutorial and example notebooks are **instructional material**. They must demonstrate correct, idiomatic usage patterns.

**Forbidden in Tutorials**:
1. ✗ Workarounds for core system bugs
2. ✗ Deprecated API patterns
3. ✗ Manual wrapping of quantities that should already have units
4. ✗ Comments explaining workarounds instead of fixing core issues

**Why This Matters**:
- Notebooks teach users how to write code
- Workarounds in tutorials become "best practices" users copy
- Hides architectural violations from developers
- Creates technical debt disguised as documentation

**Example Violation** (from Notebook 14):
```python
# BAD - Workaround in tutorial notebook
dt = uw.quantity(stokes.estimate_dt(), "dimensionless")  # Manual wrapping
v_max_nd = max(v.max())  # Assuming it's dimensionless

# CORRECT - Fix the core code, then use naturally
dt = stokes.estimate_dt()  # Already returns UWQuantity with [time] dimensionality
v_max = v.max()  # Already returns UWQuantity with [length]/[time] dimensionality
```

**When Notebooks Need Workarounds**:
1. **DO NOT** add the workaround to the notebook
2. **CREATE** a bug report/issue documenting the core violation
3. **FIX** the core code first
4. **UPDATE** the notebook to use correct pattern
5. **DOCUMENT** the fix in CLAUDE.md

### When These Principles Apply

#### Mode 1: Reference Quantities ARE Set (Units + Dimensionality Everywhere)
When you call `model.set_reference_quantities()`:
- **All inputs** require units: source terms, boundary conditions, parameters
- **All outputs** must have units: reduction operations, solver results, statistics
- **All quantities** must have dimensionality metadata, even if dimensionless
- Use: `solver.f = uw.quantity(2.0, 'K')` ✓
- Don't use: `solver.f = 2.0` ✗

**Check**: `model.has_units()` returns `True`

#### Mode 2: Reference Quantities NOT Set (Plain Numbers Everywhere)
When you DON'T set reference quantities:
- Use plain Python numbers everywhere
- No units or dimensionality required
- User responsible for dimensional consistency
- Use: `solver.f = 2.0` ✓

**Check**: `model.has_units()` returns `False`

---

## ⚠️ KNOWN VIOLATIONS - MUST FIX

### 1. MeshVariable Reduction Operations (discretisation_mesh_variables.py)

**Violating Methods**:
- `.max()` - Returns bare number, should return UWQuantity
- `.min()` - Returns bare number, should return UWQuantity
- `.mean()` - Returns bare number, should return UWQuantity
- `.std()` - Returns bare number, should return UWQuantity

**Location**: `src/underworld3/discretisation/discretisation_mesh_variables.py`

**Fix Pattern**:
```python
def max(self):
    """Return maximum value(s) with units/dimensionality preserved."""
    import underworld3 as uw
    
    # Get raw value from PETSc
    max_vals = self._vec.max()[1]
    
    # Check if units mode
    model = uw.get_default_model()
    if not model.has_units() or self.units is None:
        return max_vals  # Backward compatible - no units mode
    
    # Wrap with units + dimensionality
    if isinstance(max_vals, tuple):
        return tuple(uw.quantity(val, self.units) for val in max_vals)
    else:
        return uw.quantity(max_vals, self.units)
```

### 2. Stokes.estimate_dt() (solvers.py)

**Current**: Returns bare `float`
**Required**: Return `UWQuantity` with `[time]` dimensionality

**Location**: `src/underworld3/systems/solvers.py` (Stokes class)

**Fix Pattern**:
```python
def estimate_dt(self):
    """Estimate timestep with [time] dimensionality."""
    import underworld3 as uw
    
    v_max = self.velocityField.max()  # Now returns UWQuantity if units enabled
    if isinstance(v_max, tuple):
        v_max_val = max(v_max)
    else:
        v_max_val = v_max
    
    h_min = self.mesh.get_min_element_size()
    
    if hasattr(v_max_val, 'magnitude'):
        # Units mode - compute and wrap with [time] dimensionality
        v_mag = v_max_val.magnitude
        h_mag = h_min.magnitude if hasattr(h_min, 'magnitude') else h_min
        dt_val = 0.5 * h_mag / v_mag
        
        model = uw.get_default_model()
        return model.from_model_magnitude(dt_val, '[time]')
    else:
        # No units mode - backward compatible
        return 0.5 * h_min / v_max_val
```

### 3. Mesh.get_min_element_size() (discretisation_mesh.py)

**Current**: Likely returns bare `float`
**Required**: Return `UWQuantity` with `[length]` dimensionality when units enabled

**Location**: `src/underworld3/discretisation/discretisation_mesh.py`

---

## 📋 Action Items

### Immediate:
- [ ] Document violations in issue tracker
- [ ] Mark Notebook 14 workarounds with `# WORKAROUND - TODO: Fix core code`
- [ ] Create test cases that verify dimensionality preservation

### Short-term:
- [ ] Fix `MeshVariable.max/min/mean/std()` to return UWQuantity
- [ ] Fix `Stokes.estimate_dt()` to return UWQuantity with `[time]`
- [ ] Fix `Mesh.get_min_element_size()` to return UWQuantity with `[length]`
- [ ] Remove workarounds from tutorial notebooks

### Long-term:
- [ ] Audit ALL reduction operations for dimensionality compliance
- [ ] Create linter rule: "Reduction operations must preserve dimensionality"
- [ ] Add to test suite: Verify all solver methods return dimensioned results
- [ ] Documentation: "Dimensionality Preservation Contract for API Design"

---

## 🔍 How to Identify Violations

**Checklist when reviewing code/notebooks**:
1. Does `model.has_units()` return `True`?
2. Do ALL reduction operations return `UWQuantity`?
3. Can you call `model.to_model_magnitude()` on ALL numeric results?
4. Are there manual wrapping patterns like `uw.quantity(result, "units")`?
5. Are there comments explaining workarounds?

If any of 4-5 are true → **Core system violation**, not a documentation issue.

---

## 📚 Related Documentation

- **Current "Units Everywhere" section**: Keep existing content (lines 315-470)
- **Dimensional Analysis**: `docs/developer/ARCHITECTURAL-RESOLUTION.md`
- **Coordinate Units**: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
- **Tutorial Standards**: Add new section to developer docs

---

**Status**: Documented 2025-10-29 after discovering violations in Notebook 14
**Priority**: HIGH - Affects user-facing API and tutorial quality
**Complexity**: MEDIUM - Pattern is clear, implementation straightforward

