# UWexpression Architecture Decision

**Date:** 2025-10-28
**Status:** RESOLVED - Option A Implemented Successfully
**Issue:** `Ra * T` multiplication failures, constant unwrapping problems
**Solution:** Arithmetic method overrides to delegate to Symbol

## Problem Statement

### The Core Issue

UWexpression has conflicting roles due to its inheritance structure:

```python
class UWexpression(MathematicalMixin, UWQuantity, uw_object, Symbol):
```

**Role Conflict:**
1. **IS a SymPy Symbol** → Should use Symbol's arithmetic naturally
2. **HAS MathematicalMixin** → Tries to wrap/intercept arithmetic operations

**Result:** When multiplying `Ra * T`:
- `UWexpression.__mul__(T)` returns `NotImplemented` (Symbol can't multiply Matrix)
- Python delegates to `T.__rmul__(Ra)`
- `MathematicalMixin.__rmul__` sees `Ra` is also a MathematicalMixin
- Refuses to sympify it (to preserve "lazy evaluation")
- Tries to multiply UWexpression × Matrix directly → **TypeError**

### Why Mixins Failed Here

Mixins assume **orthogonal concerns** - features that can be independently combined. But:

1. **Arithmetic operations** (MathematicalMixin)
2. **Units/quantities** (UWQuantity)
3. **SymPy integration** (Symbol inheritance)

These are **NOT orthogonal**:
- Arithmetic DEPENDS on whether you're a Symbol
- SymPy integration CONFLICTS with custom arithmetic wrappers

**The mistake:** Applying MathematicalMixin to Symbol-based classes creates hidden conflicts.

## Three Architectural Options Considered

### Option 1: Pure Composition (No Multiple Inheritance)

**Approach:** Stop using mixins entirely, use composition.

```python
class UWExpression:
    """Wrapper around a Symbol with units"""
    def __init__(self, name, value, units=None):
        self._symbol = sympy.Symbol(name)
        self._value = value
        self._units = units
```

**Pros:**
- No inheritance conflicts
- Clear ownership

**Cons:**
- More boilerplate
- Need to implement arithmetic on each class

### Option 2: Layered Architecture (Separate Symbolic from Computational)

**Approach:** Clear split - symbols are symbols, variables hold data.

```python
# SYMBOLIC LAYER - no data
class UWExpression:
    """Named expression (e.g., Ra = alpha/kappa)"""
    def __init__(self, name, expr, units=None):
        self.name = name
        self._expr = expr  # SymPy expression
        self._units = units

# COMPUTATIONAL LAYER - has data
class MeshVariable:
    """Variable with PETSc data - NOT a Symbol"""
    def __init__(self, name, mesh, units=None):
        self._data = NDArray_With_Callback(...)
        self._sym = UWFunction(name, *mesh.coordinates)
```

**Pros:**
- Very clear conceptual model
- No role confusion
- Natural SymPy integration

**Cons:**
- Requires significant refactoring
- Each class implements arithmetic explicitly

### Option 3: Hybrid - Remove MathematicalMixin from UWexpression Only

**Approach:** Keep mixins for MeshVariable, remove from UWexpression.

```python
class UWexpression(UWQuantity, uw_object, Symbol):
    """Symbol with units - NO MathematicalMixin"""
    # Symbol provides arithmetic naturally
    # Just need _sympify_() for SymPy integration
```

**Pros:**
- Minimal change
- Fixes immediate conflict
- Preserves existing MeshVariable behavior

**Cons:**
- Still relies on mixin interactions elsewhere
- Not addressing all architectural issues

### Option A: Arithmetic Method Overrides (IMPLEMENTED ✅)

**Approach:** Keep UWexpression as Symbol, but explicitly delegate arithmetic to Symbol's implementations.

**Root Cause Analysis:**
The issue wasn't just MathematicalMixin - it was the Method Resolution Order (MRO):
```python
class UWexpression(UWQuantity, uw_object, Symbol):
    # MRO: UWexpression → UWQuantity → uw_object → Symbol → ...
```

**Problem:** `UWQuantity.__mul__` was being called before `Symbol.__mul__`, intercepting operations and preventing Symbol's natural arithmetic.

**Solution:** Override arithmetic methods to explicitly delegate to Symbol:
```python
def __mul__(self, other):
    """Multiply - delegate to Symbol to preserve symbolic expressions."""
    return Symbol.__mul__(self, other)

def __rmul__(self, other):
    """Right multiply - delegate to Symbol."""
    return Symbol.__rmul__(self, other)

# Similar for __truediv__, __rtruediv__, __add__, __radd__,
# __sub__, __rsub__, __pow__, __rpow__, __neg__
```

**Why This Works:**
1. **Bypasses MRO**: Explicitly calls Symbol's implementation, skipping UWQuantity
2. **Preserves deferred evaluation**: `_sympify_()` returns `self` (the Symbol itself)
3. **Symbol arithmetic is natural**: Symbol × Symbol → Mul(Symbol, Symbol) preserves both
4. **Minimal changes**: No architectural redesign needed

**Pros:**
- ✓ Fixes all issues (Ra * T, Ra * v, constant preservation)
- ✓ Preserves deferred evaluation pattern
- ✓ Minimal code changes
- ✓ No breaking changes to existing code
- ✓ Units tracking still works

**Cons:**
- Requires explicit override for each arithmetic operation
- Slightly more boilerplate than relying on MRO

## Decision (FINAL - 2025-10-28)

**OPTION A IMPLEMENTED AND VALIDATED ✅**

**Initial Investigation** (Option 3):
- Removed MathematicalMixin from UWexpression inheritance
- Changed `_sympify_()` to return `self` instead of `self._sym`
- Result: Fixed Ra * T, but constant preservation still failed

**Root Cause Discovery:**
- UWQuantity (parent class) has `__mul__` method that intercepts operations
- MRO puts UWQuantity before Symbol, so UWQuantity.__mul__ is called first
- UWQuantity.__mul__ doesn't preserve symbolic nature correctly

**Final Solution** (Option A):
- Added explicit arithmetic method overrides in UWexpression
- Each method delegates directly to Symbol's implementation
- Result: ALL tests pass - Ra * T works, constants preserved, deferred evaluation maintained

## Key Insights

### What "Lazy Evaluation" Actually Means

Confusion arose around "preserving lazy evaluation." Clarification:

- **Lazy = Don't access `.data` or `.array`** ✓
- **Lazy = Keep as SymPy symbols until JIT** ✓
- **Lazy ≠ Avoid creating SymPy expressions** (this is fine!)

Calling `_sympify_()` preserves laziness - it converts to SymPy symbols without accessing numeric data.

### MathematicalMixin Purpose

**Appropriate for:** MeshVariable, SwarmVariable
- These are **not** SymPy objects
- Need conversion via `_sympify_()` to participate in symbolic math
- MathematicalMixin provides this bridge

**Inappropriate for:** UWexpression
- Already **is** a SymPy Symbol
- Doesn't need arithmetic wrappers
- Symbol's native arithmetic is sufficient

### The Real Design Principle

**"Don't mix Symbol-based classes with MathematicalMixin-based classes"**

- Symbol subclasses should use Symbol's arithmetic directly
- Non-Symbol classes use MathematicalMixin to bridge to SymPy
- Never both in the same inheritance tree

## Testing Criteria

Solution succeeds if:

1. ✓ `Ra * T` works (scalar × variable)
2. ✓ `Ra * v` works (scalar × vector)
3. ✓ Constants preserved (`alpha * kappa` → `\alpha*\kappa`)
4. ✓ Lazy evaluation maintained (no `.data` access)
5. ✓ Units/dimensionality tracking works
6. ✓ Existing code doesn't break

## Option A Test Results (2025-10-28)

### ALL TESTS PASSING ✅

**Test 1: Constant Preservation**
```python
alpha = uw.expression(r"\alpha", uw.quantity(3e-5, "1/K"))
kappa = uw.expression(r"\kappa", uw.quantity(1e-6, "m^2/s"))
result = kappa * alpha

# Result: \alpha*\kappa ✓ (both symbols preserved!)
# Type: sympy.core.mul.Mul ✓
# Atoms: {\alpha, \kappa} ✓
```

**Test 2: Deferred Evaluation**
```python
sympified = alpha._sympify_()
# sympified is alpha: True ✓ (returns self, not numeric value)
```

**Test 3: Nested Constants in Compound Expressions**
```python
Ra = uw.expression("Ra", alpha / kappa)
# Ra.sym: \alpha/\kappa ✓
# Atoms in Ra.sym: {\alpha, \kappa} ✓
```

**Test 4: Ra * T Multiplication**
```python
mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2, units='K')
Ra = uw.expression("Ra", uw.quantity(1e6, "dimensionless"))

result = Ra * T
# Result type: sympy.matrices.dense.MutableDenseMatrix ✓
# Result shape: (1, 1) ✓
```

**Test 5: Ra * v Multiplication (Vector)**
```python
v = uw.discretisation.MeshVariable('v', mesh, mesh.dim, degree=2, units='m/s')
result = Ra * v
# Result type: sympy.matrices.dense.MutableDenseMatrix ✓
# Result shape: (1, 2) ✓
```

### Why Option A Succeeded

**The Key Insight:**
The issue wasn't that UWexpression's design was fundamentally broken - it was that **UWQuantity's `__mul__` method was intercepting operations before Symbol's arithmetic could run**.

**How Delegation Fixes It:**
1. **Bypasses interception**: Explicitly calling `Symbol.__mul__(self, other)` skips UWQuantity
2. **Preserves Symbol identity**: UWexpression IS still a Symbol, with all Symbol properties
3. **Natural SymPy behavior**: Symbol × Symbol → Mul(Symbol, Symbol) preserves both symbols
4. **Deferred evaluation works**: `_sympify_()` returns `self`, keeping the Symbol in the expression tree

**What We Learned:**
- Method Resolution Order (MRO) matters deeply for multiple inheritance
- Sometimes the solution is surgical delegation, not architectural redesign
- Python's operator dispatch can be complex when mixing inheritance hierarchies

## Implementation Summary

**Files Modified:**

1. **`src/underworld3/function/expressions.py`** (lines 410-480):
   - Modified `_sympify_()` to return `self` instead of `self._sym` (line 416)
   - Added arithmetic method overrides delegating to Symbol (lines 438-480):
     - `__mul__`, `__rmul__`
     - `__truediv__`, `__rtruediv__`
     - `__add__`, `__radd__`
     - `__sub__`, `__rsub__`
     - `__pow__`, `__rpow__`
     - `__neg__`

**Key Changes:**

```python
# CRITICAL: _sympify_() must return self for deferred evaluation
def _sympify_(self):
    """Return the Symbol itself for deferred evaluation."""
    return self  # NOT self._sym!

# Arithmetic delegation pattern
def __mul__(self, other):
    """Multiply - delegate to Symbol to preserve symbolic expressions."""
    return Symbol.__mul__(self, other)

def __rmul__(self, other):
    """Right multiply - delegate to Symbol."""
    return Symbol.__rmul__(self, other)
# ... similar for other operations
```

**Why Each Change Matters:**
1. **`_sympify_() returns self`**: Keeps UWexpression as Symbol in SymPy expression trees
2. **Explicit delegation**: Bypasses UWQuantity's `__mul__` in the MRO
3. **All operations covered**: Ensures consistent behavior across all arithmetic

## References

- **Context:** Ra * T multiplication bug, constant unwrapping issues
- **Discussion Date:** 2025-10-28
- **Related Files:**
  - `src/underworld3/function/expressions.py`
  - `src/underworld3/utilities/mathematical_mixin.py`
  - Tests in `test_compound_expr.py`, `test_constants_preserved.py`
