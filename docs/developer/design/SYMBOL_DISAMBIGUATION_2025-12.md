# Symbol Disambiguation in Underworld3

**Date:** 2025-12-15
**Status:** IMPLEMENTED
**Replaces:** Invisible whitespace hack (`\hspace{}`)

## Table of Contents

- [Executive Summary](#executive-summary)
- [The Problem](#the-problem)
- [The New Solution](#the-new-solution)
- [How SymPy Identity Works](#how-sympy-identity-works)
- [The sympy.Dummy Precedent](#the-sympydummy-precedent)
- [Implementation Details](#implementation-details)
- [Testing](#testing)
- [Related Documents](#related-documents)
- [Troubleshooting](#troubleshooting)
- [Related Issue: Coordinate Symbol Isolation](#related-issue-coordinate-symbol-isolation-basescalar)
- [Conclusion](#conclusion)

---

## Executive Summary

Underworld3 needs to distinguish between symbolic variables that have the same display name but belong to different meshes (e.g., `solver1.v` vs `solver2.v`). This document explains the clean, SymPy-native mechanism that replaced the previous invisible whitespace hack.

## The Problem

When users create variables on multiple meshes with the same name:

```python
mesh1 = uw.meshing.StructuredQuadBox(...)
mesh2 = uw.meshing.StructuredQuadBox(...)

v1 = uw.discretisation.MeshVariable("v", mesh1, 2)
v2 = uw.discretisation.MeshVariable("v", mesh2, 2)
```

These variables must be **symbolically distinct** so that:
1. `v1.sym != v2.sym` (different SymPy objects)
2. `v1.sym + v2.sym` keeps both terms (doesn't simplify to `2*v`)
3. Each variable can be independently substituted in expressions
4. The JIT compiler can map each symbol to the correct mesh's data

### The Old Solution (Deprecated)

Previously, Underworld3 used invisible LaTeX whitespace to make names unique:

```python
# In discretisation_mesh_variables.py (OLD CODE - REMOVED)
if mesh.instance_number > 1:
    invisible = rf"\hspace{{ {mesh.instance_number/10000}pt }}"
    self.symbol = f"{{ {invisible} {symbol} }}"
```

This created symbols like `{ \hspace{ 0.0002pt } v }` which:
- ❌ Made printed output ugly and confusing
- ❌ Complicated LaTeX rendering
- ❌ Broke serialization/deserialization
- ❌ Was hard to debug and understand
- ❌ Required cleanup regex in `mpi.py` for printing

## The New Solution

We use SymPy's native mechanisms for symbol identity:

### 1. For `UWexpression` (Symbol Subclass)

**Pattern:** Override `_hashable_content()` to include a unique ID, following the same pattern as `sympy.Dummy`.

```python
class UWexpression(Symbol):
    __slots__ = ('_uw_id',)

    def __new__(cls, name, *args, _unique_name_generation=False, **kwargs):
        # Determine unique ID
        if _unique_name_generation:
            uw_id = UWexpression._expr_count
        else:
            uw_id = None

        # CRITICAL: Use __xnew__ to bypass SymPy's internal cache
        # (The cache doesn't know about _uw_id)
        obj = Symbol.__xnew__(cls, name)
        obj._uw_id = uw_id
        return obj

    def _hashable_content(self):
        """Include _uw_id in hash for disambiguation."""
        base_content = Symbol._hashable_content(self)
        if self._uw_id is not None:
            return base_content + (self._uw_id,)
        return base_content
```

**Why `__xnew__`?**

SymPy's `Symbol.__new__` has an internal cache keyed by `(cls, name, assumptions)`. This cache runs **before** our `_hashable_content()` is called. Using `Symbol.__xnew__` bypasses this cache, ensuring each call creates a fresh object that we can customize.

### 2. For `UnderworldFunction` (Creates UndefinedFunction)

**Pattern:** Pass `_uw_id` as a keyword argument to `UndefinedFunction()`. SymPy automatically uses kwargs in `__eq__` and `__hash__` for function classes.

```python
class UnderworldFunction(sympy.Function):
    def __new__(cls, name, meshvar, vtype, component=0, ...):
        mesh = meshvar.mesh
        uw_id = mesh.instance_number if mesh.instance_number > 1 else None

        # SymPy uses _uw_id in __eq__ and __hash__ automatically!
        ourcls = sympy.core.function.UndefinedFunction(
            fname,
            bases=(UnderworldAppliedFunction,),
            _uw_id=uw_id,  # This makes functions distinct
            **options
        )
        ourcls.meshvar = weakref.ref(meshvar)
        return ourcls
```

## How SymPy Identity Works

### For Symbols

SymPy symbols use `_hashable_content()` for identity:

```python
class Symbol:
    def __hash__(self):
        return hash(self._hashable_content())

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self._hashable_content() == other._hashable_content()

    def _hashable_content(self):
        return (self.name,) + tuple(sorted(self.assumptions0.items()))
```

By adding `_uw_id` to `_hashable_content()`, we make symbols with the same name but different IDs distinct.

### For Functions (UndefinedFunction)

SymPy's `UndefinedFunction` stores kwargs and includes them in equality:

```python
# Inside sympy.core.function
class UndefinedFunction:
    def __new__(cls, name, **kwargs):
        # kwargs are stored and used in __eq__/__hash__
        ...
```

When we pass `_uw_id=mesh.instance_number`, SymPy automatically makes functions from different meshes distinct.

## The `sympy.Dummy` Precedent

Our approach mirrors `sympy.Dummy`, which uses this exact pattern:

```python
# From sympy/core/symbol.py
class Dummy(Symbol):
    _count = 0
    __slots__ = ('dummy_index',)

    def __new__(cls, name=None, dummy_index=None, **assumptions):
        if dummy_index is None:
            dummy_index = Dummy._count
            Dummy._count += 1

        cls._sanitize(assumptions, cls)
        obj = Symbol.__xnew__(cls, name, **assumptions)  # Bypass cache!
        obj.dummy_index = dummy_index
        return obj

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.dummy_index,)
```

The key insight: `Dummy` symbols with the same name are distinct because `dummy_index` is included in `_hashable_content()`.

## Implementation Details

### Files Modified

1. **`src/underworld3/function/expressions.py`**
   - Added `__slots__ = ('_uw_id',)` to `UWexpression`
   - Modified `__new__` to use `Symbol.__xnew__()` and assign `_uw_id`
   - Added `_hashable_content()` override
   - Added `__getnewargs_ex__()` for pickling support

2. **`src/underworld3/function/_function.pyx`**
   - Modified `UnderworldFunction.__new__` to pass `_uw_id` to `UndefinedFunction()`
   - Applied same fix to derivative function classes

3. **`src/underworld3/discretisation/discretisation_mesh_variables.py`**
   - Removed the `\hspace{}` hack (lines 199-201)
   - Added comment explaining new mechanism

### When `_uw_id` Is Assigned

| Object Type | When `_uw_id` is set |
|-------------|---------------------|
| `UWexpression` | When `_unique_name_generation=True` |
| `MeshVariable.sym` | When `mesh.instance_number > 1` |
| Derivative functions | Same as parent MeshVariable |

### Backward Compatibility

- Symbols created on the first mesh (`instance_number == 1`) have `_uw_id = None`
- This matches previous behavior where no disambiguation was needed
- Expressions created without `_unique_name_generation=True` are shared by name (singleton pattern)

## Testing

### Verification Script

```python
import underworld3 as uw

# Create two meshes
mesh1 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
mesh2 = uw.meshing.StructuredQuadBox(elementRes=(4, 4))

# Create variables with same name
v1 = uw.discretisation.MeshVariable("v", mesh1, 2)
v2 = uw.discretisation.MeshVariable("v", mesh2, 2)

# Test 1: Symbols are distinct
assert v1.sym != v2.sym, "Symbols should be different"

# Test 2: Display names are clean
assert '\\hspace' not in str(v1.sym), "No invisible whitespace"
assert '\\hspace' not in str(v2.sym), "No invisible whitespace"

# Test 3: Expression keeps both terms
expr = v1.sym + v2.sym
from underworld3.function._function import UnderworldAppliedFunction
atoms = expr.atoms(UnderworldAppliedFunction)
assert len(atoms) == 4, "Should have 4 atoms (2 components × 2 variables)"

# Test 4: Meshvar accessible via weakref
assert v1.sym[0,0].func.meshvar().mesh is mesh1

print("✅ All disambiguation tests passed!")
```

### Test Files

- `tests/test_symbol_disambiguation_prototype.py` - Comprehensive unit tests
- Core solver tests verify no regressions

## Benefits

| Aspect | Old (`\hspace{}`) | New (`_uw_id`) |
|--------|-------------------|----------------|
| Display names | Ugly, cluttered | Clean |
| LaTeX rendering | Artifacts possible | Perfect |
| Debugging | Confusing | Clear |
| Serialization | Problematic | Works correctly |
| SymPy integration | Hack/workaround | Native mechanism |
| Code complexity | Regex cleanup needed | Simple |

## Related Documents

- `UNITS_SIMPLIFIED_DESIGN_2025-11.md` - Units architecture (expressions are part of units system)
- `MATHEMATICAL_MIXIN_DESIGN.md` - How expressions work in arithmetic
- `ARCHITECTURE_ANALYSIS.md` - MeshVariable architecture (EnhancedMeshVariable wrapper)
- `TEMPLATE_EXPRESSION_PATTERN.md` - ExpressionProperty for solver templates
- `historical-notes.md` - Development history

### Persistence and Adaptive Meshing

The persistence layer (`src/underworld3/discretisation/persistence.py` and `enhanced_variables.py`) provides features for transferring variable data between meshes:

- **EnhancedMeshVariable**: Wrapper class that adds units, mathematical operations, and persistence capabilities
- **transfer_data_from()**: Method for copying data between variables on different meshes
- **Reserved for future**: Adaptive mesh refinement, checkpoint/restart, mesh-to-mesh interpolation

**Key Interaction with Symbol Disambiguation:**

When transferring data between meshes, the symbol disambiguation ensures that:
1. Each mesh's variables maintain distinct symbolic identity
2. Expressions containing variables from different meshes don't accidentally conflate
3. The JIT compiler correctly maps symbols to their respective mesh's data arrays

For expression transfer (not just data), explicit coordinate substitution is required—see [Current Recommendation](#current-recommendation).

## Troubleshooting

### Issue: Symbols from same mesh are incorrectly distinct

**Cause:** `_unique_name_generation=True` when it shouldn't be.

**Solution:** Only use `_unique_name_generation=True` for truly ephemeral expressions that need unique identity regardless of name.

### Issue: Symbols from different meshes are incorrectly equal

**Cause:** `mesh.instance_number` not incrementing properly.

**Solution:** Check that mesh creation increments `Mesh._instance_count`. First mesh is 1, subsequent meshes should be 2, 3, etc.

### Issue: Pickle/unpickle changes symbol identity

**Cause:** `__getnewargs_ex__()` not returning `_uw_id`.

**Solution:** Ensure `__getnewargs_ex__` includes `_uw_id` in kwargs:
```python
def __getnewargs_ex__(self):
    return ((self.name,), {'_uw_id': self._uw_id})
```

---

## Related Issue: Coordinate Symbol Isolation (BaseScalar)

### The Problem

SymPy's `BaseScalar` (used for mesh coordinates `N.x`, `N.y`, `N.z`) uses **name-based equality**:

```python
mesh1.N.x == mesh2.N.x  # True! Same name = same symbol
```

This caused SymPy's expression cache to substitute coordinates from wrong meshes, leading to subtle bugs.

### The Fix (UWCoordinate)

Modified `UWCoordinate.__eq__` and `__hash__` to use coordinate system identity, making coordinates mesh-specific:

```python
mesh1.N.x == mesh2.N.x  # Now False - different coordinate systems
```

### Design Consideration: Isolation vs Portability

**⚠️ IMPORTANT:** This isolation has trade-offs that need consideration for future work.

#### Current Behavior (Isolated)
- ✅ Prevents cross-mesh pollution bugs
- ✅ JIT compiler always gets correct mesh's coordinates
- ❌ Expressions in `x, y, z` cannot be directly shared between meshes

#### Potential Use Case: Adaptive Meshing

In adaptive meshing or mesh transfer scenarios, you might want:

```python
# Define a function on mesh1
temperature_expr = sympy.sin(x) * sympy.cos(y)

# Later, evaluate on mesh2 (refined/adapted mesh)
# QUESTION: Should this work automatically?
new_values = uw.function.evaluate(temperature_expr, mesh2.X.coords)
```

#### Questions for Future Design

1. **Is isolation always desirable?**
   - For mesh variables (`v.sym`): YES - must stay tied to their mesh's data
   - For coordinate expressions (`sin(x)`): MAYBE NOT - pure math functions could be portable

2. **Possible approaches for portable expressions:**
   - Explicit conversion: `expr.subs({mesh1.N.x: mesh2.N.x, ...})`
   - Canonical coordinate symbols: Global `x, y, z` that map to any mesh
   - Evaluation-time binding: Coordinates resolved when evaluating, not when creating

3. **When would portability be needed?**
   - Adaptive mesh refinement (AMR)
   - Mesh-to-mesh interpolation
   - Defining boundary conditions that apply to multiple meshes
   - Template expressions for material properties

### Current Recommendation

For now, the isolation is **correct and safe**. It prevents subtle bugs where expressions accidentally use wrong mesh data.

If portability is needed, use explicit substitution:

```python
# Create expression with mesh1 coordinates
expr = mesh1.N.x**2 + mesh1.N.y**2

# To use on mesh2, substitute coordinates
expr_for_mesh2 = expr.subs({
    mesh1.N.x: mesh2.N.x,
    mesh1.N.y: mesh2.N.y
})
```

This makes the mesh transfer explicit and traceable, avoiding silent bugs.

### Future Work

If adaptive meshing becomes a priority, consider:
1. A utility function `transfer_expression(expr, from_mesh, to_mesh)`
2. Documentation on expression portability patterns
3. Possibly "mesh-agnostic" coordinate symbols for pure mathematical expressions

---

## Conclusion

The `_uw_id` mechanism provides clean, maintainable symbol disambiguation using SymPy's native identity system. It eliminates the need for invisible whitespace hacks while ensuring correct behavior for multi-mesh simulations.

The related `UWCoordinate` isolation fix ensures coordinates are also mesh-specific, preventing cache pollution bugs. While this creates some friction for expression portability between meshes, it's the safer default that prevents subtle bugs. Explicit substitution provides a clear path for mesh transfer when needed.
