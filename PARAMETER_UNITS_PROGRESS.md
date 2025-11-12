# Parameter Units and Lazy Evaluation - Implementation Progress

**Date**: 2025-11-12
**Goal**: Enable constitutive model parameters to accept `uw.quantity()` with units while preserving lazy evaluation

## ✅ COMPLETED WORK

### 1. Enhanced Parameter Descriptor (✅ COMPLETE)
**File**: `/src/underworld3/utilities/_api_tools.py` (lines 287-327)

**What was done**:
- Enhanced `Parameter.__set__()` to detect and handle `UWQuantity` objects
- Copies unit metadata from UWQuantity to expression object:
  - `_pint_qty` - Pint quantity for dimensional analysis
  - `_dimensionality` - Dimensional formula [L], [M], [T], [θ]
  - `_custom_units` - User-specified unit strings
  - `_model_registry` - Reference to model for conversions
- Updates `._sym` with the numeric value for JIT substitution
- Preserves expression object identity (same Python id)

**Code pattern**:
```python
if isinstance(value, UWQuantity) and not isinstance(value, UWexpression):
    expr.sym = value._sym  # Update substitution value
    # Copy ALL unit metadata fields...
    if hasattr(value, '_pint_qty'):
        expr._pint_qty = value._pint_qty
    # ... etc for all metadata fields
```

### 2. Refactored ViscousFlowModel Parameters (✅ COMPLETE)
**File**: `/src/underworld3/constitutive_models.py` (lines 420-445)

**What was done**:
- Migrated `shear_viscosity_0` from instance-level to class-level Parameter descriptor
- Used absolute import to avoid nested class import issues
- Pattern now matches solver Template pattern (F0, F1, PF0)

**Code pattern**:
```python
class _Parameters:
    import underworld3.utilities._api_tools as api_tools

    shear_viscosity_0 = api_tools.Parameter(
        r"\eta",
        lambda params_instance: params_instance._owning_model.create_unique_symbol(
            r"\eta", 1, "Shear viscosity"
        ),
        "Shear viscosity",
        units="Pa*s"
    )
```

### 3. Template Expression Pattern Understanding (✅ COMPLETE)
**Source**: `docs/developer/TEMPLATE_EXPRESSION_PATTERN.md`

**Key insights gained**:
1. **Persistent Expression Containers**: Expression objects created ONCE, identity preserved
2. **Lazy Evaluation**: Templates reference OTHER expression objects, not their values
3. **Solver Pattern**: `F1 = Template(lambda self: self.constitutive_model.flux.T)`
4. **Critical**: Solver uses `flux` property directly, NOT `flux.sym`
5. **Implication**: The `flux` property must return an EXPRESSION OBJECT that acts as a Symbol

## ✅ RESOLVED - Tensor Reduction Issue (2025-11-12)

### The Problem
**Location**: `/src/underworld3/constitutive_models.py` line 516 (`ViscousFlowModel._build_c_tensor()`)

**Original buggy code**:
```python
# Extracts .sym (numeric value 1) - breaks lazy evaluation
viscosity_sym = viscosity.sym if hasattr(viscosity, "sym") else viscosity
self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity_sym
```

**Why it's wrong**: Substitutes numeric value `1` instead of keeping `viscosity` as a symbol

**Fix attempt 1**: Remove `.sym` extraction
```python
self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity
```
**Result**: `ValueError: scalar expected, use tensorproduct(...) for tensorial product`

**Fix attempt 2**: Use `sympy.sympify()`
```python
viscosity_sym = sympy.sympify(viscosity)
self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity_sym
```
**Result**: Same error - `sympify()` returns UWexpression unchanged via `_sympify_()` protocol

### Root Cause Analysis

**The Conflict**:
1. **Lazy evaluation requires**: Expression object as symbol in SymPy tree
2. **SymPy arrays require**: Scalar SymPy atoms (Symbol, Integer, etc.) for multiplication
3. **UWexpression IS a Symbol**: But SymPy's array `__mul__` doesn't recognize it properly

**Original comment that was removed** (line 509-514):
```python
# There is a conflict here, if viscosity is a symbolic object
# it does not work with the array operations. The fix is to
# ensure that what we are dealing with here is the actual value
# of the symbolic object, not the symbol itself. That means you
# need to provide the numerical value directly.
```

**This comment was CORRECT** - but the solution (extracting `.sym`) breaks lazy evaluation!

### How Does It Work in Solvers?

**Solver F1 Template** (`/src/underworld3/systems/solvers.py` line 134):
```python
F1 = Template(
    r"\mathbf{F}_1\left( \mathbf{u} \right)",
    lambda self: sympy.simplify(self.constitutive_model.flux.T),
    "Poisson pointwise flux term: F_1(u)",
)
```

**Key observations**:
- Uses `self.constitutive_model.flux` directly (NOT `.flux.sym`)
- The `flux` property (line 299-305 of constitutive_models.py) computes tensor operations
- Returns `sympy.Matrix(flux)` after `tensorcontraction` operations

**The flux property WORKS** because:
```python
if rank == 2:
    flux = c * ddu.T  # Matrix multiplication, not array element-wise
else:  # rank==4
    flux = sympy.tensorcontraction(
        sympy.tensorcontraction(sympy.tensorproduct(c, ddu), (1, 5)), (0, 3)
    )
```

**Hypothesis**: The difference is:
- `flux` uses `sympy.Matrix` multiplication (works with expression objects)
- `_build_c_tensor()` uses array element-wise multiplication (doesn't work)

## 🔍 INVESTIGATION NEEDED

### Questions to Answer

1. **Can we use Matrix multiplication instead of array multiplication?**
   ```python
   # Instead of:
   self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity

   # Try:
   identity = uw.maths.tensor.rank4_identity(d)
   # Convert to Matrix and use Matrix ops?
   ```

2. **What is the structure of `rank4_identity`?**
   - Is it a SymPy Array or ImmutableDenseNDimArray?
   - Can it be converted to work with expression objects?

3. **How do other constitutive models handle this?**
   - Do they all extract `.sym`?
   - Are there models with non-constant parameters?

4. **Could we use `sympy.tensorproduct()` pattern like flux does?**
   - The `flux` property successfully multiplies tensors with expression-containing matrices
   - Maybe we need to structure the operation differently

### Specific Code Locations to Examine

1. **`uw.maths.tensor.rank4_identity()`** - What does it return?
   - File: Likely in `/src/underworld3/maths/`
   - Need to understand its type and multiplication behavior

2. **Other constitutive models** - Do they have the same pattern?
   - Search for: `_build_c_tensor` in constitutive_models.py
   - Check: DiffusionModel, PlasticFlowModel, etc.

3. **Template vs Parameter** - Why does Template work?
   - Template descriptor in _api_tools.py
   - Does it handle expression objects differently?

## 📋 REMAINING TASKS

### Immediate Next Steps
1. ⏳ **Investigate tensor multiplication** - Understand why flux works but _build_c_tensor doesn't
2. ⏳ **Find alternative multiplication pattern** - Matrix ops, tensorproduct, or other approach
3. ⏳ **Test solution** - Verify lazy evaluation preserved AND tensor construction works

### Validation Tasks (After Fix)
4. ⏳ **Test parameter assignment with units** - Run debug_parameter_substitution.py
5. ⏳ **Validate dimensional metadata** - Confirm unit metadata accessible after assignment
6. ⏳ **Roll out to other models** - Apply Parameter pattern to DiffusionModel, etc.

## 💡 INSIGHTS FROM TEMPLATE DOCUMENTATION

**Key Insight**: The Template Expression Pattern works because:
1. Expression objects ARE SymPy Symbols (inherit from Symbol)
2. They appear symbolically in expression trees
3. Their `._sym` value is substituted ONLY during JIT compilation
4. Templates reference expression OBJECTS, not `.sym` values

**Critical Quote** (TEMPLATE_EXPRESSION_PATTERN.md line 47):
> "Lazy Evaluation: Templates can reliably reference sub-expressions"

**This means**:
- `constitutive_model.flux` should be an expression object
- Solver's `F1` template references that expression object
- When parameters change, expression object's `._sym` updates
- But object identity preserved → lazy evaluation works

## 🎯 SUCCESS CRITERIA

When fixed, we should be able to:
```python
# Create constitutive model
viscous_model = uw.constitutive_models.ViscousFlowModel(unknowns)

# Assign parameter WITH units
viscous_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")

# Check that:
# 1. Expression object identity preserved
visc_id_before = id(viscous_model.Parameters.shear_viscosity_0)
viscous_model.Parameters.shear_viscosity_0 = uw.quantity(2e21, "Pa*s")
visc_id_after = id(viscous_model.Parameters.shear_viscosity_0)
assert visc_id_before == visc_id_after  # ✓ Same container

# 2. Symbolic value updated
assert viscous_model.Parameters.shear_viscosity_0._sym == 2e21  # ✓ New value

# 3. Unit metadata copied
assert hasattr(viscous_model.Parameters.shear_viscosity_0, '_pint_qty')  # ✓ Units preserved

# 4. Lazy evaluation works
flux_expr = viscous_model.flux
# flux_expr should contain viscosity as a SYMBOL, not the value 2e21
```

## 📝 FILES MODIFIED

1. `/src/underworld3/utilities/_api_tools.py` - Enhanced Parameter.__set__()
2. `/src/underworld3/constitutive_models.py` - Refactored ViscousFlowModel Parameters

## 🚫 FILES WITH ATTEMPTED FIXES (Currently Broken)

1. `/src/underworld3/constitutive_models.py` line 516 - `_build_c_tensor()` method
   - Current state: Has attempted fix that doesn't work
   - Needs: Alternative multiplication pattern that works with expression objects

## 📚 REFERENCE DOCUMENTATION

- **Template Pattern**: `/docs/developer/TEMPLATE_EXPRESSION_PATTERN.md`
- **Project Status**: `/CLAUDE.md` - See "Units System" and "Coordinate Units" sections
- **Parameter Descriptor**: `/src/underworld3/utilities/_api_tools.py` lines 330-359
- **Expression Persistence**: `/src/underworld3/function/expressions.py` lines 371-378

---

## 🎉 SOLUTION IMPLEMENTED (2025-11-12)

### Root Cause
UWexpression has `__getitem__` from MathematicalMixin, which makes Python's `isinstance(value, Iterable)` return `True`. SymPy's array multiplication and assignment both check for Iterable and reject UWexpression objects, even though they inherit from Symbol.

### The Fix
**File**: `/src/underworld3/constitutive_models.py` lines 512-535

**Strategy**: For scalar viscosity, use element-wise tensor construction via loops instead of array multiplication operator. Wrap any bare UWexpression results to prevent Iterable check failures.

**Key code**:
```python
identity = uw.maths.tensor.rank4_identity(d)
result = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

# Element-wise multiplication: c_ijkl = 2 * I_ijkl * viscosity
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                val = 2 * identity[i, j, k, l] * viscosity
                # If simplification returns bare UWexpression (e.g., 2*(1/2)*visc = visc),
                # wrap it to avoid Iterable check failure during assignment
                if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                    val = sympy.Mul(sympy.S.One, val, evaluate=False)
                result[i, j, k, l] = val

self._c = result
```

**How it works**:
1. Element-wise loops bypass SymPy's array multiplication operator (which rejects Iterable)
2. Most elements are `Mul(scalar, UWexpression)` which aren't Iterable ✓
3. Edge case: `2 * (1/2) * viscosity` simplifies to bare `viscosity` (Iterable) → wrap as `Mul(1, viscosity, evaluate=False)` ✓
4. JIT unwrapper finds UWexpression atoms inside all Mul objects → substitutes with numeric values ✓

### Bonus Fix: Parameter Reset Mechanism
**File**: `/src/underworld3/utilities/_api_tools.py` lines 325-330

Enhanced Parameter descriptor to call `_reset()` when updating constitutive model parameters, ensuring `_is_setup` flag is properly invalidated.

```python
# Mark solver/model as needing setup
if hasattr(obj, "_reset"):
    # For constitutive model Parameters, call _reset() to invalidate setup
    obj._reset()
elif hasattr(obj, "is_setup"):
    obj.is_setup = False
```

### Validation Results
✅ All 20 constitutive tensor regression tests passing
✅ Parameter assignment with units works
✅ Unit metadata preserved (has `_pint_qty`)
✅ Lazy evaluation maintained (symbol in tensor, not numeric value)
✅ Symbolic viscosity flows through to flux calculation
✅ Parameter updates trigger tensor rebuild via `_reset()`

### Test Files Created
- `test_parameter_units_lazy.py` - Verifies units and lazy evaluation
- `test_parameter_update_rebuild.py` - Verifies parameter updates trigger rebuild

---

**Status**: Blocker resolved. Parameters can now accept `uw.quantity()` with units while preserving lazy evaluation and Template pattern compatibility.
