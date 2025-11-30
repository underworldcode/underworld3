# UWQuantity Unwrapping Bug Fix (2025-11-15)

## Problem

JIT compilation was failing with C syntax errors when UWQuantity constants with units were used in constitutive models:

```python
stokes.constitutive_model.Parameters.viscosity = uw.quantity(1.0, "Pa*s")
```

### Error Symptoms

1. **C Compiler Error**:
   ```
   ./cy_ext.h:236:14: error: expected expression
         |              ^
   ```

2. **Generated C Code**:
   ```c
   out[0] = 1.0/{ \eta \hspace{ 0.0006pt } };  // Invalid C syntax!
   ```

3. **Failed Tests**:
   - `test_0818_stokes_nd.py`: All 4 Stokes non-dimensionalization tests

## Root Cause

The `unwrap()` function in `src/underworld3/function/expressions.py` was **ignoring** the `keep_constants` parameter:

```python
# BEFORE (line 277-301):
def unwrap(fn, depth=None, keep_constants=True, return_self=True):
    ...
    return expand(fn, depth=depth)  # ❌ Doesn't pass keep_constants!
```

### Why This Broke JIT Compilation

1. JIT code calls: `unwrap(fn, keep_constants=False, return_self=False)` (line 414 of `_jitextension.py`)
2. `unwrap()` ignores parameters and calls `expand(fn)`
3. `expand()` hardcodes `keep_constants=True` internally
4. UWQuantity constants never get unwrapped to numeric values
5. LaTeX symbol `\eta` ends up in generated C code → compiler error

## Solution

Modified `unwrap()` to respect `keep_constants` and `return_self` parameters:

```python
# AFTER (line 277-316):
def unwrap(fn, depth=None, keep_constants=True, return_self=True):
    """..."""
    # For JIT compilation path (keep_constants=False), use _unwrap_expressions directly
    if not keep_constants or not return_self:
        import sympy
        # Get the SymPy expression
        if hasattr(fn, 'sym'):
            sym_expr = fn.sym
        elif isinstance(fn, sympy.Basic):
            sym_expr = fn
        else:
            sym_expr = sympy.sympify(fn)

        # Unwrap with parameters respected
        return _unwrap_expressions(sym_expr, keep_constants=keep_constants, return_self=return_self)

    # Default path for user-facing expansion
    return expand(fn, depth=depth)
```

## Debugging Enhancement

Added free symbol detection in `_jitextension.py` (lines 428-440) to help diagnose unwrapping failures:

```python
if verbose:
    print("Processing JIT {:4d} / {}".format(index, fn))
    # Enhanced debugging output
    free_syms = fn.free_symbols
    if free_syms:
        print("  WARNING: Free symbols remaining after unwrap:")
        for sym in free_syms:
            print(f"    - {sym} (type: {type(sym).__name__}, repr: {repr(sym)})")
            # Check if it's a UWexpression with units
            if hasattr(sym, 'units'):
                print(f"      has .units = {sym.units}")
            if hasattr(sym, 'magnitude'):
                print(f"      has .magnitude = {sym.magnitude}")
        print(f"  Original expression before unwrap: {fn_original}")
        print(f"  After unwrap: {fn}")
```

### Example Debug Output

```
Processing JIT   14 / Matrix([[1], [0], [0], [1]])
  WARNING: Free symbols remaining after unwrap:
    - 1.0 pascal * second (type: UWexpression, repr: ...)
      has .units = pascal * second
      has .magnitude = 1.0
  Original expression before unwrap: Matrix([[1/{\eta}], ...])
  After unwrap: Matrix([[1/1.0 pascal * second], ...])
```

This makes it immediately clear that:
- A UWexpression object is present
- It has units and a magnitude that should be extracted
- The unwrapping didn't properly handle it

## Testing

After the fix, all Stokes ND tests pass:

```bash
$ pixi run -e default pytest tests/test_0818_stokes_nd.py -v
...
tests/test_0818_stokes_nd.py::test_stokes_dimensional_vs_nondimensional[8] PASSED
tests/test_0818_stokes_nd.py::test_stokes_dimensional_vs_nondimensional[16] PASSED
tests/test_0818_stokes_nd.py::test_stokes_buoyancy_driven PASSED
tests/test_0818_stokes_nd.py::test_stokes_variable_viscosity PASSED
tests/test_0818_stokes_nd.py::test_stokes_scaling_derives_pressure_scale PASSED
=========================== 5 passed, 1 warning ===========================
```

## Files Modified

1. **`src/underworld3/function/expressions.py`** (line 277-316)
   - Fixed `unwrap()` to respect `keep_constants` and `return_self` parameters

2. **`src/underworld3/utilities/_jitextension.py`** (lines 414, 428-440)
   - Added `fn_original` capture for debugging
   - Added free symbol warning output with detailed type/attribute inspection

3. **`debug_stokes_jit.py`** (line 35)
   - Fixed API misuse: `ViscousFlowModel(mesh.dim)` → `ViscousFlowModel(stokes.Unknowns)`

## Lessons Learned

1. **Parameter Passing Chains**: When wrapping functions, ensure ALL parameters are passed through
2. **Debug at the Right Level**: JIT errors are hard to diagnose - add visibility before code generation
3. **Type Inspection**: Showing `type(obj).__name__` and object attributes helps identify unwrapping failures
4. **Recurring Pattern**: This is a known issue category ("unwrapping problems") - the enhanced debugging will prevent future similar issues from taking as long to diagnose

## Related Issues

- Multiple historical unwrapping bugs mentioned in conversation
- User noted: "There have been many of these errors in the past"
- Solution: Better debugging infrastructure to catch these earlier in the chain
