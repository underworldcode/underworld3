# Simplified Units Architecture (November 2025)

> **STATUS**: This document supersedes all previous units planning documents.
> Previous plans (`units_system_plan.md`, etc.) are now historical reference only.

## Core Principle: Gateway Pattern

Units are handled at **boundaries** (input/output), not during symbolic manipulation:

1. **Input Gateway**: User creates quantities with units → stored as Pint objects
2. **Symbolic Layer**: Operations produce expressions → units discoverable from atoms
3. **Output Gateway**: `evaluate()` returns dimensional `UnitAwareArray`

```
User Input          Symbolic Layer              Output
───────────         ──────────────              ──────
uw.quantity() ─┐
               ├──► UWexpression ──► unwrap() ──► evaluate() ──► UnitAwareArray
uw.expression()┘    (lazy eval)      (ND for     (dimensional
                                     solver)      for user)
```

## Type Hierarchy

### UWQuantity
- **Purpose**: Lightweight number with units (Pint-backed)
- **Use case**: Simple arithmetic between quantities
- **Properties**:
  - `.value` → dimensional (what user sees)
  - `.data` → non-dimensional (what solver sees)
  - `.units` → Pint Unit object
- **Arithmetic**: Pure Pint delegation for `UWQuantity ⊗ UWQuantity`

### UWexpression
- **Purpose**: Lazy-evaluated wrapper, the preferred user-facing object
- **Use case**: Parameters, constants, composite expressions
- **Properties**:
  - `.value` → dimensional value (from wrapped thing)
  - `.data` → non-dimensional value (from wrapped thing)
  - `.units` → discovered from wrapped thing
  - `.sym` → the wrapped SymPy expression or value
- **Arithmetic**: Returns `UWexpression` wrapping result with combined units

### MeshVariable
- **Purpose**: Field data on mesh (the template for this design)
- **Properties**:
  - `.array` → `UnitAwareArray` (dimensional)
  - `.data` → non-dimensional values
  - `.sym` → SymPy Function for symbolic use
  - `.units` → Pint Unit object
- **Pattern**: This is the model we follow for all unit-aware objects

### UnitAwareArray
- **Purpose**: NumPy array with units attached
- **Use case**: Return type from `evaluate()`, mesh coordinates
- **Properties**:
  - Inherits from `np.ndarray`
  - `.units` → Pint Unit object
  - `.to(units)` → convert to different units

## Transparent Container Principle (2025-11-26)

> **Key Insight**: A container cannot know in advance what it contains.
> All accessor methods must evaluate lazily - no cached state on composites.

### The Atomic vs Container Distinction

| Type | Role | What it stores |
|------|------|----------------|
| **UWQuantity** | Atomic leaf node | Value + Units (indivisible, this IS the data) |
| **UWexpression** | Container | Reference to contents only (derives everything) |

### Why This Matters

**UWexpression is always a container**, whether it wraps:
1. A UWQuantity (atomic) → derives `.units` from `self._value_with_units.units`
2. A SymPy tree (composite) → derives `.units` from `get_units(self._sym)`

The container never "owns" units or values - it provides access to what's inside.

```python
# Atomic: UWQuantity owns the value+units
qty = uw.quantity(3e-5, "1/K")  # This IS 3e-5 per Kelvin

# Container wrapping atomic: derives from contents
alpha = uw.expression("α", qty)
alpha.units  # → qty.units (derived, not stored separately)

# Container wrapping composite: derives from tree
product = alpha * beta  # Returns SymPy Mul containing alpha, beta
get_units(product)      # → traverses tree, finds atoms, combines units
```

### Implementation Consequences

1. **No stored units on composites** - eliminates sync issues
2. **Properties are lazy evaluations** - always reflect current state
3. **Convenience flags (e.g., `_is_constant`)** - cached queries, not owned data
4. **Arithmetic returns raw SymPy** - for `expr * expr`, let SymPy handle it

```python
class UWexpression:
    @property
    def units(self):
        # Always derived, never stored separately
        if self._value_with_units is not None:
            return self._value_with_units.units  # From contained atom
        return get_units(self._sym)  # From contained tree
```

## Arithmetic Closure Table

| Left ⊗ Right | Result Type | Units Preserved? |
|--------------|-------------|------------------|
| `UWQuantity * UWQuantity` | `UWQuantity` | ✅ Pint arithmetic |
| `UWQuantity * scalar` | `UWQuantity` | ✅ Pint arithmetic |
| `UWQuantity * UWexpression` | `UWexpression` | ✅ Wrapped with combined units |
| `UWQuantity * MeshVar.sym` | `UWexpression` | ✅ Wrapped with combined units |
| `UWQuantity * sympy.Basic` | `UWexpression` | ✅ Wrapped (qty units preserved) |
| | | |
| `UWexpression * UWexpression` | `sympy.Mul` | ✅ Discoverable from atoms (lazy) |
| `UWexpression * UWQuantity` | `UWexpression` | ✅ Wrapped with combined units |
| `UWexpression * scalar` | `UWexpression` | ✅ Preserves expression units |
| | | |
| `MeshVar.sym * MeshVar.sym` | `sympy.Mul` | ✅ Discoverable from atoms |
| `MeshVar.sym * scalar` | `sympy.Mul` | ✅ MeshVar units preserved |
| `MeshVar.sym * sympy.Basic` | `sympy.Mul` | ✅ Discoverable from atoms |

**Key Rule**: Any operation involving `UWQuantity` or `UWexpression` returns a type that preserves units. Pure SymPy operations between MeshVariable symbols are OK because `get_units()` can discover units from the atoms.

## User-Facing Recommendations

### Preferred Pattern
```python
# PREFERRED: Use expressions for parameters
alpha = uw.expression(r"\alpha", uw.quantity(3e-5, "1/K"), "thermal expansivity")
rho0 = uw.expression(r"\rho_0", uw.quantity(3300, "kg/m^3"), "reference density")

# Arithmetic preserves units
buoyancy = rho0 * alpha * g * dT  # Returns UWexpression with correct units
```

### Acceptable Pattern
```python
# OK for quick calculations, but prefer expressions for model parameters
viscosity = uw.quantity(1e21, "Pa*s")
velocity = uw.quantity(5, "cm/year")
```

### Avoid
```python
# AVOID: Raw quantities in symbolic expressions without wrapping
# This works but is less clear and loses symbolic meaning
result = uw.quantity(5, "m/s") * mesh.X[0]  # Works but prefer expression
```

## Implementation Requirements

### 1. UWQuantity.__mul__ (and other operators)
```python
def __mul__(self, other):
    if isinstance(other, UWQuantity):
        # Pure Pint arithmetic
        return UWQuantity(result.magnitude, result.units)
    elif isinstance(other, (int, float)):
        # Scalar - Pint handles
        return UWQuantity(result.magnitude, result.units)
    else:
        # Everything else: wrap in UWexpression
        # Compute combined units, wrap result
        return UWexpression(name, UWQuantity(value, combined_units))
```

### 2. UWexpression.__mul__ (and other operators)
```python
def __mul__(self, other):
    if isinstance(other, UWQuantity):
        # Delegate to UWQuantity which returns UWexpression
        return other.__rmul__(self)
    elif isinstance(other, UWexpression):
        # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26):
        # Return raw SymPy product - units derived on demand via get_units()
        # This preserves lazy evaluation and eliminates sync issues
        return Symbol.__mul__(self, other)
    elif isinstance(other, (int, float)):
        # Scalar: wrap result preserving self's units
        return UWexpression(name, UWQuantity(self.value * other, self.units))
    else:
        # Default to SymPy multiplication
        return Symbol.__mul__(self, other)
```

### 3. unwrap() / unwrap_for_evaluate()
This is where all unit handling converges:
- Extract units from expression atoms
- Compute non-dimensional values for solver
- Track result dimensionality for re-dimensionalization

### 4. evaluate()
Gateway function that:
- Calls unwrap to get ND expression
- Evaluates numerically
- Re-dimensionalizes result
- Returns `UnitAwareArray` with correct units

## What We Deleted

The following are **no longer used** (preserved as `*_old.py` for reference):
- `UnitAwareExpression` (~1500 lines) - tracked units through all arithmetic
- Complex unit-tracking in expression arithmetic
- Multiple inheritance from UWQuantity in UWexpression

## Benefits of This Design

1. **Simplicity**: Units handled in one place (unwrap/evaluate)
2. **Consistency**: Same pattern as MeshVariable
3. **User Experience**: Expressions are self-documenting with LaTeX names
4. **Lazy Evaluation**: Parameters can change (time-stepping)
5. **Solver Compatibility**: ND values for PETSc, dimensional for user

## Files Affected

- `src/underworld3/function/quantities.py` - Simplified UWQuantity
- `src/underworld3/function/expressions.py` - Simplified UWexpression
- `src/underworld3/function/functions_unit_system.py` - evaluate() gateway
- `src/underworld3/units.py` - get_units(), non_dimensionalise(), dimensionalise()

## Testing Strategy

1. **Tier A (Production)**: Core Stokes ND tests, basic quantity/expression tests
2. **Tier B (Validated)**: Mixed arithmetic, evaluate combinations
3. **Tier C (Experimental)**: Edge cases being developed

See `docs/developer/TESTING-RELIABILITY-SYSTEM.md` for test classification.
