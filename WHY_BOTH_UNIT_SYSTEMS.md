# Why Both UnitAwareMixin AND UnitAwareArray?

## The Core Question

If `UnitAwareArray` can attach units to numpy arrays, why do we need `UnitAwareMixin` on variables?

## The Answer: Persistent vs Transient Unit Tracking

### UnitAwareMixin = **Persistent Source of Truth**

```python
# Variable created with units
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")

# Units stored in the VARIABLE OBJECT
velocity.units              # "m/s" - ALWAYS available
velocity._units             # Stored as attribute
velocity._units_backend     # Pint backend for operations

# Operations on the VARIABLE use unit information
velocity.to_units("km/s")                   # Conversion
velocity.check_units_compatibility(other)   # Checking
velocity.non_dimensional_value()            # Scaling for solvers
```

**Key point**: The variable **owns** its units. They persist across operations, saves/loads, and define what the variable represents.

### UnitAwareArray = **Transient Metadata Copy**

```python
# Evaluate COPIES unit information to result
vel = uw.function.evaluate(velocity, coords)

# Array has a COPY of the unit string
uw.get_units(vel)  # "m/s" - copied from velocity

# But it's just metadata for inspection
vel.to_units("km/s")  # ❌ Not available! It's just a string
```

**Key point**: The array **carries** units for inspection, but doesn't own them or support operations.

## Why This Distinction Matters

### 1. Variables Need Full Unit Operations

```python
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="cm/year")

# These require UnitAwareMixin capabilities:
velocity.to_units("m/s")                    # Unit conversion
velocity.non_dimensional_value()            # Scaling for PETSc solvers
velocity.check_units_compatibility(density) # Validation
velocity.create_quantity(100.0)             # Create Pint quantities
```

**UnitAwareArray cannot do this** - it only carries a unit STRING, not a Pint backend with conversion capabilities.

### 2. Arrays Are Transient, Variables Are Persistent

```python
# Variable: PERSISTENT unit storage
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")
# Save to disk
velocity.save("checkpoint.h5")
# Later: Load from disk
velocity.load_from_h5_plex_vector("checkpoint.h5")
# Units PERSIST - stored in HDF5 metadata

# Array: TRANSIENT unit information
vel = uw.function.evaluate(velocity, coords)
magnitudes = np.linalg.norm(vel, axis=1)  # Units lost (correctly!)
# magnitudes is plain numpy array - units were transient metadata
```

**Why this is correct**: Numpy operations change dimensions. `np.linalg.norm()` converts `[m/s]` to a magnitude - what are the units now? Unknown! So arrays **should** lose units through operations.

### 3. Variables Support Symbolic Math, Arrays Don't

```python
# Variable: Used in symbolic expressions
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")
density = uw.discretisation.MeshVariable("rho", mesh, 1, units="kg/m^3")

# Symbolic expression with unit propagation
momentum_expr = density * velocity  # Units: kg/(m^2 * s)

# This is where UnitAwareMixin + MathematicalMixin shine:
# - Mathematical operations return SymPy expressions
# - Unit compatibility is checked
# - Units propagate through symbolic math

# Array: Just numerical data
vel = uw.function.evaluate(velocity, coords)  # Array[m/s]
rho = uw.function.evaluate(density, coords)   # Array[kg/m^3]
mom = rho * vel  # Plain numpy broadcasting, NO unit checking
```

**UnitAwareMixin integrates with MathematicalMixin** for symbolic operations. Arrays don't participate in symbolic math.

## Concrete Example: What Would Break Without UnitAwareMixin?

Let's remove UnitAwareMixin and only use UnitAwareArray:

```python
# Create variable (no UnitAwareMixin)
velocity = uw.discretisation.MeshVariable("V", mesh, 2)
# ❌ Problem 1: Where do we store units="m/s"?
# Can't store on the PETSc vector, can't store on mesh

# Evaluate - let's say we somehow attached units to the array
vel = uw.function.evaluate(velocity, coords)  # UnitAwareArray with "m/s"

# ❌ Problem 2: How do we do unit conversion?
vel_km_s = vel.to_units("km/s")  # UnitAwareArray has no conversion backend!

# ❌ Problem 3: How do we scale for solvers?
stokes = uw.systems.Stokes(mesh, velocityField=velocity)
# Solver needs to know units to scale properly - where does it get them?
# velocity has no .units attribute without UnitAwareMixin!

# ❌ Problem 4: How do symbolic expressions track units?
expr = velocity * density
# MathematicalMixin creates sympy expression, but how does it know units?
# velocity.sym has no unit information without UnitAwareMixin!

# ❌ Problem 5: How do we persist units?
velocity.save("checkpoint.h5")
# The HDF5 save code checks velocity.units attribute - doesn't exist!

# ❌ Problem 6: How do we validate assignments?
temperature = uw.discretisation.MeshVariable("T", mesh, 1)
temperature.array[:] = vel  # Should this be allowed? m/s → temperature?
# Without UnitAwareMixin, no way to check compatibility
```

## The Real Architecture

```
┌─────────────────────────────────────────────────┐
│           MeshVariable (Object)                 │
│  ┌──────────────────────────────────────────┐  │
│  │        UnitAwareMixin                    │  │
│  │  - units: "m/s"                          │  │
│  │  - _units_backend: PintBackend           │  │
│  │  - .to_units()                           │  │
│  │  - .non_dimensional_value()              │  │
│  │  - .check_units_compatibility()          │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  .sym → SymPy expression (for symbolic math)   │
│  .array → NDArray_With_Callback (for data)     │
│  .save() → Writes units to HDF5 metadata       │
└─────────────────────────────────────────────────┘
                      │
                      │ uw.function.evaluate(velocity, coords)
                      ↓
           ┌──────────────────────────┐
           │  UnitAwareArray          │
           │  - _units: "m/s"         │  ← Simple string copy
           │  - numpy operations OK   │
           │  - Transient metadata    │
           └──────────────────────────┘
```

**Key insight**:
- **Variable** = persistent, operational unit system (source of truth)
- **Array** = transient, informational unit metadata (for inspection)

## Question 2: Why not use NDArray_With_Callback?

**Great question!** NDArray_With_Callback is for **PETSc synchronization**, not unit tracking.

### NDArray_With_Callback Purpose

```python
# From discretisation_mesh_variables.py
array_obj = uw.utilities.NDArray_With_Callback(
    initial_data,
    owner=self,
    disable_inplace_operators=False,
)

def canonical_data_callback(array, change_context):
    """Callback to sync variable changes back to PETSc"""
    data_changed = change_context.get("data_has_changed", True)
    if not data_changed:
        return

    # Sync to PETSc using established method
    self.pack_raw_data_to_petsc(canonical_array, sync=True)
```

**Purpose**: Automatically sync array modifications to PETSc vectors.

### Why UnitAwareArray Doesn't Use Callbacks

```python
# Evaluate returns NEW array, not a view of MeshVariable data
vel = uw.function.evaluate(velocity, coords)

# This is a COPY of data at specific coordinates
# It has NO connection to the original MeshVariable
# There's nothing to "sync back" to!

# These are independent:
velocity.array[0, 0, 0] = 99.0  # Changes MeshVariable (syncs to PETSc)
vel[0, 0, 0] = 99.0             # Changes local copy (doesn't sync anywhere)
```

**Key difference**:
- **`velocity.array`** = View of MeshVariable's PETSc data (needs NDArray_With_Callback)
- **`vel` from evaluate()** = Independent copy at sample points (no PETSc connection)

### Could We Add Unit Info to NDArray_With_Callback?

Theoretically yes, but:

```python
# Problem 1: NDArray_With_Callback is for MeshVariable.array
velocity.array  # Returns NDArray_With_Callback
# This already has access to velocity.units! No need to duplicate.

# Problem 2: Evaluate results aren't NDArray_With_Callback
vel = uw.function.evaluate(velocity, coords)
# This is a NEW array, not a view of velocity.array
# It can't be NDArray_With_Callback because there's no callback needed!
# It's not connected to PETSc, so what would the callback do?

# Problem 3: Different ownership models
velocity.array._owner = velocity  # Owned by MeshVariable
vel._owner = ???  # Owned by... nothing? It's just a computed result.
```

**Architectural mismatch**:
- NDArray_With_Callback = "I'm a view that needs to sync to PETSc"
- UnitAwareArray = "I'm a computed result that happens to know its units"

## Summary

### Why UnitAwareMixin Is Essential

1. **Persistent unit storage** - Variables own their units across their lifetime
2. **Unit operations** - Conversion, scaling, compatibility checking
3. **Backend integration** - Pint backend for full dimensional analysis
4. **Symbolic math** - Units propagate through SymPy expressions
5. **Persistence** - Units saved/loaded with HDF5 checkpoints
6. **Solver integration** - Non-dimensionalization for PETSc

### Why UnitAwareArray Is Different

1. **Transient metadata** - Just carries a copy of the unit string
2. **Informational only** - For inspection, not operations
3. **Numpy compatibility** - Lightweight subclass that doesn't break numpy
4. **Evaluate-specific** - Only for results from `evaluate()`, not variable data

### Why Not NDArray_With_Callback For Units?

1. **Different purpose** - Callbacks for PETSc sync, not unit tracking
2. **Different scope** - Callbacks for variable views, not evaluate results
3. **No connection** - Evaluate results aren't connected to PETSc
4. **Wrong abstraction** - Units are metadata, not synchronization events

## The Bottom Line

**We need THREE things**:
1. **UnitAwareMixin** - Operational unit system for variables (persistent source of truth)
2. **UnitAwareArray** - Informational unit metadata for evaluate results (transient copy)
3. **NDArray_With_Callback** - PETSc synchronization for variable arrays (unrelated to units)

These are orthogonal concerns serving different needs. You couldn't replace any one with another without breaking functionality.
