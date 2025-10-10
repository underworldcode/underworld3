# Hybrid SymPy+Pint Architecture: Solutions to Original Questions

This document addresses each question from our conversation using the hybrid SymPy+Pint approach.

## Original Questions and Hybrid Solutions

### **A) Flexible Reference Quantities (Your Point)**

**Problem**: Hard-coding "viscosity" categories is too restrictive for different domains.

**Hybrid Solution**: ✅ **PERFECT FIT**
```python
# Works for any domain - user provides what matters to them
mantle_model.set_reference_quantities(
    mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
    plate_velocity=5*uw.units.cm/uw.units.year,
    lithosphere_thickness=100*uw.units.km
)

ice_model.set_reference_quantities(
    ice_viscosity=1e13*uw.units.Pa*uw.units.s,
    glacier_velocity=100*uw.units.m/uw.units.year,
    ice_thickness=1*uw.units.km
)

# System parses Pint dimensionalities, converts to SymPy, solves constraints
```

**Benefits**:
- User-friendly Pint input
- Automatic dimensional analysis
- Domain-agnostic approach

---

### **1) Variable Creation After Units Set**

**Question**: Should `uw.MeshVariable("u", mesh)` automatically infer it's velocity?

**Your Answer**: Explicitly require it - can't rely on names.

**Hybrid Solution**: ✅ **ENFORCED**
```python
# After model commits to units, this is REQUIRED:
velocity = uw.MeshVariable("u", mesh, units="m/s")  # Must specify
pressure = uw.MeshVariable("p", mesh, units="Pa")   # Must specify

# RuntimeError if not provided:
velocity = uw.MeshVariable("u", mesh)  # Error: units required

# The variable creates SymPy expressions with units:
velocity.sym[0]  # Returns: u_0 * meter/second (SymPy expression)
```

**Benefits**:
- No name-based inference (unreliable)
- Forces explicit thinking about dimensions
- Works with any solver type (Stokes, elasticity, etc.)

---

### **2) Mixed Variable Types**

**Question**: How handle variables that don't fit standard categories?

**Your Answer**: Everything has units or None (dimensionless).

**Hybrid Solution**: ✅ **ELEGANT**
```python
# Physical quantities
velocity = uw.MeshVariable("u", mesh, units="m/s")
temperature = uw.MeshVariable("T", mesh, units="K")

# Dimensionless quantities
rayleigh_number = uw.MeshVariable("Ra", mesh, units=None)
custom_ratio = uw.MeshVariable("ratio", mesh, units=None)

# SymPy expressions handle this naturally:
# velocity.sym → u * meter/second
# rayleigh_number.sym → Ra (dimensionless)
```

**Benefits**:
- Clean binary choice: units or None
- Works in both dimensionless and units modes
- SymPy expressions automatically handle dimensionless terms

---

### **3) Expression Units**

**Question**: Should `velocity[0].diff(x)` know it has units of 1/time?

**Your Answer**: Yes, we'll have to address this.

**Hybrid Solution**: ✅ **NATIVE SUPPORT**
```python
# Variables create SymPy expressions with units
velocity = uw.MeshVariable("u", mesh, units="m/s")
x = mesh.X[0]  # Position coordinate with length units

# Automatic unit propagation:
strain_rate = velocity.sym[0].diff(x)
# Result: meter*Derivative(u_0, x)/second / meter = Derivative(u_0, x)/second
# Units: 1/second ✓

# Unit arithmetic in expressions:
stress = viscosity * strain_rate  # Automatically has Pa units
divergence = velocity.sym.div()   # Automatically has 1/second units
```

**Benefits**:
- SymPy natively handles unit arithmetic
- Automatic dimensional analysis
- Expression units flow naturally through calculations

---

### **4) Boundary Conditions**

**Question**: How handle `stokes.add_dirichlet_bc(5*uw.units.cm/uw.units.year, "Top")`?

**Your Answer**: Make this acceptable, but think about JIT compilation.

**Hybrid Solution**: ✅ **JIT COMPATIBLE**
```python
# User provides natural units (Pint or SymPy):
stokes.add_dirichlet_bc(5*uw.units.cm/uw.units.year, "Top")

# OR complex functions:
def inflow_velocity(x, y, t):
    return sp.sin(x) * 5*su.meter/(100*su.year)  # SymPy units

stokes.add_dirichlet_bc(inflow_velocity, "Top")

# unwrap() handles conversion BEFORE JIT:
def enhanced_unwrap(user_function):
    def compiled_function(*args):
        # Call user function (returns SymPy expression with units)
        result = user_function(*args)

        # Separate numerical and unit parts:
        numerical_expr, unit_scale = separate_units_for_jit(result)

        # Convert unit scale to model base units:
        scale_factor = convert_to_model_units(unit_scale)

        # Generate JIT code for numerical part:
        jit_code = ccode(numerical_expr)

        return f"({jit_code}) * {scale_factor}"

    return compiled_function
```

**Benefits**:
- User writes in natural units
- Conversion happens in unwrap (isolated, testable)
- JIT compiler sees dimensionless code
- No changes needed elsewhere in codebase

---

### **5) Serialization Format**

**Question**: Include units metadata in HDF5? Separate units file?

**Hybrid Solution**: ✅ **BOTH PINT + SYMPY METADATA**
```python
# Model serialization includes:
{
    "reference_quantities": {
        # Original Pint quantities for user reconstruction
        "mantle_viscosity": {"value": 1e21, "units": "pascal * second"},
        "plate_velocity": {"value": 5, "units": "centimeter / year"}
    },
    "sympy_scaling": {
        # Derived SymPy scaling for internal use
        "length_scale": {"value": 1e6, "units": "meter"},
        "time_scale": {"value": 3.16e13, "units": "second"},
        "mass_scale": {"value": 3.16e40, "units": "kilogram"}
    },
    "units_committed": True
}

# Variable metadata:
velocity_metadata = {
    "units": "m/s",
    "sympy_expression": "u_0 * meter/second"
}
```

**Benefits**:
- Complete reconstruction capability
- Both user-friendly (Pint) and internal (SymPy) formats
- Version compatibility tracking

---

## Additional Advantages of Hybrid Approach

### **Performance** (Your Key Concern)
```python
# Fast array access (unchanged):
velocity.array[...] = values  # Direct numpy, no units overhead

# Explicit conversion only when needed:
vel_ms = velocity.to_array("m/s")  # Convert for analysis
```

### **Backwards Compatibility** (Your Requirement)
```python
# Existing dimensionless code works unchanged:
model = uw.Model()  # No units commitment
velocity = uw.MeshVariable("u", mesh)  # No units= required
velocity.array[...] = 0.05  # Plain numbers, works as before
```

### **JIT Isolation** (Your Strategy)
```python
# All unit handling isolated in unwrap():
# - Testable independently
# - No changes to solver code
# - Verifiable conversion accuracy
```

## Summary: Perfect Alignment

The hybrid SymPy+Pint approach addresses **every single concern** from our conversation:

| Original Concern | Hybrid Solution |
|------------------|-----------------|
| **Flexible categories** | ✅ Reference quantities, not hardcoded |
| **Performance** | ✅ Array access unchanged |
| **Expression units** | ✅ SymPy native support |
| **JIT compatibility** | ✅ Separable parts in unwrap |
| **Backwards compatibility** | ✅ Dimensionless mode preserved |
| **User experience** | ✅ Pint input, SymPy expressions |
| **Boundary conditions** | ✅ Natural units, unwrap converts |
| **Serialization** | ✅ Both Pint + SymPy metadata |

This architecture gives us the **best of both worlds** while solving all the fundamental integration challenges we identified.