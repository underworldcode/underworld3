# Final Units Architecture Recommendation

## Summary

After extensive exploration, the **Hybrid SymPy+Pint approach** emerges as the clear winner for Underworld3 units integration. This approach addresses every concern from our conversation while providing a clean, performant, and backwards-compatible solution.

## Key Findings

### ✅ **SymPy Units Solve the Fundamental Problem**

The core insight: **SymPy expressions can natively carry units**, eliminating the wrapper/conversion complexity that makes Pint-only approaches problematic.

```python
# The magic: SymPy expressions naturally include units
velocity_expr = velocity.sym[0] * su.meter / su.second
strain_rate = sp.diff(velocity_expr, x)  # Automatically has units 1/s
stress = viscosity * strain_rate         # Automatically has units Pa
```

### ✅ **Pint Provides User Experience**

Pint excels at user-friendly input and conversion:

```python
# User provides familiar Pint quantities
model.set_reference_quantities(
    mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
    plate_velocity=5*uw.units.cm/uw.units.year
)
```

### ✅ **JIT Compatibility via Separation**

SymPy expressions can be separated into numerical and unit parts for code generation:

```python
# Expression: 1.0e+21*kilogram*Derivative(u_x, x)/second**2
numerical_part = Derivative(u_x, x)  # For JIT compilation
unit_scale = 1.0e+21*kilogram/second**2  # Convert to model units
```

## Architecture Overview

```
User Input (Pint) → Internal (SymPy) → JIT (Dimensionless)
     ↓                   ↓                  ↓
Familiar units    Native expressions   Fast code
Easy conversion   Unit arithmetic      No overhead
```

## Design Decisions Addressed

### **1. Array Access Performance** ✅
```python
velocity.array[...] = values  # Fast numpy, no units overhead
velocity.to_array("m/s")      # Explicit conversion when needed
```

### **2. Dimensionless by Default** ✅
```python
# Current behavior preserved
model = uw.Model()  # No units
velocity = uw.MeshVariable("u", mesh)  # Works as before

# Units are lifetime commitment
model.set_reference_quantities(...)  # Point of no return
```

### **3. Explicit Units Required** ✅
```python
# After units commitment, explicit specification required
velocity = uw.MeshVariable("u", mesh, units="m/s")  # Must specify
rayleigh = uw.MeshVariable("Ra", mesh, units=None)  # Dimensionless OK
```

### **4. Flexible Reference Quantities** ✅
```python
# Domain-agnostic, user specifies what matters
mantle_model.set_reference_quantities(
    mantle_viscosity=1e21*Pa*s, plate_velocity=5*cm/year
)
ice_model.set_reference_quantities(
    ice_viscosity=1e13*Pa*s, glacier_velocity=100*m/year
)
```

### **5. JIT Boundary Conditions** ✅
```python
# User writes natural functions
def inflow(x, y, t):
    return sp.sin(x) * 5*su.meter/(100*su.year)

# unwrap() handles unit separation before compilation
stokes.add_dirichlet_bc(inflow, "Top")
```

## Implementation Strategy

### **Phase 1: Foundation**
1. **Pint ↔ SymPy conversion utilities**
2. **Model.set_reference_quantities()** method
3. **Enhanced MeshVariable** with units awareness
4. **Basic expression unit handling**

### **Phase 2: Integration**
1. **Modified unwrap()** for unit separation
2. **JIT code generation** with unit scaling
3. **Serialization** with units metadata
4. **Testing framework** for unit validation

### **Phase 3: Polish**
1. **Error handling** and user guidance
2. **Performance optimization**
3. **Documentation** and examples
4. **Migration tools** for existing code

## Benefits Summary

| Aspect | Hybrid Approach |
|--------|-----------------|
| **Expression Integration** | ✅ Native SymPy support |
| **User Experience** | ✅ Familiar Pint input |
| **Performance** | ✅ Array access unchanged |
| **JIT Compatibility** | ✅ Separable components |
| **Backwards Compatibility** | ✅ Dimensionless mode preserved |
| **Domain Flexibility** | ✅ Reference quantities approach |
| **Error Prevention** | ✅ Dimensional analysis |
| **Serialization** | ✅ Complete metadata |

## Code Examples

### **Complete User Workflow**
```python
# 1. Set up problem with natural units
model = uw.Model()
model.set_reference_quantities(
    mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
    plate_velocity=5*uw.units.cm/uw.units.year,
    domain_depth=3000*uw.units.km
)

# 2. Create variables (units required)
mesh = uw.meshing.Box(elementRes=(32, 32))
velocity = uw.MeshVariable("u", mesh, units="m/s")
pressure = uw.MeshVariable("p", mesh, units="Pa")
temperature = uw.MeshVariable("T", mesh, units="K")

# 3. Set up physics with unit-aware expressions
stokes = uw.systems.Stokes(mesh, velocity, pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Boundary conditions in natural units
def plate_velocity_bc(x, y, t):
    return 5*su.meter/(100*su.year) * sp.tanh(x/100)

stokes.add_dirichlet_bc(plate_velocity_bc, "Top")

# 4. Solve (automatic unit handling)
stokes.solve()

# 5. Analyze results in any units
vel_cmyr = velocity.to_array("cm/year")
temp_celsius = temperature.to_array("celsius")

plt.quiver(mesh.points.to_array("km")[:, 0],
           mesh.points.to_array("km")[:, 1],
           vel_cmyr[:, 0], vel_cmyr[:, 1])
```

### **Expression Units Flow Naturally**
```python
# Variables automatically create unit-aware SymPy expressions
strain_rate = velocity.sym[0].diff(x)        # Has units 1/s
stress = viscosity_field * strain_rate       # Has units Pa
heat_production = stress * strain_rate       # Has units Pa/s
buoyancy = thermal_expansion * density * gravity * temperature  # Units kg/m³
```

## Conclusion

The hybrid SymPy+Pint architecture provides:

1. **Complete solution** to all identified problems
2. **Performance preservation** through smart design
3. **Natural user experience** with familiar Pint input
4. **Robust implementation** using proven technologies
5. **Future-proof design** that scales across domains

This approach transforms Underworld3 from a dimensionless system to a fully units-aware platform while maintaining all the performance and compatibility benefits users expect.

## Next Steps

1. **Review this architecture** for any remaining concerns
2. **Begin implementation** starting with Phase 1 components
3. **Create test cases** validating all design decisions
4. **Develop migration strategy** for existing users

The foundation is solid - we now have a clear path to implementation.