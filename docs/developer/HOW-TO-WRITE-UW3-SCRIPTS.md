# How to Write Underworld3 Scripts

**Last Updated**: 2025-11-15
**Status**: Living Document - Updated as new patterns emerge

This guide captures critical lessons learned from writing and debugging Underworld3 tests and scripts. It serves as a reference for common patterns, pitfalls, and best practices.

---

## Table of Contents

1. [Critical Ordering Rules](#critical-ordering-rules)
2. [Swarm Usage Patterns](#swarm-usage-patterns)
3. [Units System Integration](#units-system-integration)
4. [Mesh and Variable Creation](#mesh-and-variable-creation)
5. [Solver Setup and Execution](#solver-setup-and-execution)
6. [Common Pitfalls and Anti-Patterns](#common-pitfalls-and-anti-patterns)
7. [Testing Best Practices](#testing-best-practices)
8. [Debugging Techniques](#debugging-techniques)

---

## Critical Ordering Rules

### ⚠️ RULE #1: Swarm Variables Before Population

**THE MOST COMMON ERROR**: Creating swarm variables AFTER populating the swarm.

```python
# ❌ WRONG - Will cause PETSc DM structure errors
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)  # ERROR: No variables defined yet!
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1)  # Too late!

# ✅ CORRECT - Variables must exist before population
swarm = uw.swarm.Swarm(mesh)
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1)  # Create first
v_var = uw.swarm.SwarmVariable("vector", swarm, 2)  # All variables
swarm.populate(fill_param=3)  # Now populate
```

**Why this matters:**
- `populate()` and `add_particles_with_coordinates()` set up the PETSc DM structure
- The DM must know about all fields (variables) before particle allocation
- Creating variables after population causes DM inconsistencies and cryptic PETSc errors

**Error symptoms:**
- `petsc4py.PETSc.Error: error code 63`
- `PETSC ERROR: Argument out of range`
- Long comma-separated coordinate strings in error messages

**Applies to:**
- `swarm.populate()`
- `swarm.add_particles_with_coordinates()`
- Any operation that adds particles to the swarm

### RULE #2: Reference Quantities Before Mesh Creation

When using units, set reference quantities BEFORE creating meshes:

```python
# ✅ CORRECT - Reference quantities first
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),
    material_density=uw.quantity(3300, "kg/m**3"),
)
mesh = uw.meshing.StructuredQuadBox(...)  # Mesh inherits reference quantities

# ❌ WRONG - Setting after mesh creation won't apply to existing mesh
mesh = uw.meshing.StructuredQuadBox(...)
model.set_reference_quantities(...)  # Too late for this mesh!
```

**Why this matters:**
- Reference quantities are immutable after mesh creation
- Coordinate units are established during mesh initialization
- Late setting won't retroactively apply to existing objects

### RULE #3: Model Reset Between Tests

```python
# ✅ CORRECT - Reset at start of each test
def test_something_with_units():
    uw.reset_default_model()  # Clean slate
    model = uw.get_default_model()
    model.set_reference_quantities(...)
    # ... rest of test

# ❌ WRONG - Reusing model from previous test
def test_something_with_units():
    model = uw.get_default_model()  # May have stale state!
    # ... rest of test
```

**Why this matters:**
- Tests can pollute each other's model state
- Strict units mode and reference quantities persist across tests
- Prevents mysterious failures from test ordering dependencies

---

## Swarm Usage Patterns

### Basic Swarm Creation

```python
# Complete swarm setup pattern
mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
swarm = uw.swarm.Swarm(mesh)

# Create ALL variables BEFORE populating
scalar_var = uw.swarm.SwarmVariable("material", swarm, 1)
vector_var = uw.swarm.SwarmVariable("velocity", swarm, 2)

# Create proxy variables if needed for integration/derivatives
proxy_var = uw.swarm.SwarmVariable("proxy", swarm, 1, proxy_degree=2)

# NOW populate the swarm
swarm.populate(fill_param=3)  # Uniform layout-based population

# OR use specific coordinates
coords = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
swarm.add_particles_with_coordinates(coords)
```

### Proxy Variables for Integration

When you need to integrate swarm data or compute derivatives:

```python
# Create proxy variable during initialization
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)

# Populate swarm
swarm.populate(fill_param=3)

# Set swarm data
x_coords = swarm._particle_coordinates.data[:, 0]
s_var.data[:, 0] = 2.0 + x_coords

# Use symbolic representation for integration
I_f = uw.maths.Integral(mesh, fn=s_var.sym[0])  # Integrates proxy
result = I_f.evaluate()
```

**Key points:**
- Proxy variables create mesh-based RBF interpolations of swarm data
- Required for integrals (can't integrate point data directly)
- Required for derivatives (need continuous field representation)
- `proxy_degree` controls RBF interpolation quality

### Swarm Data Access Patterns

```python
# ✅ CORRECT - Direct array access
scalar_var.data[:, 0] = values  # Single variable, automatic sync

# ✅ CORRECT - Batch updates for multiple variables
with uw.synchronised_array_update():
    scalar_var.data[:, 0] = values1
    vector_var.data[:, 0] = values2
    vector_var.data[:, 1] = values3

# ❌ AVOID - Old access context pattern (legacy)
with swarm.access(scalar_var):
    scalar_var.data[:, 0] = values  # Unnecessary, use direct access
```

---

## Units System Integration

### Units Everywhere or Nowhere Principle

**Core Principle**: Either ALL quantities have units (when reference quantities set) OR all are plain numbers (when not set). No mixing.

```python
# Mode 1: Units Everywhere
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),  # Provides [M] dimension
)

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.f = uw.quantity(2.0, "K")  # ✓ Units required
poisson.f = 2.0  # ✗ ERROR - plain number not allowed

# Mode 2: Plain Numbers Everywhere
uw.reset_default_model()
uw.use_nondimensional_scaling(False)

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)  # No units

poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.f = 2.0  # ✓ Plain number allowed
```

### Understanding [M] Dimension Requirements

If you use quantities with mass dimension (pressure, stress, viscosity), you MUST provide [M]:

```python
# ✅ CORRECT - Complete dimensional specification
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),         # [L]
    material_density=uw.quantity(3300, "kg/m**3"),  # [M]
)

# Create pressure variable (has [M L⁻¹ T⁻²] dimensions)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # Works!

# ❌ WRONG - Missing [M] dimension
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),  # Only [L], no [M]
)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # ERROR at use-time!
```

**How to provide [M]:**
1. Explicit: `material_density=uw.quantity(3300, "kg/m**3")`
2. Via proxy: `mantle_viscosity=uw.quantity(1e21, "Pa*s")` (Pa contains kg)
3. Direct scale: `pressure_scale=uw.quantity(1e7, "Pa")` (no dimensional analysis)

### Checking Units Mode

```python
model = uw.get_default_model()

if model.has_units():
    # Units mode - all quantities need units
    value = uw.quantity(100, "m")
else:
    # Plain numbers mode
    value = 100
```

### Unit-Aware Variable Creation

```python
# With reference quantities set:
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2, units="m/s")

# Coordinates automatically get units from reference quantities
x, y = mesh.X  # These have units from domain_depth
u = uw.get_units(x)  # Returns 'kilometer' if domain_depth in km

# Symbolic operations preserve units
grad_T = T.sym.diff(y)  # Has units: kelvin / kilometer
flux = grad_T * v.sym[1]  # Units: kelvin * m / (s * km)
```

---

## Mesh and Variable Creation

### Basic Mesh Creation

```python
# 2D structured mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
)

# With units (requires reference quantities set first!)
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(uw.quantity(1000, "km"), uw.quantity(500, "km")),
    units="kilometer",  # Explicit coordinate units
)

# 3D structured mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(8, 8, 8),
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
)
```

### Variable Creation Best Practices

```python
# Scalar variable
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Vector variable (2D)
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)

# Vector variable (3D)
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)

# Pressure (typically degree=1 for Stokes)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# With units
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2, units="m/s")

# Private variables (not saved/loaded with model)
temp_var = uw.discretisation.MeshVariable("_temp", mesh, 1, _register=False)
```

### Variable Naming Convention

```python
# ✅ GOOD - Clear, descriptive names
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# ⚠️ AVOID - Ambiguous 'model' variable name
# 'model' is ambiguous: uw.Model vs constitutive models
# Use specific names:
constitutive_model = stokes.constitutive_model  # ✓
viscous_model = poisson.constitutive_model  # ✓
model = stokes.constitutive_model  # ✗ Ambiguous
```

### Direct Data Access (Modern Pattern)

```python
# ✅ MODERN - Direct array access
var.array[..., 0] = values  # Single variable

# ✅ MODERN - Batch updates
with uw.synchronised_array_update():
    var1.array[..., 0] = values1
    var2.array[..., 0] = values2

# ❌ LEGACY - Old access context (still works but not needed)
with mesh.access(var):
    var.data[...] = values
```

---

## Solver Setup and Execution

### Poisson Solver

```python
# Basic setup
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)

poisson = uw.systems.Poisson(mesh, u_Field=u)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0

# Source term (respects units mode!)
if model.has_units():
    poisson.f = uw.quantity(1.0, "appropriate_units")
else:
    poisson.f = 1.0

# Boundary conditions
poisson.add_dirichlet_bc(T_bottom, "Bottom")
poisson.add_dirichlet_bc(T_top, "Top")

# Solve
poisson.solve()
```

### Stokes Solver

```python
# Create variables BEFORE solver
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# Setup solver
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

# Boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, 0.0), "Top")

# Solve
stokes.solve()

# Access results
velocity_data = v.array
pressure_data = p.array
```

### Constitutive Model Access

```python
# ✅ CORRECT - Use 'constitutive_model' variable name
constitutive_model = stokes.constitutive_model
constitutive_model.Parameters.viscosity = 1e21

# ✅ CORRECT - Inline access is fine too
stokes.constitutive_model.Parameters.viscosity = 1e21

# ⚠️ AVOID - Ambiguous 'model' variable name
model = stokes.constitutive_model  # Confusing with uw.Model
```

---

## Common Pitfalls and Anti-Patterns

### ❌ Swarm Variable Creation After Population

```python
# ❌ WRONG - Most common error!
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)
var = uw.swarm.SwarmVariable("s", swarm, 1)  # ERROR!

# ✅ CORRECT
swarm = uw.swarm.Swarm(mesh)
var = uw.swarm.SwarmVariable("s", swarm, 1)
swarm.populate(fill_param=3)
```

### ❌ Mixing Units and Plain Numbers

```python
# ❌ WRONG - Inconsistent unit usage
model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
poisson.f = 2.0  # ERROR: Units required when reference quantities set

# ✅ CORRECT - Consistent approach
model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
poisson.f = uw.quantity(2.0, "kelvin")
```

### ❌ Missing Reference Quantities for [M] Dimension

```python
# ❌ WRONG - Incomplete dimensional specification
model.set_reference_quantities(domain_depth=uw.quantity(500, "m"))
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # Needs [M]!

# ✅ CORRECT - Complete specification
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),
    material_density=uw.quantity(3300, "kg/m**3"),
)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")
```

### ❌ Batman Pattern (Anti-Pattern - DO NOT USE)

**The Batman Pattern**: Declaring all variables upfront before any computational work, "just in case."

```python
# ❌ BATMAN PATTERN - DO NOT DO THIS
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2)
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)  # Don't need yet!
flux = uw.discretisation.MeshVariable('flux', mesh, mesh.dim, degree=1)  # Don't need yet!
# ... 20 more variables you MIGHT need later

# Solve the actual problem (far from declarations)
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# Use pre-declared variables
proj = uw.systems.Projection(mesh, gradT, ...)
proj.solve()
```

**Why this is wrong:**
- Unnatural workflow - requires predicting all future needs
- Poor software design - violates principle of locality
- Breaks exploratory analysis - can't create variables on demand
- **WAS** required due to old DM state corruption bug
- **NOW FIXED** (2025-10-14) - variables can be created anytime

```python
# ✅ CORRECT - Natural workflow
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2)

# Solve
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# Create gradient variable WHEN YOU NEED IT
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)
proj = uw.systems.Projection(mesh, gradT, ...)
proj.solve()
```

**See**: `CLAUDE.md` "NO BATMAN" section for full history

### ❌ Not Resetting Model Between Tests

```python
# ❌ WRONG - Tests can interfere
def test_with_units():
    model = uw.get_default_model()  # May have state from previous test!
    # ...

# ✅ CORRECT - Clean slate each test
def test_with_units():
    uw.reset_default_model()
    model = uw.get_default_model()
    # ...
```

---

## Testing Best Practices

### Test Structure Template

```python
import pytest
import underworld3 as uw
import numpy as np

@pytest.mark.level_2  # Complexity: 1=quick, 2=intermediate, 3=physics
@pytest.mark.tier_a   # Reliability: a=production, b=validated, c=experimental
def test_descriptive_name():
    """
    Clear docstring explaining what is being tested and why.

    Include any important context about expected behavior.
    """
    # ALWAYS reset model at start of test
    uw.reset_default_model()

    # Setup with units if needed
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "m"),
        material_density=uw.quantity(3300, "kg/m**3"),
    )

    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))

    # Create variables
    var = uw.discretisation.MeshVariable("var", mesh, 1, degree=2, units="kelvin")

    # Test operations
    # ...

    # Assertions with clear failure messages
    assert result == expected, f"Expected {expected}, got {result}"
```

### Test Markers Usage

```python
# Level markers (complexity/runtime)
@pytest.mark.level_1  # Quick - imports, setup, no solving (~seconds)
@pytest.mark.level_2  # Intermediate - integration, simple solves (~minutes)
@pytest.mark.level_3  # Physics - benchmarks, complex solves (~minutes to hours)

# Tier markers (reliability/trust)
@pytest.mark.tier_a   # Production-ready - trusted for TDD, CI
@pytest.mark.tier_b   # Validated - use with caution, needs more testing
@pytest.mark.tier_c   # Experimental - development only, not for automation

# Expected failures
@pytest.mark.xfail(reason="Clear explanation of why this fails")

# Multiple markers
@pytest.mark.level_2
@pytest.mark.tier_b
@pytest.mark.xfail(reason="Unit-aware derivative bug: UnitAwareDerivativeMatrix * NegativeOne")
def test_something():
    pass
```

### Testing Swarms (Critical!)

```python
def test_swarm_functionality():
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    swarm = uw.swarm.Swarm(mesh)

    # ⚠️ CRITICAL: Create ALL variables BEFORE populating!
    scalar_var = uw.swarm.SwarmVariable("scalar", swarm, 1)
    vector_var = uw.swarm.SwarmVariable("vector", swarm, 2)

    # NOW populate
    swarm.populate(fill_param=3)

    # Set data and test
    scalar_var.data[:, 0] = 1.0
    assert scalar_var.array.mean() == pytest.approx(1.0)
```

---

## Debugging Techniques

### JIT Compilation Issues

If you see generated C code with symbolic expressions instead of numbers:

```c
// ERROR symptom in generated code:
out[0] = 1.0/{ \eta \hspace{ 0.0006pt } };  // Should be numeric!
```

**Cause**: `unwrap(fn, keep_constants=False)` not properly unwrapping constants.

**Solution**: Check that constants (like UWQuantity) are being unwrapped to numeric values.

### PETSc DM Errors with Swarms

Error pattern:
```
[0]PETSC ERROR: Argument out of range
[0]PETSC ERROR: Input string 0.5488135039273248,0.7151893663724195,...
```

**Cause**: Swarm variables created AFTER `populate()` or `add_particles_with_coordinates()`.

**Solution**: Create ALL swarm variables BEFORE adding particles.

### Units Dimension Errors

Error pattern:
```
ValueError: Cannot create variable with units when model has no reference quantities
```

**Cause**: Missing reference quantities, or trying to use [M] dimension without providing [M].

**Solution**:
1. Set reference quantities before mesh creation
2. Ensure [M] dimension available if using pressure/stress/viscosity units

### Test State Pollution

Symptom: Test passes in isolation but fails in suite, or vice versa.

**Cause**: Model state persisting between tests.

**Solution**: ALWAYS call `uw.reset_default_model()` at start of each test.

### Derivative Units Bugs

Error pattern:
```
TypeError: unsupported operand type(s) for *: 'UnitAwareDerivativeMatrix' and 'NegativeOne'
```

**Cause**: Unit-aware derivative arithmetic not fully implemented.

**Status**: Known bug as of 2025-11-15, mark tests with xfail.

---

## Quick Reference Checklist

### Starting a New Script

- [ ] Import underworld3: `import underworld3 as uw`
- [ ] Decide units mode: With reference quantities or plain numbers?
- [ ] If using units: Set reference quantities BEFORE mesh creation
- [ ] If using units with [M]: Provide material_density or equivalent

### Creating a Swarm

- [ ] Create mesh first
- [ ] Create swarm: `swarm = uw.swarm.Swarm(mesh)`
- [ ] Create ALL swarm variables (regular + proxy)
- [ ] THEN populate: `swarm.populate()` or `add_particles_with_coordinates()`
- [ ] Set swarm data
- [ ] Use `.sym` for symbolic operations (integration, derivatives)

### Writing a Test

- [ ] Add `@pytest.mark.level_N` marker (1, 2, or 3)
- [ ] Add `@pytest.mark.tier_X` marker (a, b, or c)
- [ ] Start with `uw.reset_default_model()`
- [ ] Set reference quantities if using units
- [ ] Follow correct ordering (mesh → variables → populate for swarms)
- [ ] Clear docstring explaining what's being tested
- [ ] Meaningful assertion messages

### Setting Up Units

- [ ] Call `uw.reset_default_model()` for clean state
- [ ] Get model: `model = uw.get_default_model()`
- [ ] Set reference quantities: `model.set_reference_quantities(...)`
- [ ] Include [M] dimension if needed (density, viscosity, or pressure_scale)
- [ ] Create mesh (coordinates inherit units)
- [ ] Create variables with `units=` parameter
- [ ] Use `uw.quantity()` for all numerical values in solver parameters

---

## Version History

- **2025-11-15**: Initial version
  - Swarm ordering rules from test_0850/0851 debugging
  - Units everywhere-or-nowhere principle
  - [M] dimension requirements
  - Batman pattern documentation
  - Test classification system integration
  - JIT unwrapping debugging patterns

---

## See Also

- `CLAUDE.md`: Project status, coding conventions, "NO BATMAN" section
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md`: Test tier classification
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`: Coordinate units implementation
- `docs/beginner/tutorials/12-Units_System.ipynb`: Units system tutorial
- `docs/beginner/tutorials/13-Non_Dimensional_Scaling.ipynb`: Dimensional analysis
