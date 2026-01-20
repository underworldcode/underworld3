# Underworld3 Script Writing Cheat Sheet

**Quick reference for common UW3 patterns - use this to avoid repeated mistakes!**

---

## ⚠️ CRITICAL: Constitutive Model Instantiation

### ✅ CORRECT Pattern
```python
# Assign the CLASS itself (not instantiated!)
solver.constitutive_model = uw.constitutive_models.DiffusionModel
solver.constitutive_model.Parameters.diffusivity = 1.0

# For Stokes
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1e21
```

### ❌ WRONG Pattern
```python
# DO NOT instantiate with arguments!
solver.constitutive_model = uw.constitutive_models.DiffusionModel(mesh.dim)  # ✗ WRONG
solver.constitutive_model = uw.constitutive_models.ViscousFlowModel()        # ✗ WRONG
```

**Why**: The solver framework handles instantiation internally. You assign the CLASS, then set parameters.

---

## Mesh Creation

### ⚠️ IMPORTANT: Prefer Simplex Meshes

**Quadrilateral elements can be problematic** (especially with `evaluate()` and `global_evaluate()`).
**Prefer simplex (triangular/tetrahedral) meshes for robust performance.**

### ✅ PREFERRED: Unstructured Simplex Box
```python
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
    regular=False  # Use regular=True for structured triangulation
)
```

### ⚠️ Use with Caution: Structured Quad Box
```python
# Quadrilateral elements - can have issues with evaluate/global_evaluate
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0)
)
```

---

## Mesh Variables

### Scalar Field
```python
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[...] = 1.0  # Direct assignment
```

### Vector Field
```python
velocity = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
velocity.array[...] = 0.0
```

### With Units
```python
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="K")
```

---

## Poisson Solver

### Basic Setup
```python
poisson = uw.systems.Poisson(mesh, u_Field=T)

# Constitutive model (diffusivity)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0

# Source term
poisson.f = 1.0

# Solve
poisson.solve()
```

---

## Stokes Solver

### Basic Setup
```python
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# Constitutive model (viscosity)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

# Body force
stokes.bodyforce = sympy.Matrix([0, -1])

# Boundary conditions
stokes.add_dirichlet_bc((0.0,), "Bottom", (0, 1))

# Solve
stokes.solve()
```

---

## Advection-Diffusion

### Basic Setup
```python
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=velocity.sym,
    solver_name="adv_diff"
)

# Constitutive model (diffusivity)
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = kappa

# Source term
adv_diff.f = 0.0

# Solve with timestep
dt = 0.01
adv_diff.solve(timestep=dt)
```

---

## Swarms and Particle Tracking

### Create Swarm
```python
swarm = uw.swarm.Swarm(mesh)
material = uw.swarm.SwarmVariable("M", swarm, size=1, proxy_degree=0, dtype="int")

# Populate
swarm.populate(fill_param=4)
```

### Advect Particles
```python
# Update particle positions
swarm.advection(v_uw=velocity.sym, delta_t=dt, corrector=False)
```

---

## Function Evaluation

### At Specific Points
```python
import numpy as np

coords = np.array([[0.5, 0.5], [0.25, 0.25]])
result = uw.function.evaluate(T.sym, coords, rbf=False)
```

### On Mesh Coordinates
```python
# Dimensional coordinates
result = uw.function.evaluate(T.sym, mesh.X.coords, rbf=False)

# Non-dimensional coordinates
result = uw.function.evaluate(T.sym, mesh.data[:, :mesh.dim], rbf=False)
```

---

## Boundary Conditions

### Dirichlet BC
```python
# Scalar field
poisson.add_dirichlet_bc(1.0, "Top")
poisson.add_dirichlet_bc(0.0, "Bottom")

# Vector field - specific components
stokes.add_dirichlet_bc((0.0,), "Left", (0,))      # x-component only
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom", (0, 1))  # both components
```

### Natural BC (Neumann)
```python
# Flux boundary condition
poisson.add_natural_bc(-1.0, "Right")  # Outward flux
```

---

## Units System

### Setting Reference Quantities
```python
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s")
)
```

### Using Units in Code
```python
# Create variable with units
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="K")

# Set values with units
poisson.f = uw.quantity(2.0, "K")

# Non-dimensionalize
value_nd = uw.non_dimensionalise(uw.quantity(1500, "K"))

# Dimensionalize
value_dim = uw.dimensionalise(0.5, "temperature")
```

---

## Data Access Patterns

### Single Variable
```python
# Direct assignment
var.array[...] = values
```

### Multiple Variables (Batch Operations)
```python
with uw.synchronised_array_update():
    var1.array[...] = values1
    var2.array[...] = values2
    var3.array[...] = values3
```

---

## Timing and Profiling

### Enable Timing
```python
import underworld3 as uw

uw.timing.start()

# ... run simulation ...

uw.timing.print_table()  # Show results
```

### Decorator for Custom Functions
```python
@uw.timing.routine_timer_decorator
def my_expensive_function():
    # ... computation ...
    pass
```

---

## Symbolic Expressions

### Using Mesh Coordinates
```python
x, y = mesh.X  # Coordinate symbols

# Define spatially-varying source term
poisson.f = sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)
```

### Using Variable Symbols
```python
# Temperature-dependent viscosity
eta = 1.0 * sympy.exp(-T.sym / 1000.0)
stokes.constitutive_model.Parameters.viscosity = eta
```

---

## Common Mistakes to Avoid

### 1. Constitutive Model Instantiation
❌ `solver.constitutive_model = uw.constitutive_models.DiffusionModel(mesh.dim)`
✅ `solver.constitutive_model = uw.constitutive_models.DiffusionModel`

### 2. Mesh Element Type
❌ `mesh = uw.meshing.StructuredQuadBox(...)`  # Quadrilateral - can be problematic
✅ `mesh = uw.meshing.UnstructuredSimplexBox(...)`  # Simplex - robust

**Why**: Quadrilateral elements can have issues with `evaluate()` and `global_evaluate()`. Prefer simplex meshes.

### 3. Variable Naming
❌ `model = stokes.constitutive_model`  # Ambiguous!
✅ `constitutive_model = stokes.constitutive_model`  # Clear

### 4. Access Contexts (Legacy)
❌ `with mesh.access(var): var.data[...] = values`  # Old pattern
✅ `var.array[...] = values`  # New pattern

### 5. Mesh Coordinates
❌ `mesh.data`  # Deprecated
✅ `mesh.X.coords`  # Current

### 6. Units Everywhere or Nowhere
When `model.has_units()` is True:
❌ `poisson.f = 2.0`  # Missing units
✅ `poisson.f = uw.quantity(2.0, "K")`  # With units

---

## Example: Complete Poisson Problem

```python
import underworld3 as uw
import numpy as np
import sympy

# Create mesh (use simplex for robustness!)
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.05,
    regular=True
)

# Create variable
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Create solver
poisson = uw.systems.Poisson(mesh, u_Field=T)

# Set constitutive model (ASSIGN CLASS, NOT INSTANCE!)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0

# Set source term
x, y = mesh.X
poisson.f = sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)

# Boundary conditions
poisson.add_dirichlet_bc(0.0, "Bottom")
poisson.add_dirichlet_bc(0.0, "Top")

# Solve
poisson.solve()

# Evaluate result
coords = np.array([[0.5, 0.5]])
result = uw.function.evaluate(T.sym, coords, rbf=False)
print(f"T at center: {result[0]}")
```

---

## Example: Complete Stokes Problem

```python
import underworld3 as uw
import sympy

# Create mesh (use simplex for robustness!)
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
    regular=True
)

# Create variables
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# Create solver
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# Set constitutive model (ASSIGN CLASS, NOT INSTANCE!)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

# Body force (buoyancy)
stokes.bodyforce = sympy.Matrix([0, -1])

# Boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))   # No horizontal velocity on left
stokes.add_dirichlet_bc((0.0,), "Right", (0,))  # No horizontal velocity on right
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,)) # No vertical velocity on bottom
stokes.add_dirichlet_bc((0.0,), "Top", (1,))    # No vertical velocity on top

# Solve
stokes.solve()

# Check velocity
print(f"Max velocity: {v.array.max()}")
```

---

**Remember**: When in doubt, check existing working examples in `docs/examples/` or tests!
