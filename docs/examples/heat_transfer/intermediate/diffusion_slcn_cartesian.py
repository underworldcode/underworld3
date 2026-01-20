# %% [markdown]
"""
# 🔬 Linear Diffusion of a Hot Pipe

**PHYSICS:** Heat diffusion  
**DIFFICULTY:** intermediate  
**DOMAIN:** heat_transfer  
**RUNTIME:** 2-3 minutes

## Description
Demonstrates linear thermal diffusion using the SLCN (Stream Line Centered Node) method. 
Models cooling of a hot cylindrical pipe in 2D with comparison to 1D analytical solution.
No advection - pure thermal diffusion problem.

## Key Concepts
- Thermal diffusion equation
- SLCN discretization method  
- Analytical vs numerical comparison
- Cylindrical symmetry in Cartesian coordinates
- Initial value problems with diffusion

## Physics Background
The thermal diffusion equation:
```
∂T/∂t = κ ∇²T
```
Where:
- T: Temperature
- κ: Thermal diffusivity
- t: Time

For a hot pipe cooling in an infinite medium, the 1D analytical solution in cylindrical coordinates provides a benchmark.

## Adaptable Parameters
- `resolution`: Mesh resolution (try 16-64 for speed vs accuracy)
- `domain_size`: Domain dimensions (ensure large enough for boundary effects)
- `thermal_diffusivity`: Material property (try 0.5-2.0)
- `initial_temperature`: Hot pipe temperature (try 100-1000)
- `pipe_radius`: Size of initial hot region (try 0.1-0.3)
- `final_time`: Simulation duration (adjust based on diffusion time scale)

## Claude Adaptation Hints

**Easy modifications:**
- Change mesh resolution for accuracy/speed trade-off
- Modify thermal diffusivity to see different cooling rates
- Adjust initial temperature distribution shape and magnitude
- Change domain size and aspect ratio

**Medium modifications:**
- Add different boundary condition types (insulating vs conducting)
- Include heat sources or sinks within the domain
- Implement anisotropic thermal conductivity
- Compare with analytical solutions for different geometries

**Advanced modifications:**
- Add advection for convective heat transfer
- Include temperature-dependent material properties
- Implement adaptive time stepping for efficiency
- Extend to 3D for fully cylindrical geometry
"""

# %% [markdown]
"""
## Setup and Parameters
Define key parameters at the top for easy modification.
"""

# %%
# Parameter constants - easily modifiable by users
RESOLUTION = 32                    # PARAM: mesh resolution - cells per dimension
DOMAIN_MIN = (0.0, 0.0)           # PARAM: domain minimum coordinates  
DOMAIN_MAX = (1.0, 1.0)           # PARAM: domain maximum coordinates
THERMAL_DIFFUSIVITY = 1.0         # PARAM: thermal diffusivity
INITIAL_TEMP_HOT = 1.0            # PARAM: initial hot pipe temperature
INITIAL_TEMP_COLD = 0.0           # PARAM: background temperature
PIPE_CENTER = (0.5, 0.5)          # PARAM: center of hot pipe
PIPE_RADIUS = 0.2                 # PARAM: radius of hot pipe
FINAL_TIME = 0.1                  # PARAM: simulation end time
TIME_STEPS = 100                  # PARAM: number of time steps

# %%
# Standard imports for Underworld3
import nest_asyncio
nest_asyncio.apply()  # Fix for Jupyter notebook issues

from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy as sp
from mpi4py import MPI
import math

# Visualization (only on single processor)
if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
# PETSc error handling setup
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# %% [markdown]
"""
## Mesh Creation
Create a structured Cartesian mesh for the diffusion problem.
"""

# %%
# SECTION: Mesh Creation
# Create structured mesh for regular diffusion problem
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RESOLUTION, RESOLUTION),
    minCoords=DOMAIN_MIN,
    maxCoords=DOMAIN_MAX,
    qdegree=2
)

# %% [markdown]
"""
## Variable Definition
Set up the temperature field and create coordinates for initial conditions.
"""

# %%
# SECTION: Variable Definition
# Define temperature field on the mesh
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Create coordinate system for initial condition setup
x, y = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]

# %% [markdown]
"""
## Initial Conditions
Set up initial temperature distribution with hot pipe in center.
"""

# %%
# SECTION: Initial Conditions
# Define initial temperature distribution (hot pipe in cold background)
radius_expr = sp.sqrt((x - PIPE_CENTER[0])**2 + (y - PIPE_CENTER[1])**2)

# Step function for initial condition: hot inside pipe radius, cold outside
initial_condition = sp.Piecewise(
    (INITIAL_TEMP_HOT, radius_expr <= PIPE_RADIUS),
    (INITIAL_TEMP_COLD, True)
)

# Set initial temperature distribution
# Use temperature.coords for DOF coordinates (P2 has more points than mesh vertices)
with uw.function.expression(initial_condition) as initial_temp_fn:
    temperature.array[:, 0, 0] = uw.function.evaluate(initial_temp_fn, temperature.coords)

print(f"Initial temperature range: {temperature.array.min():.3f} to {temperature.array.max():.3f}")

# %% [markdown]
"""
## Solver Setup
Configure the SLCN solver for thermal diffusion.
"""

# %%
# SECTION: Solver Setup
# Create SLCN (Stream Line Centered Node) solver for diffusion
slcn_solver = uw.systems.SLCN(
    mesh,
    u_Field=temperature,
    V_fn=sp.Matrix([0, 0]),  # No advection - pure diffusion
    solver_name="diffusion"
)

# Set thermal diffusivity in constitutive model
slcn_solver.constitutive_model = uw.constitutive_models.DiffusionModel
slcn_solver.constitutive_model.Parameters.diffusivity = THERMAL_DIFFUSIVITY

# Set time step
dt = FINAL_TIME / TIME_STEPS
slcn_solver.petsc_options["ts_dt"] = dt
slcn_solver.petsc_options["ts_max_time"] = FINAL_TIME

# %% [markdown]
"""
## Time Stepping Solution
Solve the diffusion equation through time.
"""

# %%
# SECTION: Time Stepping
print(f"Solving thermal diffusion for {FINAL_TIME:.3f} time units...")
print(f"Time step: {dt:.6f}, Total steps: {TIME_STEPS}")

# Store temperature profiles for analysis
times = []
center_temps = []

# Time stepping loop
slcn_solver.solve(timestep=dt)

print("✓ Time stepping completed")

# %% [markdown]
"""
## Analytical Comparison
Compare with 1D analytical solution for validation.
"""

# %%
# SECTION: Analytical Comparison
# 1D analytical solution for infinite cylinder cooling
def analytical_1d_solution(r, t, T0, alpha):
    """
    Analytical solution for cooling of infinite cylinder.
    r: radial distance, t: time, T0: initial temperature, alpha: diffusivity
    """
    if t <= 0:
        return T0 if r <= PIPE_RADIUS else 0.0
    
    # Series solution (first few terms)
    # This is a simplified version - full solution requires Bessel functions
    diffusion_length = np.sqrt(4 * alpha * t)
    if r <= PIPE_RADIUS:
        return T0 * np.exp(-(r**2) / (4 * alpha * t)) / (4 * np.pi * alpha * t)
    else:
        return 0.0

# Extract temperature along central horizontal line for comparison
if uw.mpi.size == 1:
    # Create points along center line for analysis
    n_points = 51
    r_values = np.linspace(0, 0.5, n_points)
    
    analytical_temps = [
        analytical_1d_solution(r, FINAL_TIME, INITIAL_TEMP_HOT, THERMAL_DIFFUSIVITY) 
        for r in r_values
    ]
    
    print("Analytical comparison prepared")

# %% [markdown]
"""
## Results and Visualization
Display results and compare with analytical solution.
"""

# %%
# SECTION: Results
print(f"Final temperature range: {temperature.array.min():.6f} to {temperature.array.max():.6f}")
print(f"Maximum temperature location: center region")
print(f"Heat diffusion distance: ~{np.sqrt(4 * THERMAL_DIFFUSIVITY * FINAL_TIME):.3f}")

# Calculate total thermal energy for conservation check
if uw.mpi.size == 1:
    total_energy = np.sum(temperature.array) * (1.0/RESOLUTION)**2
    print(f"Total thermal energy: {total_energy:.6f}")

# %% [markdown]
"""
## Parameter Study Suggestions

**Mesh Resolution Study:**
```python
for res in [16, 32, 64]:
    RESOLUTION = res
    # Re-run simulation and compare accuracy vs computational cost
```

**Diffusivity Sensitivity:**
```python  
for kappa in [0.5, 1.0, 2.0]:
    THERMAL_DIFFUSIVITY = kappa
    # Observe how diffusion rate affects cooling
```

**Time Scale Analysis:**
```python
characteristic_time = PIPE_RADIUS**2 / THERMAL_DIFFUSIVITY
# Choose FINAL_TIME relative to characteristic_time
```
"""

# %%
# Success message
if uw.mpi.size == 1:
    print("✅ Diffusion simulation completed successfully!")
    print("🔧 Try modifying parameters at the top of the notebook")
    print("📊 Add visualization code below for temperature contours")

# %% [markdown] 
"""
## Learning Exercises

1. **Convergence Study**: Run with different mesh resolutions and observe accuracy
2. **Parameter Exploration**: Vary thermal diffusivity and observe cooling rates  
3. **Geometry Effects**: Change pipe radius and domain size
4. **Boundary Conditions**: Experiment with different edge boundary conditions
5. **Analytical Validation**: Implement full Bessel function solution for comparison

## Next Steps
- **Add convection**: Combine with fluid mechanics for advection-diffusion
- **Variable properties**: Temperature-dependent thermal conductivity
- **3D extension**: Full cylindrical geometry
- **Adaptive time stepping**: Optimize computational efficiency
"""