# %% [markdown]
"""
# 📚 Basic Stokes Flow: Driven Cavity

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** basic  
**RUNTIME:** < 2 minutes

## Description
Classic driven cavity benchmark - a square box with a moving lid.
This is the "Hello World" of computational fluid dynamics.

## Key Concepts
- Stokes equations (low Reynolds number flow)
- Velocity-pressure coupling
- Lid-driven boundary conditions
- Stream function visualization

## Learning Objectives
- Set up a simple Stokes flow problem
- Apply velocity boundary conditions
- Understand velocity-pressure saddle point system
- Visualize flow patterns
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
# Parameters - easy to modify
RESOLUTION = 32                        # PARAM: mesh resolution
DOMAIN_SIZE = 1.0                      # PARAM: domain size
VISCOSITY = 1.0                        # PARAM: fluid viscosity
LID_VELOCITY = 1.0                     # PARAM: lid velocity magnitude

import underworld3 as uw
import numpy as np
import sympy as sp

# %% [markdown]
"""
## Mesh and Variables
"""

# %%
# Create mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RESOLUTION, RESOLUTION),
    minCoords=(0.0, 0.0),
    maxCoords=(DOMAIN_SIZE, DOMAIN_SIZE),
    qdegree=2
)

# Define velocity and pressure fields
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %% [markdown]
"""
## Stokes System Setup
"""

# %%
# Create Stokes solver
stokes = uw.systems.Stokes(
    mesh,
    velocityField=velocity,
    pressureField=pressure
)

# Set viscosity
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = VISCOSITY

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
# Moving lid (top boundary)
stokes.add_essential_bc([LID_VELOCITY, 0.0], "Top", [0, 1])

# No-slip on other walls
stokes.add_essential_bc([0.0, 0.0], "Bottom", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Left", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Right", [0, 1])

# %% [markdown]
"""
## Solve and Analyze
"""

# %%
stokes.solve()

if uw.mpi.size == 1:
    v_max = np.max(np.sqrt(velocity.array[:, 0, 0]**2 + velocity.array[:, 0, 1]**2))
    print(f"✓ Maximum velocity: {v_max:.3f}")
    print(f"✓ Driven cavity flow solved!")
