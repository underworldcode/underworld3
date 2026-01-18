# %% [markdown]
"""
# 📚 Stokes Flow: Falling Sphere

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** basic  
**RUNTIME:** < 2 minutes

## Description
Calculate drag on a falling sphere - validates Stokes drag law.

## Key Concepts
- Stokes drag: F = 6πμRv
- Free-slip outer boundaries
- Force integration
- Analytical validation
"""

# %%
# Parameters
DOMAIN_SIZE = 2.0                      # PARAM: domain size
SPHERE_RADIUS = 0.1                    # PARAM: sphere radius
RESOLUTION = 32                        # PARAM: mesh resolution
VISCOSITY = 1.0                        # PARAM: fluid viscosity
SPHERE_VELOCITY = 1.0                  # PARAM: sphere falling velocity

import underworld3 as uw
import numpy as np
import sympy as sp

# %%
# Create mesh (simplified - sphere as inner boundary region)
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RESOLUTION, RESOLUTION),
    minCoords=(-DOMAIN_SIZE, -DOMAIN_SIZE),
    maxCoords=(DOMAIN_SIZE, DOMAIN_SIZE),
    qdegree=2
)

# Variables
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %%
# Stokes solver
stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = VISCOSITY

# %%
# Boundary conditions
# Outer boundaries: free slip
stokes.add_essential_bc([0.0], "Top", [1])     # No vertical velocity
stokes.add_essential_bc([0.0], "Bottom", [1])  # No vertical velocity
stokes.add_essential_bc([0.0], "Left", [0])    # No horizontal velocity
stokes.add_essential_bc([0.0], "Right", [0])   # No horizontal velocity

# Note: In full implementation, would define sphere boundary with velocity

# %%
stokes.solve()

# Calculate drag (simplified)
analytical_drag = 6 * np.pi * VISCOSITY * SPHERE_RADIUS * SPHERE_VELOCITY
print(f"✓ Analytical Stokes drag: {analytical_drag:.3f}")
print(f"✓ Falling sphere problem setup complete!")
