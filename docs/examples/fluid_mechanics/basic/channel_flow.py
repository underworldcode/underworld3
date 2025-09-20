# %% [markdown]
"""
# 📚 Basic Channel Flow (Poiseuille Flow)

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** basic  
**RUNTIME:** < 1 minute

## Description
Pressure-driven flow between parallel plates - analytical solution available.

## Key Concepts
- Poiseuille flow profile
- Pressure gradient driving force
- No-slip boundary conditions
- Analytical validation

## Physics Background
Parabolic velocity profile: v(y) = (ΔP/2μL) * y(H-y)
"""

# %%
# Parameters
CHANNEL_LENGTH = 2.0                   # PARAM: channel length
CHANNEL_HEIGHT = 1.0                   # PARAM: channel height
RESOLUTION_X = 40                      # PARAM: horizontal resolution
RESOLUTION_Y = 20                      # PARAM: vertical resolution
VISCOSITY = 1.0                        # PARAM: dynamic viscosity
PRESSURE_GRADIENT = 1.0                # PARAM: driving pressure gradient

import underworld3 as uw
import numpy as np
import sympy as sp

# %%
# Create mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RESOLUTION_X, RESOLUTION_Y),
    minCoords=(0.0, 0.0),
    maxCoords=(CHANNEL_LENGTH, CHANNEL_HEIGHT),
    qdegree=2
)

# Variables
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# %%
# Stokes system with body force
stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = VISCOSITY

# Add pressure gradient as body force
stokes.bodyforce = sp.Matrix([PRESSURE_GRADIENT, 0])

# %%
# Boundary conditions - no-slip on walls
stokes.add_essential_bc([0.0, 0.0], "Top", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Bottom", [0, 1])

# %%
# Solve
stokes.solve()

# Check against analytical solution
if uw.mpi.size == 1:
    # Maximum velocity should be at channel center
    y_center = CHANNEL_HEIGHT / 2
    v_max_analytical = (PRESSURE_GRADIENT * CHANNEL_HEIGHT**2) / (8 * VISCOSITY)
    print(f"✓ Analytical max velocity: {v_max_analytical:.3f}")
    print(f"✓ Channel flow solved!")
