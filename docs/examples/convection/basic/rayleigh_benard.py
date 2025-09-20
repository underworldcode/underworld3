# %% [markdown]
"""
# 📚 Basic Rayleigh-Benard Convection

**PHYSICS:** convection  
**DIFFICULTY:** basic  
**RUNTIME:** < 3 minutes

## Description
Thermal convection in a heated layer - fundamental geophysics problem.

## Key Concepts
- Rayleigh number and convection onset
- Coupled thermal-mechanical system
- Boussinesq approximation
- Critical Rayleigh number ~1708
"""

# %%
# Parameters
RESOLUTION = 32                        # PARAM: mesh resolution
RAYLEIGH_NUMBER = 1e4                  # PARAM: Rayleigh number
ASPECT_RATIO = 2.0                     # PARAM: width/height ratio

import underworld3 as uw
import numpy as np
import sympy as sp

# %%
# Create mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(RESOLUTION*ASPECT_RATIO), RESOLUTION),
    minCoords=(0.0, 0.0),
    maxCoords=(ASPECT_RATIO, 1.0),
    qdegree=2
)

# Variables
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# %%
# Stokes solver with buoyancy
stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

# Buoyancy force (Boussinesq approximation)
stokes.bodyforce = sp.Matrix([0, -RAYLEIGH_NUMBER * temperature.sym[0]])

# %%
# Thermal solver
thermal = uw.systems.Poisson(mesh, u_Field=temperature)
thermal.constitutive_model = uw.constitutive_models.DiffusionModel
thermal.constitutive_model.Parameters.diffusivity = 1.0

# %%
# Boundary conditions
# Velocity: free-slip sides, no-slip top/bottom
stokes.add_essential_bc([0.0], "Left", [0])
stokes.add_essential_bc([0.0], "Right", [0])
stokes.add_essential_bc([0.0, 0.0], "Top", [0, 1])
stokes.add_essential_bc([0.0, 0.0], "Bottom", [0, 1])

# Temperature: hot bottom, cold top
thermal.add_essential_bc([1.0], "Bottom")
thermal.add_essential_bc([0.0], "Top")

# %%
# Initial temperature with perturbation
with mesh.access(temperature):
    temperature.array[:] = 1.0 - mesh.data[:, 1]  # Linear profile
    # Add small perturbation to trigger convection
    temperature.array[:] += 0.01 * np.sin(np.pi * mesh.data[:, 0] / ASPECT_RATIO)

# %%
# Solve coupled system
thermal.solve()
stokes.solve()

print(f"✓ Rayleigh number: {RAYLEIGH_NUMBER}")
print(f"✓ Convection system solved!")
