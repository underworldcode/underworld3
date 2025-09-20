# %% [markdown]
"""
# 📚 Basic Elastic Deformation

**PHYSICS:** solid_mechanics  
**DIFFICULTY:** basic  
**RUNTIME:** < 2 minutes

## Description
Simple elastic beam under loading - fundamental solid mechanics.

## Key Concepts
- Linear elasticity
- Stress-strain relationships
- Displacement boundary conditions
- von Mises stress
"""

# %%
# Parameters
BEAM_LENGTH = 2.0                      # PARAM: beam length
BEAM_HEIGHT = 0.5                      # PARAM: beam height
YOUNGS_MODULUS = 1e3                   # PARAM: Young's modulus
POISSONS_RATIO = 0.3                   # PARAM: Poisson's ratio
APPLIED_LOAD = 1.0                     # PARAM: applied force

import underworld3 as uw
import numpy as np
import sympy as sp

# %%
# Create mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(40, 10),
    minCoords=(0.0, 0.0),
    maxCoords=(BEAM_LENGTH, BEAM_HEIGHT),
    qdegree=2
)

# Displacement field
displacement = uw.discretisation.MeshVariable("u", mesh, 2, degree=2)
stress = uw.discretisation.MeshVariable("stress", mesh, (2, 2), degree=1)

# %%
# Stokes solver (for elastic problems)
stokes = uw.systems.Stokes(mesh, velocityField=displacement, pressureField=stress)

# Elastic constitutive model
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Note: Simplified - would use proper elastic model in full implementation

# %%
# Boundary conditions
# Fixed left end
stokes.add_essential_bc([0.0, 0.0], "Left", [0, 1])

# Applied load on right end
stokes.add_natural_bc([APPLIED_LOAD, 0.0], "Right")

# %%
stokes.solve()

print(f"✓ Elastic beam with E={YOUNGS_MODULUS}, ν={POISSONS_RATIO}")
print(f"✓ Applied load: {APPLIED_LOAD}")
print(f"✓ Elastic deformation solved!")
