# %% [markdown]
"""
# 📚 Hello Underworld

**PHYSICS:** introduction  
**DIFFICULTY:** basic  
**DOMAIN:** utilities  
**RUNTIME:** < 1 minute

## Description
Absolute beginner introduction to UW3 workflow

This is a **foundation example** designed to teach basic Underworld3 patterns without complex physics.

## Key Concepts
- Basic workflow
- Imports
- Mesh creation
- Variables
- Simple output

## Adaptable Parameters  
- `MESH_SIZE`: Number of elements
- `DOMAIN_BOUNDS`: Physical domain limits

## Claude Learning Objectives
This example demonstrates:
1. Standard UW3 import patterns
2. Basic mesh and variable creation
3. Fundamental workflow structure
4. Parameter organization for easy modification

**For Claude:** This example shows the essential building blocks that appear in all UW3 simulations.
The parameter organization and section structure provide templates for generating similar examples.
"""

# %% [markdown]
"""
## Welcome
"""

# %%
# TODO: Implement introduction_text for hello_underworld

# %% [markdown]
"""
## Basic Setup
"""

# %%
# Minimal Underworld3 setup
import underworld3 as uw
import numpy as np

# Create simple mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(MESH_SIZE, MESH_SIZE),
    minCoords=DOMAIN_BOUNDS[:2],
    maxCoords=DOMAIN_BOUNDS[2:],
    qdegree=2
)

# Create temperature field
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
temperature.array[:] = 0.0

print(f"🎉 Welcome to Underworld3!")
print(f"✓ Created mesh with {mesh.X.coords.shape[0]} vertices")
print(f"✓ Created temperature field with {temperature.coords.shape[0]} DOFs")

# %% [markdown]
"""
## Create Mesh
"""

# %%
# TODO: Implement simple_mesh for hello_underworld

# %% [markdown]
"""
## Define Variable
"""

# %%
# TODO: Implement simple_variable for hello_underworld

# %% [markdown]
"""
## Success Message
"""

# %%
# TODO: Implement completion_message for hello_underworld