# %% [markdown]
"""
# 📚 Create Structured Mesh

**PHYSICS:** meshing  
**DIFFICULTY:** basic  
**DOMAIN:** utilities  
**RUNTIME:** < 1 minute

## Description
Demonstrate structured mesh creation with different resolutions

This is a **foundation example** designed to teach basic Underworld3 patterns without complex physics.

## Key Concepts
- Mesh creation
- Element types
- Domain definition
- Resolution effects

## Adaptable Parameters  
- `RESOLUTION`: Number of elements per dimension
- `DOMAIN_SIZE`: Physical domain dimensions
- `ELEMENT_DEGREE`: Polynomial degree of elements

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
## Setup
"""

# %%
# Standard Underworld3 imports
import underworld3 as uw
import numpy as np

# Parameter constants - modify these for different behavior
RESOLUTION = 32                    # PARAM: Number of elements per dimension
DOMAIN_SIZE = 1.0                   # PARAM: Physical domain dimensions
ELEMENT_DEGREE = 2                    # PARAM: Polynomial degree of elements

print(f"✓ Parameters loaded for create_structured_mesh")

# %% [markdown]
"""
## Mesh Creation
"""

# %%
# Create structured quadrilateral mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RESOLUTION, RESOLUTION),
    minCoords=(0.0, 0.0),
    maxCoords=(DOMAIN_SIZE, DOMAIN_SIZE),
    qdegree=ELEMENT_DEGREE
)

print(f"✓ Created {mesh.X.coords.shape[0]} vertex structured mesh")
print(f"  Elements: {RESOLUTION}×{RESOLUTION}")
print(f"  Domain: {DOMAIN_SIZE}×{DOMAIN_SIZE}")

# %% [markdown]
"""
## Mesh Analysis
"""

# %%
# TODO: Implement mesh_properties for create_structured_mesh

# %% [markdown]
"""
## Visualization
"""

# %%
# TODO: Implement basic_mesh_plot for create_structured_mesh