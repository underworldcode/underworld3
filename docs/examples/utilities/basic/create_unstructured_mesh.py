# %% [markdown]
"""
# 📚 Create Unstructured Mesh

**PHYSICS:** meshing  
**DIFFICULTY:** basic  
**DOMAIN:** utilities  
**RUNTIME:** < 1 minute

## Description
Demonstrate unstructured mesh creation and quality assessment

This is a **foundation example** designed to teach basic Underworld3 patterns without complex physics.

## Key Concepts
- Unstructured meshes
- Mesh quality
- Boundary conforming
- Adaptive sizing

## Adaptable Parameters  
- `CELL_SIZE`: Target element size
- `DOMAIN_GEOMETRY`: Domain shape and boundaries
- `QUALITY_THRESHOLD`: Mesh quality acceptance criteria

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
CELL_SIZE = 1.0                   # PARAM: Target element size
DOMAIN_GEOMETRY = 1.0                   # PARAM: Domain shape and boundaries
QUALITY_THRESHOLD = 1.0                   # PARAM: Mesh quality acceptance criteria

print(f"✓ Parameters loaded for create_unstructured_mesh")

# %% [markdown]
"""
## Geometry Definition
"""

# %%
# TODO: Implement domain_geometry for create_unstructured_mesh

# %% [markdown]
"""
## Mesh Generation
"""

# %%
# Create unstructured triangular mesh  
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(DOMAIN_GEOMETRY, DOMAIN_GEOMETRY),
    cellSize=CELL_SIZE,
    qdegree=2
)

print(f"✓ Created {mesh.X.coords.shape[0]} vertex unstructured mesh")
print(f"  Target cell size: {CELL_SIZE}")
print(f"  Actual elements: ~{mesh.X.coords.shape[0] // 3}")

# %% [markdown]
"""
## Quality Assessment
"""

# %%
# TODO: Implement mesh_quality_analysis for create_unstructured_mesh