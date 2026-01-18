# %% [markdown]
"""
# 📚 Define Scalar Variable

**PHYSICS:** variables  
**DIFFICULTY:** basic  
**DOMAIN:** utilities  
**RUNTIME:** < 1 minute

## Description
Create and manipulate scalar fields on meshes

This is a **foundation example** designed to teach basic Underworld3 patterns without complex physics.

## Key Concepts
- Scalar variables
- Degrees of freedom
- Data access
- Field initialization

## Adaptable Parameters  
- `VARIABLE_DEGREE`: Polynomial degree of finite element space
- `INITIAL_VALUE`: Starting field values
- `FIELD_NAME`: Variable identifier

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
## Mesh Setup
"""

# %%
# TODO: Implement basic_mesh for define_scalar_variable

# %% [markdown]
"""
## Variable Definition
"""

# %%
# Create scalar field variable
scalar_field = uw.discretisation.MeshVariable(
    FIELD_NAME, 
    mesh, 
    num_components=1,
    degree=VARIABLE_DEGREE
)

# Initialize with constant value
scalar_field.array[:] = INITIAL_VALUE

print(f"✓ Created scalar field '{FIELD_NAME}'")
print(f"  Degrees of freedom: {scalar_field.array.shape[0]}")
print(f"  Initial value: {INITIAL_VALUE}")
print(f"  Current range: {scalar_field.array.min():.3f} to {scalar_field.array.max():.3f}")

# %% [markdown]
"""
## Data Access
"""

# %%
# TODO: Implement array_data_access for define_scalar_variable

# %% [markdown]
"""
## Field Operations
"""

# %%
# TODO: Implement basic_field_operations for define_scalar_variable