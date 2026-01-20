# %% [markdown]
"""
# 📚 Define Vector Variable

**PHYSICS:** variables  
**DIFFICULTY:** basic  
**DOMAIN:** utilities  
**RUNTIME:** < 1 minute

## Description
Create and manipulate vector fields with proper indexing

This is a **foundation example** designed to teach basic Underworld3 patterns without complex physics.

## Key Concepts
- Vector variables
- Component access
- Vector operations
- Coordinate systems

## Adaptable Parameters  
- `VECTOR_DIMENSION`: Number of vector components
- `COORDINATE_SYSTEM`: Spatial coordinate system
- `INITIAL_FIELD`: Initial vector field expression

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
## Mesh and Coordinates
"""

# %%
# TODO: Implement coordinate_system_setup for define_vector_variable

# %% [markdown]
"""
## Vector Variable
"""

# %%
# Create vector field variable
vector_field = uw.discretisation.MeshVariable(
    "velocity",
    mesh,
    num_components=VECTOR_DIMENSION,
    degree=2
)

# Initialize components separately
vector_field.array[:, 0, 0] = 1.0  # x-component
if VECTOR_DIMENSION > 1:
    vector_field.array[:, 0, 1] = 0.0  # y-component

print(f"✓ Created {VECTOR_DIMENSION}D vector field")
print(f"  Shape: {vector_field.array.shape}")
print(f"  Components: {VECTOR_DIMENSION}")
print(f"  Magnitude range: {np.linalg.norm(vector_field.array.reshape(-1, VECTOR_DIMENSION), axis=1).max():.3f}")

# %% [markdown]
"""
## Component Access
"""

# %%
# TODO: Implement vector_component_access for define_vector_variable

# %% [markdown]
"""
## Vector Operations
"""

# %%
# TODO: Implement vector_math_operations for define_vector_variable