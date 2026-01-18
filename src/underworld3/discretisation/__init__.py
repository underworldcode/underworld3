r"""
Mesh and variable discretisation classes.

This module provides the core discretisation infrastructure for finite
element computations in Underworld3.

Classes
-------
Mesh : class
    Unstructured mesh with PETSc DMPlex backend. Supports various cell
    types and coordinate systems.
MeshVariable : class
    Field variable defined on mesh (nodal or cell-based storage).

Functions
---------
checkpoint_xdmf : function
    Save mesh and variables to XDMF format for visualization.
meshVariable_lookup_by_symbol : function
    Find mesh variable by its symbolic representation.

See Also
--------
underworld3.meshing : Mesh generation utilities.
underworld3.swarm : Particle-based discretisation.
"""
from .discretisation_mesh import Mesh
from .enhanced_variables import EnhancedMeshVariable as MeshVariable
from .discretisation_mesh import checkpoint_xdmf
from .discretisation_mesh import meshVariable_lookup_by_symbol
from .discretisation_mesh import petsc_dm_find_labeled_points_local
from .discretisation_mesh import _from_gmsh
