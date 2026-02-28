r"""
Mesh and variable discretisation classes.

This module provides the core discretisation infrastructure for finite
element computations in Underworld3.

**Mesh** -- Unstructured mesh with PETSc DMPlex backend.

**MeshVariable** -- Field variable defined on mesh (nodal or cell-based storage).

**checkpoint_xdmf** -- Save mesh and variables to XDMF format.

**meshVariable_lookup_by_symbol** -- Find mesh variable by its symbol.

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
