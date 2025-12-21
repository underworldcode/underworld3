r"""
Mathematical utilities for continuum mechanics.

This module provides mathematical functions and operations needed for
finite element formulations, including tensor algebra, vector calculus
in various coordinate systems, and integration.

Submodules
----------
tensor : module
    Tensor notation conversions (Voigt, Mandel, full tensor forms).
vector_calculus : module
    Gradient, divergence, curl in Cartesian coordinates.
vector_calculus_cylindrical : module
    Vector calculus operators in cylindrical coordinates.
vector_calculus_spherical : module
    Vector calculus operators in spherical coordinates.

Functions
---------
delta_function : function
    Regularized delta function for localized source terms.
L2_norm : function
    L2 norm computation for fields.
Integral : class
    Domain integration over mesh.
CellWiseIntegral : class
    Cell-by-cell integration.

See Also
--------
underworld3.coordinates : Coordinate system definitions.
underworld3.discretisation : Mesh classes for integration.
"""
from . import tensors as tensor
from .vector_calculus import mesh_vector_calculus as vector_calculus
from .vector_calculus import (
    mesh_vector_calculus_cylindrical as vector_calculus_cylindrical,
)

from .functions import delta as delta_function
from .functions import L2_norm as L2_norm

# from .vector_calculus import (
#     mesh_vector_calculus_spherical_lonlat as vector_calculus_spherical_lonlat,
# )

from .vector_calculus import (
    mesh_vector_calculus_spherical as vector_calculus_spherical,
)
from .vector_calculus import (
    mesh_vector_calculus_spherical_surface2D_lonlat as vector_calculus_spherical_surface2D_lonlat,
)

# These could be wrapped so that they can be documented along with the math module
from underworld3.cython.petsc_maths import Integral
from underworld3.cython.petsc_maths import CellWiseIntegral
