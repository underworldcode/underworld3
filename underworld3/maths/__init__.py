from . import tensors as tensor
from .vector_calculus import mesh_vector_calculus as vector_calculus
from .vector_calculus import mesh_vector_calculus_cylindrical as vector_calculus_cylindrical
from .vector_calculus import mesh_vector_calculus_spherical_lonlat as vector_calculus_spherical_lonlat

# These could be wrapped so that they can be documented along with the math module
from underworld3.cython.petsc_maths import Integral
from underworld3.cython.petsc_maths import CellWiseIntegral
