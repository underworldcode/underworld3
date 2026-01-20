r"""
Cython extensions for PETSc integration.

This module contains Cython-compiled extensions that provide the low-level
interface between Underworld3 and PETSc. These modules are compiled during
package installation.

Key Extensions
--------------
petsc_generic_snes_solvers
    Base SNES solver classes (SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt).
generic_solvers
    High-level solver wrappers.
petsc_maths
    Integration and mathematical operations via PETSc.
petsc_discretisation
    Mesh and variable discretisation interface.

Notes
-----
These modules require PETSc and petsc4py to be installed. The Cython
sources (.pyx files) are compiled during ``pip install``.

See Also
--------
underworld3.systems : High-level solver interface.
underworld3.discretisation : Mesh classes using these extensions.
"""
