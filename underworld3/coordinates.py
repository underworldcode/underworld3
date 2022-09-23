from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
from petsc4py import PETSc

import underworld3
from underworld3 import VarType
import sympy


class CoordinateSystemType(Enum):
    """
    Meshes can have natural coordinate systems which lie on top of the Cartesian
    coordinate system that we use for constructing the solvers (usually)
    """

    CARTESIAN = 0
    CYLINDRICAL2D = 1  # Cyl2D and Polar are equivalent here
    POLAR = 1  #
    CYLINDRICAL3D = 2  # (Not really used for anything)
    SPHERICAL = 3


# Maybe break this out into it's own file - this needs to cover, basis vectors,
# coordinate arrays in the natural coordinate system (and setting them once the other
# setup is complete), direction of the vertical, implementation of
# rotated boundary conditions, null spaces ...


class CoordinateSystem:
    R"""This class is attached to a mesh to provide programmatic access to coordinate system operations.

    `CoordinateSystem.R` - the coordinates in the natural reference (sympy symbols)
    `CoordinateSystem.xR` - the coordinates in the natural reference (sympy symbols in terms of mesh.X)
    `CoordinateSystem._Rot` - the matrix to rotate from X to R, written symbolically
    `CoordinateSystem._xRot` - the matrix to rotate from X to R, written in terms of mesh.X

    """

    def __init__(self, mesh, system: Optional[CoordinateSystemType] = None):

        self.mesh = mesh

        if system == CoordinateSystemType.CYLINDRICAL2D and self.mesh.dim == 2:
            self.type = "Cylindrical 2D"
            self._R = sympy.Matrix([sympy.symbols(R"r, \theta")], real=True)

            x, y = self.X
            r = sympy.sqrt(x**2 + y**2)
            t = sympy.atan2(y, x)
            self._xR = sympy.Matrix([[r, t]])

            th = self.R[1]
            self._Rot = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th)],
                    [-sympy.sin(th), sympy.cos(th)],
                ]
            )

            self._xRot = self._Rot.subs(th, sympy.atan2(y, x))

        elif system == CoordinateSystemType.CYLINDRICAL3D and self.mesh.dim == 3:
            self.type = "Cylindrical 3D"

            self._R = sympy.Matrix([sympy.symbols(R"r, \theta, z")], real=True)

            x, y, z = self.X
            r = sympy.sqrt(x**2 + y**2)
            t = sympy.atan2(y, x)
            self._xR = sympy.Matrix([[r, t, z]])

            th = self._R[1]
            self._Rot = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th), 0],
                    [-sympy.sin(th), sympy.cos(th), 0],
                    [0, 0, 1],
                ]
            )

            self._xRot = self._Rot.subs(th, sympy.atan2(y, x))

        elif system == CoordinateSystemType.SPHERICAL and self.mesh.dim == 3:
            self.type = "Spherical"

            self._R = sympy.Matrix([sympy.symbols(R"r, \lambda_{1}, \lambda_{2}")])

            x, y, z = self.X
            r = sympy.sqrt(x**2 + y**2 + z**2)
            l1 = sympy.atan2(y, x)
            l2 = sympy.asin(z / r)

            self._xR = sympy.Matrix([[r, l1, l2]])

            # l1 is longitude, l2 is latitude
            rl1 = self._R[1]
            rl2 = self._R[2]
            self._Rot = sympy.Matrix(
                [
                    [+sympy.cos(rl1) * sympy.cos(rl2), +sympy.sin(rl1) * sympy.cos(rl2), sympy.sin(rl2)],
                    [-sympy.sin(rl1) * sympy.cos(rl2), +sympy.cos(rl1) * sympy.cos(rl2), 0],
                    [-sympy.cos(rl1) * sympy.sin(rl2), -sympy.cos(rl1) * sympy.sin(rl2), sympy.cos(rl2)],
                ]
            )

            self._xRot = self._Rot.subs([(rl1, l1), (rl2, l2)])

        else:  # Cartesian by default
            self.type = f"Cartesian {self.mesh.dim}D"

            self._R = self.X
            self._xR = self.X
            self._Rot = sympy.eye(self.mesh.dim)
            self._xRot = sympy.eye(self.mesh.dim)

        # For all meshes

        return

    @property
    def X(self) -> sympy.Matrix:
        return sympy.Matrix(self.mesh.r).T

    @property
    def R(self) -> sympy.Matrix:
        return self._R

    @property
    def xR(self) -> sympy.Matrix:
        return self._xR

    @property
    def Rot(self) -> sympy.Matrix:
        return self._Rot

    @property
    def xRot(self) -> sympy.Matrix:
        return self._xRot

    @property
    def unit_e_0(self) -> sympy.Matrix:
        return self._xRot[0, :]

    @property
    def unit_e_1(self) -> sympy.Matrix:
        return self._xRot[1, :]

    @property
    def unit_e_2(self) -> sympy.Matrix:
        if self.mesh.dim == 3:
            return self._xRot[2, :]
        else:
            return None

    ## Here we can add an ipython_display method to add the class documentation and a description of the
    ## entities that are defined (use sympy printing to make that work automatically)
