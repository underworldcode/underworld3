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
    CYLINDRICAL2D_NATIVE = 2  # Cyl2D and Polar are equivalent here
    POLAR_NATIVE = 2  #
    CYLINDRICAL3D = 3  # (Not really used for anything)
    CYLINDRICAL3D_NATIVE = 4  # (Not really used for anything)
    SPHERICAL = 5
    SPHERICAL_NATIVE = 6
    SPHERE_SURFACE_NATIVE = 7  # theta / phi only R = 1 ...


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

    Need these:

      Coordinates N -> Native, xi_1, xi_2 by default, but over-written with meaningful symbols if possible
      Coordinates R -> Natural XYZ or R Theta Phi or R Theta z
      Coordinates X -> Cartesian XYZ

      nRotr - Matrix to map N to R
      rRotn - nRotr.T
      nRotx - Matrix to map N to X
      xRotn - nRotx.T
      xRotr = xRotn * nRotr
      rRotx = rRotn * nRotx

      Either R or X will be an alias for N depending on whether the DM is Cartesian  (Alias has the same vectors but different names)


    """

    def __init__(self, mesh, system: Optional[CoordinateSystemType] = CoordinateSystemType.CARTESIAN):

        self.mesh = mesh
        self.coordinate_type = system

        # are the mesh coordinates XYZ or have they been replaced by
        # "Natural" coordinates like r, theta, z ?
        self.CartesianDM = True

        # This is the default: Cartesian
        self._N = sympy.Matrix(self.mesh.r).T
        self._N[0]._latex_form = r"\mathrm{x}"
        self._N[1]._latex_form = r"\mathrm{y}"
        if self.mesh.cdim == 3:
            self._N[2]._latex_form = r"\mathrm{z}"

        ## Change specific coordinates systems as required

        if system == CoordinateSystemType.CYLINDRICAL2D and self.mesh.dim == 2:
            """
            This describes the situation for an annulus mesh with base coordinates
            in Cartesian (x,y,z).
            """

            self.type = "Cylindrical 2D"

            self._X = self._N.copy()
            self._x = self._X

            x, y = self.N
            r = sympy.sqrt(x**2 + y**2)
            t = sympy.atan2(y, x)
            self._R = sympy.Matrix([[r, t]])

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta")], real=True)
            th = self._r[1]
            self._rRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th)],
                    [-sympy.sin(th), sympy.cos(th)],
                ]
            )
            self._rRotN = self._rRotN_sym.subs(th, sympy.atan2(y, x))
            self._xRotN = sympy.eye(self.mesh.dim)

        ## The following is for the situation where the DM has r/theta coordinates loaded up
        elif system == CoordinateSystemType.CYLINDRICAL2D_NATIVE and self.mesh.dim == 2:
            self.type = "Cylindrical 2D Native"
            self.CartesianDM = False

            self._N[0]._latex_form = R"r"
            self._N[1]._latex_form = R"\theta"
            self._R = self._N.copy()
            self._r = self._R

            r, t = self.N
            x = r * sympy.cos(t)
            y = r * sympy.sin(t)

            self._X = sympy.Matrix([[x, y]])
            self._x = sympy.Matrix([sympy.symbols(R"x, y")], real=True)

            th = self.R[1]
            self._xRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), -sympy.sin(th)],
                    [sympy.sin(th), sympy.cos(th)],
                ]
            )

            self._xRotN = self._xRotN_sym
            self._rRotN = sympy.eye(self.mesh.dim)

        elif system == CoordinateSystemType.CYLINDRICAL3D and self.mesh.dim == 3:
            self.type = "Cylindrical 3D"

            self._X = self._N.copy()
            self._x = self._X

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, z")], real=True)

            x, y, z = self.X
            r = sympy.sqrt(x**2 + y**2)
            t = sympy.atan2(y, x)
            self._R = sympy.Matrix([[r, t, z]])

            th = self._r[1]
            self._rRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th), 0],
                    [-sympy.sin(th), sympy.cos(th), 0],
                    [0, 0, 1],
                ]
            )

            self._rRotN = self._rRotN_sym.subs(th, sympy.atan2(y, x))
            self._xRotN = sympy.eye(self.mesh.dim)

        elif system == CoordinateSystemType.CYLINDRICAL3D_NATIVE and self.mesh.dim == 3:
            self.type = "Cylindrical 3D Native"
            self.CartesianDM = False

            self._N[0]._latex_form = R"r"
            self._N[1]._latex_form = R"\theta"
            self._R = self._N.copy()
            self._r = self._R

            r, t, z = self.N
            x = r * sympy.cos(t)
            y = r * sympy.sin(t)

            self._X = sympy.Matrix([[x, y, z]])
            self._x = sympy.Matrix([sympy.symbols(R"x, y, z")], real=True)
            self._x[2] = z  # we should use the real one ?

            th = self.R[1]
            self._xRotN_sym = sympy.Matrix(
                [
                    [+sympy.cos(th), sympy.sin(th), 0],
                    [-sympy.sin(th), sympy.cos(th), 0],
                    [0, 0, 1],
                ]
            )

            self._xRotN = self._xRotN_sym
            self._rRotN = sympy.eye(self.mesh.dim)

        elif system == CoordinateSystemType.SPHERICAL and self.mesh.dim == 3:
            self.type = "Spherical"

            self._X = self._N.copy()
            self._x = self._X

            self._r = sympy.Matrix([sympy.symbols(R"r, \lambda_{1}, \lambda_{2}")])

            x, y, z = self.X
            r = sympy.sqrt(x**2 + y**2 + z**2)
            l1 = sympy.atan2(y, x)
            l2 = sympy.asin(z / r)

            self._R = sympy.Matrix([[r, l1, l2]])

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

            self._X = self._N.copy()
            self._x = self._X

            self._xRotN = sympy.eye(self.mesh.dim)
            self._rRotN = sympy.eye(self.mesh.dim)

        # For all meshes

        return

    @property
    def X(self) -> sympy.Matrix:
        return self._X

    @property
    def x(self) -> sympy.Matrix:
        return self._x

    @property
    def N(self) -> sympy.Matrix:
        return self._N

    @property
    def R(self) -> sympy.Matrix:
        return self._R

    @property
    def r(self) -> sympy.Matrix:
        return self._r

    @property  # alias for backward compat
    def xR(self) -> sympy.Matrix:
        return self._R

    @property
    def rRotN(self) -> sympy.Matrix:
        return self._rRotN

    @property
    def xRotN(self) -> sympy.Matrix:
        return self._xRotN

    @property
    def unit_e_0(self) -> sympy.Matrix:
        return self._rRotN[0, :]

    @property
    def unit_e_1(self) -> sympy.Matrix:
        return self._rRotN[1, :]

    @property
    def unit_e_2(self) -> sympy.Matrix:
        if self.mesh.dim == 3:
            return self._rRotN[2, :]
        else:
            return None

    @property
    def unit_i(self) -> sympy.Matrix:
        return self._xRotN[0, :]

    @property
    def unit_j(self) -> sympy.Matrix:
        return self._xRotN[1, :]

    @property
    def unit_k(self) -> sympy.Matrix:
        if self.mesh.dim == 3:
            return self._xRotN[2, :]
        else:
            return None

    ## Here we can add an ipython_display method to add the class documentation and a description of the
    ## entities that are defined (use sympy printing to make that work automatically)
