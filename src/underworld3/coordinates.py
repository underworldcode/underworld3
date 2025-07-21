from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
from petsc4py import PETSc

import underworld3
from underworld3 import VarType
import sympy

expression = lambda *x, **X: underworld3.function.expressions.UWexpression(
    *x, _unique_name_generation=True, **X
)


class CoordinateSystemType(Enum):
    """
    Meshes can have natural coordinate systems which lie on top of the Cartesian
    coordinate system that we use for constructing the solvers (usually)
    """

    CARTESIAN = 0
    CYLINDRICAL2D = 10  # Cyl2D and Polar are equivalent here
    POLAR = 10  #
    CYLINDRICAL2D_NATIVE = 11  # Cyl2D and Polar are equivalent here
    POLAR_NATIVE = 11  #

    CYLINDRICAL3D = 100  # (Not really used for anything)
    CYLINDRICAL3D_NATIVE = 101  # (Not really used for anything)
    SPHERICAL = 200
    SPHERICAL_NATIVE = 201
    SPHERICAL_NATIVE_RTP = 201
    # SPHERICAL_NATIVE_RLONLAT = 7
    SPHERE_SURFACE_NATIVE = 301  # theta / phi only R = 1 ...
    # SPHERE_SURFACE_NATIVE_RLONLAT = 302  # theta / phi only R = 1 ...


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

    def __init__(
        self,
        mesh,
        system: Optional[CoordinateSystemType] = CoordinateSystemType.CARTESIAN,
    ):
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

        self._R = self._N.copy()

        # We need this to define zeros in the coordinate transforms
        # (since they need to indicate they are coordinate functions even
        # if they are independent of all coordinates)

        if self.mesh.cdim == 3:
            self.independent_of_N = expression(
                r"0(x,y,z)",
                underworld3.maths.functions.vanishing
                * self._N[0]
                * self._N[1]
                * self._N[2],
                "independent of N0, N1, N2",
            )
        else:
            self.independent_of_N = expression(
                r"0(x,y,z)",
                underworld3.maths.functions.vanishing * self._N[0] * self._N[1],
                "independent of N0, N1",
            )

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
            r = expression(R"r", sympy.sqrt(x**2 + y**2), "Radial Coordinate")

            t = expression(
                R"\theta",
                sympy.Piecewise((0, x == 0), (sympy.atan2(y, x), True)),
                "Angular coordinate",
            )

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
            t = sympy.Piecewise((0, x == 0), (sympy.atan2(y, x), True))

            self._R = sympy.Matrix([[r, t, z]])

            th = self._r[1]
            self._rRotN_sym = sympy.Matrix(
                [
                    [sympy.cos(th), sympy.sin(th), 0],
                    [-sympy.sin(th), sympy.cos(th), 0],
                    [0, 0, 1],
                ]
            )

            self._rRotN = self._rRotN_sym.subs(th, t)
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
                    [+sympy.cos(th), sympy.sin(th), self.independent_of_N],
                    [-sympy.sin(th), sympy.cos(th), self.independent_of_N],
                    [self.independent_of_N, self.independent_of_N, 1],
                ]
            )

            self._xRotN = self._xRotN_sym
            self._rRotN = sympy.eye(self.mesh.dim)

        elif system == CoordinateSystemType.SPHERICAL and self.mesh.dim == 3:
            self.type = "Spherical"

            self._X = self._N.copy()
            self._x = self._X

            self._r = sympy.Matrix([sympy.symbols(R"r, \theta, \phi")])

            x, y, z = self.X

            r = expression(
                R"r",
                sympy.sqrt(x**2 + y**2 + z**2),
                "Radial coordinate",
            )

            th = expression(
                R"\theta",
                sympy.acos(z / r),
                "co-latitude",
            )

            ph = expression(
                R"\phi",
                sympy.atan2(y, x),
                "longitude",
            )

            self._R = sympy.Matrix([[r, th, ph]])

            r1 = self._r[1]
            r2 = self._r[2]
            rRotN_sym = sympy.Matrix(
                [
                    [
                        sympy.sin(r1) * sympy.cos(r2),
                        sympy.sin(r1) * sympy.sin(r2),
                        sympy.cos(r1),
                    ],
                    [
                        sympy.cos(r1) * sympy.cos(r2),
                        sympy.cos(r1) * sympy.sin(r2),
                        -sympy.sin(r1),
                    ],
                    [
                        -sympy.sin(r2),
                        +sympy.cos(r2),
                        self.independent_of_N,
                    ],
                ]
            )

            rz = sympy.sqrt(x**2 + y**2)
            r_x_rz = sympy.sqrt((x**2 + y**2 + z**2) * (x**2 + y**2))

            rRotN = sympy.Matrix(
                [
                    [
                        x / r,
                        y / r,
                        z / r,
                    ],
                    [
                        (x * z) / r_x_rz,
                        (y * z) / r_x_rz,
                        -(x**2 + y**2) / r_x_rz,
                    ],
                    [
                        -y / rz,
                        +x / rz,
                        self.independent_of_N,
                    ],
                ]
            )

            self._rRotN_sym = rRotN_sym
            self._rRotN = rRotN

            self._xRotN = sympy.eye(self.mesh.dim)

        elif system == CoordinateSystemType.SPHERICAL_NATIVE and self.mesh.dim == 3:
            self.type = "Spherical Native"
            self.CartesianDM = False

            self._N[0]._latex_form = R"r"
            self._N[1]._latex_form = R"\theta"
            self._N[2]._latex_form = R"\varphi"
            self._R = self._N.copy()
            self._r = self._R

            r, t, p = self.N

            x = r * sympy.sin(t) * sympy.cos(p)
            y = r * sympy.sin(t) * sympy.sin(p)
            z = r * sympy.cos(t)

            self._X = sympy.Matrix([[x, y, z]])
            self._x = sympy.Matrix([sympy.symbols(R"x, y, z")], real=True)

            r1 = self._R[1]
            r2 = self._R[2]
            self._Rot = sympy.Matrix(  ## Check this next !
                [
                    [
                        sympy.sin(r1) * sympy.cos(r2),
                        sympy.sin(r1) * sympy.sin(r2),
                        sympy.cos(r2),
                    ],
                    [
                        sympy.cos(r1) * sympy.cos(r2),
                        sympy.cos(r1) * sympy.sin(r2),
                        -sympy.cos(r2),
                    ],
                    [
                        -sympy.sin(r2),
                        +sympy.cos(r2),
                        self.independent_of_N,
                    ],
                ]
            )

            self._xRotN = self._Rot.subs([(r1, t), (r2, p)])
            self._rRotN = sympy.eye(self.mesh.dim)

        elif (
            system == CoordinateSystemType.SPHERE_SURFACE_NATIVE and self.mesh.dim == 2
        ):
            self.type = "Spherical Native"
            self.CartesianDM = False

            self._N[0]._latex_form = R"\lambda_1"
            self._N[1]._latex_form = R"\lambda_2"
            self._R = self._N.copy()
            self._r = self._R
            l1, l2 = self.N

            r = sympy.sympify(1)  # Maybe we will need to change this value
            x = r * sympy.cos(l1) * sympy.cos(l2)
            y = r * sympy.sin(l1) * sympy.cos(l2)
            z = r * sympy.sin(l2)

            self._X = sympy.Matrix([[x, y, z]])
            self._x = sympy.Matrix([sympy.symbols(R"x, y, z")], real=True)

            # l1 is longitude, l2 is latitude
            rl1 = self._R[0]
            rl2 = self._R[1]
            self._Rot = sympy.Matrix(
                [
                    [
                        +sympy.cos(rl1) * sympy.cos(rl2),
                        +sympy.sin(rl1) * sympy.cos(rl2),
                        sympy.sin(rl2),
                    ],
                    [
                        -sympy.sin(rl1) * sympy.cos(rl2),
                        +sympy.cos(rl1) * sympy.cos(rl2),
                        self.independent_of_N,
                    ],
                    [
                        -sympy.cos(rl1) * sympy.sin(rl2),
                        -sympy.cos(rl1) * sympy.sin(rl2),
                        sympy.cos(rl2),
                    ],
                ]
            )

            self._xRotN = self._Rot.subs([(rl1, l1), (rl2, l2)])
            self._rRotN = sympy.eye(self.mesh.dim)

        else:  # Cartesian by default
            self.type = f"Cartesian {self.mesh.dim}D"

            self._X = self._N  # .copy()
            self._x = self._X

            self._xRotN = sympy.eye(self.mesh.dim)
            self._rRotN = sympy.eye(self.mesh.dim)

        # For all meshes
        #

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

    # Should validate on dim
    def unit_ijk(self, dirn) -> sympy.Matrix:
        if dirn <= self.mesh.dim:
            return self._xRotN[dirn, :]
        else:
            return None

    ## Here we can add an ipython_display method to add the class documentation and a description of the
    ## entities that are defined (use sympy printing to make that work automatically)
