# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

import os

os.environ["SYMPY_USE_CACHE"] = "no"

res = 0.1
r_o = 1.0
r_i = 0.5
free_slip_upper = True


# +
meshdisc_xyz = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)

mesh = meshdisc_xyz
PXs = sympy.Function(r"\mathcal{P}")(*mesh.N.base_scalars()[0 : mesh.dim])
VXs = sympy.Matrix.zeros(1, mesh.dim)
for i in range(mesh.dim):
    VXs[i] = sympy.Function(rf"\mathcal{{V}}_{i}")(*mesh.N.base_scalars()[0 : mesh.dim])
display(PXs)
display(VXs)
# -

# ## Symbolic forms v. mesh variables
#
# Mesh variables are `sympy.Function` objects that have the additional property of carrying data that allows them to be interploated to any point on a mesh and numerically differentiated (once).
#
# The symbolic forms allow us to undertake all the manipulations and simplifications available in sympy which are not all available for mesh variables (such as higher-order derivatives).
#
# For example, we can demonstrate some vector calculus results which we can use while we develop our equation systems. We can substitute for the mesh variables later if we choose.

# +
gradPXs = meshdisc_xyz.vector.gradient(PXs)
display(gradPXs)

divgradPXs = meshdisc_xyz.vector.divergence(gradPXs)
display(divgradPXs)

curlgradPXs = meshdisc_xyz.vector.curl(gradPXs)
display(curlgradPXs)

# +
## Create a Cylindrical Mesh with a Native Coordinate System

meshball_xyz_tmp = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)

xy_vec = meshball_xyz_tmp.dm.getCoordinates()
xy = xy_vec.array.reshape(-1, 2)
dmplex = meshball_xyz_tmp.dm.clone()
rtheta = np.empty_like(xy)
rtheta[:, 0] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
rtheta[:, 1] = np.arctan2(xy[:, 1] + 1.0e-16, xy[:, 0] + 1.0e-16)
rtheta_vec = xy_vec.copy()
rtheta_vec.array[...] = rtheta.reshape(-1)[...]
dmplex.setCoordinates(rtheta_vec)
meshball_xyz_tmp.vtk("tmp_disk.vtk")
del meshball_xyz_tmp

meshdisc = uw.meshing.Mesh(
    dmplex,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D_NATIVE,
)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshdisc.dm, [0.0, 1.0], [0.0, 0.0], [0.0, 2 * np.pi]
)


## Add some mesh variables (Vector and scalar)

VC = uw.discretisation.MeshVariable(r"U^c", meshdisc, 2, degree=2)
PC = uw.discretisation.MeshVariable(r"P^c", meshdisc, 1, degree=1)

## Create some symbolic equivalents

mesh = meshdisc
PCs = sympy.Function(r"\mathcal{P}")(*mesh.N.base_scalars()[0 : mesh.dim])
VCs = sympy.Matrix.zeros(1, mesh.dim)
for i in range(mesh.dim):
    VCs[i] = sympy.Function(rf"\mathcal{{V}}_{i}")(*mesh.N.base_scalars()[0 : mesh.dim])
display(PCs)
display(VCs)

# +
gradPCs = meshdisc.vector.gradient(PCs)
display(gradPCs)

divgradPCs = meshdisc.vector.divergence(gradPCs)
display(divgradPCs)

curlgradPCs = meshdisc.vector.curl(gradPCs)
display(curlgradPCs)

# +
## Create a Spherical Mesh with a Native Coordinate System

## NOTE: this only works if numElements is an odd number

meshball_xyz_tmp = uw.meshing.CubedSphere(
    radiusOuter=r_o, radiusInner=r_i, numElements=7, simplex=True
)

xyz_vec = meshball_xyz_tmp.dm.getCoordinates()
xyz = xyz_vec.array.reshape(-1, 3)
dmplex = meshball_xyz_tmp.dm.clone()

rl1l2 = np.empty_like(xyz)
rl1l2[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
rl1l2[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])
rl1l2[:, 2] = (
    np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2]) - np.pi / 2
)

rl1l2_vec = xyz_vec.copy()
rl1l2_vec.array[...] = rl1l2.reshape(-1)[...]
dmplex.setCoordinates(rl1l2_vec)

meshball_xyz_tmp.vtk("tmp_sphere.vtk")
del meshball_xyz_tmp

meshball = uw.meshing.Mesh(
    dmplex, coordinate_system_type=uw.coordinates.CoordinateSystemType.SPHERICAL_NATIVE
)
uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
    meshball.dm, [0.0, 6.28, 0.0], [0.0, 0.0, 0.0], [0.0, 2 * np.pi, 0.0]
)


## Add some mesh variables (Vector and scalar)

VS = uw.discretisation.MeshVariable(r"U^s", meshball, 3, degree=2)
PS = uw.discretisation.MeshVariable(r"P^s", meshball, 1, degree=1)

## Create some symbolic equivalents

mesh = meshball
PSs = sympy.Function(r"\mathcal{P}")(*mesh.N.base_scalars())
VSs = sympy.Matrix.zeros(1, mesh.dim)
for i in range(mesh.dim):
    VSs[i] = sympy.Function(rf"\mathcal{{V}}_{i}")(*mesh.N.base_scalars())
display(PSs)
display(VSs)

# +
gradPSs = meshball.vector.gradient(PSs)
display(gradPSs)

divVSs = meshball.vector.divergence(VSs)
display(divVSs)

divgradPSs = meshball.vector.divergence(gradPSs)
display(divgradPSs)

curlgradPSs = meshball.vector.curl(gradPSs)
display(curlgradPSs)

# Note
sympy.simplify(curlgradPSs)

# +
# if uw.mpi.size == 1:

#     import numpy as np
#     import pyvista as pv
#     import vtk

#     pv.global_theme.background = "white"
#     pv.global_theme.window_size = [1000, 1000]
#     pv.global_theme.antialiasing = True
#     pv.global_theme.jupyter_backend = "panel"
#     pv.global_theme.smooth_shading = True

#     pvmesh = pv.read("tmp_sphere.vtk")

#     pl = pv.Plotter(window_size=(750, 750))

#     pl.add_mesh(pvmesh,'Black', 'wireframe')

#     pl.show(cpos="xy")
# -

display(meshball.CoordinateSystem.rRotN)
display(meshball.CoordinateSystem.xRotN)

# +
# We can validate using the pure symbolic forms

gradPSs = meshball.vector.gradient(PSs)
divVSs = meshball.vector.divergence(VSs)
curlVSs = meshball.vector.curl(VSs)

sympy.simplify(meshball.vector.divergence(curlVSs))
# -

sympy.simplify(meshball.vector.gradient(divVSs))

gradPSs

divVSs


gradPs = meshball.vector.gradient(PS.sym)
gradPs

divVs = meshball.vector.divergence(VS.sym)
divVs

curlVs = meshball.vector.curl(VS.sym)
curlVs

meshball.vector.strain_tensor(VS.sym)

PS.sym.diff(meshball.CoordinateSystem.N[0])

PS.sym

sympy.simplify(meshball.vector.divergence(meshball.vector.gradient(P)))

meshball.N.base_vectors()
