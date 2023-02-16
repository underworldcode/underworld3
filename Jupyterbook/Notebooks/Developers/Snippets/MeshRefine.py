# +
## Mesh refinement ... 

import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing


import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# +
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547
res = 500 / 6730 

mesh0 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_low_r.msh")

r = uw.discretisation.MeshVariable("R", mesh0, 1)
U = uw.discretisation.MeshVariable("U", mesh0, mesh0.dim, degree=2)
# -

ds = mesh0.dm.getDS()


ds.getNumFields()

# +
# mesh0.CoordinateSystem.R[0]
# with mesh0.access(r):
#     r.data[:,0] = uw.function.evaluate(mesh0.CoordinateSystem.R[0], mesh0.data, mesh0.N)
# -


mesh0.dm.view()
dm1 = mesh0.dm

mesh1 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_low_r1.msh",
                           refinement=2)

mesh1.dm.view()

# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    mesh1.vtk("tmp_meshball.vtk")
    pvmesh = pv.read("tmp_meshball.vtk")
    


# -


pvmesh.points *= 0.999

# +
pl = pv.Plotter(window_size=[1000, 1000])
pl.add_axes()

pl.add_mesh(
    pvmesh, 
    cmap="coolwarm",
    clim=[0.997, 1.0],
    edge_color="Black",
    style="surface",
    show_edges=True,
)

pl.add_mesh(
    pvmesh0, 
    edge_color="Blue",
    style="wireframe",
    color="Blue", 
    render_lines_as_tubes=True,
)


# pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
# OR
pl.show(cpos="xy")
# -




