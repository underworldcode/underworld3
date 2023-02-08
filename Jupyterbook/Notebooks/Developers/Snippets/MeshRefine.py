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
# mesh0.dm.computeGradientClementInterpolant()
# -

0/0

dm_r1 = mesh0.dm.refine()
dm_r2 = dm_r1.refine()

dms = mesh0.dm.refineHierarchy(2)

# +
dm2 = dm_r2

c2 = dm2.getCoordinatesLocal()
coords = c2.array.reshape(-1,3)
Rdm2 = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2 )

upperIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(dm2, "Upper")
coords[upperIndices] *= r_o / Rdm2[upperIndices].reshape(-1,1)

lowerIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(dm2, "Lower")
coords[lowerIndices] *= r_i / Rdm2[lowerIndices].reshape(-1,1)

c2.array[...] = coords.reshape(-1)
dm2.setCoordinatesLocal(c2)

# -





0/0

# +
from underworld3.coordinates import CoordinateSystemType
mesh2 = uw.discretisation.Mesh(dm2, coordinate_system_type=CoordinateSystemType.SPHERICAL)

r2 = uw.discretisation.MeshVariable("R", mesh2, 1)

with mesh2.access(r2):
    r2.data[:,0] = Rdm2[:]

# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    mesh2.vtk("tmp_meshball.vtk")
    pvmesh = pv.read("tmp_meshball.vtk")
    
    with mesh2.access():
        pvmesh.point_data["R"] = r2.data.copy()
        

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
    scalars="R",
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




