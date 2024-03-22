# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy
import meshio
import gmsh

import os

os.environ["SYMPY_USE_CACHE"] = "no"


# -


bubblemesh = uw.meshing.SegmentedSphere(
    radiusOuter=1.0,
    radiusInner=0.98,
    cellSize=0.1,
    numSegments=6, 
    qdegree=3, 
    filename="tmpWedge.msh",
    coordinatesNative=False,
)


bubblemesh.dm.view()

# ## Equation systems
#
# We have the surface mesh, two dimensional coordinates in lat, lon, and an appropriate coordinate system definition. We can now define variables etc. 
#
# Note, the Cartesian coords are 3D, the lat / lon are 2D
#

V = uw.discretisation.MeshVariable("U", bubblemesh, 3, degree=2)
P = uw.discretisation.MeshVariable("P", bubblemesh, 1, degree=1, continuous=False)
Pc = uw.discretisation.MeshVariable("Pc",bubblemesh, 1, degree=2)
T  = uw.discretisation.MeshVariable("T", bubblemesh, 1, degree=3)
Tnode = uw.discretisation.MeshVariable("Tn", bubblemesh, 1, degree=1)
Tdiff  = uw.discretisation.MeshVariable("dT", bubblemesh, 1, degree=1, continuous=False)
Tdiffc  = uw.discretisation.MeshVariable("dTc", bubblemesh, 1, degree=2, continuous=True)

r, lon,lat = bubblemesh.CoordinateSystem.R

# +
t_init = sympy.sympify(sympy.sin(6 * lon) * sympy.cos(5 * lat) * sympy.cos(lat))

with bubblemesh.access(T):
    T.data[:,0] = uw.function.evaluate(t_init, T.coords, bubblemesh.N)
# -

slope = sympy.sqrt(T.gradient().dot(T.gradient()))
slope

# ### SNES example
#
# The simplest possible solver implementation is just a projection operator

# +
projector = uw.systems.Projection(bubblemesh, Tnode)
projector.uw_function = T.sym[0]
projector.smoothing = 1.0e-3

options = projector.petsc_options
options.setValue("snes_rtol",1.0e-4)
# options.setValue("pc_gamg_type","agg")
# options.setValue("pc_gamg_agg_nsmooths", 1)
# options.setValue("pc_gamg_threshold", 0.25)

projector.solve()

# +
projector = uw.systems.Projection(bubblemesh, Tdiff)
projector.uw_function = slope # sympy.diff(T.sym[0], lon)        
projector.smoothing = 1.0e-6

projector.add_dirichlet_bc(0.0, "PoleAxisN", 0)
projector.add_dirichlet_bc(0.0, "PolePtNo", 0)
projector.add_dirichlet_bc(0.0, "PolePtNi", 0)
projector.add_dirichlet_bc(0.0, "PoleAxisS", 0)
projector.add_dirichlet_bc(0.0, "PolePtSo", 0)
projector.add_dirichlet_bc(0.0, "PolePtSi", 0)

options = projector.petsc_options
options.setValue("snes_rtol",1.0e-4)

projector.solve()
# +
projector = uw.systems.Projection(bubblemesh, Tdiffc)
projector.uw_function = slope # sympy.diff(T.sym[0], lon)        
projector.smoothing = 1.0e-6

projector.add_dirichlet_bc(0.0, "PoleAxisN", 0)
projector.add_dirichlet_bc(0.0, "PolePtNo", 0)
projector.add_dirichlet_bc(0.0, "PolePtNi", 0)
projector.add_dirichlet_bc(0.0, "PoleAxisS", 0)
projector.add_dirichlet_bc(0.0, "PolePtSo", 0)
projector.add_dirichlet_bc(0.0, "PolePtSi", 0)

options = projector.petsc_options
options.setValue("snes_rtol",1.0e-4)


projector.solve()
# -


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis
 
    pvmesh = vis.mesh_to_pv_mesh(bubblemesh)
    pvmesh.point_data["nT"] = vis.scalar_fn_to_pv_points(pvmesh, Tnode.sym)
    pvmesh.point_data["dT"] = vis.scalar_fn_to_pv_points(pvmesh, Tdiff.sym)
    pvmesh.point_data["dTc"] = vis.scalar_fn_to_pv_points(pvmesh, Tdiffc.sym)

    pl = pv.Plotter(window_size=(750, 750))
  
    pl.add_mesh(
        pvmesh,
        show_edges=True,
        scalars="dT",
        cmap="RdYlBu",
        opacity=1.0,  
        # clim=[0,1]
    )

    
    pl.add_axes(labels_off=False)


    pl.show(cpos="xy")

# +
## Below is some useful stuff for dmplex that we should put somewhere
# How to get element edges etc. 

# +
# pstart, pend = dmplex.getDepthStratum(0)
# estart, eend = dmplex.getDepthStratum(1)
# cstart, cend = dmplex.getDepthStratum(2)

# dmlonlat = dmplex.getCoordinates().array.reshape(-1,2)

# del_th_max = 0.0
# for i in range(estart, eend):
#     p1 , p2 = dmplex.getCone(i) - (pstart, pstart)
#     del_th = np.abs(dmlonlat[p1, 0] - dmlonlat[p2,0]) / np.pi
#     del_th_max  = max(del_th, del_th_max)
#     if np.sign(dmlonlat[p1, 0]) != np.sign(dmlonlat[p2, 0]) :
#         if del_th > 1:
#             print(f"(dt-> {del_th:.3f} ",
#                   f"{dmlonlat[p1, 0]:+4.2f} <-> {dmlonlat[p2, 0]:+4.2f} ",
#                   f"{xyz[p1,2]}"
#                  )

# del_th   

# +
# for cell in range(cstart, cend):
#     points = set()
#     edges = dmplex.getCone(cell)
#     for edge in edges:
#         points = points.union(dmplex.getCone(edge)-pstart)
    
#     print( f"{cell}",
#            f"{dmlonlat[tuple(points),0]/np.pi}, {xyz[tuple(points),2]}"
#          )


# +
# Some dmplex options

# https://petsc.org/release/src/dm/impls/plex/plexgmsh.c.html

# options = PETSc.Options()
# options["dm_plex_gmsh_multiple_tags"] = None
# options["dm_plex_gmsh_spacedim"] = 2
# options["dm_plex_gmsh_use_regions"] = None
# options["dm_plex_gmsh_mark_vertices"] = None
