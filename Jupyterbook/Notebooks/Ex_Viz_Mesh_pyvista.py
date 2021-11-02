# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np


# %%
# This (guy) sets up the visualisation defaults

import numpy as np
import pyvista as pv
import vtk

pv.global_theme.background = 'white'
pv.global_theme.window_size = [1000, 500]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = 'panel'
pv.global_theme.smooth_shading = True

# %%
minCoords=(-3.0,-1.0,-1.0)
maxCoords=( 1.0, 1.1, 1.2)

elementRes=(10,2,3)

qmesh2 = uw.mesh.Hex_Box(dim=3, 
                         elementRes=elementRes, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         cell_size=10.0)

pyvtk_check = qmesh2.mesh2pyvista().plot(show_edges=True)

# %%
minCoords=(-3.0,-1.0,-1.0)
maxCoords=( 1.0, 1.1, 1.2)

elementRes=(12,8,5)

s_s_mesh = uw.mesh.Simplex_Box(dim=3, 
                         elementRes=elementRes, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         cell_size=0.05)


s_s_mesh.mesh2pyvista().plot(show_edges=True)


# %%
minCoords=(-3.0,-1.0,-1.0)
maxCoords=( 1.0, 1.1, 1.2)

elementRes=(12,8,5)

u_s_mesh = uw.mesh.Unstructured_Simplex_Box(dim=3, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         coarse_cell_size=0.5,
                         global_cell_size=0.05,           
                                         )


u_s_mesh.mesh2pyvista().plot(show_edges=True)


# %%
minCoords=(-1.0, 0.0, 0.0)
maxCoords=( 1.0, 1.0, 0.0)
elementRes=(16,8,0)

u_s_2d_mesh = uw.mesh.Unstructured_Simplex_Box(dim=2, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         coarse_cell_size=0.2,
                         global_cell_size=0.01,           
                                         )

s_s_2d_mesh = uw.mesh.Simplex_Box(dim=2, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         elementRes=elementRes,
                         cell_size=0.01,           
                                         )

s_q_2d_mesh = uw.mesh.Hex_Box(dim=2, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         elementRes=elementRes,
                         cell_size=0.025,     # This is equivalent to the 0.01 of the simplex mesh
                                         )



# %%
u_s_2d_mesh.mesh2pyvista().plot(show_edges=True)


# %%
s_s_2d_mesh.mesh2pyvista().plot(show_edges=True)

# %%
s_q_2d_mesh.mesh2pyvista().plot(show_edges=True)

# %%
mesh_to_test = u_s_2d_mesh
viz_mesh = mesh_to_test.mesh2pyvista()

# %%
# Set up a field
dens_ball = 10.
dens_other = 1.
position_ball = 0.75*mesh_to_test.N.j
radius_ball = 0.1

# %%
# Create a density profile

import sympy
off_rvec = mesh_to_test.rvec - position_ball
abs_r = off_rvec.dot(off_rvec)
density = sympy.Piecewise( ( dens_ball,    abs_r < radius_ball**2 ),
                           ( dens_other,                   True ) )
density

# %%
# Write density into a variable for saving
densvar = uw.mesh.MeshVariable("density",mesh_to_test,1)
with mesh_to_test.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)

# %%
viz_mesh.point_data["density"] = uw.function.evaluate(density, viz_mesh.points )

pl1 = pv.Plotter()
pl1.add_mesh(viz_mesh, scalars="density", cmap="coolwarm", edge_color="Black", show_edges=True)
pl1.show()

# %%
# body force
unit_rvec = mesh_to_test.rvec / sympy.sqrt(mesh_to_test.rvec.dot(mesh_to_test.rvec))


# %%
## empty arrays
u_mesh_points = np.zeros((mesh_to_test.data.shape[0], 3))
u_mesh_vec = np.zeros_like(u_mesh_points)

u_mesh_points[:,0:2] = mesh_to_test.data 
u_mesh_vec[:,0:2] = uw.function.evaluate(unit_rvec, u_mesh_points[:,0:2] * 0.9999)


# %%
pl1 = pv.Plotter()
pl1.add_mesh(viz_mesh, scalars="density", cmap="coolwarm", edge_color="Black", show_edges=True)
pl1.add_points(u_mesh_points, point_size=3.0, color="White", render_points_as_spheres=True)
pl1.add_arrows(u_mesh_points, -u_mesh_vec, mag=0.025)

pl1.show()

# %%
## Bug that needs to be fixed

## The elements in a refined Hex mesh are not correctly ordered using the procedure that
## works for unrefined hexes, quads, tris and tets. It is an element ordering issue but 
## I am not sure how to 

minCoords=(-3.0,-1.0,-1.0)
maxCoords=( 1.0, 1.1, 1.2)

elementRes=(5,2,2)

qmesh2 = uw.mesh.Hex_Box(dim=3, 
                         elementRes=elementRes, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         cell_size=0.5)

qmesh2.mesh2pyvista().plot(show_edges=True)


# %%
