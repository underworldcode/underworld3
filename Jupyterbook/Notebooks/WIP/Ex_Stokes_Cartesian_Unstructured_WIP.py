# %% [markdown]
# # Cartesian Stokes, Unstructured Mesh
#
# This demonstrates the box solve for 2d / 3d using unstructured tetrahedral meshes
# and comparison with structured tetrahedra. This is the sinking (isoviscous) ball.
#
# Meshes are generated using gmsh / pygmsh and visualised with pyvista 
#
# ## Yet to implement
#
#  - Boundary conditions are all no-slip 
#
#  - Boundary labels on unstructured meshing has not been implemented
#
#  - FAS in SNES does not work automatically because there is no automatic coarsening pathway that is set up when a mesh is refined. I expect that can be figured out by trawling through the C examples !

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] =  1.0e-5
options["ksp_monitor_short"] = None
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_max_it"] = 1
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_factorization_type"] ="full"
options["pc_fieldsplit_schur_precondition"] = "a11"
options["fieldsplit_velocity_pc_type"] = "lu"
options["fieldsplit_pressure_ksp_rtol"] = 1.e-5
options["fieldsplit_pressure_pc_type"] = "lu"

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

elementRes=(12,8,5)

s_s_mesh = uw.discretisation.Simplex_Box(dim=3, 
                         elementRes=elementRes, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         cell_size=0.05)


s_s_mesh.mesh2pyvista().plot(show_edges=True)


# %%
minCoords=(-3.0,-1.0,-1.0)
maxCoords=( 1.0, 1.1, 1.2)

elementRes=(12,8,5)

u_s_mesh = uw.discretisation.Unstructured_Simplex_Box(dim=3, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         coarse_cell_size=0.25,
                         global_cell_size=0.1,           
                                         )


u_s_mesh.mesh2pyvista().plot(show_edges=True)


# %%
minCoords=(-1.0, 0.0, 0.0)
maxCoords=( 1.0, 1.0, 0.0)
elementRes=(16,8,0)

u_s_2d_mesh = uw.discretisation.Unstructured_Simplex_Box(dim=2, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         coarse_cell_size=0.2,
                         global_cell_size=0.01,           
                                         )

s_s_2d_mesh = uw.discretisation.Simplex_Box(dim=2, 
                         minCoords=minCoords, 
                         maxCoords=maxCoords,
                         elementRes=elementRes,
                         cell_size=0.01,           
                                         )

# %%
u_s_2d_mesh.mesh2pyvista().plot(show_edges=True)


# %%
s_s_2d_mesh.mesh2pyvista().plot(show_edges=True)

# %%
mesh_to_test = s_s_2d_mesh
viz_mesh = mesh_to_test.mesh2pyvista()

# %%
# Create Stokes object
stokes = Stokes(mesh_to_test, u_degree=2, p_degree=1)
# Constant visc
stokes.viscosity = 1.
# No slip boundary conditions
stokes.add_dirichlet_bc( (0.,0.), mesh_to_test.boundary.ALL_BOUNDARIES, (0,1) )

# %%
stokes.u.fn

# %%
stokes.bcs

# %%
# Set more some things
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
densvar = uw.discretisation.MeshVariable("density",mesh_to_test,1)
with mesh_to_test.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)

# %%
viz_mesh.point_data["density"] = uw.function.evaluate(density, viz_mesh.points )

pl1 = pv.Plotter()
pl1.add_mesh(viz_mesh, scalars="density", cmap="coolwarm", edge_color="Black", show_edges=True)
pl1.show()

# %%

# %%
# body force
unit_rvec = mesh_to_test.rvec / sympy.sqrt(mesh_to_test.rvec.dot(mesh_to_test.rvec))
stokes.bodyforce = -unit_rvec*density
stokes.bodyforce

# %%
# Solve time
stokes.solve()

# %%
umag = stokes.u.fn.dot(stokes.u.fn) 
viz_mesh.point_data["umag"] = uw.function.evaluate(umag, viz_mesh.points[:,0:2]*0.9999999 )

## empty arrays
u_mesh_points = np.zeros((mesh_to_test.data.shape[0], 3))
u_mesh_vec = np.zeros_like(u_mesh_points)

u_mesh_points[:,0:2] = mesh_to_test.data 
u_mesh_vec[:,0:2] = uw.function.evaluate(stokes.u.fn, u_mesh_points[:,0:2] * 0.9999)

p_data = pv.PolyData(u_mesh_points)
p_data["umag"] = uw.function.evaluate(umag, mesh_to_test.data*0.9999 )

# %%
viz_mesh.points[:,0:2]


uw.function.evaluate(stokes.u.fn, np.array([(0.0,0.0),]))
# viz_mesh.point_data["umag"] = uw.function.evaluate(umag,
#                                                    viz_mesh.points[:,0:2]*0.9999999 )


# %%
pl1 = pv.Plotter()
pl1.add_mesh(viz_mesh, scalars="umag", cmap="coolwarm", edge_color="Black", show_edges=True)
# pl1.add_points(p_data, point_size=3.0, scalars="umag", cmap="coolwarm")
# pl1.add_points(u_mesh_points, point_size=3.0, color="Black", render_points_as_spheres=True)
pl1.add_arrows(u_mesh_points, u_mesh_vec, mag=3.0)

pl1.show()

# %%
## Strange bug 





# %%
