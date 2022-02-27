# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
## Checking that this runs / is sane. This is not a demo yet !!

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np

options = PETSc.Options()
# options["pc_type"]  = "svd"

# options["ksp_rtol"] = 1.0e-7
# # options["ksp_monitor_short"] = None

# # options["snes_type"]  = "fas"
# options["snes_converged_reason"] = None
# options["snes_monitor_short"] = None
# # options["snes_view"]=None
# options["snes_rtol"] = 1.0e-7

# %%
mesh = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0), 
                                          maxCoords=(1.0,1,0), cell_size=0.05) 

p_soln = uw.mesh.MeshVariable('P',  mesh, 1, degree=3 )
v_soln  = uw.mesh.MeshVariable('U', mesh, mesh.dim,  degree=2 )


# %%
# Create Poisson object
darcy = uw.systems.SteadyStateDarcy(mesh, u_Field=p_soln)

# %%
import sympy
sympy.MutableDenseNDimArray

# %%
# Set some things
darcy.k = 2.0+sympy.sin(mesh.N.x*sympy.pi)
darcy.f = 0.
darcy.add_dirichlet_bc( 1., "Bottom" )  
darcy.add_dirichlet_bc( 0., "Top" )  

darcy._setup_terms() # check this

# %%
darcy._f0

# %%
darcy._f1

# %%
darcy._G3

# %%
# Solve time
darcy.solve()

# %%
darcy_flux = sympy.vector.gradient(p_soln.fn)


# %%
# Fluxes

vector_projection = uw.systems.Vector_Projection(mesh, v_soln)
vector_projection.uw_function = darcy_flux
vector_projection.smoothing = 1.0e-3  # see how well it works !

vector_projection.solve()

# %%
with mesh.access():
    print(p_soln.data)

# %%
with mesh.access():
    print(v_soln.data)

# %%
vector_projection._f0

# %%
# time to plot it ... 

# %%

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pv.start_xvfb()
    
    pvmesh = mesh.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
    


    with mesh.access():
        usol = v_soln.data.copy()
  
    pvmesh.point_data["P"]  = uw.function.evaluate(p_soln.fn, mesh.data)
 
    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]
    
    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter()
   
    pl.add_arrows(arrow_loc, arrow_length, mag=0.05, opacity=0.75)


    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                  use_transparency=False, opacity=0.5, clim=[0.0,1.0])
    


    pl.show(cpos="xy")

# %%
