# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw

import numpy as np
import sympy

# %%

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                 maxCoords=(1.0,1.0), 
                                 cellSize=1.0/32.0)


# %%
# mesh variables

t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3 )
dTdX   = uw.discretisation.MeshVariable("dTdX", mesh, 1, degree=2 )
gradT  = uw.discretisation.MeshVariable("gradT",mesh, mesh.dim, degree=2 )


# %%
# Create Poisson object


gradient = uw.systems.Projection(mesh, dTdX)
gradient.uw_function = sympy.diff(t_soln.sym, mesh.N.x)
gradient.smoothing = 1.0e-3

# These are both SNES Scalar objects

gradT_projector = uw.systems.Vector_Projection(mesh, gradT)
gradT_projector.uw_function = mesh.vector.gradient(t_soln.sym)

poisson = uw.systems.Poisson(mesh, u_Field=t_soln)


# %%
gradT_projector.uw_function

# %%
# Set some things
poisson.k = 1. 
poisson.f = 0.0
poisson.add_dirichlet_bc( 1., "Bottom" )  
poisson.add_dirichlet_bc( 0., "Top" )  

# %%
poisson.petsc_fe_u.view()

# %%
# Solve time
poisson.solve()

# %%
gradT_projector.solve()

# %%
gradient.petsc_fe_u.view()

# %%
gradient.uw_function = sympy.diff(t_soln.sym, mesh.N.x)
gradient.solve()

# %%
gradient.uw_function

# %%
# non-linear smoothing term (probably not needed especially at the boundary)

# gradient.uw_function = sympy.diff(t_soln.fn, mesh.N.y) 
# gradient.solve(_force_setup=True)

# %%
gradT_projector.solve()

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(gradient.u.sym[0], mesh.data)
    # if not np.allclose(mesh_numerical_soln, -1.0, rtol=0.01):
    #     raise RuntimeError("Unexpected values encountered.")

# %%
mesh_numerical_soln

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")
    
    with mesh.access():
        pvmesh.point_data["T"] = mesh_numerical_soln
        pvmesh.point_data["dTdy"] = uw.function.evaluate(gradient.u.fn, mesh.data) 

    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="dTdy",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
with mesh.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(sympy.sin(mesh.N.x * np.pi), poisson.u.coords)
    
gradient.solve()

# %%
uw.function.evaluate(gradient.u.sym[0], mesh.data)

# %%
uw.function.evaluate(sympy.sin(mesh.N.x * np.pi), poisson.u.coords)

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")


    with mesh.access():
        pvmesh.point_data["dTdy"] = uw.function.evaluate(gradient.u.fn-np.pi*sympy.cos(mesh.N.x*np.pi), mesh.data) 
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="dTdy",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
pvmesh.point_data["dTdy"].min(), pvmesh.point_data["dTdy"].max()

# %%

# %%
