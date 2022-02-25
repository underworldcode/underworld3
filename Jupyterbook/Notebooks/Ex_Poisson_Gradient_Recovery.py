# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw

import numpy as np
import sympy

# %%
mesh = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0),
                                          maxCoords=(1.0,1,0), 
                                          cell_size=0.05,regular=False) 


# %%

# %%
# Create Poisson object
poisson = uw.systems.Poisson(mesh, degree=3)
gradient = uw.systems.Projection(mesh, degree=2)

# %%
# Set some things
poisson.k = 1. 
poisson.f = 0.0
poisson.add_dirichlet_bc( 1., "Bottom" )  
poisson.add_dirichlet_bc( 0., "Top" )  

# %%
# Solve time
poisson.solve()

# %%

# %%
gradient.F0 = gradient.u.fn - sympy.diff(poisson.u.fn, mesh.N.x)
gradient.solve()

# %%
# non-linear smoothing term (probably not needed especially at the boundary)
gradient.F0  = gradient.u.fn - sympy.diff(poisson.u.fn, mesh.N.y) 
# gradient.f += (gradient._L[0]**2 + gradient._L[1]**2) / 1000

gradient.solve(zero_init_guess=True)

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(gradient.u.fn, mesh.data)
    if not np.allclose(mesh_numerical_soln, -1.0, rtol=0.01):
        raise RuntimeError("Unexpected values encountered.")

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
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = mesh.mesh2pyvista()

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
with mesh.access(poisson.u):
    poisson.u.data[:,0] = uw.function.evaluate(sympy.sin(mesh.N.y*np.pi), poisson.u.coords)
    
gradient.f = gradient.u.fn - sympy.vector.gradient(poisson.u.fn).to_matrix(mesh.N)[1]
gradient.petsc_options["snes_rtol"] = 1.0e-8
gradient.petsc_options["ksp_rtol"] = 1.0e-8
gradient.solve()

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
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = mesh.mesh2pyvista()

    with mesh.access():
        pvmesh.point_data["T"]  = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln

        pvmesh.point_data["dTdy"] = uw.function.evaluate(gradient.u.fn-np.pi*sympy.cos(mesh.N.y*np.pi), mesh.data) 

        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="DT",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
pvmesh.point_data["dTdy"].min(), pvmesh.point_data["dTdy"].max()

# %%

# %%
